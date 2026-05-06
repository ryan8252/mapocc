import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
import logging

from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class MapToOccLayoutAdapter(BaseModule):
    """Build a masked map-layout residual for selected OCC semantic queries."""

    def __init__(self,
                 in_channels,
                 query_channels,
                 num_map_classes=6,
                 num_occ_queries=18,
                 target_occ_indices=(11, 13, 14, 15),
                 relation_mask=None,
                 map_prior_detach=True,
                 eps=1e-4,
                 init_alpha=0.0,
                 relation_init_std=0.02,
                 gate_mode='source_only',
                 gate_hidden_channels=None,
                 debug=False,
                 debug_max_calls=10,
                 init_cfg=None):
        super(MapToOccLayoutAdapter, self).__init__(init_cfg)
        self.in_channels = int(in_channels)
        self.query_channels = int(query_channels)
        self.num_map_classes = int(num_map_classes)
        self.num_occ_queries = int(num_occ_queries)
        self.map_prior_detach = bool(map_prior_detach)
        self.eps = float(eps)
        self.gate_mode = str(gate_mode)
        self.debug = bool(debug)
        self.debug_max_calls = int(debug_max_calls)
        self._debug_calls = 0
        if self.gate_mode != 'source_only':
            raise NotImplementedError(
                'MapToOccLayoutAdapter currently implements only '
                'gate_mode="source_only". Use a separate Stage 5 ablation '
                'for target-aware gating.')

        target_occ_indices = [int(index) for index in target_occ_indices]
        if not target_occ_indices:
            raise ValueError('target_occ_indices must not be empty.')
        if len(set(target_occ_indices)) != len(target_occ_indices):
            raise ValueError('target_occ_indices must not contain duplicates.')
        if any(index < 0 or index >= self.num_occ_queries
               for index in target_occ_indices):
            raise ValueError(
                'target_occ_indices contains an index outside '
                f'[0, {self.num_occ_queries}).')
        self.register_buffer(
            'target_occ_indices',
            torch.tensor(target_occ_indices, dtype=torch.long),
            persistent=False)
        target_selector = torch.zeros(
            len(target_occ_indices), self.num_occ_queries)
        target_selector[
            torch.arange(len(target_occ_indices)),
            torch.tensor(target_occ_indices, dtype=torch.long)] = 1.0
        self.register_buffer(
            'target_selector', target_selector, persistent=False)

        if relation_mask is None:
            relation_mask = torch.ones(
                self.num_map_classes, len(target_occ_indices))
        else:
            relation_mask = torch.as_tensor(relation_mask, dtype=torch.float32)
        expected_shape = (self.num_map_classes, len(target_occ_indices))
        if tuple(relation_mask.shape) != expected_shape:
            raise ValueError(
                'relation_mask must have shape '
                f'{expected_shape}, got {tuple(relation_mask.shape)}.')
        if torch.any(relation_mask.sum(dim=0) <= 0):
            raise ValueError(
                'Each target OCC query must have at least one allowed '
                'map-class relation.')
        self.register_buffer(
            'relation_mask', relation_mask.bool(), persistent=False)

        self.channel_proj = nn.Linear(self.in_channels, self.query_channels)
        self.relation = nn.Parameter(
            torch.empty(self.num_map_classes, len(target_occ_indices)))
        gate_hidden_channels = gate_hidden_channels or self.query_channels
        self.source_gate = nn.Sequential(
            nn.Linear(self.query_channels, gate_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_channels, 1))
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        nn.init.xavier_uniform_(self.channel_proj.weight)
        nn.init.zeros_(self.channel_proj.bias)
        nn.init.normal_(self.relation, mean=0.0, std=float(relation_init_std))
        for module in self.source_gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _masked_average_pool(self, map_feature, map_prob):
        if map_feature.shape[-2:] != map_prob.shape[-2:]:
            map_prob = F.interpolate(
                map_prob,
                size=map_feature.shape[-2:],
                mode='bilinear',
                align_corners=False)

        weight_sum = map_prob.flatten(2).sum(dim=-1)
        weighted_feature = torch.einsum('bchw,bmhw->bmc',
                                        map_feature, map_prob)
        token = weighted_feature / weight_sum.clamp_min(self.eps).unsqueeze(-1)
        valid = weight_sum > self.eps
        token = token * valid.unsqueeze(-1).to(token.dtype)
        return token, valid

    def forward(self, map_feature, map_logits):
        if map_feature.dim() != 4:
            raise ValueError(
                'map_feature must have shape [B, C, H, W], got '
                f'{tuple(map_feature.shape)}.')
        if map_logits.dim() != 4:
            raise ValueError(
                'map_logits must have shape [B, M, H, W], got '
                f'{tuple(map_logits.shape)}.')
        if map_feature.size(0) != map_logits.size(0):
            raise ValueError(
                'map_feature and map_logits batch sizes must match: '
                f'{map_feature.size(0)} vs {map_logits.size(0)}.')
        if map_feature.size(1) != self.in_channels:
            raise ValueError(
                f'Expected map_feature channels={self.in_channels}, got '
                f'{map_feature.size(1)}.')
        if map_logits.size(1) != self.num_map_classes:
            raise ValueError(
                f'Expected {self.num_map_classes} map logits, got '
                f'{map_logits.size(1)}.')

        if self.map_prior_detach:
            map_feature = map_feature.detach()
            map_logits = map_logits.detach()

        map_prob = map_logits.sigmoid()
        raw_tokens, valid_tokens = self._masked_average_pool(
            map_feature, map_prob)
        map_tokens = self.channel_proj(raw_tokens)

        relation = self.relation * self.relation_mask.to(
            dtype=self.relation.dtype)
        delta = torch.einsum('bmd,mk->bkd', map_tokens, relation)

        source_valid = torch.einsum(
            'bm,mk->bk',
            valid_tokens.to(delta.dtype),
            self.relation_mask.to(delta.dtype)) > 0
        gate = torch.sigmoid(self.source_gate(delta))
        delta = delta * gate * source_valid.unsqueeze(-1).to(delta.dtype)
        delta = self.alpha.to(delta.dtype) * delta

        query_residual = torch.einsum(
            'bkd,kq->bqd',
            delta,
            self.target_selector.to(dtype=delta.dtype, device=delta.device))

        if self.debug and self._debug_calls < self.debug_max_calls:
            logger = logging.getLogger('mmdet')
            with torch.no_grad():
                raw_token_norm = raw_tokens.detach().norm(dim=-1).mean(dim=0)
                map_token_norm = map_tokens.detach().norm(dim=-1).mean(dim=0)
                delta_norm = delta.detach().norm(dim=-1).mean(dim=0)
                residual_norm = query_residual.detach().norm(dim=-1).mean(dim=0)
                gate_mean = gate.detach().mean(dim=0).flatten()
                valid_ratio = source_valid.detach().float().mean(dim=0)
            logger.warning(
                '[LGMG adapter debug] call=%d alpha=%.6f '
                'delta_mean_norm=%.6f residual_mean_norm=%.6f '
                'raw_token_norm=%s map_token_norm=%s delta_norm=%s '
                'gate_mean=%s source_valid_ratio=%s',
                self._debug_calls,
                float(self.alpha.detach()),
                float(delta.detach().norm(dim=-1).mean()),
                float(query_residual.detach().norm(dim=-1).mean()),
                [round(float(x), 6) for x in raw_token_norm],
                [round(float(x), 6) for x in map_token_norm],
                [round(float(x), 6) for x in delta_norm],
                [round(float(x), 6) for x in gate_mean],
                [round(float(x), 6) for x in valid_ratio])
            self._debug_calls += 1

        return query_residual
