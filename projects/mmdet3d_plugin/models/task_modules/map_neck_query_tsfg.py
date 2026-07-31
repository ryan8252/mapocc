"""Map-neck Query-TSFG adapter.

Use detached ProtoOcc PQD background scene-aware queries as prototype signals
directly on the protected 128-channel map BEV feature.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class MapNeckQueryTSFG(BaseModule):
    """Background-query-guided map feature generation.

    The module keeps the map-specific neck as the main path and adds a
    zero-initialized residual delta conditioned on PQD background queries.
    """

    requires_pqd_query_info = True

    def __init__(self,
                 query_channels=48,
                 map_channels=128,
                 occ_num_classes=18,
                 background_class_ids=(11, 12, 13, 14, 15, 16),
                 prototype_source='query_norm_real_bqc',
                 detach_query=True,
                 prototype_hidden_channels=128,
                 similarity_mode='cosine',
                 use_prototype_wise=True,
                 use_prototype_aware=True,
                 residual=True,
                 fixed_gamma=1.0,
                 zero_init_delta=True,
                 zero_init_gate=True,
                 debug=False,
                 debug_interval=100,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if prototype_source not in (
                'query_norm_real_bqc',
                'query_embed_real_bqc',
                'mask_embed_real_bqc'):
            raise ValueError(
                "prototype_source must be 'query_norm_real_bqc', "
                "'query_embed_real_bqc', or 'mask_embed_real_bqc'; got "
                f'{prototype_source!r}.')
        if similarity_mode not in ('cosine', 'dot'):
            raise ValueError(
                "similarity_mode must be 'cosine' or 'dot'; got "
                f'{similarity_mode!r}.')
        if not use_prototype_wise and not use_prototype_aware:
            raise ValueError(
                'At least one of use_prototype_wise/use_prototype_aware '
                'must be True.')

        self.query_channels = query_channels
        self.map_channels = map_channels
        self.occ_num_classes = occ_num_classes
        self.background_class_ids = list(background_class_ids)
        self.prototype_source = prototype_source
        self.detach_query = detach_query
        self.similarity_mode = similarity_mode
        self.use_prototype_wise = use_prototype_wise
        self.use_prototype_aware = use_prototype_aware
        self.residual = residual
        self.fixed_gamma = float(fixed_gamma)
        self.debug = debug
        self.debug_interval = max(int(debug_interval), 1)
        self._debug_step = 0

        if not self.background_class_ids:
            raise ValueError('background_class_ids cannot be empty.')
        if min(self.background_class_ids) < 0:
            raise ValueError('background_class_ids must be non-negative.')
        if max(self.background_class_ids) >= occ_num_classes:
            raise ValueError(
                f'background_class_ids {self.background_class_ids} exceed '
                f'occ_num_classes={occ_num_classes}.')

        num_bg = len(self.background_class_ids)
        self.prototype_proj = nn.Sequential(
            nn.Linear(query_channels, prototype_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(prototype_hidden_channels, map_channels),
        )

        self.gate_mlp = None
        if use_prototype_aware:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * map_channels, prototype_hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(prototype_hidden_channels, map_channels),
            )
            if zero_init_gate:
                nn.init.zeros_(self.gate_mlp[-1].weight)
                if self.gate_mlp[-1].bias is not None:
                    nn.init.zeros_(self.gate_mlp[-1].bias)

        fused_in_channels = map_channels
        if use_prototype_aware:
            fused_in_channels += map_channels
        if use_prototype_wise:
            fused_in_channels += num_bg

        self.fusion = nn.Sequential(
            nn.Conv2d(fused_in_channels, map_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(map_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(map_channels, map_channels, kernel_size=1, bias=True),
        )
        if zero_init_delta:
            nn.init.zeros_(self.fusion[-1].weight)
            if self.fusion[-1].bias is not None:
                nn.init.zeros_(self.fusion[-1].bias)

    def _get_query_source(self,
                          query_norm_real_bqc=None,
                          query_embed_real_bqc=None,
                          mask_embed_real_bqc=None):
        source_map = dict(
            query_norm_real_bqc=query_norm_real_bqc,
            query_embed_real_bqc=query_embed_real_bqc,
            mask_embed_real_bqc=mask_embed_real_bqc,
        )
        query = source_map[self.prototype_source]
        if query is None:
            raise ValueError(
                f'{self.prototype_source} is required by MapNeckQueryTSFG.')
        if query.dim() != 3:
            raise ValueError(
                f'{self.prototype_source} must have shape [B, C_cls, C]; '
                f'got {tuple(query.shape)}.')
        if query.shape[1] < self.occ_num_classes:
            raise ValueError(
                f'{self.prototype_source} has {query.shape[1]} classes, '
                f'expected at least {self.occ_num_classes}.')
        if query.shape[2] != self.query_channels:
            raise ValueError(
                f'{self.prototype_source} has {query.shape[2]} channels, '
                f'expected {self.query_channels}.')
        return query[:, self.background_class_ids, :]

    def _make_background_prototypes(self, query_bg):
        query_source = query_bg.detach() if self.detach_query else query_bg
        return self.prototype_proj(query_source)

    def _prototype_response(self, p_bg_128, map_feature):
        if not self.use_prototype_wise:
            return None
        if self.similarity_mode == 'cosine':
            p_sim = F.normalize(p_bg_128, dim=-1)
            f_sim = F.normalize(map_feature, dim=1)
        else:
            p_sim = p_bg_128
            f_sim = map_feature
        return torch.einsum('bnc,bchw->bnhw', p_sim, f_sim)

    def _prototype_aware_feature(self, p_bg_128, map_feature):
        if not self.use_prototype_aware:
            return None, None
        p_avg = p_bg_128.mean(dim=1)
        p_max = p_bg_128.max(dim=1).values
        gate = 1.0 + self.gate_mlp(torch.cat([p_avg, p_max], dim=-1))
        return map_feature * gate[:, :, None, None], gate

    def _offdiag_stats(self, matrix):
        n = matrix.shape[-1]
        if n <= 1:
            nan = matrix.new_tensor(float('nan'))
            return nan, nan
        mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
        values = matrix[:, mask]
        return values.mean(), values.max()

    def _response_correlation_stats(self, response):
        if response is None:
            nan = torch.tensor(float('nan'))
            return nan, nan
        b, n, h, w = response.shape
        if n <= 1:
            nan = response.new_tensor(float('nan'))
            return nan, nan
        flat = response.reshape(b, n, h * w)
        flat = flat - flat.mean(dim=-1, keepdim=True)
        flat = F.normalize(flat, dim=-1)
        corr = torch.einsum('bnv,bmv->bnm', flat, flat)
        return self._offdiag_stats(corr)

    def _log_debug(self, query_bg, response, gate, delta, map_feature):
        if not self.debug:
            return
        self._debug_step += 1
        if self._debug_step % self.debug_interval != 0:
            return

        logger = logging.getLogger('mmdet')
        with torch.no_grad():
            p_norm = F.normalize(query_bg.detach(), dim=-1)
            query_cos = torch.einsum('bnc,bmc->bnm', p_norm, p_norm)
            query_offdiag_mean, query_offdiag_max = self._offdiag_stats(
                query_cos)
            resp_corr_mean, resp_corr_max = self._response_correlation_stats(
                response)

            if response is None:
                a_min = a_mean = a_max = a_std = delta.new_tensor(
                    float('nan'))
            else:
                a_min = response.min()
                a_mean = response.mean()
                a_max = response.max()
                a_std = response.std()

            if gate is None:
                gate_delta = delta.new_tensor(float('nan'))
            else:
                gate_delta = (gate - 1.0).abs().mean()

            delta_abs_mean = delta.abs().mean()
            delta_abs_max = delta.abs().max()
            fmap_abs_mean = map_feature.abs().mean().clamp_min(1e-6)
            logger.info(
                '[MapNeckQueryTSFG] source=%s sim=%s '
                'query_cos_offdiag(mean/max)=%.4f/%.4f '
                'A[min/mean/max/std]=%.4f/%.4f/%.4f/%.4f '
                'A_corr_offdiag(mean/max)=%.4f/%.4f '
                'gate_minus_1_abs=%.6f delta_abs(mean/max)=%.6f/%.6f '
                'fmap_abs=%.6f delta/fmap=%.6f',
                self.prototype_source,
                self.similarity_mode,
                query_offdiag_mean.item(),
                query_offdiag_max.item(),
                a_min.item(),
                a_mean.item(),
                a_max.item(),
                a_std.item(),
                resp_corr_mean.item(),
                resp_corr_max.item(),
                gate_delta.item(),
                delta_abs_mean.item(),
                delta_abs_max.item(),
                fmap_abs_mean.item(),
                (delta_abs_mean / fmap_abs_mean).item())

    def forward(self,
                map_feature,
                voxel_feature=None,
                query_norm_real_bqc=None,
                query_embed_real_bqc=None,
                mask_embed_real_bqc=None,
                **kwargs):
        if map_feature.dim() != 4:
            raise ValueError(
                'MapNeckQueryTSFG expects map_feature shape [B, C, H, W]; '
                f'got {tuple(map_feature.shape)}.')
        if map_feature.shape[1] != self.map_channels:
            raise ValueError(
                f'map_feature has {map_feature.shape[1]} channels, '
                f'expected {self.map_channels}.')

        query_bg = self._get_query_source(
            query_norm_real_bqc=query_norm_real_bqc,
            query_embed_real_bqc=query_embed_real_bqc,
            mask_embed_real_bqc=mask_embed_real_bqc)
        p_bg_128 = self._make_background_prototypes(query_bg)

        response = self._prototype_response(p_bg_128, map_feature)
        aware_feature, gate = self._prototype_aware_feature(
            p_bg_128, map_feature)

        fused_inputs = [map_feature]
        if aware_feature is not None:
            fused_inputs.append(aware_feature)
        if response is not None:
            fused_inputs.append(response)
        delta = self.fusion(torch.cat(fused_inputs, dim=1))

        self._log_debug(query_bg, response, gate, delta, map_feature)

        gamma = map_feature.new_tensor(self.fixed_gamma)
        if self.residual:
            return map_feature + gamma * delta
        return gamma * delta
