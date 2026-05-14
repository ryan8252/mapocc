"""CFV Prototype-TSFG map fusion.

This module routes detached ProtoOcc CFV/background prototype semantics into
the protected 128-channel map neck as a scheduled residual prior.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class CFVPrototypeTSFGMapFusion(BaseModule):
    """Background-query-guided CFV-to-map residual adapter."""

    requires_pqd_query_info = True

    def __init__(self,
                 voxel_in_channels=48,
                 query_channels=48,
                 map_channels=128,
                 occ_num_classes=18,
                 background_class_ids=(11, 12, 13, 14, 15, 16),
                 z_collapse_mode='avg_max_concat',
                 detach_cfv=True,
                 detach_query=True,
                 query_source='scene_aware',
                 query_to_filter='pqd_mask_embed_detached',
                 prototype_mlp_hidden_channels=128,
                 mlp_proto_init='residual_zero',
                 normalize_similarity=True,
                 activation_mode='sigmoid_cosine',
                 temperature=10.0,
                 c_gate_mode='prototype',
                 fusion_mode='residual_two_layer',
                 residual=True,
                 zero_init_delta=True,
                 gamma_schedule=None,
                 initial_gamma=1.0,
                 refine_kernel_size=1,
                 debug=False,
                 debug_interval=100,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if z_collapse_mode not in ('avg', 'max', 'avg_max_concat'):
            raise ValueError(
                "z_collapse_mode must be one of 'avg', 'max', "
                f"'avg_max_concat'; got {z_collapse_mode!r}")
        if activation_mode not in ('sigmoid_cosine',
                                   'backgroundness_softmax_over_k'):
            raise ValueError(
                "activation_mode must be 'sigmoid_cosine' or "
                f"'backgroundness_softmax_over_k'; got {activation_mode!r}")
        if c_gate_mode not in ('none', 'prototype', 'cfv', 'hybrid'):
            raise ValueError(
                "c_gate_mode must be one of 'none', 'prototype', 'cfv', "
                f"'hybrid'; got {c_gate_mode!r}")
        if fusion_mode not in ('scalar', 'residual_two_layer',
                               'gated_residual', 'direct'):
            raise ValueError(
                "fusion_mode must be one of 'scalar', 'residual_two_layer', "
                f"'gated_residual', 'direct'; got {fusion_mode!r}")
        if query_to_filter not in ('pqd_mask_embed_detached',
                                   'local_residual_mlp',
                                   'local_mlp',
                                   'local_random_mlp'):
            raise ValueError(
                "query_to_filter must be 'pqd_mask_embed_detached', "
                "'local_residual_mlp', 'local_mlp', or "
                f"'local_random_mlp'; got {query_to_filter!r}")
        if refine_kernel_size not in (1, 3):
            raise ValueError(
                f'refine_kernel_size must be 1 or 3; got {refine_kernel_size}')

        self.voxel_in_channels = voxel_in_channels
        self.query_channels = query_channels
        self.map_channels = map_channels
        self.occ_num_classes = occ_num_classes
        self.background_class_ids = list(background_class_ids)
        self.z_collapse_mode = z_collapse_mode
        self.detach_cfv = detach_cfv
        self.detach_query = detach_query
        self.query_source = query_source
        self.query_to_filter = query_to_filter
        self.mlp_proto_init = mlp_proto_init
        self.normalize_similarity = normalize_similarity
        self.activation_mode = activation_mode
        self.temperature = float(temperature)
        self.c_gate_mode = c_gate_mode
        self.fusion_mode = fusion_mode
        self.residual = residual
        self.gamma_schedule = gamma_schedule
        self.gamma = float(initial_gamma)
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

        if query_to_filter != 'pqd_mask_embed_detached':
            self.query_to_filter_mlp = self._build_query_to_filter_mlp(
                prototype_mlp_hidden_channels,
                zero_init=(query_to_filter == 'local_residual_mlp' or
                           mlp_proto_init == 'residual_zero'))
        else:
            self.query_to_filter_mlp = None

        num_bg = len(self.background_class_ids)
        refine_in_channels = voxel_in_channels + num_bg
        refine_padding = refine_kernel_size // 2
        self.refine = nn.Sequential(
            nn.Conv3d(refine_in_channels,
                      voxel_in_channels,
                      kernel_size=refine_kernel_size,
                      padding=refine_padding,
                      bias=False),
            nn.BatchNorm3d(voxel_in_channels),
            nn.ReLU(inplace=True),
        )

        z_collapsed_channels = (
            2 * voxel_in_channels
            if z_collapse_mode == 'avg_max_concat'
            else voxel_in_channels)
        self.project_to_map = nn.Sequential(
            nn.Conv2d(z_collapsed_channels, map_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(map_channels),
            nn.ReLU(inplace=True),
        )

        self.prototype_gate = None
        self.cfv_gate = None
        self.hybrid_gate = None
        if c_gate_mode == 'prototype':
            self.prototype_gate = self._build_gate_mlp(
                2 * voxel_in_channels,
                prototype_mlp_hidden_channels,
                voxel_in_channels)
        elif c_gate_mode == 'cfv':
            self.cfv_gate = self._build_gate_mlp(
                2 * voxel_in_channels,
                prototype_mlp_hidden_channels,
                voxel_in_channels)
        elif c_gate_mode == 'hybrid':
            self.hybrid_gate = self._build_gate_mlp(
                4 * voxel_in_channels,
                prototype_mlp_hidden_channels,
                voxel_in_channels)

        fused_in_channels = map_channels * 2
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

        self.gate = None
        if fusion_mode == 'gated_residual':
            self.gate = nn.Sequential(
                nn.Conv2d(fused_in_channels, map_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(map_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(map_channels, map_channels,
                          kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

    def _build_query_to_filter_mlp(self, hidden_channels, zero_init=False):
        mlp = nn.Sequential(
            nn.Linear(self.query_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, self.voxel_in_channels),
        )
        if zero_init:
            nn.init.zeros_(mlp[-1].weight)
            if mlp[-1].bias is not None:
                nn.init.zeros_(mlp[-1].bias)
        return mlp

    def _build_gate_mlp(self, in_channels, hidden_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )
        nn.init.zeros_(mlp[-1].weight)
        if mlp[-1].bias is not None:
            nn.init.zeros_(mlp[-1].bias)
        return mlp

    def set_gamma(self, value):
        self.gamma = float(value)

    def _select_background(self, class_tensor, name):
        if class_tensor is None:
            raise ValueError(f'{name} is required by CFVPrototypeTSFGMapFusion.')
        if class_tensor.dim() != 3:
            raise ValueError(
                f'{name} must have shape [B, num_classes, C]; got '
                f'{tuple(class_tensor.shape)}.')
        if class_tensor.shape[1] < self.occ_num_classes:
            raise ValueError(
                f'{name} has {class_tensor.shape[1]} classes, expected at '
                f'least {self.occ_num_classes}.')
        return class_tensor[:, self.background_class_ids, :]

    def _make_filters(self, query_embed_real_bqc, mask_embed_real_bqc):
        if self.query_to_filter == 'pqd_mask_embed_detached':
            p_bg = self._select_background(
                mask_embed_real_bqc, 'mask_embed_real_bqc')
            return p_bg.detach()

        query = self._select_background(
            query_embed_real_bqc, 'query_embed_real_bqc')
        if self.detach_query:
            query = query.detach()
        projected = self.query_to_filter_mlp(query)
        if (self.query_to_filter == 'local_residual_mlp' and
                query.shape[-1] == self.voxel_in_channels):
            return query + projected
        return projected

    def _prototype_gate(self, p_bg):
        p_avg = p_bg.mean(dim=1)
        p_max = p_bg.max(dim=1).values
        return 1.0 + self.prototype_gate(torch.cat([p_avg, p_max], dim=-1))

    def _cfv_gate(self, cfv):
        cfv_avg = cfv.mean(dim=(2, 3, 4))
        cfv_max = cfv.amax(dim=(2, 3, 4))
        return 1.0 + self.cfv_gate(torch.cat([cfv_avg, cfv_max], dim=-1))

    def _hybrid_gate(self, p_bg, cfv):
        p_avg = p_bg.mean(dim=1)
        p_max = p_bg.max(dim=1).values
        cfv_avg = cfv.mean(dim=(2, 3, 4))
        cfv_max = cfv.amax(dim=(2, 3, 4))
        gate_input = torch.cat([p_avg, p_max, cfv_avg, cfv_max], dim=-1)
        return 1.0 + self.hybrid_gate(gate_input)

    def _apply_channel_gate(self, cfv, p_bg):
        if self.c_gate_mode == 'none':
            return cfv
        if self.c_gate_mode == 'prototype':
            gate = self._prototype_gate(p_bg)
        elif self.c_gate_mode == 'cfv':
            gate = self._cfv_gate(cfv)
        else:
            gate = self._hybrid_gate(p_bg, cfv)
        return cfv * gate[:, :, None, None, None]

    def _activation(self, cfv, p_bg):
        if self.normalize_similarity:
            cfv = F.normalize(cfv, dim=1)
            p_bg = F.normalize(p_bg, dim=-1)
        sim = torch.einsum('bcxyz,bnc->bnxyz', cfv, p_bg)
        scaled = self.temperature * sim
        if self.activation_mode == 'sigmoid_cosine':
            return torch.sigmoid(scaled)
        class_competition = F.softmax(scaled, dim=1)
        backgroundness = torch.sigmoid(scaled.max(dim=1, keepdim=True).values)
        return backgroundness * class_competition

    def _z_collapse(self, feature):
        if self.z_collapse_mode == 'avg':
            return feature.mean(dim=-1)
        if self.z_collapse_mode == 'max':
            return feature.max(dim=-1).values
        return torch.cat([
            feature.mean(dim=-1),
            feature.max(dim=-1).values,
        ], dim=1)

    def _log_debug(self, p_bg, activation, delta, map_feature):
        if not self.debug:
            return
        self._debug_step += 1
        if self._debug_step % self.debug_interval != 0:
            return
        logger = logging.getLogger('mmdet')
        with torch.no_grad():
            delta_norm = delta.abs().mean()
            map_norm = map_feature.abs().mean().clamp_min(1e-6)
            logger.info(
                '[CFVProtoTSFG] gamma=%.4f p_bg_norm=%.4f '
                'A[min/mean/max]=%.4f/%.4f/%.4f '
                'delta_abs=%.6f scaled_delta/map=%.6f',
                self.gamma,
                p_bg.norm(dim=-1).mean().item(),
                activation.min().item(),
                activation.mean().item(),
                activation.max().item(),
                delta_norm.item(),
                (abs(self.gamma) * delta_norm / map_norm).item())

    def forward(self,
                map_feature,
                voxel_feature,
                query_embed_real_bqc=None,
                mask_embed_real_bqc=None,
                **kwargs):
        if voxel_feature.dim() != 5:
            raise ValueError(
                'CFVPrototypeTSFGMapFusion expects voxel_feature shape '
                f'[B, C, X, Y, Z]; got {tuple(voxel_feature.shape)}.')
        if map_feature.dim() != 4:
            raise ValueError(
                'CFVPrototypeTSFGMapFusion expects map_feature shape '
                f'[B, C, H, W]; got {tuple(map_feature.shape)}.')
        if voxel_feature.shape[1] != self.voxel_in_channels:
            raise ValueError(
                f'voxel_feature has {voxel_feature.shape[1]} channels, '
                f'expected {self.voxel_in_channels}.')
        if map_feature.shape[1] != self.map_channels:
            raise ValueError(
                f'map_feature has {map_feature.shape[1]} channels, '
                f'expected {self.map_channels}.')

        cfv_source = voxel_feature.detach() if self.detach_cfv else voxel_feature
        p_bg = self._make_filters(query_embed_real_bqc, mask_embed_real_bqc)
        activation = self._activation(cfv_source, p_bg)
        aware_cfv = self._apply_channel_gate(cfv_source, p_bg)

        refined = self.refine(torch.cat([activation, aware_cfv], dim=1))
        proto_cfv = self.project_to_map(self._z_collapse(refined))
        if proto_cfv.shape[-2:] != map_feature.shape[-2:]:
            proto_cfv = F.interpolate(
                proto_cfv,
                size=map_feature.shape[-2:],
                mode='bilinear',
                align_corners=False)

        gamma = map_feature.new_tensor(self.gamma)
        if self.fusion_mode == 'scalar':
            delta = proto_cfv
        else:
            fused_in = torch.cat([map_feature, proto_cfv], dim=1)
            delta = self.fusion(fused_in)
            if self.fusion_mode == 'gated_residual':
                delta = self.gate(fused_in) * delta
            elif self.fusion_mode == 'direct':
                self._log_debug(p_bg, activation, delta, map_feature)
                return delta

        self._log_debug(p_bg, activation, delta, map_feature)
        if self.residual:
            return map_feature + gamma * delta
        return gamma * delta
