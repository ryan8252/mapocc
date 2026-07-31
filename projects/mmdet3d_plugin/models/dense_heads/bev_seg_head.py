import logging
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.runner import BaseModule

from mmdet3d.models.builder import HEADS, build_loss


def _zero_init_conv(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class _BEVSegResBlock(nn.Module):
    def __init__(self, channels, norm_cfg=dict(type='BN')):
        super(_BEVSegResBlock, self).__init__()
        self.conv1 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv2.apply(_zero_init_conv)

    def forward(self, x):
        return F.relu(x + self.conv2(self.conv1(x)), inplace=True)


class _BEVSegConvNeXtBlock(nn.Module):
    def __init__(self,
                 channels,
                 expansion=4,
                 kernel_size=7,
                 norm_cfg=dict(type='BN')):
        super(_BEVSegConvNeXtBlock, self).__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels)
        if norm_cfg is None:
            self.norm = nn.Identity()
        else:
            self.norm = build_norm_layer(norm_cfg, channels)[1]
        self.pointwise1 = nn.Conv2d(channels, channels * expansion, 1)
        self.act = nn.GELU()
        self.pointwise2 = nn.Conv2d(channels * expansion, channels, 1)
        _zero_init_conv(self.pointwise2)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise2(self.act(self.pointwise1(x)))
        return residual + x


@HEADS.register_module()
class BEVSegHead(BaseModule):
    """Lightweight BEV segmentation head for naive multi-task learning.

    This mirrors the common BEVFusion-style setup: shared BEV features are fed
    into a shallow 2D CNN decoder that predicts one binary mask per map class.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=128,
                 num_classes=6,
                 num_convs=2,
                 bevseg_head_type='simple',
                 bevseg_num_refine_blocks=2,
                 use_dual_area_line_head=False,
                 area_class_indices=None,
                 line_class_indices=None,
                 line_head_use_detail=False,
                 line_detail_channels=0,
                 use_map_aux_loss=False,
                 map_aux_levels=('100', '50'),
                 map_aux_weight_100=0.4,
                 map_aux_weight_50=0.2,
                 map_aux_downsample='maxpool',
                 map_aux_in_channels=None,
                 map_aux_hidden_channels=None,
                 use_thin_boundary_aux_loss=False,
                 thin_boundary_class_indices=None,
                 thin_boundary_dilation=2,
                 thin_boundary_loss_weight=0.2,
                 thin_boundary_use_focal=True,
                 thin_boundary_use_dice=True,
                 use_thin_roi_refinement=False,
                 thin_roi_class_indices=None,
                 thin_roi_threshold=0.35,
                 thin_roi_min_pixels=64,
                 thin_roi_dilation=2,
                 thin_roi_hidden_channels=None,
                 use_map_active_enhance_gate=False,
                 map_active_gate_channels=1,
                 map_active_gate_hidden_channels=None,
                 map_active_gate_class_groups=None,
                 map_active_gate_dilations=0,
                 map_active_gate_beta=0.5,
                 map_active_gate_group_betas=None,
                 map_active_gate_loss_weight=0.2,
                 map_active_gate_use_focal=True,
                 map_active_gate_use_dice=True,
                 map_active_gate_init_bias=-4.0,
                 use_query_refinement=False,
                 query_refine_num_bins=5,
                 query_refine_heads=8,
                 query_refine_ffn_ratio=2,
                 query_refine_attn_dropout=0.0,
                 query_refine_alpha_init=0.0,
                 query_refine_detach_logits=True,
                 query_refine_scale_dot=True,
                 use_occ2map_active_gate_prior=False,
                 occ2map_prior_channels=2,
                 occ2map_prior_zero_init=True,
                 occ2map_prior_gate_scale=1.0,
                 occ2map_prior_gate_channel_mask=None,
                 map_loss_type=None,
                 map_dice_weight=1.0,
                 map_focal_gamma=2.0,
                 map_focal_alpha=0.25,
                 map_focal_loss_mode='shared',
                 map_focal_loss_class_names=None,
                 map_lovasz_weight=1.0,
                 use_map_class_weights=False,
                 use_bev_coordconv=False,
                 bev_coord_type='xy',
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 loss_bce=None,
                 loss_dice=None,
                 loss_focal=None,
                 map_loss_balance_mode='none',
                 map_class_weights=None,
                 overlay_class_indices=None,
                 overlay_pos_weight=None,
                 dynamic_overlay_ref_pos_ratio=None,
                 dynamic_overlay_gamma=0.5,
                 dynamic_overlay_min_weight=1.0,
                 dynamic_overlay_max_weight=5.0,
                 dynamic_overlay_eps=1e-6,
                 map_balance_debug=False,
                 map_balance_debug_interval=50,
                 map_balance_class_names=None):
        super(BEVSegHead, self).__init__()
        self.with_cp = with_cp
        self.num_classes = num_classes
        self.bevseg_head_type = self._normalize_head_type(bevseg_head_type)
        self.bevseg_num_refine_blocks = int(bevseg_num_refine_blocks)
        self.use_dual_area_line_head = bool(use_dual_area_line_head)
        self.line_head_use_detail = bool(line_head_use_detail)
        self.line_detail_channels = int(line_detail_channels or 0)
        self.use_map_aux_loss = bool(use_map_aux_loss)
        self.map_aux_levels = self._parse_aux_levels(map_aux_levels)
        self.map_aux_weights = {
            '100': float(map_aux_weight_100),
            '50': float(map_aux_weight_50),
        }
        self.map_aux_downsample = self._normalize_aux_downsample(
            map_aux_downsample)
        self.use_thin_boundary_aux_loss = bool(use_thin_boundary_aux_loss)
        self.thin_boundary_class_indices = self._parse_optional_class_indices(
            thin_boundary_class_indices, 'thin_boundary_class_indices')
        self.thin_boundary_dilation = int(thin_boundary_dilation)
        self.thin_boundary_loss_weight = float(thin_boundary_loss_weight)
        self.thin_boundary_use_focal = bool(thin_boundary_use_focal)
        self.thin_boundary_use_dice = bool(thin_boundary_use_dice)
        self.use_thin_roi_refinement = bool(use_thin_roi_refinement)
        self.thin_roi_class_indices = self._parse_optional_class_indices(
            thin_roi_class_indices, 'thin_roi_class_indices')
        self.thin_roi_threshold = float(thin_roi_threshold)
        self.thin_roi_min_pixels = int(thin_roi_min_pixels)
        self.thin_roi_dilation = int(thin_roi_dilation)
        self.thin_roi_hidden_channels = (
            None if thin_roi_hidden_channels is None
            else int(thin_roi_hidden_channels))
        self.use_map_active_enhance_gate = bool(use_map_active_enhance_gate)
        self.map_active_gate_channels = int(map_active_gate_channels)
        self.map_active_gate_hidden_channels = (
            None if map_active_gate_hidden_channels is None
            else int(map_active_gate_hidden_channels))
        self.map_active_gate_beta = float(map_active_gate_beta)
        self.map_active_gate_group_betas = self._parse_active_gate_group_betas(
            map_active_gate_group_betas)
        self.map_active_gate_loss_weight = float(map_active_gate_loss_weight)
        self.map_active_gate_use_focal = bool(map_active_gate_use_focal)
        self.map_active_gate_use_dice = bool(map_active_gate_use_dice)
        self.map_active_gate_init_bias = float(map_active_gate_init_bias)
        self.use_query_refinement = bool(use_query_refinement)
        self.query_refine_num_bins = int(query_refine_num_bins)
        self.query_refine_heads = int(query_refine_heads)
        self.query_refine_ffn_ratio = int(query_refine_ffn_ratio)
        self.query_refine_attn_dropout = float(query_refine_attn_dropout)
        self.query_refine_detach_logits = bool(query_refine_detach_logits)
        self.query_refine_scale_dot = bool(query_refine_scale_dot)
        self.use_occ2map_active_gate_prior = bool(
            use_occ2map_active_gate_prior)
        self.occ2map_prior_channels = int(occ2map_prior_channels)
        self.occ2map_prior_zero_init = bool(occ2map_prior_zero_init)
        self.occ2map_prior_gate_scale = float(occ2map_prior_gate_scale)
        self.occ2map_prior_gate_channel_mask = (
            None if occ2map_prior_gate_channel_mask is None
            else [float(v) for v in occ2map_prior_gate_channel_mask])
        self.map_active_gate_class_groups = self._parse_active_gate_groups(
            map_active_gate_class_groups)
        self.map_active_gate_dilations = self._parse_active_gate_dilations(
            map_active_gate_dilations)
        self.map_loss_type = self._normalize_map_loss_type(map_loss_type)
        self.use_map_lovasz = self.map_loss_type == 'focal_lovasz'
        self.map_lovasz_weight = float(map_lovasz_weight)
        self.map_focal_loss_mode = self._normalize_focal_loss_mode(
            map_focal_loss_mode)
        self.use_map_class_weights = bool(use_map_class_weights)
        self.use_bev_coordconv = bool(use_bev_coordconv)
        self.bev_coord_type = self._normalize_coord_type(bev_coord_type)
        self.map_loss_balance_mode = self._normalize_balance_mode(
            map_loss_balance_mode)
        self.map_class_weights = self._parse_class_weights(map_class_weights)
        if (self.use_map_class_weights
                and self.map_loss_balance_mode == 'none'):
            self.map_loss_balance_mode = 'class_static'
        self.overlay_class_indices = self._parse_overlay_indices(
            overlay_class_indices)
        self.overlay_pos_weight = self._parse_overlay_values(
            overlay_pos_weight, 'overlay_pos_weight', default=None)
        self.dynamic_overlay_ref_pos_ratio = self._parse_overlay_values(
            dynamic_overlay_ref_pos_ratio,
            'dynamic_overlay_ref_pos_ratio',
            default=None)
        self.dynamic_overlay_gamma = float(dynamic_overlay_gamma)
        self.dynamic_overlay_min_weight = float(dynamic_overlay_min_weight)
        self.dynamic_overlay_max_weight = float(dynamic_overlay_max_weight)
        self.dynamic_overlay_eps = float(dynamic_overlay_eps)
        self.map_balance_debug = bool(map_balance_debug)
        self.map_balance_debug_interval = int(map_balance_debug_interval)
        self.map_balance_class_names = self._parse_class_names(
            map_balance_class_names, 'map_balance_class_names')
        self.map_focal_loss_class_names = self._parse_class_names(
            map_focal_loss_class_names, 'map_focal_loss_class_names')
        self._map_balance_debug_step = 0
        self._validate_thin_cfg()
        self._validate_active_gate_cfg()
        self._validate_balance_cfg()
        loss_bce, loss_dice, loss_focal = self._resolve_map_loss_cfgs(
            loss_bce,
            loss_dice,
            loss_focal,
            map_dice_weight,
            map_focal_gamma,
            map_focal_alpha)

        if self.use_bev_coordconv:
            coord_channels = self._num_coord_channels()
            self.coord_proj = ConvModule(
                in_channels + coord_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True))
        else:
            self.coord_proj = None

        if self.use_dual_area_line_head:
            self.area_class_indices = self._parse_head_indices(
                area_class_indices, 'area_class_indices')
            self.line_class_indices = self._parse_head_indices(
                line_class_indices, 'line_class_indices')
            self._validate_dual_head_indices()
            if self.line_head_use_detail and self.line_detail_channels <= 0:
                raise ValueError(
                    'line_head_use_detail=True requires '
                    'line_detail_channels > 0.')

            self.decoder = None
            self.predictor = None
            self.area_decoder, area_channels = self._build_decoder(
                in_channels,
                hidden_channels,
                num_convs,
                norm_cfg)
            line_in_channels = in_channels
            if self.line_head_use_detail:
                line_in_channels += self.line_detail_channels
            self.line_decoder, line_channels = self._build_decoder(
                line_in_channels,
                hidden_channels,
                num_convs,
                norm_cfg)
            self.area_predictor = nn.Conv2d(
                area_channels, len(self.area_class_indices), kernel_size=1)
            self.line_predictor = nn.Conv2d(
                line_channels, len(self.line_class_indices), kernel_size=1)
        else:
            self.area_class_indices = []
            self.line_class_indices = []
            self.decoder, current_channels = self._build_decoder(
                in_channels,
                hidden_channels,
                num_convs,
                norm_cfg)
            self.predictor = nn.Conv2d(
                current_channels, num_classes, kernel_size=1)
            if self.use_map_active_enhance_gate:
                gate_hidden = (
                    current_channels
                    if self.map_active_gate_hidden_channels is None
                    else self.map_active_gate_hidden_channels)
                self.map_active_gate = nn.Sequential(
                    ConvModule(
                        current_channels,
                        gate_hidden,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    nn.Conv2d(
                        gate_hidden,
                        self.map_active_gate_channels,
                        kernel_size=1))
                nn.init.constant_(
                    self.map_active_gate[-1].bias,
                    self.map_active_gate_init_bias)
                if self.use_occ2map_active_gate_prior:
                    self.occ2map_gate_proj = nn.Conv2d(
                        self.occ2map_prior_channels,
                        self.map_active_gate_channels,
                        kernel_size=1)
                    if self.occ2map_prior_zero_init:
                        self.occ2map_gate_proj.apply(_zero_init_conv)
                else:
                    self.occ2map_gate_proj = None
            else:
                self.map_active_gate = None
                self.occ2map_gate_proj = None
            if self.use_thin_roi_refinement:
                roi_hidden = (
                    current_channels if self.thin_roi_hidden_channels is None
                    else self.thin_roi_hidden_channels)
                self.thin_roi_refiner = nn.Sequential(
                    ConvModule(
                        current_channels,
                        roi_hidden,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    nn.Conv2d(
                        roi_hidden,
                        len(self.thin_roi_class_indices),
                        kernel_size=1))
                self.thin_roi_refiner[-1].apply(_zero_init_conv)
            else:
                self.thin_roi_refiner = None

        self.query_refine_class_embed = None
        self.query_refine_bin_embed = None
        self.query_refine_query_norm = None
        self.query_refine_attn = None
        self.query_refine_attn_norm = None
        self.query_refine_ffn = None
        self.query_refine_ffn_norm = None
        self.query_refine_mask_embed = None
        if self.use_query_refinement:
            if self.use_dual_area_line_head:
                raise ValueError(
                    'use_query_refinement=True currently supports only the '
                    'single BEVSegHead decoder path.')
            if self.query_refine_num_bins <= 0:
                raise ValueError('query_refine_num_bins must be positive.')
            if self.query_refine_heads <= 0:
                raise ValueError('query_refine_heads must be positive.')
            if current_channels % self.query_refine_heads != 0:
                raise ValueError(
                    'query_refine_heads must divide the BEVSegHead decoder '
                    f'channels: {self.query_refine_heads} vs '
                    f'{current_channels}.')
            ffn_channels = current_channels * max(self.query_refine_ffn_ratio, 1)
            self.query_refine_class_embed = nn.Embedding(
                num_classes, current_channels)
            self.query_refine_bin_embed = nn.Embedding(
                self.query_refine_num_bins, current_channels)
            self.query_refine_query_norm = nn.LayerNorm(current_channels)
            self.query_refine_attn = nn.MultiheadAttention(
                current_channels,
                self.query_refine_heads,
                dropout=self.query_refine_attn_dropout,
                batch_first=True)
            self.query_refine_attn_norm = nn.LayerNorm(current_channels)
            self.query_refine_ffn = nn.Sequential(
                nn.Linear(current_channels, ffn_channels),
                nn.ReLU(inplace=True),
                nn.Linear(ffn_channels, current_channels))
            self.query_refine_ffn_norm = nn.LayerNorm(current_channels)
            self.query_refine_mask_embed = nn.Sequential(
                nn.Linear(current_channels, current_channels),
                nn.ReLU(inplace=True),
                nn.Linear(current_channels, current_channels),
                nn.ReLU(inplace=True),
                nn.Linear(current_channels, current_channels))
            self.query_refine_alpha = nn.Parameter(
                torch.tensor(float(query_refine_alpha_init)))
        else:
            self.register_parameter('query_refine_alpha', None)

        self.map_aux_heads = nn.ModuleDict()
        if self.use_map_aux_loss:
            aux_in_channels = self._parse_aux_in_channels(
                map_aux_in_channels, in_channels)
            aux_hidden_channels = (
                hidden_channels if map_aux_hidden_channels is None
                else int(map_aux_hidden_channels))
            for level in self.map_aux_levels:
                self.map_aux_heads[level] = nn.Sequential(
                    ConvModule(
                        aux_in_channels[level],
                        aux_hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    nn.Conv2d(
                        aux_hidden_channels, num_classes, kernel_size=1))

        self.loss_bce = build_loss(loss_bce) if loss_bce is not None else None
        self.loss_dice = build_loss(loss_dice) if loss_dice is not None else None
        self.loss_focal = build_loss(loss_focal) if loss_focal is not None else None

    def _normalize_map_loss_type(self, loss_type):
        if loss_type is None:
            return None
        loss_type = str(loss_type).lower()
        aliases = {
            'ce': 'bce',
            'sigmoid_bce': 'bce',
            'dice_bce': 'bce_dice',
            'focal+dice': 'focal_dice',
            'bce+dice': 'bce_dice',
        }
        loss_type = aliases.get(loss_type, loss_type)
        valid_types = {'bce', 'bce_dice', 'focal', 'focal_dice',
                       'focal_lovasz'}
        if loss_type not in valid_types:
            raise ValueError(
                f'Unsupported map_loss_type: {loss_type}. '
                f'Expected one of {sorted(valid_types)}.')
        return loss_type

    def _normalize_focal_loss_mode(self, mode):
        if mode is None:
            return 'shared'
        mode = str(mode).lower()
        aliases = {
            'default': 'shared',
            'global': 'shared',
            'bevfusion': 'classwise_sum',
            'classwise': 'classwise_sum',
            'classwise_bevfusion': 'classwise_sum',
            'classwise_avg': 'classwise_mean',
            'classwise_average': 'classwise_mean',
        }
        mode = aliases.get(mode, mode)
        valid_modes = {'shared', 'classwise_sum', 'classwise_mean'}
        if mode not in valid_modes:
            raise ValueError(
                f'Unsupported map_focal_loss_mode: {mode}. '
                f'Expected one of {sorted(valid_modes)}.')
        return mode

    def _resolve_map_loss_cfgs(self,
                               loss_bce,
                               loss_dice,
                               loss_focal,
                               map_dice_weight,
                               map_focal_gamma,
                               map_focal_alpha):
        if self.map_loss_type is None:
            return loss_bce, loss_dice, loss_focal

        bce_cfg = dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0)
        dice_cfg = dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            loss_weight=float(map_dice_weight))
        focal_cfg = dict(
            type='BinaryMaskFocalLoss',
            use_sigmoid=True,
            gamma=float(map_focal_gamma),
            alpha=float(map_focal_alpha),
            reduction='mean',
            loss_weight=1.0)

        loss_bce = bce_cfg if self.map_loss_type in ('bce', 'bce_dice') else None
        loss_dice = dice_cfg if self.map_loss_type in ('bce_dice', 'focal_dice') else None
        loss_focal = (
            focal_cfg
            if self.map_loss_type in ('focal', 'focal_dice', 'focal_lovasz')
            else None)
        return loss_bce, loss_dice, loss_focal

    def _parse_aux_levels(self, levels):
        if levels is None:
            return []
        levels = [str(level) for level in levels]
        valid_levels = {'100', '50'}
        for level in levels:
            if level not in valid_levels:
                raise ValueError(
                    f'Unsupported map aux level: {level}. '
                    f'Expected one of {sorted(valid_levels)}.')
        if len(set(levels)) != len(levels):
            raise ValueError('map_aux_levels must not contain duplicates.')
        return levels

    def _normalize_aux_downsample(self, downsample):
        downsample = 'maxpool' if downsample is None else str(downsample).lower()
        valid = {'maxpool', 'nearest'}
        if downsample not in valid:
            raise ValueError(
                f'Unsupported map_aux_downsample: {downsample}. '
                f'Expected one of {sorted(valid)}.')
        return downsample

    def _parse_aux_in_channels(self, aux_in_channels, in_channels):
        default_channels = int(in_channels) * 2
        if aux_in_channels is None:
            return {level: default_channels for level in self.map_aux_levels}
        if isinstance(aux_in_channels, int):
            return {level: int(aux_in_channels) for level in self.map_aux_levels}
        parsed = {}
        for level in self.map_aux_levels:
            if level not in aux_in_channels:
                raise ValueError(
                    f'map_aux_in_channels must define level {level}.')
            parsed[level] = int(aux_in_channels[level])
        return parsed

    def _normalize_coord_type(self, coord_type):
        coord_type = 'xy' if coord_type is None else str(coord_type).lower()
        valid_types = {'xy', 'xyr', 'xyrtheta'}
        if coord_type not in valid_types:
            raise ValueError(
                f'Unsupported bev_coord_type: {coord_type}. '
                f'Expected one of {sorted(valid_types)}.')
        return coord_type

    def _num_coord_channels(self):
        if self.bev_coord_type == 'xy':
            return 2
        if self.bev_coord_type == 'xyr':
            return 3
        return 4

    def _normalize_head_type(self, head_type):
        head_type = 'simple' if head_type is None else str(head_type).lower()
        valid_types = {'simple', 'res_refine', 'convnext'}
        if head_type not in valid_types:
            raise ValueError(
                f'Unsupported bevseg_head_type: {head_type}. '
                f'Expected one of {sorted(valid_types)}.')
        return head_type

    def _build_decoder(self,
                       in_channels,
                       hidden_channels,
                       num_convs,
                       norm_cfg):
        blocks = []
        current_channels = in_channels
        if self.bevseg_head_type == 'simple':
            for _ in range(num_convs):
                blocks.append(
                    ConvModule(
                        current_channels,
                        hidden_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ))
                current_channels = hidden_channels
            return nn.Sequential(*blocks), current_channels

        blocks.append(
            ConvModule(
                current_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True),
            ))
        current_channels = hidden_channels
        for _ in range(self.bevseg_num_refine_blocks):
            if self.bevseg_head_type == 'res_refine':
                blocks.append(_BEVSegResBlock(current_channels, norm_cfg))
            else:
                blocks.append(_BEVSegConvNeXtBlock(current_channels, norm_cfg=norm_cfg))
        return nn.Sequential(*blocks), current_channels

    def _parse_head_indices(self, indices, name):
        if indices is None:
            raise ValueError(
                f'{name} must be set when use_dual_area_line_head=True.')
        indices = [int(index) for index in indices]
        if not indices:
            raise ValueError(f'{name} must not be empty.')
        for index in indices:
            if index < 0 or index >= self.num_classes:
                raise ValueError(
                    f'{name} contains out-of-range index {index} for '
                    f'num_classes={self.num_classes}.')
        if len(set(indices)) != len(indices):
            raise ValueError(f'{name} must not contain duplicate indices.')
        return indices

    def _parse_optional_class_indices(self, indices, name):
        if indices is None:
            return []
        indices = [int(index) for index in indices]
        for index in indices:
            if index < 0 or index >= self.num_classes:
                raise ValueError(
                    f'{name} contains out-of-range index {index} for '
                    f'num_classes={self.num_classes}.')
        if len(set(indices)) != len(indices):
            raise ValueError(f'{name} must not contain duplicate indices.')
        return indices

    def _parse_active_gate_groups(self, groups):
        if groups is None:
            if self.map_active_gate_channels == 1:
                return [list(range(self.num_classes))]
            raise ValueError(
                'map_active_gate_class_groups must be set when '
                'map_active_gate_channels > 1.')
        parsed = []
        for group in groups:
            indices = [int(index) for index in group]
            if not indices:
                raise ValueError(
                    'map_active_gate_class_groups must not contain an empty '
                    'group.')
            for index in indices:
                if index < 0 or index >= self.num_classes:
                    raise ValueError(
                        'map_active_gate_class_groups contains '
                        f'out-of-range index {index} for '
                        f'num_classes={self.num_classes}.')
            if len(set(indices)) != len(indices):
                raise ValueError(
                    'map_active_gate_class_groups must not contain duplicate '
                    'indices inside one group.')
            parsed.append(indices)
        return parsed

    def _parse_active_gate_dilations(self, dilations):
        if isinstance(dilations, int):
            return [int(dilations) for _ in range(self.map_active_gate_channels)]
        return [int(dilation) for dilation in dilations]

    def _parse_active_gate_group_betas(self, group_betas):
        if group_betas is None:
            return [self.map_active_gate_beta
                    for _ in range(self.map_active_gate_channels)]
        try:
            return [float(beta) for beta in group_betas]
        except (TypeError, ValueError) as exc:
            raise TypeError(
                'map_active_gate_group_betas must be a sequence of numbers.'
            ) from exc

    def _validate_thin_cfg(self):
        if (self.use_thin_boundary_aux_loss
                and not self.thin_boundary_class_indices):
            raise ValueError(
                'use_thin_boundary_aux_loss=True requires '
                'thin_boundary_class_indices.')
        if self.thin_boundary_dilation < 0:
            raise ValueError('thin_boundary_dilation must be >= 0.')
        if self.thin_boundary_loss_weight < 0:
            raise ValueError('thin_boundary_loss_weight must be >= 0.')
        if self.use_thin_roi_refinement:
            if self.use_dual_area_line_head:
                raise ValueError(
                    'use_thin_roi_refinement=True currently supports only '
                    'the single BEVSegHead decoder path.')
            if not self.thin_roi_class_indices:
                raise ValueError(
                    'use_thin_roi_refinement=True requires '
                    'thin_roi_class_indices.')
        if self.thin_roi_min_pixels < 0:
            raise ValueError('thin_roi_min_pixels must be >= 0.')
        if self.thin_roi_dilation < 0:
            raise ValueError('thin_roi_dilation must be >= 0.')

    def _validate_active_gate_cfg(self):
        if (self.use_occ2map_active_gate_prior
                and not self.use_map_active_enhance_gate):
            raise ValueError(
                'use_occ2map_active_gate_prior=True requires '
                'use_map_active_enhance_gate=True.')
        if not self.use_map_active_enhance_gate:
            return
        if self.use_dual_area_line_head:
            raise ValueError(
                'use_map_active_enhance_gate=True currently supports only '
                'the single BEVSegHead decoder path.')
        if self.map_active_gate_channels <= 0:
            raise ValueError('map_active_gate_channels must be positive.')
        if len(self.map_active_gate_class_groups) != self.map_active_gate_channels:
            raise ValueError(
                'map_active_gate_class_groups length must match '
                'map_active_gate_channels.')
        if len(self.map_active_gate_dilations) != self.map_active_gate_channels:
            raise ValueError(
                'map_active_gate_dilations length must match '
                'map_active_gate_channels.')
        if len(self.map_active_gate_group_betas) != self.map_active_gate_channels:
            raise ValueError(
                'map_active_gate_group_betas length must match '
                'map_active_gate_channels.')
        if any(dilation < 0 for dilation in self.map_active_gate_dilations):
            raise ValueError('map_active_gate_dilations must be >= 0.')
        if (not math.isfinite(self.map_active_gate_beta)
                or self.map_active_gate_beta < 0):
            raise ValueError('map_active_gate_beta must be finite and >= 0.')
        if any(not math.isfinite(beta) or beta < 0
               for beta in self.map_active_gate_group_betas):
            raise ValueError(
                'map_active_gate_group_betas must be finite and >= 0.')
        if self.map_active_gate_loss_weight < 0:
            raise ValueError('map_active_gate_loss_weight must be >= 0.')
        if self.use_occ2map_active_gate_prior:
            if self.occ2map_prior_channels <= 0:
                raise ValueError('occ2map_prior_channels must be positive.')
            if self.occ2map_prior_gate_scale < 0:
                raise ValueError('occ2map_prior_gate_scale must be >= 0.')
            if (self.occ2map_prior_gate_channel_mask is not None
                    and len(self.occ2map_prior_gate_channel_mask)
                    != self.map_active_gate_channels):
                raise ValueError(
                    'occ2map_prior_gate_channel_mask length must match '
                    'map_active_gate_channels.')

    def _validate_dual_head_indices(self):
        all_indices = self.area_class_indices + self.line_class_indices
        if len(set(all_indices)) != len(all_indices):
            raise ValueError(
                'area_class_indices and line_class_indices must not overlap.')
        expected = set(range(self.num_classes))
        if set(all_indices) != expected:
            raise ValueError(
                'area_class_indices and line_class_indices must cover every '
                f'map class exactly once; got {sorted(all_indices)} for '
                f'num_classes={self.num_classes}.')

    def _normalize_balance_mode(self, mode):
        if mode is None:
            return 'none'
        mode = str(mode).lower()
        aliases = {
            'off': 'none',
            'disable': 'none',
            'disabled': 'none',
            'static': 'class_static',
            'overlay': 'overlay_static',
            'dynamic': 'overlay_dynamic',
            'v3': 'overlay_dynamic',
        }
        mode = aliases.get(mode, mode)
        valid_modes = {'none', 'class_static', 'overlay_static',
                       'overlay_dynamic'}
        if mode not in valid_modes:
            raise ValueError(
                f'Unsupported map_loss_balance_mode: {mode}. '
                f'Expected one of {sorted(valid_modes)}.')
        return mode

    def _parse_class_weights(self, weights):
        if weights is None:
            return None
        weights = [float(weight) for weight in weights]
        if len(weights) != self.num_classes:
            raise ValueError(
                'map_class_weights length must match num_classes: '
                f'{len(weights)} vs {self.num_classes}.')
        return weights

    def _parse_overlay_indices(self, indices):
        if indices is None:
            return []
        indices = [int(index) for index in indices]
        for index in indices:
            if index < 0 or index >= self.num_classes:
                raise ValueError(
                    'overlay_class_indices contains an out-of-range index: '
                    f'{index} for num_classes={self.num_classes}.')
        if len(set(indices)) != len(indices):
            raise ValueError('overlay_class_indices must not contain duplicates.')
        return indices

    def _parse_overlay_values(self, values, name, default=None):
        if not self.overlay_class_indices:
            if values is not None:
                raise ValueError(
                    f'{name} requires overlay_class_indices to be set.')
            return None
        if values is None:
            if default is None:
                return None
            return [float(default) for _ in self.overlay_class_indices]
        if isinstance(values, (int, float)):
            return [float(values) for _ in self.overlay_class_indices]

        values = [float(value) for value in values]
        if len(values) == len(self.overlay_class_indices):
            return values
        if len(values) == self.num_classes:
            return [values[index] for index in self.overlay_class_indices]
        raise ValueError(
            f'{name} must be a scalar, have length '
            f'{len(self.overlay_class_indices)} for overlay classes, or have '
            f'length {self.num_classes} for all map classes.')

    def _parse_class_names(self, class_names, cfg_name='class_names'):
        if class_names is None:
            return [f'class_{index}' for index in range(self.num_classes)]
        class_names = list(class_names)
        if len(class_names) != self.num_classes:
            raise ValueError(
                f'{cfg_name} length must match num_classes: '
                f'{len(class_names)} vs {self.num_classes}.')
        return class_names

    def _validate_balance_cfg(self):
        mode = self.map_loss_balance_mode
        if mode == 'class_static' and self.map_class_weights is None:
            raise ValueError(
                'class_static map balancing requires map_class_weights.')
        if mode == 'overlay_static':
            if not self.overlay_class_indices:
                raise ValueError(
                    'overlay_static map balancing requires '
                    'overlay_class_indices.')
            if self.overlay_pos_weight is None:
                raise ValueError(
                    'overlay_static map balancing requires overlay_pos_weight.')
        if mode == 'overlay_dynamic':
            if not self.overlay_class_indices:
                raise ValueError(
                    'overlay_dynamic map balancing requires '
                    'overlay_class_indices.')
            if self.dynamic_overlay_ref_pos_ratio is None:
                raise ValueError(
                    'overlay_dynamic map balancing requires '
                    'dynamic_overlay_ref_pos_ratio.')
            if any(ratio <= 0 for ratio in self.dynamic_overlay_ref_pos_ratio):
                raise ValueError(
                    'dynamic_overlay_ref_pos_ratio values must be positive.')

        if self.dynamic_overlay_gamma < 0:
            raise ValueError('dynamic_overlay_gamma must be non-negative.')
        if self.dynamic_overlay_min_weight <= 0:
            raise ValueError(
                'dynamic_overlay_min_weight must be greater than zero.')
        if self.dynamic_overlay_max_weight < self.dynamic_overlay_min_weight:
            raise ValueError(
                'dynamic_overlay_max_weight must be >= '
                'dynamic_overlay_min_weight.')
        if self.dynamic_overlay_eps <= 0:
            raise ValueError('dynamic_overlay_eps must be greater than zero.')

    def forward(self, bev_feature):
        bev_feature, detail_feature, aux_features, occ2map_prior = (
            self._split_forward_inputs(bev_feature))
        bev_feature = self._apply_coordconv(bev_feature)

        if self.use_dual_area_line_head:
            seg_logits = self._forward_dual_head(bev_feature, detail_feature)
            gate_logits = None
        else:
            decoded_feature = self._run_decoder(self.decoder, bev_feature)
            decoded_feature, gate_logits = self._apply_map_active_gate(
                decoded_feature, occ2map_prior)
            cnn_logits = self.predictor(decoded_feature)
            seg_logits = self._apply_query_refinement(
                decoded_feature, cnn_logits)
            seg_logits = self._apply_thin_roi_refinement(
                decoded_feature, seg_logits)

        if self.training and (
                self.use_map_aux_loss or self.use_map_active_enhance_gate):
            outputs = dict(bev_seg_logits=seg_logits)
            if gate_logits is not None:
                outputs['map_active_gate_logits'] = gate_logits
            if self.use_map_aux_loss:
                for level in self.map_aux_levels:
                    if level not in aux_features:
                        raise ValueError(
                            f'Missing aux feature level `{level}` for '
                            'use_map_aux_loss=True.')
                    outputs[f'aux_logits_{level}'] = self.map_aux_heads[level](
                        aux_features[level])
            return outputs

        return seg_logits

    def _apply_map_active_gate(self, decoded_feature, occ2map_prior=None):
        if not self.use_map_active_enhance_gate:
            return decoded_feature, None
        gate_logits = self.map_active_gate(decoded_feature)
        if self.occ2map_gate_proj is not None and occ2map_prior is not None:
            occ2map_prior = occ2map_prior.to(
                device=gate_logits.device,
                dtype=gate_logits.dtype)
            if occ2map_prior.shape[-2:] != gate_logits.shape[-2:]:
                occ2map_prior = F.interpolate(
                    occ2map_prior,
                    size=gate_logits.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            prior_logits = self.occ2map_gate_proj(occ2map_prior)
            if self.occ2map_prior_gate_channel_mask is not None:
                channel_mask = prior_logits.new_tensor(
                    self.occ2map_prior_gate_channel_mask).view(1, -1, 1, 1)
                prior_logits = prior_logits * channel_mask
            gate_logits = gate_logits + (
                self.occ2map_prior_gate_scale * prior_logits)
        gate_prob = gate_logits.sigmoid()
        group_betas = gate_prob.new_tensor(
            self.map_active_gate_group_betas).view(1, -1, 1, 1)
        spatial_gain = (gate_prob * group_betas).amax(dim=1, keepdim=True)
        enhanced = decoded_feature * (1.0 + spatial_gain)
        return enhanced, gate_logits

    def _run_decoder(self, decoder, x):
        if self.with_cp and self.training:
            return checkpoint(decoder, x)
        return decoder(x)

    def _forward_dual_head(self, bev_feature, detail_feature=None):
        area_feat = self._run_decoder(self.area_decoder, bev_feature)
        area_logits = self.area_predictor(area_feat)

        line_input = bev_feature
        if self.line_head_use_detail:
            detail_feature = self._prepare_detail_feature(
                detail_feature, bev_feature)
            line_input = torch.cat([bev_feature, detail_feature], dim=1)
        line_feat = self._run_decoder(self.line_decoder, line_input)
        line_logits = self.line_predictor(line_feat)

        seg_logits = bev_feature.new_zeros(
            bev_feature.shape[0],
            self.num_classes,
            bev_feature.shape[2],
            bev_feature.shape[3])
        area_indices = torch.tensor(
            self.area_class_indices,
            device=bev_feature.device,
            dtype=torch.long)
        line_indices = torch.tensor(
            self.line_class_indices,
            device=bev_feature.device,
            dtype=torch.long)
        seg_logits.index_copy_(1, area_indices, area_logits)
        seg_logits.index_copy_(1, line_indices, line_logits)
        return seg_logits

    def _split_forward_inputs(self, bev_feature):
        detail_feature = None
        aux_features = {}
        occ2map_prior = None
        if isinstance(bev_feature, dict):
            detail_feature = bev_feature.get('detail_feature', None)
            aux_features = bev_feature.get('aux_features', {}) or {}
            occ2map_prior = bev_feature.get('occ2map_prior', None)
            bev_feature = bev_feature.get('bev_feature', None)
            if bev_feature is None:
                raise ValueError(
                    'BEVSegHead dict input must contain `bev_feature`.')
        elif isinstance(bev_feature, (list, tuple)):
            if not bev_feature:
                raise ValueError('BEVSegHead tuple/list input must not be empty.')
            detail_feature = bev_feature[1] if len(bev_feature) > 1 else None
            aux_features = bev_feature[2] if len(bev_feature) > 2 else {}
            occ2map_prior = bev_feature[3] if len(bev_feature) > 3 else None
            bev_feature = bev_feature[0]
        aux_features = {str(level): feat
                        for level, feat in aux_features.items()}
        return bev_feature, detail_feature, aux_features, occ2map_prior

    def _build_bev_coord(self, bev_feature):
        batch_size, _, height, width = bev_feature.shape
        device = bev_feature.device
        dtype = bev_feature.dtype
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        try:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        except TypeError:
            grid_y, grid_x = torch.meshgrid(ys, xs)
        coords = [grid_x, grid_y]
        if self.bev_coord_type in ('xyr', 'xyrtheta'):
            radius = torch.sqrt(grid_x * grid_x + grid_y * grid_y)
            radius = radius / radius.new_tensor(2.0).sqrt()
            coords.append(radius)
        if self.bev_coord_type == 'xyrtheta':
            theta = torch.atan2(grid_y, grid_x) / grid_x.new_tensor(3.141592653589793)
            coords.append(theta)
        coord = torch.stack(coords, dim=0).unsqueeze(0)
        return coord.expand(batch_size, -1, -1, -1)

    def _apply_coordconv(self, bev_feature):
        if not self.use_bev_coordconv:
            return bev_feature
        coord = self._build_bev_coord(bev_feature)
        return self.coord_proj(torch.cat([bev_feature, coord], dim=1))

    def _prepare_detail_feature(self, detail_feature, bev_feature):
        if detail_feature is None:
            shape = (
                bev_feature.shape[0],
                self.line_detail_channels,
                bev_feature.shape[2],
                bev_feature.shape[3])
            return bev_feature.new_zeros(shape)
        if detail_feature.shape[2:] != bev_feature.shape[2:]:
            detail_feature = F.interpolate(
                detail_feature,
                size=bev_feature.shape[2:],
                mode='bilinear',
                align_corners=True)
        if detail_feature.shape[1] != self.line_detail_channels:
            raise ValueError(
                'line detail feature channel mismatch: expected '
                f'{self.line_detail_channels}, got {detail_feature.shape[1]}.')
        return detail_feature

    def _apply_thin_roi_refinement(self, decoded_feature, seg_logits):
        if not self.use_thin_roi_refinement:
            return seg_logits
        indices = torch.tensor(
            self.thin_roi_class_indices,
            device=seg_logits.device,
            dtype=torch.long)
        roi = self._build_thin_roi(seg_logits, indices)
        delta = self.thin_roi_refiner(decoded_feature) * roi
        refined = seg_logits.clone()
        refined[:, indices] = refined[:, indices] + delta
        return refined

    def _build_thin_roi(self, seg_logits, indices):
        scores = seg_logits[:, indices].detach().sigmoid().amax(
            dim=1, keepdim=True)
        roi = scores >= self.thin_roi_threshold
        if self.thin_roi_min_pixels > 0:
            flat_scores = scores.flatten(1)
            topk = min(self.thin_roi_min_pixels, flat_scores.shape[1])
            threshold = flat_scores.topk(topk, dim=1).values[:, -1]
            threshold = threshold.view(-1, 1, 1, 1)
            roi = roi | (scores >= threshold)
        roi = roi.float()
        if self.thin_roi_dilation > 0:
            kernel = 2 * self.thin_roi_dilation + 1
            roi = F.max_pool2d(
                roi,
                kernel_size=kernel,
                stride=1,
                padding=self.thin_roi_dilation)
        return roi

    def _query_refine_bin_bounds(self, height):
        num_bins = min(self.query_refine_num_bins, height)
        bounds = []
        for bin_index in range(num_bins):
            start = int(round(float(bin_index) * height / num_bins))
            end = int(round(float(bin_index + 1) * height / num_bins))
            end = max(end, start + 1)
            bounds.append((start, min(end, height)))
        return bounds

    def _gather_query_refine_queries(self, decoded_feature, seg_logits):
        batch_size, channels, height, width = decoded_feature.shape
        _, num_classes, _, _ = seg_logits.shape
        select_logits = (
            seg_logits.detach() if self.query_refine_detach_logits
            else seg_logits)
        feature_flat = decoded_feature.flatten(2)
        query_list = []
        class_ids = []
        bin_ids = []
        bounds = self._query_refine_bin_bounds(height)

        for bin_index, (start, end) in enumerate(bounds):
            for class_index in range(num_classes):
                bin_logits = select_logits[
                    :, class_index, start:end, :].flatten(1)
                local_index = bin_logits.argmax(dim=1)
                flat_index = local_index + start * width
                gather_index = flat_index.view(batch_size, 1, 1).expand(
                    -1, channels, 1)
                query_list.append(
                    feature_flat.gather(2, gather_index).squeeze(-1))
                class_ids.append(class_index)
                bin_ids.append(bin_index)

        queries = torch.stack(query_list, dim=1)
        class_ids = torch.tensor(
            class_ids, device=decoded_feature.device, dtype=torch.long)
        bin_ids = torch.tensor(
            bin_ids, device=decoded_feature.device, dtype=torch.long)
        return queries, class_ids, bin_ids, bounds

    def _apply_query_refinement(self, decoded_feature, seg_logits):
        if not self.use_query_refinement:
            return seg_logits

        batch_size, channels, height, width = decoded_feature.shape
        if seg_logits.shape[1] != self.num_classes:
            raise ValueError(
                'query refinement expects seg_logits channel count to match '
                f'num_classes={self.num_classes}, got {seg_logits.shape[1]}.')

        queries, class_ids, bin_ids, bounds = self._gather_query_refine_queries(
            decoded_feature, seg_logits)
        class_embed = self.query_refine_class_embed(class_ids).to(
            dtype=queries.dtype)
        bin_embed = self.query_refine_bin_embed(bin_ids).to(
            dtype=queries.dtype)
        queries = queries + class_embed.unsqueeze(0) + bin_embed.unsqueeze(0)
        queries = self.query_refine_query_norm(queries)

        attn_out, _ = self.query_refine_attn(
            queries, queries, queries, need_weights=False)
        queries = self.query_refine_attn_norm(queries + attn_out)
        ffn_out = self.query_refine_ffn(queries)
        queries = self.query_refine_ffn_norm(queries + ffn_out)

        mask_embed = self.query_refine_mask_embed(queries)
        if self.query_refine_scale_dot:
            mask_embed = mask_embed * (channels ** -0.5)
        all_query_logits = torch.einsum(
            'bqc,bchw->bqhw', mask_embed, decoded_feature)

        query_logits = seg_logits.new_zeros(
            batch_size, self.num_classes, height, width)
        offset = 0
        for start, end in bounds:
            next_offset = offset + self.num_classes
            query_logits[:, :, start:end, :] = all_query_logits[
                :, offset:next_offset, start:end, :]
            offset = next_offset

        alpha = self.query_refine_alpha.to(dtype=seg_logits.dtype)
        return seg_logits + alpha * query_logits

    def loss(self, seg_logits, gt_masks_bev):
        gt_masks_bev = gt_masks_bev.float()
        if isinstance(seg_logits, dict):
            final_logits = seg_logits.get('bev_seg_logits', None)
            if final_logits is None:
                raise ValueError(
                    'BEVSegHead loss dict must contain `bev_seg_logits`.')
            losses = self._loss_single(final_logits, gt_masks_bev)
            losses.update(self._thin_boundary_aux_loss(
                final_logits, gt_masks_bev))
            losses.update(self._map_active_gate_loss(
                seg_logits.get('map_active_gate_logits', None),
                gt_masks_bev))
            for level in self.map_aux_levels:
                key = f'aux_logits_{level}'
                if key not in seg_logits:
                    continue
                aux_gt = self._downsample_aux_target(
                    gt_masks_bev, seg_logits[key].shape[-2:])
                aux_losses = self._loss_single(
                    seg_logits[key],
                    aux_gt,
                    prefix=f'loss_map_aux_{level}',
                    loss_scale=self.map_aux_weights[level])
                losses.update(aux_losses)
            return losses

        losses = self._loss_single(seg_logits, gt_masks_bev)
        losses.update(self._thin_boundary_aux_loss(seg_logits, gt_masks_bev))
        return losses

    def _build_map_active_gate_target(self, gt_masks_bev, target_size):
        targets = []
        for indices, dilation in zip(
                self.map_active_gate_class_groups,
                self.map_active_gate_dilations):
            index_tensor = torch.tensor(
                indices,
                device=gt_masks_bev.device,
                dtype=torch.long)
            target = gt_masks_bev[:, index_tensor].amax(
                dim=1, keepdim=True).float()
            if dilation > 0:
                kernel = 2 * dilation + 1
                target = F.max_pool2d(
                    target,
                    kernel_size=kernel,
                    stride=1,
                    padding=dilation)
            if tuple(target.shape[-2:]) != tuple(target_size):
                target = F.interpolate(
                    target, size=target_size, mode='nearest')
            targets.append(target)
        return torch.cat(targets, dim=1)

    def _map_active_gate_loss(self, gate_logits, gt_masks_bev):
        if not self.use_map_active_enhance_gate or gate_logits is None:
            return {}
        target = self._build_map_active_gate_target(
            gt_masks_bev, gate_logits.shape[-2:])
        losses = {}
        weight = self.map_active_gate_loss_weight
        if weight <= 0:
            return losses

        if self.map_active_gate_use_focal and self.loss_focal is not None:
            losses['loss_map_active_gate_focal'] = (
                weight * self.loss_focal(gate_logits, target))
        elif self.loss_bce is not None:
            losses['loss_map_active_gate_bce'] = (
                weight * self.loss_bce(gate_logits, target))
        else:
            losses['loss_map_active_gate_bce'] = (
                weight * F.binary_cross_entropy_with_logits(
                    gate_logits, target))

        if self.map_active_gate_use_dice and self.loss_dice is not None:
            losses['loss_map_active_gate_dice'] = (
                weight * self.loss_dice(
                    gate_logits.reshape(-1, *gate_logits.shape[2:]),
                    target.reshape(-1, *target.shape[2:])))
        return losses

    def _loss_single(self,
                     seg_logits,
                     gt_masks_bev,
                     prefix='loss_map',
                     loss_scale=1.0):
        if self.map_loss_balance_mode != 'none':
            return self._balanced_loss(
                seg_logits, gt_masks_bev, prefix, loss_scale)

        losses = {}

        if self.loss_bce is not None:
            losses[f'{prefix}_bce'] = (
                self.loss_bce(seg_logits, gt_masks_bev) * loss_scale)

        if self.loss_focal is not None:
            losses.update(self._focal_loss(
                seg_logits, gt_masks_bev, prefix, loss_scale))

        if self.loss_dice is not None:
            losses[f'{prefix}_dice'] = self.loss_dice(
                seg_logits.reshape(-1, *seg_logits.shape[2:]),
                gt_masks_bev.reshape(-1, *gt_masks_bev.shape[2:]),
            ) * loss_scale

        if self.use_map_lovasz:
            losses[f'{prefix}_lovasz'] = (
                self.map_lovasz_weight
                * self._lovasz_hinge_loss_per_class(
                    seg_logits, gt_masks_bev).mean()
                * loss_scale)

        return losses

    def _focal_loss(self,
                    seg_logits,
                    gt_masks_bev,
                    prefix='loss_map',
                    loss_scale=1.0):
        if self.map_focal_loss_mode == 'shared':
            return {
                f'{prefix}_focal':
                self.loss_focal(seg_logits, gt_masks_bev) * loss_scale
            }

        if seg_logits.shape[1] != self.num_classes:
            raise ValueError(
                'Classwise map focal loss expects seg_logits channel count '
                f'to match num_classes={self.num_classes}, but got '
                f'{seg_logits.shape[1]}.')

        class_scale = float(loss_scale)
        if self.map_focal_loss_mode == 'classwise_mean':
            class_scale = class_scale / max(self.num_classes, 1)

        losses = {}
        for class_index, class_name in enumerate(
                self.map_focal_loss_class_names):
            losses[f'{prefix}_{class_name}_focal'] = (
                self.loss_focal(
                    seg_logits[:, class_index],
                    gt_masks_bev[:, class_index]) * class_scale)
        return losses

    def _thin_boundary_aux_loss(self, seg_logits, gt_masks_bev):
        if not self.use_thin_boundary_aux_loss:
            return {}
        indices = torch.tensor(
            self.thin_boundary_class_indices,
            device=seg_logits.device,
            dtype=torch.long)
        thin_logits = seg_logits[:, indices]
        thin_target = gt_masks_bev[:, indices].float()
        if self.thin_boundary_dilation > 0:
            kernel = 2 * self.thin_boundary_dilation + 1
            thin_target = F.max_pool2d(
                thin_target,
                kernel_size=kernel,
                stride=1,
                padding=self.thin_boundary_dilation)

        losses = {}
        weight = self.thin_boundary_loss_weight
        if self.thin_boundary_use_focal and self.loss_focal is not None:
            losses['loss_map_thin_boundary_focal'] = (
                weight * self.loss_focal(thin_logits, thin_target))
        elif self.loss_bce is not None:
            losses['loss_map_thin_boundary_bce'] = (
                weight * self.loss_bce(thin_logits, thin_target))
        else:
            losses['loss_map_thin_boundary_bce'] = (
                weight * F.binary_cross_entropy_with_logits(
                    thin_logits, thin_target))

        if self.thin_boundary_use_dice and self.loss_dice is not None:
            losses['loss_map_thin_boundary_dice'] = (
                weight * self.loss_dice(
                    thin_logits.reshape(-1, *thin_logits.shape[2:]),
                    thin_target.reshape(-1, *thin_target.shape[2:])))
        return losses

    def _downsample_aux_target(self, gt_masks_bev, target_size):
        if tuple(gt_masks_bev.shape[-2:]) == tuple(target_size):
            return gt_masks_bev
        if self.map_aux_downsample == 'nearest':
            return F.interpolate(
                gt_masks_bev.float(), size=target_size, mode='nearest')
        in_h, in_w = gt_masks_bev.shape[-2:]
        out_h, out_w = target_size
        if in_h % out_h == 0 and in_w % out_w == 0:
            return F.max_pool2d(
                gt_masks_bev.float(),
                kernel_size=(in_h // out_h, in_w // out_w),
                stride=(in_h // out_h, in_w // out_w))
        return F.adaptive_max_pool2d(gt_masks_bev.float(), target_size)

    def _balanced_loss(self,
                       seg_logits,
                       gt_masks_bev,
                       prefix='loss_map',
                       loss_scale=1.0):
        if seg_logits.shape != gt_masks_bev.shape:
            raise ValueError(
                'BEVSegHead balanced loss expects seg_logits and '
                'gt_masks_bev to have the same shape, but got '
                f'{tuple(seg_logits.shape)} and {tuple(gt_masks_bev.shape)}.')

        losses = {}
        bce_weight, dice_weight, balance_info = self._build_balance_weights(
            gt_masks_bev)

        bce_contrib = None
        if self.loss_bce is not None:
            if not getattr(self.loss_bce, 'use_sigmoid', True):
                raise NotImplementedError(
                    'BEVSegHead balanced BCE currently supports only '
                    'sigmoid BCE.')
            bce_per_pixel = F.binary_cross_entropy_with_logits(
                seg_logits, gt_masks_bev, reduction='none')
            bce_weighted = bce_per_pixel * bce_weight
            bce_loss_weight = float(getattr(self.loss_bce, 'loss_weight', 1.0))
            losses[f'{prefix}_bce'] = (
                loss_scale * bce_loss_weight * bce_weighted.mean())
            bce_contrib = bce_loss_weight * bce_weighted.mean(dim=(0, 2, 3))

        focal_contrib = None
        if self.loss_focal is not None:
            losses[f'{prefix}_focal'] = loss_scale * self.loss_focal(
                seg_logits,
                gt_masks_bev,
                weight=bce_weight,
                reduction_override='mean')
            focal_contrib = self.loss_focal(
                seg_logits,
                gt_masks_bev,
                weight=bce_weight,
                reduction_override='none').mean(dim=(0, 2, 3))

        dice_contrib = None
        if self.loss_dice is not None:
            dice_per_class = self._dice_loss_per_class(
                seg_logits, gt_masks_bev)
            dice_weighted = dice_per_class * dice_weight
            dice_loss_weight = float(
                getattr(self.loss_dice, 'loss_weight', 1.0))
            losses[f'{prefix}_dice'] = (
                loss_scale * dice_loss_weight * dice_weighted.mean())
            dice_contrib = dice_loss_weight * dice_weighted.mean(dim=0)

        lovasz_contrib = None
        if self.use_map_lovasz:
            lovasz_per_class = self._lovasz_hinge_loss_per_class(
                seg_logits, gt_masks_bev)
            lovasz_weights = dice_weight.mean(dim=0)
            lovasz_weighted = lovasz_per_class * lovasz_weights
            losses[f'{prefix}_lovasz'] = (
                loss_scale * self.map_lovasz_weight * lovasz_weighted.mean())
            lovasz_contrib = self.map_lovasz_weight * lovasz_weighted

        self._log_balance_debug(
            balance_info,
            bce_contrib,
            dice_contrib,
            focal_contrib,
            lovasz_contrib)
        return losses

    def _build_balance_weights(self, gt_masks_bev):
        bce_weight = torch.ones_like(gt_masks_bev)
        dice_weight = gt_masks_bev.new_ones(
            (gt_masks_bev.shape[0], gt_masks_bev.shape[1]))
        pos_pixels = gt_masks_bev.sum(dim=(0, 2, 3))
        valid_pixels = max(
            int(gt_masks_bev.shape[0] * gt_masks_bev.shape[2]
                * gt_masks_bev.shape[3]), 1)
        pos_ratio = pos_pixels / float(valid_pixels)

        class_weights = None
        overlay_weights_full = gt_masks_bev.new_ones(self.num_classes)
        ref_ratio_full = gt_masks_bev.new_zeros(self.num_classes)
        present_full = pos_pixels > 0

        if self.map_loss_balance_mode == 'class_static':
            class_weights = gt_masks_bev.new_tensor(self.map_class_weights)
            bce_weight = bce_weight * class_weights.view(1, -1, 1, 1)
            dice_weight = dice_weight * class_weights.view(1, -1)

        if self.map_loss_balance_mode in ('overlay_static',
                                          'overlay_dynamic'):
            overlay_indices = torch.tensor(
                self.overlay_class_indices,
                device=gt_masks_bev.device,
                dtype=torch.long)
            present_overlay = pos_pixels[overlay_indices] > 0

            if self.map_loss_balance_mode == 'overlay_static':
                overlay_weights = gt_masks_bev.new_tensor(
                    self.overlay_pos_weight)
            else:
                ref_ratio = gt_masks_bev.new_tensor(
                    self.dynamic_overlay_ref_pos_ratio)
                batch_ratio = pos_ratio[overlay_indices]
                overlay_weights = (
                    ref_ratio / (batch_ratio + self.dynamic_overlay_eps)
                ).pow(self.dynamic_overlay_gamma)
                overlay_weights = torch.clamp(
                    overlay_weights,
                    min=self.dynamic_overlay_min_weight,
                    max=self.dynamic_overlay_max_weight)
                ref_ratio_full[overlay_indices] = ref_ratio

            overlay_weights = torch.where(
                present_overlay,
                overlay_weights,
                torch.ones_like(overlay_weights))
            overlay_weights_full[overlay_indices] = overlay_weights

            for offset, class_index in enumerate(self.overlay_class_indices):
                if bool(present_overlay[offset].item()):
                    weight = overlay_weights[offset]
                    bce_weight[:, class_index] = (
                        1.0 + (weight - 1.0) * gt_masks_bev[:, class_index])
                    dice_weight[:, class_index] = weight

        balance_info = dict(
            mode=self.map_loss_balance_mode,
            pos_pixels=pos_pixels.detach(),
            pos_ratio=pos_ratio.detach(),
            class_weights=class_weights.detach()
            if class_weights is not None else None,
            overlay_weights=overlay_weights_full.detach(),
            ref_ratio=ref_ratio_full.detach(),
            present=present_full.detach())
        return bce_weight, dice_weight, balance_info

    def _dice_loss_per_class(self, seg_logits, gt_masks_bev):
        pred = seg_logits
        if getattr(self.loss_dice, 'activate', True):
            if getattr(self.loss_dice, 'use_sigmoid', True):
                pred = pred.sigmoid()
            else:
                raise NotImplementedError(
                    'BEVSegHead balanced Dice currently supports only '
                    'sigmoid activation.')

        pred = pred.flatten(2)
        target = gt_masks_bev.flatten(2).float()
        eps = float(getattr(self.loss_dice, 'eps', 1e-3))
        intersection = torch.sum(pred * target, dim=2)

        if getattr(self.loss_dice, 'naive_dice', False):
            denominator = torch.sum(pred, dim=2) + torch.sum(target, dim=2)
            dice_score = (2 * intersection + eps) / (denominator + eps)
        else:
            pred_area = torch.sum(pred * pred, dim=2) + eps
            target_area = torch.sum(target * target, dim=2) + eps
            dice_score = (2 * intersection) / (pred_area + target_area)

        return 1 - dice_score

    def _lovasz_grad(self, gt_sorted):
        p = gt_sorted.numel()
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1.0 - gt_sorted.float()).cumsum(0)
        jaccard = 1.0 - intersection / union.clamp_min(1e-6)
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def _lovasz_hinge_flat(self, logits, labels):
        if labels.numel() == 0:
            return logits.sum() * 0.0
        labels = labels.float()
        signs = 2.0 * labels - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        return torch.dot(F.relu(errors_sorted), grad)

    def _lovasz_hinge_loss_per_class(self, seg_logits, gt_masks_bev):
        per_class_losses = []
        for class_index in range(seg_logits.shape[1]):
            class_losses = []
            for batch_index in range(seg_logits.shape[0]):
                class_losses.append(self._lovasz_hinge_flat(
                    seg_logits[batch_index, class_index].reshape(-1),
                    gt_masks_bev[batch_index, class_index].reshape(-1)))
            per_class_losses.append(torch.stack(class_losses).mean())
        return torch.stack(per_class_losses)

    def _format_class_values(self, values, precision=4):
        if values is None:
            return 'none'
        values = values.detach().cpu().tolist()
        parts = []
        for name, value in zip(self.map_balance_class_names, values):
            if isinstance(value, bool):
                formatted = str(value)
            elif float(value).is_integer():
                formatted = str(int(value))
            else:
                formatted = f'{float(value):.{precision}g}'
            parts.append(f'{name}:{formatted}')
        return ', '.join(parts)

    def _log_balance_debug(self,
                           balance_info,
                           bce_contrib,
                           dice_contrib,
                           focal_contrib=None,
                           lovasz_contrib=None):
        if not self.map_balance_debug:
            return
        step = self._map_balance_debug_step
        self._map_balance_debug_step += 1
        if (self.map_balance_debug_interval > 1
                and step % self.map_balance_debug_interval != 0):
            return

        logger = logging.getLogger('mmdet')
        logger.info(
            '[BEVSegHead map balance] '
            f'step={step} mode={balance_info["mode"]} '
            'pos_pixels=('
            f'{self._format_class_values(balance_info["pos_pixels"], 0)}) '
            'pos_ratio=('
            f'{self._format_class_values(balance_info["pos_ratio"], 6)}) '
            'weights=('
            f'{self._format_class_values(balance_info["overlay_weights"], 4)}) '
            'ref_ratio=('
            f'{self._format_class_values(balance_info["ref_ratio"], 6)}) '
            'bce_contrib=('
            f'{self._format_class_values(bce_contrib, 4)}) '
            'focal_contrib=('
            f'{self._format_class_values(focal_contrib, 4)}) '
            'lovasz_contrib=('
            f'{self._format_class_values(lovasz_contrib, 4)}) '
            'dice_contrib=('
            f'{self._format_class_values(dice_contrib, 4)})')

    def predict(self, seg_logits):
        return torch.sigmoid(seg_logits)
