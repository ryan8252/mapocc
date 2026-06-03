# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from torch.utils.checkpoint import checkpoint
from mmcv.cnn.bricks import ConvModule
from mmdet.models import NECKS
import torch.nn.functional as F


def _zero_init_conv(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@NECKS.register_module()
class FPN_LSS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 ):
        super(FPN_LSS, self).__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )

        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True)
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.extra_upsample:
            x = self.up2(x)
        return x

@NECKS.register_module()
class Custom_FPN_LSS(nn.Module):
    def __init__(self,
                 catconv_in_channels1,
                 catconv_in_channels2,
                 out_channels,
                 scale_factor=2,
                 input_feature_index=(0, 1, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 with_cp=False,
                 use_fpn_lateral_projection=False,
                 fpn_lateral_in_channels=None,
                 fpn_projection_channels=256,
                 use_fpn_global_context=False,
                 fpn_global_context_type='aspp',
                 fpn_context_channels=None,
                 return_map_aux_features=False,
                 map_aux_feature_levels=('100', '50')):
        super(Custom_FPN_LSS, self).__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        self.with_cp = with_cp
        self.use_fpn_lateral_projection = bool(use_fpn_lateral_projection)
        self.use_fpn_global_context = bool(use_fpn_global_context)
        self.return_map_aux_features = bool(return_map_aux_features)
        self.map_aux_feature_levels = tuple(str(level)
                                            for level in map_aux_feature_levels)
        self.fpn_global_context_type = str(fpn_global_context_type).lower()
        self.fpn_projection_channels = int(fpn_projection_channels)
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1
        self.fused_channels = out_channels * channels_factor
        self.fpn_lateral_in_channels = self._resolve_lateral_in_channels(
            fpn_lateral_in_channels,
            catconv_in_channels1,
            catconv_in_channels2,
            self.fused_channels)

        if self.use_fpn_global_context:
            low_channels = self.fpn_lateral_in_channels[2]
            self.fpn_global_context = self._build_global_context(
                low_channels, fpn_context_channels, norm_cfg)
        else:
            self.fpn_global_context = None

        if self.use_fpn_lateral_projection:
            self.fpn_lateral_projs = nn.ModuleList([
                ConvModule(
                    in_channels,
                    self.fpn_projection_channels,
                    kernel_size=1,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU', inplace=True))
                for in_channels in self.fpn_lateral_in_channels
            ])
            catconv_in_channels1 = self.fpn_projection_channels * 2
            catconv_in_channels2 = (
                self.fused_channels + self.fpn_projection_channels)
        else:
            self.fpn_lateral_projs = None

        self.cat_conv1 = self._make_cat_conv(
            catconv_in_channels1, self.fused_channels, norm_cfg)
        self.cat_conv2 = self._make_cat_conv(
            catconv_in_channels2, self.fused_channels, norm_cfg)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(self.fused_channels, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )

    def _make_cat_conv(self, in_channels, out_channels, norm_cfg):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        x3, x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]], feats[self.input_feature_index[2]]
        if self.fpn_global_context is not None:
            x1 = self._apply_checkpoint(self.fpn_global_context, x1)

        if self.fpn_lateral_projs is not None:
            x3 = self._apply_checkpoint(self.fpn_lateral_projs[0], x3)
            x2 = self._apply_checkpoint(self.fpn_lateral_projs[1], x2)
            x1 = self._apply_checkpoint(self.fpn_lateral_projs[2], x1)

        x1_up = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x2_cat = torch.cat([x2, x1_up], dim=1)
        x2_cat = self._apply_checkpoint(self.cat_conv1, x2_cat)

        x2_up = self._apply_checkpoint(self.up, x2_cat)

        x3_cat = torch.cat([x3, x2_up], dim=1)
        x3_cat = self._apply_checkpoint(self.cat_conv2, x3_cat)

        if self.extra_upsample:
            x4_out = self.up2(x3_cat)
        else:
            x4_out = None

        if self.return_map_aux_features:
            aux_features = {}
            if '100' in self.map_aux_feature_levels:
                aux_features['100'] = x3_cat
            if '50' in self.map_aux_feature_levels:
                aux_features['50'] = x2_cat
            return dict(bev_feature=x4_out, aux_features=aux_features)

        return [x4_out, x4_out, x4_out, x4_out]

    def _apply_checkpoint(self, module, *inputs):
        if self.with_cp and self.training:
            return checkpoint(module, *inputs)
        else:
            return module(*inputs)

    def _resolve_lateral_in_channels(self,
                                     fpn_lateral_in_channels,
                                     catconv_in_channels1,
                                     catconv_in_channels2,
                                     fused_channels):
        if fpn_lateral_in_channels is not None:
            fpn_lateral_in_channels = tuple(
                int(channel) for channel in fpn_lateral_in_channels)
            if len(fpn_lateral_in_channels) != 3:
                raise ValueError(
                    'fpn_lateral_in_channels must contain three values for '
                    'level0, level1, and level2.')
            return fpn_lateral_in_channels

        level0_channels = int(catconv_in_channels2) - int(fused_channels)
        if level0_channels <= 0:
            raise ValueError(
                'Cannot infer level0 channels for Custom_FPN_LSS from '
                f'catconv_in_channels2={catconv_in_channels2} and '
                f'fused_channels={fused_channels}.')

        # The local ProtoOcc BEV backbone uses [C, 2C, 4C] pyramid channels.
        level1_channels = level0_channels * 2
        level2_channels = int(catconv_in_channels1) - level1_channels
        if level2_channels <= 0:
            raise ValueError(
                'Cannot infer level2 channels for Custom_FPN_LSS from '
                f'catconv_in_channels1={catconv_in_channels1} and '
                f'level1_channels={level1_channels}. Set '
                '`fpn_lateral_in_channels` explicitly.')
        return (level0_channels, level1_channels, level2_channels)

    def _build_global_context(self, in_channels, context_channels, norm_cfg):
        if self.fpn_global_context_type == 'aspp':
            return _FPNASPPContext(
                in_channels,
                context_channels=context_channels,
                norm_cfg=norm_cfg)
        if self.fpn_global_context_type == 'ppm':
            return _FPNPPMContext(
                in_channels,
                context_channels=context_channels,
                norm_cfg=norm_cfg)
        raise ValueError(
            'Unsupported fpn_global_context_type: '
            f'{self.fpn_global_context_type}. Expected "aspp" or "ppm".')


class _FPNASPPContext(nn.Module):
    def __init__(self, in_channels, context_channels=None, norm_cfg=dict(type='BN')):
        super(_FPNASPPContext, self).__init__()
        if context_channels is None:
            context_channels = max(in_channels // 4, 32)
        self.branches = nn.ModuleList()
        for dilation in (1, 2, 4):
            self.branches.append(
                ConvModule(
                    in_channels,
                    context_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU', inplace=True)))
        self.pool_branch = ConvModule(
            in_channels,
            context_channels,
            kernel_size=1,
            bias=False,
            norm_cfg=None,
            act_cfg=dict(type='ReLU', inplace=True))
        self.project = ConvModule(
            context_channels * 4,
            in_channels,
            kernel_size=1,
            bias=True,
            norm_cfg=None,
            act_cfg=None)
        self.project.apply(_zero_init_conv)

    def forward(self, x):
        out = [branch(x) for branch in self.branches]
        pooled = F.adaptive_avg_pool2d(x, 1)
        pooled = self.pool_branch(pooled)
        pooled = F.interpolate(
            pooled, size=x.shape[2:], mode='bilinear', align_corners=True)
        out.append(pooled)
        return x + self.project(torch.cat(out, dim=1))


class _FPNPPMContext(nn.Module):
    def __init__(self,
                 in_channels,
                 context_channels=None,
                 pool_scales=(1, 2, 3, 6),
                 norm_cfg=dict(type='BN')):
        super(_FPNPPMContext, self).__init__()
        if context_channels is None:
            context_channels = max(in_channels // len(pool_scales), 32)
        self.pool_scales = tuple(pool_scales)
        self.branches = nn.ModuleList()
        for scale in self.pool_scales:
            branch_norm_cfg = None if scale == 1 else norm_cfg
            self.branches.append(
                ConvModule(
                    in_channels,
                    context_channels,
                    kernel_size=1,
                    bias=False,
                    norm_cfg=branch_norm_cfg,
                    act_cfg=dict(type='ReLU', inplace=True)))
        self.project = ConvModule(
            in_channels + context_channels * len(self.pool_scales),
            in_channels,
            kernel_size=1,
            bias=True,
            norm_cfg=None,
            act_cfg=None)
        self.project.apply(_zero_init_conv)

    def forward(self, x):
        out = [x]
        for scale, branch in zip(self.pool_scales, self.branches):
            pooled = F.adaptive_avg_pool2d(x, scale)
            pooled = branch(pooled)
            pooled = F.interpolate(
                pooled, size=x.shape[2:], mode='bilinear', align_corners=True)
            out.append(pooled)
        return x + self.project(torch.cat(out, dim=1))
