import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.utils.checkpoint import checkpoint

from mmdet3d.models import BACKBONES
from mmdet3d.models import builder


def _should_checkpoint(with_cp, training, *inputs):
    return with_cp and training and any(
        torch.is_tensor(input_) and input_.requires_grad for input_ in inputs)


@BACKBONES.register_module()
class OccFeature3DBranch(BaseModule):
    """Small 3D feature branch for MAESTRO OCC TSFG features.

    Input and output use the MAESTRO voxel layout: (B, C, X, Y, Z).
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 hidden_channels=None,
                 num_convs=2,
                 norm_cfg=dict(type='BN3d'),
                 with_cp=False,
                 init_cfg=None):
        super(OccFeature3DBranch, self).__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or out_channels
        self.with_cp = with_cp
        self.use_residual = in_channels == out_channels

        layers = []
        cur_channels = in_channels
        for _ in range(max(num_convs - 1, 0)):
            layers.append(
                ConvModule(
                    cur_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU', inplace=True)))
            cur_channels = hidden_channels
        layers.append(
            ConvModule(
                cur_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        self.branch = nn.Sequential(*layers)

    def forward(self, occ_feature):
        occ_feature_conv = occ_feature.permute(0, 1, 4, 2, 3).contiguous()
        if _should_checkpoint(self.with_cp, self.training, occ_feature_conv):
            out = checkpoint(self.branch, occ_feature_conv)
        else:
            out = self.branch(occ_feature_conv)
        if self.use_residual:
            out = out + occ_feature_conv
        return out.permute(0, 1, 3, 4, 2).contiguous()


@BACKBONES.register_module()
class MapFeature2DBranch(BaseModule):
    """Small 2D feature branch for MAESTRO map TSFG features."""

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 hidden_channels=None,
                 num_convs=2,
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 init_cfg=None):
        super(MapFeature2DBranch, self).__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or out_channels
        self.with_cp = with_cp
        self.use_residual = in_channels == out_channels

        layers = []
        cur_channels = in_channels
        for _ in range(max(num_convs - 1, 0)):
            layers.append(
                ConvModule(
                    cur_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU', inplace=True)))
            cur_channels = hidden_channels
        layers.append(
            ConvModule(
                cur_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        self.branch = nn.Sequential(*layers)

    def forward(self, map_feature):
        if _should_checkpoint(self.with_cp, self.training, map_feature):
            out = checkpoint(self.branch, map_feature)
        else:
            out = self.branch(map_feature)
        if self.use_residual:
            out = out + map_feature
        return out


@BACKBONES.register_module()
class MapOccHFMFusion3D(BaseModule):
    """Lift BEV map TSFG features into OCC TSFG features with residual fusion."""

    def __init__(self,
                 occ_channels,
                 map_channels,
                 hidden_channels=None,
                 norm_cfg_2d=dict(type='BN'),
                 norm_cfg_3d=dict(type='BN3d'),
                 zero_init=True,
                 with_cp=False,
                 init_cfg=None):
        super(MapOccHFMFusion3D, self).__init__(init_cfg)
        hidden_channels = hidden_channels or occ_channels
        self.with_cp = with_cp
        self.map_project = ConvModule(
            map_channels,
            occ_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg_2d,
            act_cfg=dict(type='ReLU', inplace=True))
        self.fuse = nn.Sequential(
            ConvModule(
                occ_channels * 2,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg_3d,
                act_cfg=dict(type='ReLU', inplace=True)),
            nn.Conv3d(hidden_channels, occ_channels, kernel_size=1))
        if zero_init:
            nn.init.zeros_(self.fuse[-1].weight)
            if self.fuse[-1].bias is not None:
                nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, occ_feature, map_feature):
        b, _, x, y, z = occ_feature.shape
        if map_feature.shape[-2:] != (x, y):
            raise ValueError(
                f'Map feature size {map_feature.shape[-2:]} does not match '
                f'OCC feature XY size {(x, y)}.')

        if _should_checkpoint(self.with_cp, self.training, map_feature):
            map_projected = checkpoint(self.map_project, map_feature)
        else:
            map_projected = self.map_project(map_feature)

        map_lift = map_projected.unsqueeze(-1).expand(-1, -1, -1, -1, z)
        occ_conv = occ_feature.permute(0, 1, 4, 2, 3).contiguous()
        map_lift = map_lift.permute(0, 1, 4, 2, 3).contiguous()
        fusion_input = torch.cat([occ_conv, map_lift], dim=1)
        if _should_checkpoint(self.with_cp, self.training, fusion_input):
            residual = checkpoint(self.fuse, fusion_input)
        else:
            residual = self.fuse(fusion_input)
        fused = occ_conv + residual
        return fused.permute(0, 1, 3, 4, 2).contiguous()


@BACKBONES.register_module()
class MapFeatureFPNMixer(BaseModule):
    """Reuse the local BEV backbone/FPN pattern for a 200x200 map feature."""

    def __init__(self,
                 bev_encoder_backbone,
                 bev_encoder_neck,
                 init_cfg=None):
        super(MapFeatureFPNMixer, self).__init__(init_cfg)
        self.bev_encoder_backbone = builder.build_backbone(bev_encoder_backbone)
        self.bev_encoder_neck = builder.build_neck(bev_encoder_neck)

    def forward(self, map_feature):
        multi_scale_bev = self.bev_encoder_backbone(map_feature)
        mixed_feature = self.bev_encoder_neck(multi_scale_bev)
        if isinstance(mixed_feature, (list, tuple)):
            return mixed_feature[0]
        return mixed_feature
