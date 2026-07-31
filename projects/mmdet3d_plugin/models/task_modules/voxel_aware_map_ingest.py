"""Voxel-aware Map Ingest (VAMI).

Route the OCC-supervised `comprehensive_voxel_feature` produced by
`Dual_Branch_Encoder` / `cnn3d_decoder` back into the map branch, which
currently consumes only `multi_scale_bev` upstream of the hierarchical fusion
module and therefore never sees the voxel-level CE+Lovasz signal.

Design notes (see ``auxiliary_voxel_semantic_supervision_plan.md``):

* Detach the voxel source by default so map loss does not propagate back
  through HFM/DBE and disturb the OCC main path; VAMI's own projection and
  fusion weights remain trainable.
* Residual fusion with a zero-initialised final conv keeps the initial
  forward identical to the baseline map path (delta == 0 at step 0). The
  branch only starts contributing when fusion learns a useful update.
* Shapes are aligned via bilinear resize on the voxel side so that the map
  side proceeds untouched into the segmentation head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class VoxelAwareMapIngest(BaseModule):
    """Voxel-aware feature ingest for the map branch.

    Args:
        voxel_in_channels (int): channels of ``comprehensive_voxel_feature``
            (= ``Dual_Branch_Encoder.voxel_out_channels``, default 48).
        map_channels (int): channels of ``map_bev_feature`` produced by the
            map-specific BEV neck (default 128 for the cnn-map-neck baseline).
        z_collapse_mode (str): how to collapse the Z axis of the voxel
            feature. One of ``'avg'``, ``'max'``, ``'avg_max_concat'``.
        project_channels (int): channels of the projected voxel feature
            before fusion concat. Should match the order of ``map_channels``.
        fusion_hidden_channels (int): hidden channels for the fusion conv.
        detach_voxel_source (bool): if True, ``voxel_feature.detach()`` is
            taken before the projection. The projection/fusion weights stay
            trainable. Recommended default.
        residual (bool): if True, the module returns
            ``map_feature + fusion(concat(map_feature, projected))``.
        zero_init_output (bool): if True, the last conv of the fusion stack
            is zero-initialised so the initial residual is exactly 0.
    """

    def __init__(self,
                 voxel_in_channels=48,
                 map_channels=128,
                 z_collapse_mode='avg_max_concat',
                 project_channels=128,
                 fusion_hidden_channels=128,
                 detach_voxel_source=True,
                 residual=True,
                 zero_init_output=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if z_collapse_mode not in ('avg', 'max', 'avg_max_concat'):
            raise ValueError(
                "z_collapse_mode must be one of 'avg', 'max', "
                f"'avg_max_concat'; got {z_collapse_mode!r}")
        self.voxel_in_channels = voxel_in_channels
        self.map_channels = map_channels
        self.z_collapse_mode = z_collapse_mode
        self.project_channels = project_channels
        self.detach_voxel_source = detach_voxel_source
        self.residual = residual

        z_collapsed_channels = (
            2 * voxel_in_channels
            if z_collapse_mode == 'avg_max_concat'
            else voxel_in_channels)

        self.project = nn.Sequential(
            nn.Conv2d(z_collapsed_channels, project_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(project_channels),
            nn.ReLU(inplace=True),
        )

        fused_in = map_channels + project_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_in, fusion_hidden_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_hidden_channels, map_channels,
                      kernel_size=1, bias=True),
        )

        if zero_init_output:
            nn.init.zeros_(self.fusion[-1].weight)
            if self.fusion[-1].bias is not None:
                nn.init.zeros_(self.fusion[-1].bias)

    def _z_collapse(self, voxel_feature):
        # voxel_feature: [B, C, X, Y, Z] -> collapse along the last axis.
        if self.z_collapse_mode == 'avg':
            return voxel_feature.mean(dim=-1)
        if self.z_collapse_mode == 'max':
            return voxel_feature.max(dim=-1).values
        avg = voxel_feature.mean(dim=-1)
        mx = voxel_feature.max(dim=-1).values
        return torch.cat([avg, mx], dim=1)

    def forward(self, map_feature, voxel_feature, **kwargs):
        """Fuse the voxel-supervised feature into the map BEV feature.

        Args:
            map_feature (Tensor): ``[B, C_map, H, W]``, output of the map
                BEV neck. Will be returned with the same shape.
            voxel_feature (Tensor): ``[B, C_v, X, Y, Z]``, the
                ``comprehensive_voxel_feature`` from ``Dual_Branch_Encoder``.

        Returns:
            Tensor: ``[B, C_map, H, W]``.
        """
        if voxel_feature.dim() != 5:
            raise ValueError(
                'VAMI expects a 5D voxel_feature (B, C, X, Y, Z); '
                f'got shape {tuple(voxel_feature.shape)}.')
        if map_feature.dim() != 4:
            raise ValueError(
                'VAMI expects a 4D map_feature (B, C, H, W); '
                f'got shape {tuple(map_feature.shape)}.')
        if voxel_feature.shape[1] != self.voxel_in_channels:
            raise ValueError(
                f'voxel_feature has {voxel_feature.shape[1]} channels, '
                f'but VAMI was configured with voxel_in_channels='
                f'{self.voxel_in_channels}.')
        if map_feature.shape[1] != self.map_channels:
            raise ValueError(
                f'map_feature has {map_feature.shape[1]} channels, '
                f'but VAMI was configured with map_channels='
                f'{self.map_channels}.')

        voxel_source = (
            voxel_feature.detach()
            if self.detach_voxel_source else voxel_feature)

        z_flat = self._z_collapse(voxel_source)
        projected = self.project(z_flat)

        if projected.shape[-2:] != map_feature.shape[-2:]:
            projected = F.interpolate(
                projected,
                size=map_feature.shape[-2:],
                mode='bilinear',
                align_corners=False)

        fused_in = torch.cat([map_feature, projected], dim=1)
        delta = self.fusion(fused_in)
        if self.residual:
            return map_feature + delta
        return delta
