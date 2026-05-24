from torch import nn
import torch

from mmdet3d.models import BACKBONES, builder


@BACKBONES.register_module()
class MapTopologyEncoder(nn.Module):
    """Map-specific BEV topology encoder with configurable Z projection."""

    def __init__(self,
                 in_channels=80,
                 z_channels=None,
                 z_projection='weighted_pool',
                 mid_channels=160,
                 num_channels=(160, 320, 640),
                 num_layer=(1, 1, 1),
                 stride=(2, 2, 2),
                 ConvNext_kernel_size=7,
                 with_cp=True,
                 zero_init_z_logits=True):
        super().__init__()
        valid_z_projection = ('weighted_pool', 'cat_z')
        if z_projection not in valid_z_projection:
            raise ValueError(
                'MapTopologyEncoder supports z_projection in '
                f'{valid_z_projection}, got {z_projection!r}.')
        if z_projection == 'cat_z' and z_channels is None:
            raise ValueError(
                "MapTopologyEncoder with z_projection='cat_z' requires "
                'z_channels to determine the 2D projection input channels.')

        self.z_channels = z_channels
        self.z_projection = z_projection

        if z_projection == 'weighted_pool':
            proj_in_channels = in_channels
            self.z_logit = nn.Conv3d(in_channels, 1, kernel_size=1)
        else:
            proj_in_channels = in_channels * z_channels
            self.z_logit = None

        if self.z_logit is not None and zero_init_z_logits:
            nn.init.constant_(self.z_logit.weight, 0)
            nn.init.constant_(self.z_logit.bias, 0)

        self.input_proj = nn.Sequential(
            nn.Conv2d(proj_in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))

        self.bev_backbone = builder.build_backbone(
            dict(
                type='CustomBEVBackbone',
                numC_input=mid_channels,
                num_channels=list(num_channels),
                num_layer=list(num_layer),
                stride=list(stride),
                ConvNext_kernel_size=ConvNext_kernel_size,
                with_cp=with_cp))

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(
                'MapTopologyEncoder expects input shape [B, C, Z, H, W], '
                f'but got {tuple(x.shape)}.')
        if self.z_channels is not None and x.shape[2] != self.z_channels:
            raise ValueError(
                'MapTopologyEncoder was configured with '
                f'z_channels={self.z_channels}, but input has Z={x.shape[2]}.')

        if self.z_projection == 'weighted_pool':
            z_weight = torch.softmax(self.z_logit(x), dim=2)
            x_bev = torch.sum(x * z_weight, dim=2)
        else:
            x_bev = torch.cat(x.unbind(dim=2), dim=1)
        x_bev = self.input_proj(x_bev)
        return self.bev_backbone(x_bev)
