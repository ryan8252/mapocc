from torch import nn

from mmdet3d.models import BACKBONES
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmdet3d.models import builder
import torch

class voxelize_module(nn.Module):
    def __init__(
            self,
            bev_z = 16,
            in_dim=48,
    ):
        super(voxelize_module, self).__init__()
        self.bev_z = bev_z
        self.linear = nn.Sequential(
                nn.Linear(in_dim, 2 * in_dim),
                nn.ReLU(),
                nn.Linear(2*in_dim, self.bev_z * in_dim),
            )
    def forward(self, x):
        x_linear = self.linear(x.permute(0,2,3,1))
        B, H, W, C_Z = x_linear.shape
        x = x_linear.reshape(B, H, W, self.bev_z, -1).permute(0,4,1,2,3)
        return x


class MapHFMFusionLayer(nn.Module):
    """Fuse one BEV pyramid level with one early voxel-branch feature."""

    def __init__(self, bev_channels, voxel_channels, hidden_channels=None):
        super().__init__()
        self.bev_channels = bev_channels
        self.voxel_channels = voxel_channels
        hidden_channels = hidden_channels or bev_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(
                bev_channels + voxel_channels * 2,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                bev_channels,
                kernel_size=1,
                padding=0,
                bias=True),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        if self.fusion[-1].bias is not None:
            nn.init.zeros_(self.fusion[-1].bias)

    def _collapse_voxel_bev(self, voxel_feature):
        if voxel_feature.dim() != 5:
            raise ValueError(
                'MapHFMFusionLayer expects voxel feature in '
                f'(B, C, Z, H, W), got shape {tuple(voxel_feature.shape)}.')
        if voxel_feature.shape[1] != self.voxel_channels:
            raise ValueError(
                f'MapHFMFusionLayer expected {self.voxel_channels} voxel '
                f'channels, got {voxel_feature.shape[1]}.')
        return torch.cat([
            voxel_feature.mean(dim=2),
            voxel_feature.max(dim=2).values,
        ], dim=1)

    def forward(self, bev_feature, voxel_feature):
        if bev_feature.dim() != 4:
            raise ValueError(
                'MapHFMFusionLayer expects BEV feature in (B, C, H, W), '
                f'got shape {tuple(bev_feature.shape)}.')
        if bev_feature.shape[1] != self.bev_channels:
            raise ValueError(
                f'MapHFMFusionLayer expected {self.bev_channels} BEV '
                f'channels, got {bev_feature.shape[1]}.')

        voxel_bev = self._collapse_voxel_bev(voxel_feature)
        if voxel_bev.shape[-2:] != bev_feature.shape[-2:]:
            voxel_bev = F.interpolate(
                voxel_bev,
                size=bev_feature.shape[-2:],
                mode='bilinear',
                align_corners=True)
        delta = self.fusion(torch.cat([bev_feature, voxel_bev], dim=1))
        return bev_feature + delta


@BACKBONES.register_module()
class Dual_Branch_Encoder(nn.Module):
    def __init__(
        self,
        z_size = [4, 8, 16],
        vox_feat1=80,
        vox_feat2=24,
        vox_feat3=48*4,
        vox_feat4=48*8,
        voxel_out_channels=48,
        bev_feat_ch1=160,
        bev_feat_ch2=320,
        bev_feat_ch3=640,
        vox_kernel_size=3,
        with_cp=True,
        bev_encoder_backbone=None,
        bev_encoder_neck=None,
        map_bev_encoder_neck=None,
        down_sample_for_3d_pooling=None,
        return_bev_feature=False,
        return_map_feature=False,
        detach_map_feature=False,
        use_map_hfm=False,
        map_hfm_lower_source='vox3',
        map_hfm_with_cp=None,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.return_bev_feature = return_bev_feature
        self.return_map_feature = return_map_feature
        self.detach_map_feature = detach_map_feature
        self.use_map_hfm = use_map_hfm
        if map_hfm_lower_source not in ('vox2', 'vox3'):
            raise ValueError(
                "map_hfm_lower_source must be one of 'vox2' or 'vox3', "
                f'got {map_hfm_lower_source!r}.')
        self.map_hfm_lower_source = map_hfm_lower_source
        self.map_hfm_with_cp = with_cp if map_hfm_with_cp is None else map_hfm_with_cp
        if self.return_map_feature and not self.return_bev_feature:
            raise ValueError('return_map_feature=True requires return_bev_feature=True.')
        if self.return_map_feature and map_bev_encoder_neck is None:
            raise ValueError('return_map_feature=True requires map_bev_encoder_neck.')
        if self.use_map_hfm and map_bev_encoder_neck is None:
            raise ValueError('use_map_hfm=True requires map_bev_encoder_neck.')
        
        # BEV encoder
        self.down_sample_for_3d_pooling = \
                    nn.Conv2d(down_sample_for_3d_pooling[0],
                    down_sample_for_3d_pooling[1],
                    kernel_size=1,
                    padding=0,
                    stride=1)
        self.bev_encoder_backbone = builder.build_backbone(bev_encoder_backbone)

        # Voxel encoder
        self.vox_branch_first = self._make_vox_branch_first(vox_feat1, vox_feat2, vox_kernel_size)
        self.vox_branch1 = self._make_downsample_branch(vox_feat2, vox_feat2, vox_kernel_size)
        self.vox_branch2 = self._make_vox_branch(vox_feat2, vox_feat3, vox_kernel_size, downsample=True)
        self.vox_branch3 = self._make_vox_branch(vox_feat3, vox_feat4, vox_kernel_size, downsample=True)
        self.vox_branch4 = self._make_vox_branch(vox_feat4, vox_feat3, vox_kernel_size)

        # Hierarchical Fusion Module
        self.vox_up_branch2 = self._make_upsample_branch(vox_feat3, vox_feat2, vox_kernel_size)
        self.vox_up_branch3 = self._make_upsample_branch(vox_feat2, vox_feat2, vox_kernel_size)

        self.vox_up_branch_final = nn.Sequential(
            nn.Conv3d(vox_feat2, voxel_out_channels * 2, kernel_size=1, padding=0, stride=1),
            nn.Softplus(),
            nn.Conv3d(voxel_out_channels * 2, voxel_out_channels, kernel_size=1, padding=0, stride=1),
        )

        self.bev_ch1 = self._make_bev_branch(bev_feat_ch3, vox_feat3, z_size[0])
        self.bev_ch2 = self._make_bev_branch(bev_feat_ch2, vox_feat2, z_size[1])
        self.bev_ch3 = self._make_bev_branch(bev_feat_ch1, vox_feat2, z_size[2])

        # for high performance
        self.bev_encoder_neck = builder.build_neck(bev_encoder_neck)
        self.map_bev_encoder_neck = (
            builder.build_neck(map_bev_encoder_neck)
            if map_bev_encoder_neck is not None else None
        )
        if self.use_map_hfm:
            lower_vox_channels = vox_feat3
            self.map_hfm_layers = nn.ModuleList([
                MapHFMFusionLayer(bev_feat_ch1, vox_feat2),
                MapHFMFusionLayer(bev_feat_ch2, vox_feat2),
                MapHFMFusionLayer(bev_feat_ch3, lower_vox_channels),
            ])
        else:
            self.map_hfm_layers = None
        self.voxelize_module = voxelize_module(
            in_dim = voxel_out_channels,
            )
        
    def _make_vox_branch_first(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def _make_vox_branch(self, in_channels, out_channels, kernel_size, downsample=False):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]
        if downsample:
            layers.extend([
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
            ])
        return nn.Sequential(*layers)

    def _make_downsample_branch(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def _make_upsample_branch(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def _make_bev_branch(self, bev_channels, vox_channels, z_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(bev_channels, vox_channels * z_size, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(vox_channels * z_size),
            nn.ReLU(),
        )


    def apply_checkpoint(self, layer, x):
        if self.with_cp and self.training:
            return cp.checkpoint(layer, x)
        else:
            return layer(x)

    def _apply_checkpoint_multi(self, layer, *inputs):
        use_checkpoint = (
            self.training and
            self.map_hfm_with_cp and
            any(torch.is_tensor(t) and t.requires_grad for t in inputs)
        )
        if use_checkpoint:
            return cp.checkpoint(layer, *inputs)
        return layer(*inputs)

    def _build_map_hfm_features(self, multi_scale_bev, vox_res, vox1, vox2, vox3):
        if len(multi_scale_bev) < 3:
            raise ValueError(
                'Map-HFM expects at least 3 BEV pyramid levels, got '
                f'{len(multi_scale_bev)}.')
        lower_vox = vox3 if self.map_hfm_lower_source == 'vox3' else vox2
        bev_sources = multi_scale_bev
        voxel_sources = [vox_res, vox1, lower_vox]
        if self.detach_map_feature:
            bev_sources = [feat.detach() for feat in bev_sources]
            voxel_sources = [feat.detach() for feat in voxel_sources]
        map_hfm_features = []
        for layer, bev_feature, voxel_feature in zip(
                self.map_hfm_layers, bev_sources, voxel_sources):
            map_hfm_features.append(
                self._apply_checkpoint_multi(layer, bev_feature, voxel_feature))
        return map_hfm_features

    def forward(self, x):
        # Voxel Branch
        # checkpoint is crucial for reducing GPU memory usage during training, but require longer training time.
        vox_res = self.apply_checkpoint(self.vox_branch_first, x)
        vox1 = self.apply_checkpoint(self.vox_branch1, vox_res)
        vox2 = self.apply_checkpoint(self.vox_branch2, vox1)
        vox3 = self.apply_checkpoint(self.vox_branch3, vox2)
        vox3 = F.interpolate(vox3, size=vox2.shape[2:], mode='trilinear', align_corners=True)
        vox3 = self.apply_checkpoint(self.vox_branch4, vox3)
        
        # BEV Branch
        pooled_x = torch.cat(x.unbind(dim=2), 1)
        pooled_x = self.down_sample_for_3d_pooling(pooled_x)
        multi_scale_bev = self.bev_encoder_backbone(pooled_x)

        map_bev_feature = None
        if self.map_bev_encoder_neck is not None:
            if self.use_map_hfm:
                map_source = self._build_map_hfm_features(
                    multi_scale_bev, vox_res, vox1, vox2, vox3)
            elif self.detach_map_feature:
                map_source = [feat.detach() for feat in multi_scale_bev]
            else:
                map_source = multi_scale_bev
            map_bev = self.map_bev_encoder_neck(map_source)
            map_bev_feature = map_bev[0] if isinstance(map_bev, (list, tuple)) else map_bev

        # Hierarchical Fusion Module
        B, C, Z, H, W = vox3.shape
        vox3 = vox3 + vox2 + self.bev_ch1(multi_scale_bev[2]).view(B, -1, Z, H, W).contiguous()
        vox_2 = self.apply_checkpoint(self.vox_up_branch2, vox3)
        B, C, Z, H, W = vox_2.shape
        vox_2 = vox_2 + vox1 + self.bev_ch2(multi_scale_bev[1]).view(B, -1, Z, H, W).contiguous()
        vox_raw = self.apply_checkpoint(self.vox_up_branch3, vox_2)
        B, C, Z, H, W = vox_res.shape
        vox_raw = self.apply_checkpoint(self.vox_up_branch_final, vox_res + vox_raw + self.bev_ch3(multi_scale_bev[0]).view(B, -1, Z, H, W).contiguous())
        vox_raw = vox_raw.permute(0,1,3,4,2)

        # residual connection (for high performance)
        bev = self.bev_encoder_neck(multi_scale_bev)
        bev_feature = bev[0]
        vox = self.voxelize_module(bev_feature)
        comprehensive_voxel_feature = vox + vox_raw

        if self.return_map_feature:
            return comprehensive_voxel_feature, bev_feature, map_bev_feature

        if self.return_bev_feature:
            return comprehensive_voxel_feature, bev_feature

        return comprehensive_voxel_feature


@BACKBONES.register_module()
class MapOnly_BEV_Encoder(nn.Module):
    """BEV encoder for map-only upper-bound experiments.

    This keeps the BEV feature path used by `Dual_Branch_Encoder` but omits
    the voxel branch, hierarchical 3D fusion, and occupancy BEV projection.
    """

    def __init__(self,
                 down_sample_for_3d_pooling=None,
                 bev_encoder_backbone=None,
                 map_bev_encoder_neck=None,
                 detach_map_feature=False):
        super().__init__()
        if down_sample_for_3d_pooling is None:
            raise ValueError('MapOnly_BEV_Encoder requires down_sample_for_3d_pooling.')
        if map_bev_encoder_neck is None:
            raise ValueError('MapOnly_BEV_Encoder requires map_bev_encoder_neck.')

        self.detach_map_feature = detach_map_feature
        self.down_sample_for_3d_pooling = nn.Conv2d(
            down_sample_for_3d_pooling[0],
            down_sample_for_3d_pooling[1],
            kernel_size=1,
            padding=0,
            stride=1)
        self.bev_encoder_backbone = builder.build_backbone(bev_encoder_backbone)
        self.map_bev_encoder_neck = builder.build_neck(map_bev_encoder_neck)

    def forward(self, x):
        pooled_x = torch.cat(x.unbind(dim=2), 1)
        pooled_x = self.down_sample_for_3d_pooling(pooled_x)
        multi_scale_bev = self.bev_encoder_backbone(pooled_x)
        if self.detach_map_feature:
            multi_scale_bev = [feat.detach() for feat in multi_scale_bev]
        map_bev = self.map_bev_encoder_neck(multi_scale_bev)
        return map_bev[0] if isinstance(map_bev, (list, tuple)) else map_bev
