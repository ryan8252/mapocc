from torch import nn

from mmdet3d.models import BACKBONES
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmdet3d.models import builder
import torch


def _gate_logit(init_value):
    init_value = float(init_value)
    init_value = min(max(init_value, 1e-4), 1.0 - 1e-4)
    value = torch.tensor(init_value / (1.0 - init_value))
    return torch.log(value).item()


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


@BACKBONES.register_module()
class PerScaleMapResidualAdapter(nn.Module):
    """Zero-init per-scale residual adapter for the map BEV branch."""

    def __init__(self,
                 in_channels=[160, 320, 640],
                 residual_scale=1.0,
                 with_cp=False,
                 detach_input=False,
                 learnable_residual_gate=False,
                 residual_gate_init=0.1):
        super().__init__()
        self.in_channels = list(in_channels)
        self.residual_scale = residual_scale
        self.with_cp = with_cp
        self.detach_input = detach_input
        self.learnable_residual_gate = bool(learnable_residual_gate)
        self.blocks = nn.ModuleList([
            self._make_block(channels) for channels in self.in_channels
        ])
        if self.learnable_residual_gate:
            gate_logit = _gate_logit(residual_gate_init)
            self.residual_gates = nn.ParameterList([
                nn.Parameter(torch.full((1,), gate_logit))
                for _ in self.in_channels
            ])
        else:
            self.residual_gates = None

    def _make_block(self, channels):
        block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        nn.init.constant_(block[-1].weight, 0)
        if block[-1].bias is not None:
            nn.init.constant_(block[-1].bias, 0)
        return block

    def _forward_block(self, block, feat):
        if self.with_cp and self.training and feat.requires_grad:
            return cp.checkpoint(block, feat)
        return block(feat)

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            raise TypeError('PerScaleMapResidualAdapter expects a list/tuple of BEV features.')
        if len(feats) != len(self.blocks):
            raise ValueError(
                f'Expected {len(self.blocks)} BEV feature scales, got {len(feats)}.')

        outputs = []
        gates = (
            self.residual_gates if self.residual_gates is not None
            else [None] * len(self.blocks))
        for feat, block, channels, gate in zip(
                feats, self.blocks, self.in_channels, gates):
            if feat.shape[1] != channels:
                raise ValueError(
                    f'Expected {channels} channels, got {feat.shape[1]}.')
            source = feat.detach() if self.detach_input else feat
            residual = self._forward_block(block, source)
            scale = self.residual_scale
            if gate is not None:
                scale = scale * torch.sigmoid(gate).view(1, 1, 1, 1)
            outputs.append(source + scale * residual)

        return tuple(outputs) if isinstance(feats, tuple) else outputs


class MapPreBackboneAdapter(nn.Module):
    """Zero-init residual adapter on the 200x200 BEV feature before backbone."""

    def __init__(self,
                 channels=160,
                 adapter_type='res_bottleneck',
                 hidden_channels=64,
                 residual_scale=1.0,
                 detach_input=False,
                 with_cp=False):
        super().__init__()
        valid_adapter_types = ('res_bottleneck', 'convnext')
        if adapter_type not in valid_adapter_types:
            raise ValueError(
                'MapPreBackboneAdapter supports adapter_type in '
                f'{valid_adapter_types}, got {adapter_type!r}.')
        self.channels = channels
        self.adapter_type = adapter_type
        self.residual_scale = residual_scale
        self.detach_input = detach_input
        self.with_cp = with_cp

        if adapter_type == 'res_bottleneck':
            self.block = self._make_res_bottleneck(channels, hidden_channels)
        else:
            self.block = self._make_convnext_block(channels)

    def _make_res_bottleneck(self, channels, hidden_channels):
        block = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        nn.init.zeros_(block[-1].weight)
        if block[-1].bias is not None:
            nn.init.zeros_(block[-1].bias)
        return block

    def _make_convnext_block(self, channels):
        block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=7,
                padding=3,
                groups=channels,
                bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
        )
        nn.init.zeros_(block[-1].weight)
        if block[-1].bias is not None:
            nn.init.zeros_(block[-1].bias)
        return block

    def _forward_block(self, x):
        if self.with_cp and self.training and x.requires_grad:
            return cp.checkpoint(self.block, x)
        return self.block(x)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                'MapPreBackboneAdapter expects input shape [B, C, H, W], '
                f'got {tuple(x.shape)}.')
        if x.shape[1] != self.channels:
            raise ValueError(
                f'MapPreBackboneAdapter expected {self.channels} channels, '
                f'got {x.shape[1]}.')
        source = x.detach() if self.detach_input else x
        residual = self._forward_block(source)
        return source + self.residual_scale * residual


class MapHFMFusionLayer(nn.Module):
    """Fuse one BEV pyramid level with one early voxel-branch feature.

    Injects voxel-branch geometry (mean + max over height) into the map BEV
    path as a zero-init residual, so the map branch sees 3D geometry that the
    pure-2D BEV path lacks. Plain ``nn.Module`` (not registered) because it is
    only instantiated internally by ``Dual_Branch_Encoder``.
    """

    def __init__(self,
                 bev_channels,
                 voxel_channels,
                 hidden_channels=None,
                 learnable_residual_gate=False,
                 residual_gate_init=0.1):
        super().__init__()
        self.bev_channels = bev_channels
        self.voxel_channels = voxel_channels
        self.learnable_residual_gate = bool(learnable_residual_gate)
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
        if self.learnable_residual_gate:
            self.residual_gate = nn.Parameter(
                torch.full((1,), _gate_logit(residual_gate_init)))
        else:
            self.residual_gate = None

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
        if self.residual_gate is not None:
            delta = torch.sigmoid(self.residual_gate).view(1, 1, 1, 1) * delta
        return bev_feature + delta


@BACKBONES.register_module()
class MapZResidualLayer(nn.Module):
    """Zero-init map residual from the full-resolution LSS voxel feature.

    This preserves the shared BEV/map-neck path and adds only an optional
    map-specific correction before BEVSegHead.
    """

    def __init__(self,
                 in_channels=80,
                 z_channels=16,
                 out_channels=128,
                 hidden_channels=128,
                 z_projection='cat_z',
                 num_height_patterns=4,
                 residual_scale=1.0,
                 detach_input=True,
                 with_cp=True,
                 zero_init_z_logits=True):
        super().__init__()
        if z_projection == 'weighted_pool':
            z_projection = 'learned_pool'
        valid_z_projection = ('cat_z', 'learned_pool')
        if z_projection not in valid_z_projection:
            raise ValueError(
                'MapZResidualLayer supports z_projection in '
                f'{valid_z_projection}, got {z_projection!r}.')
        if z_projection == 'learned_pool' and num_height_patterns < 1:
            raise ValueError('num_height_patterns must be >= 1.')

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.z_projection = z_projection
        self.num_height_patterns = num_height_patterns
        self.residual_scale = residual_scale
        self.detach_input = detach_input
        self.with_cp = with_cp

        self.refine3d = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
        if z_projection == 'learned_pool':
            self.height_logits = nn.Conv3d(
                in_channels, num_height_patterns, kernel_size=1)
            project_in_channels = in_channels * num_height_patterns
            if zero_init_z_logits:
                nn.init.zeros_(self.height_logits.weight)
                if self.height_logits.bias is not None:
                    nn.init.zeros_(self.height_logits.bias)
        else:
            self.height_logits = None
            project_in_channels = in_channels * z_channels

        self.project2d = nn.Sequential(
            nn.Conv2d(
                project_in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.project2d[-1].weight)
        if self.project2d[-1].bias is not None:
            nn.init.zeros_(self.project2d[-1].bias)

    def _forward_refine3d(self, x):
        if self.with_cp and self.training and x.requires_grad:
            return cp.checkpoint(self.refine3d, x)
        return self.refine3d(x)

    def _forward_project2d(self, x):
        if self.with_cp and self.training and x.requires_grad:
            return cp.checkpoint(self.project2d, x)
        return self.project2d(x)

    def _z_project(self, x):
        if self.z_projection == 'cat_z':
            return torch.cat(x.unbind(dim=2), dim=1)

        z_weight = torch.softmax(self.height_logits(x), dim=2)
        pooled = []
        for k in range(self.num_height_patterns):
            weight = z_weight[:, k:k + 1]
            pooled.append(torch.sum(x * weight, dim=2))
        return torch.cat(pooled, dim=1)

    def forward(self, x, output_size=None):
        if x.dim() != 5:
            raise ValueError(
                'MapZResidualLayer expects input shape [B, C, Z, H, W], '
                f'got {tuple(x.shape)}.')
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f'MapZResidualLayer expected {self.in_channels} channels, '
                f'got {x.shape[1]}.')
        if x.shape[2] != self.z_channels:
            raise ValueError(
                f'MapZResidualLayer expected Z={self.z_channels}, '
                f'got Z={x.shape[2]}.')

        source = x.detach() if self.detach_input else x
        refined = source + self._forward_refine3d(source)
        bev = self._z_project(refined)
        residual = self._forward_project2d(bev)
        if output_size is not None and residual.shape[-2:] != output_size:
            residual = F.interpolate(
                residual, size=output_size, mode='bilinear',
                align_corners=True)
        return self.residual_scale * residual


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
        map_residual_adapter=None,
        down_sample_for_3d_pooling=None,
        return_bev_feature=False,
        return_map_feature=False,
        detach_map_feature=False,
        use_map_hfm=False,
        map_hfm_lower_source='vox3',
        map_hfm_with_cp=None,
        map_hfm_learnable_residual_gate=False,
        map_hfm_residual_gate_init=0.1,
        map_highres_skip=False,
        map_highres_hidden=128,
        map_highres_detach=True,
        map_highres_fusion='add',
        use_map_z_aware_compression=False,
        map_z_compression_type='conv3d',
        map_height_attention_groups=4,
        map_z_compression_detach=False,
        map_z_compression_with_cp=None,
        map_z_residual=None,
        use_map_pre_backbone_adapter=False,
        map_pre_backbone_adapter_type='res_bottleneck',
        map_pre_backbone_adapter_hidden=64,
        map_pre_backbone_adapter_detach=False,
        map_pre_backbone_adapter_with_cp=None,
        use_map_adapter=None,
        map_adapter_type=None,
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
        self.map_hfm_learnable_residual_gate = bool(
            map_hfm_learnable_residual_gate)
        self.map_hfm_residual_gate_init = float(map_hfm_residual_gate_init)
        self.map_highres_skip = map_highres_skip
        self.map_highres_detach = map_highres_detach
        valid_highres_fusions = ('add', 'concat', 'gated')
        if map_highres_fusion not in valid_highres_fusions:
            raise ValueError(
                'map_highres_fusion must be one of '
                f'{valid_highres_fusions}, got {map_highres_fusion!r}.')
        self.map_highres_fusion = map_highres_fusion
        valid_z_compression_types = ('conv3d', 'height_attention', 'sum')
        if map_z_compression_type not in valid_z_compression_types:
            raise ValueError(
                'map_z_compression_type must be one of '
                f'{valid_z_compression_types}, got {map_z_compression_type!r}.')
        if map_height_attention_groups < 1:
            raise ValueError('map_height_attention_groups must be >= 1.')
        self.use_map_z_aware_compression = use_map_z_aware_compression
        self.map_z_compression_type = map_z_compression_type
        self.map_height_attention_groups = map_height_attention_groups
        self.map_z_compression_detach = map_z_compression_detach
        self.map_z_compression_with_cp = (
            with_cp if map_z_compression_with_cp is None
            else map_z_compression_with_cp)
        if use_map_adapter is not None:
            use_map_pre_backbone_adapter = use_map_adapter
        if map_adapter_type is not None:
            map_pre_backbone_adapter_type = map_adapter_type
        self.use_map_pre_backbone_adapter = use_map_pre_backbone_adapter
        self.map_pre_backbone_adapter_type = map_pre_backbone_adapter_type
        self.map_pre_backbone_adapter_with_cp = (
            with_cp if map_pre_backbone_adapter_with_cp is None
            else map_pre_backbone_adapter_with_cp)
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
        map_x0_channels = down_sample_for_3d_pooling[1]
        if self.use_map_z_aware_compression:
            if self.map_z_compression_type == 'conv3d':
                self.map_z_refine_3d = self._make_map_z_refine3d(vox_feat1)
                self.map_height_attn = None
                self.map_height_project = None
                self.map_z_sum_project = None
            elif self.map_z_compression_type == 'sum':
                self.map_z_refine_3d = None
                self.map_height_attn = None
                self.map_height_project = None
                self.map_z_sum_project = nn.Conv2d(
                    vox_feat1,
                    map_x0_channels,
                    kernel_size=1,
                    padding=0)
            else:
                self.map_z_refine_3d = None
                self.map_height_attn = nn.Conv3d(
                    vox_feat1, map_height_attention_groups, kernel_size=1)
                self.map_height_project = nn.Conv2d(
                    vox_feat1 * map_height_attention_groups,
                    map_x0_channels,
                    kernel_size=1)
                nn.init.zeros_(self.map_height_attn.weight)
                if self.map_height_attn.bias is not None:
                    nn.init.zeros_(self.map_height_attn.bias)
                self.map_z_sum_project = None
        else:
            self.map_z_refine_3d = None
            self.map_height_attn = None
            self.map_height_project = None
            self.map_z_sum_project = None
        self.map_pre_backbone_adapter = (
            MapPreBackboneAdapter(
                channels=map_x0_channels,
                adapter_type=map_pre_backbone_adapter_type,
                hidden_channels=map_pre_backbone_adapter_hidden,
                residual_scale=1.0,
                detach_input=map_pre_backbone_adapter_detach,
                with_cp=self.map_pre_backbone_adapter_with_cp)
            if self.use_map_pre_backbone_adapter else None
        )
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
        self.map_residual_adapter = (
            builder.build_backbone(map_residual_adapter)
            if map_residual_adapter is not None else None
        )
        self.map_z_residual = (
            builder.build_backbone(map_z_residual)
            if map_z_residual is not None else None
        )
        if self.use_map_hfm:
            lower_vox_channels = vox_feat3
            self.map_hfm_layers = nn.ModuleList([
                MapHFMFusionLayer(
                    bev_feat_ch1,
                    vox_feat2,
                    learnable_residual_gate=self.map_hfm_learnable_residual_gate,
                    residual_gate_init=self.map_hfm_residual_gate_init),
                MapHFMFusionLayer(
                    bev_feat_ch2,
                    vox_feat2,
                    learnable_residual_gate=self.map_hfm_learnable_residual_gate,
                    residual_gate_init=self.map_hfm_residual_gate_init),
                MapHFMFusionLayer(
                    bev_feat_ch3,
                    lower_vox_channels,
                    learnable_residual_gate=self.map_hfm_learnable_residual_gate,
                    residual_gate_init=self.map_hfm_residual_gate_init),
            ])
        else:
            self.map_hfm_layers = None
        if self.map_highres_skip:
            if self.map_bev_encoder_neck is None:
                raise ValueError('map_highres_skip=True requires map_bev_encoder_neck.')
            highres_in = down_sample_for_3d_pooling[1]
            highres_out = getattr(self.map_bev_encoder_neck, 'out_channels', None)
            if highres_out is None:
                raise ValueError(
                    'map_highres_skip requires the map neck to expose out_channels.')
            # Shallow stride-1 stack on the full-resolution pre-backbone BEV
            # feature (pooled_x, 200x200). Gives the map head real high-res
            # spatial detail instead of bilinearly upsampled 100x100 content.
            # Final conv zero-init -> step-0 equals the baseline map feature.
            self.map_highres_block = nn.Sequential(
                nn.Conv2d(highres_in, map_highres_hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(map_highres_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(map_highres_hidden, map_highres_hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(map_highres_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(map_highres_hidden, highres_out, kernel_size=1),
            )
            nn.init.zeros_(self.map_highres_block[-1].weight)
            if self.map_highres_block[-1].bias is not None:
                nn.init.zeros_(self.map_highres_block[-1].bias)
            if self.map_highres_fusion == 'gated':
                self.map_highres_gate = nn.Conv2d(
                    highres_out * 2, highres_out, kernel_size=1)
                self.map_highres_concat = None
            elif self.map_highres_fusion == 'concat':
                self.map_highres_gate = None
                self.map_highres_concat = nn.Conv2d(
                    highres_out * 2, highres_out, kernel_size=1)
                nn.init.zeros_(self.map_highres_concat.weight)
                if self.map_highres_concat.bias is not None:
                    nn.init.zeros_(self.map_highres_concat.bias)
            else:
                self.map_highres_gate = None
                self.map_highres_concat = None
        else:
            self.map_highres_block = None
            self.map_highres_gate = None
            self.map_highres_concat = None
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

    def _make_map_z_refine3d(self, channels):
        block = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        nn.init.zeros_(block[-1].weight)
        nn.init.zeros_(block[-1].bias)
        return block

    def _flatten_z_as_channels(self, x):
        return torch.cat(x.unbind(dim=2), dim=1)

    def _forward_map_z_refine3d(self, x):
        if (self.map_z_compression_with_cp and self.training
                and x.requires_grad):
            return cp.checkpoint(self.map_z_refine_3d, x)
        return self.map_z_refine_3d(x)

    def _build_map_z_aware_x0(self, x):
        source = x.detach() if self.map_z_compression_detach else x
        if self.map_z_compression_type == 'conv3d':
            refined = source + self._forward_map_z_refine3d(source)
            return self.down_sample_for_3d_pooling(
                self._flatten_z_as_channels(refined))
        if self.map_z_compression_type == 'sum':
            return self.map_z_sum_project(source.sum(dim=2))

        height_logits = self.map_height_attn(source)
        height_weight = torch.softmax(height_logits, dim=2)
        pooled = []
        for k in range(self.map_height_attention_groups):
            pooled.append(torch.sum(
                source * height_weight[:, k:k + 1], dim=2))
        return self.map_height_project(torch.cat(pooled, dim=1))


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

    def _get_map_bev_tensor(self, map_bev_feature):
        if isinstance(map_bev_feature, dict):
            map_bev_tensor = map_bev_feature.get('bev_feature', None)
            if map_bev_tensor is None:
                raise ValueError(
                    'map_bev_feature dict must contain `bev_feature`.')
            return map_bev_tensor
        return map_bev_feature

    def _set_map_bev_tensor(self, map_bev_feature, map_bev_tensor):
        if isinstance(map_bev_feature, dict):
            map_bev_feature = dict(map_bev_feature)
            map_bev_feature['bev_feature'] = map_bev_tensor
            return map_bev_feature
        return map_bev_tensor

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
        pooled_x = self._flatten_z_as_channels(x)
        pooled_x = self.down_sample_for_3d_pooling(pooled_x)
        use_map_specific_backbone = (
            self.map_bev_encoder_neck is not None and (
                self.use_map_z_aware_compression
                or self.map_pre_backbone_adapter is not None))
        multi_scale_bev = self.bev_encoder_backbone(pooled_x)
        map_pooled_x = pooled_x
        map_multi_scale_bev = multi_scale_bev
        if self.map_bev_encoder_neck is not None:
            if self.use_map_z_aware_compression:
                map_pooled_x = self._build_map_z_aware_x0(x)
            if self.map_pre_backbone_adapter is not None:
                map_pooled_x = self.map_pre_backbone_adapter(map_pooled_x)
            map_multi_scale_bev = (
                self.bev_encoder_backbone(map_pooled_x)
                if use_map_specific_backbone else multi_scale_bev)

        map_bev_feature = None
        if self.map_bev_encoder_neck is not None:
            if self.use_map_hfm:
                # Inject voxel-branch geometry into the map BEV path. Detach of
                # the shared sources (if enabled) is handled inside the builder.
                map_source = self._build_map_hfm_features(
                    map_multi_scale_bev, vox_res, vox1, vox2, vox3)
            elif self.detach_map_feature:
                map_source = [feat.detach() for feat in map_multi_scale_bev]
            else:
                map_source = map_multi_scale_bev
            if self.map_residual_adapter is not None:
                # 2D zero-init refinement, applied after the geometry injection.
                map_source = self.map_residual_adapter(map_source)
            map_bev = self.map_bev_encoder_neck(map_source)
            map_bev_feature = map_bev[0] if isinstance(map_bev, (list, tuple)) else map_bev
            if self.map_highres_skip:
                # Real full-resolution (200x200) detail from the pre-backbone
                # BEV. Source detached by default so map gradients don't reshape
                # the occ-shared pooling; the block itself stays trainable.
                map_bev_tensor = self._get_map_bev_tensor(map_bev_feature)
                highres_source = (
                    map_pooled_x.detach()
                    if self.map_highres_detach else map_pooled_x)
                if (self.with_cp and self.training
                        and highres_source.requires_grad):
                    highres = cp.checkpoint(self.map_highres_block, highres_source)
                else:
                    highres = self.map_highres_block(highres_source)
                if highres.shape[-2:] != map_bev_tensor.shape[-2:]:
                    highres = F.interpolate(
                        highres, size=map_bev_tensor.shape[-2:],
                        mode='bilinear', align_corners=True)
                if self.map_highres_fusion == 'gated':
                    gate = torch.sigmoid(self.map_highres_gate(torch.cat(
                        [map_bev_tensor, highres], dim=1)))
                    map_bev_tensor = map_bev_tensor + gate * highres
                elif self.map_highres_fusion == 'concat':
                    delta = self.map_highres_concat(torch.cat(
                        [map_bev_tensor, highres], dim=1))
                    map_bev_tensor = map_bev_tensor + delta
                else:
                    map_bev_tensor = map_bev_tensor + highres
                map_bev_feature = self._set_map_bev_tensor(
                    map_bev_feature, map_bev_tensor)
            if self.map_z_residual is not None:
                map_bev_tensor = self._get_map_bev_tensor(map_bev_feature)
                z_residual = self.map_z_residual(
                    x, output_size=map_bev_tensor.shape[-2:])
                map_bev_feature = self._set_map_bev_tensor(
                    map_bev_feature, map_bev_tensor + z_residual)

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


@BACKBONES.register_module()
class MapOnly_BEVFusion_Encoder(nn.Module):
    """Map-only BEVFusion-style 2D decoder.

    The intended input is a 2D LSS BEV feature [B, C, H, W], produced by a
    one-bin zbound and collapse_z=True. A 5D fallback is kept for diagnostics.
    """

    def __init__(self,
                 bev_decoder_backbone=None,
                 bev_decoder_neck=None,
                 z_collapse='sum'):
        super().__init__()
        if bev_decoder_backbone is None:
            raise ValueError(
                'MapOnly_BEVFusion_Encoder requires bev_decoder_backbone.')
        if bev_decoder_neck is None:
            raise ValueError(
                'MapOnly_BEVFusion_Encoder requires bev_decoder_neck.')
        self.z_collapse = str(z_collapse)
        self.bev_decoder_backbone = builder.build_backbone(
            bev_decoder_backbone)
        self.bev_decoder_neck = builder.build_neck(bev_decoder_neck)

    def _collapse_if_needed(self, x):
        if x.dim() == 4:
            return x
        if x.dim() != 5:
            raise ValueError(
                'MapOnly_BEVFusion_Encoder expects a 4D BEV feature or a '
                f'5D voxel feature, got shape {tuple(x.shape)}.')
        if self.z_collapse == 'sum':
            return x.sum(dim=2)
        if self.z_collapse == 'mean':
            return x.mean(dim=2)
        if self.z_collapse == 'max':
            return x.max(dim=2).values
        if self.z_collapse == 'cat':
            return torch.cat(x.unbind(dim=2), dim=1)
        raise ValueError(
            f'Unsupported z_collapse={self.z_collapse}. Expected one of '
            '"sum", "mean", "max", or "cat".')

    def forward(self, x):
        x = self._collapse_if_needed(x)
        multi_scale_bev = self.bev_decoder_backbone(x)
        return self.bev_decoder_neck(multi_scale_bev)


@BACKBONES.register_module()
class MapOnly_MTE_Encoder(nn.Module):
    """Map-only encoder using a MapTopologyEncoder (learned softmax-Z pooling)
    in place of ``MapOnly_BEV_Encoder``'s fixed cat-Z flatten + shared BEV
    backbone.

    Diagnostic for Path B: does a map-private, learned-Z BEV backbone raise the
    map-only ceiling above the shared-backbone map-only upper bound (48.34)? If
    yes, the map-private branch is worth building; if no, learned-Z pooling
    loses too much height detail versus cat-Z and we should stay with residuals.
    """

    def __init__(self,
                 map_topology_encoder=None,
                 map_bev_encoder_neck=None,
                 detach_map_feature=False):
        super().__init__()
        if map_topology_encoder is None:
            raise ValueError('MapOnly_MTE_Encoder requires map_topology_encoder.')
        if map_bev_encoder_neck is None:
            raise ValueError('MapOnly_MTE_Encoder requires map_bev_encoder_neck.')
        self.detach_map_feature = detach_map_feature
        self.map_topology_encoder = builder.build_backbone(map_topology_encoder)
        self.map_bev_encoder_neck = builder.build_neck(map_bev_encoder_neck)

    def forward(self, x):
        # x: LSS voxel feature [B, C, Z, H, W]
        multi_scale_map = self.map_topology_encoder(x)
        if self.detach_map_feature:
            multi_scale_map = [feat.detach() for feat in multi_scale_map]
        map_bev = self.map_bev_encoder_neck(multi_scale_map)
        return map_bev[0] if isinstance(map_bev, (list, tuple)) else map_bev
