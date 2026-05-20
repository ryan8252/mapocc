import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.utils.checkpoint import checkpoint

from mmdet3d.models import BACKBONES


@BACKBONES.register_module()
class MAESTROTaskSpecificFeatureGenerator(BaseModule):
    """Task-specific feature generator used by MAESTRO 2-task ProtoOcc.

    Paper mapping:
    - Task-dependent Feature Transformation: F_t = T_t(F_s)
    - Prototype-wise Features: F_wise = G_t dot F_t
    - Prototype-aware Features: F_aware = gamma_t_scale * F_t
    - Adaptive Feature Enhancement: F_tilde = Conv([F_wise, F_aware])
    - Feature Suppression: F_TS = F_tilde * S_t_supp

    Input uses ProtoOcc's voxel layout (B, C, X, Y, Z). The map branch
    collapses height Z into BEV channels; the occupancy branch keeps 3D layout
    internally and converts back to (B, C, X, Y, Z) before returning.
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 prototype_channels=None,
                 num_prototypes=None,
                 task='occ',
                 voxel_z=16,
                 hidden_channels=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 loss_supp_weight=0.0,
                 loss_name='loss_maestro_supp',
                 init_cfg=None):
        super(MAESTROTaskSpecificFeatureGenerator, self).__init__(init_cfg)
        out_channels = out_channels or in_channels
        prototype_channels = prototype_channels or in_channels
        hidden_channels = hidden_channels or out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prototype_channels = prototype_channels
        self.num_prototypes = num_prototypes or out_channels
        self.task = task
        self.voxel_z = voxel_z
        self.with_cp = with_cp
        self.loss_supp_weight = loss_supp_weight
        self.loss_name = loss_name
        self.is_bev_task = task in ('bev', 'map')

        if self.is_bev_task:
            self.feature_transform = ConvModule(
                in_channels * voxel_z,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True),
            )
            self.fuse = ConvModule(
                out_channels + self.num_prototypes,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True),
            )
            self.score_predictor = nn.Conv2d(out_channels, 1, kernel_size=1)
        else:
            self.feature_transform = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=True),
            )
            self.fuse = ConvModule(
                out_channels + self.num_prototypes,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=True),
            )
            self.score_predictor = nn.Conv3d(out_channels, 1, kernel_size=1)

        self.prototype_proj = nn.Sequential(
            nn.Linear(prototype_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )
        self.prototype_gate = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid(),
        )

    def _transform_feature(self, F_s):
        # Task-dependent Feature Transformation.
        if self.is_bev_task:
            b, c, x, y, z = F_s.shape
            if z != self.voxel_z:
                raise ValueError(
                    f'Expected voxel_z={self.voxel_z}, but got feature Z={z}.')
            bev_feature = F_s.permute(0, 1, 4, 2, 3).reshape(
                b, c * z, x, y)
            if self.with_cp and self.training:
                return checkpoint(self.feature_transform, bev_feature)
            return self.feature_transform(bev_feature)

        conv_feature = F_s.permute(0, 1, 4, 2, 3).contiguous()
        if self.with_cp and self.training:
            return checkpoint(self.feature_transform, conv_feature)
        return self.feature_transform(conv_feature)

    def _flatten_feature(self, feature):
        return feature.flatten(2)

    def _unflatten_feature(self, feature_flat, feature_like):
        return feature_flat.reshape(
            feature_like.shape[0], feature_flat.shape[1], *feature_like.shape[2:])

    def _adaptive_feature_enhancement(self, F_t, G_t):
        F_t_flat = self._flatten_feature(F_t)
        b, c = F_t_flat.shape[:2]
        if G_t is None or G_t.size(1) == 0:
            F_wise = F_t_flat.new_zeros(b, self.num_prototypes, F_t_flat.size(-1))
            F_aware = F_t_flat.new_zeros(F_t_flat.shape)
            return F_wise, F_aware

        if G_t.size(1) != self.num_prototypes:
            raise ValueError(
                f'Expected {self.num_prototypes} task prototypes for {self.task}, '
                f'but got {G_t.size(1)}.')

        G_t = self.prototype_proj(G_t)

        # Prototype-wise Features: dot product between each prototype and F_t.
        F_wise = torch.einsum('bnc,bcs->bns', G_t, F_t_flat)

        # Prototype-aware Features: pooled prototype group predicts gamma_t_scale.
        G_t_mean = G_t.mean(dim=1)
        G_t_max = G_t.max(dim=1).values
        gamma_t_scale = self.prototype_gate(torch.cat([G_t_mean, G_t_max], dim=-1))
        F_aware = F_t_flat * gamma_t_scale.view(b, c, 1)
        return F_wise, F_aware

    def _resize_target(self, target_mask, score_logits):
        if target_mask is None:
            return None

        target = target_mask.float()
        if self.is_bev_task:
            if target.dim() == 4:
                target = target.max(dim=1).values
            if target.dim() != 3:
                raise ValueError(
                    f'Expected BEV target with 3 or 4 dims, got {target.dim()}.')
            target = target.unsqueeze(1)
            if target.shape[-2:] != score_logits.shape[-2:]:
                target = F.interpolate(
                    target, size=score_logits.shape[-2:], mode='nearest')
            return target

        if target.dim() != 4:
            raise ValueError(
                f'Expected occupancy target with 4 dims, got {target.dim()}.')
        target = target.permute(0, 3, 1, 2).unsqueeze(1)
        if target.shape[-3:] != score_logits.shape[-3:]:
            target = F.interpolate(
                target, size=score_logits.shape[-3:], mode='nearest')
        return target

    def _suppression_loss(self, score_logits, target_mask):
        if self.loss_supp_weight <= 0 or target_mask is None:
            return {}

        target = self._resize_target(target_mask, score_logits)
        loss = F.binary_cross_entropy_with_logits(score_logits, target)
        return {self.loss_name: loss * self.loss_supp_weight}

    def _crop_score_logits(self, score_logits, target_crop_slices=None):
        if target_crop_slices is None:
            return score_logits

        x_slice, y_slice = target_crop_slices
        if self.is_bev_task:
            return score_logits[:, :, x_slice, y_slice]
        return score_logits[:, :, :, x_slice, y_slice]

    def forward(self,
                voxel_feature,
                prototypes,
                target_mask=None,
                target_crop_slices=None):
        # Shared Voxel Feature (F_s) -> transformed task feature (F_t).
        F_s = voxel_feature
        G_t = prototypes
        F_t = self._transform_feature(F_s)

        # Adaptive Feature Enhancement.
        F_wise_flat, F_aware_flat = self._adaptive_feature_enhancement(F_t, G_t)
        F_wise = self._unflatten_feature(F_wise_flat, F_t)
        F_aware = self._unflatten_feature(F_aware_flat, F_t)
        enhanced_input = torch.cat([F_wise, F_aware], dim=1)

        if self.with_cp and self.training:
            F_tilde = checkpoint(self.fuse, enhanced_input)
        else:
            F_tilde = self.fuse(enhanced_input)

        # Feature Suppression Score: S_t_supp is predicted from F_aware.
        S_t_supp_logits = self.score_predictor(F_aware)
        S_t_supp = torch.sigmoid(S_t_supp_logits)

        # Task-Specific Feature: F_TS = F_tilde * S_t_supp.
        F_TS = F_tilde * S_t_supp
        loss_logits = self._crop_score_logits(
            S_t_supp_logits, target_crop_slices)
        losses = self._suppression_loss(loss_logits, target_mask)

        if self.is_bev_task:
            return F_TS, losses

        F_TS = F_TS.permute(0, 1, 3, 4, 2).contiguous()
        return F_TS, losses
