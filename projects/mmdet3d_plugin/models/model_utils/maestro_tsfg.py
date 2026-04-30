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

    The input convention is ProtoOcc's voxel feature layout (B, C, X, Y, Z).
    For map segmentation, the height axis is collapsed into channels and a 2D
    feature is produced. For occupancy, the 3D layout is kept and the output is
    converted back to (B, C, X, Y, Z).
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 prototype_channels=None,
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
                out_channels * 3,
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
                out_channels * 3,
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
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def _transform_feature(self, voxel_feature):
        if self.is_bev_task:
            b, c, x, y, z = voxel_feature.shape
            if z != self.voxel_z:
                raise ValueError(
                    f'Expected voxel_z={self.voxel_z}, but got feature Z={z}.')
            bev_feature = voxel_feature.permute(0, 1, 4, 2, 3).reshape(
                b, c * z, x, y)
            if self.with_cp and self.training:
                return checkpoint(self.feature_transform, bev_feature)
            return self.feature_transform(bev_feature)

        conv_feature = voxel_feature.permute(0, 1, 4, 2, 3).contiguous()
        if self.with_cp and self.training:
            return checkpoint(self.feature_transform, conv_feature)
        return self.feature_transform(conv_feature)

    def _flatten_feature(self, feature):
        return feature.flatten(2)

    def _unflatten_feature(self, feature_flat, feature_like):
        return feature_flat.reshape(
            feature_like.shape[0], feature_flat.shape[1], *feature_like.shape[2:])

    def _prototype_context(self, feature, prototypes):
        b, c = feature.shape[:2]
        feature_flat = self._flatten_feature(feature)
        if prototypes is None or prototypes.size(1) == 0:
            zeros = feature_flat.new_zeros(feature_flat.shape)
            return zeros, zeros

        projected = self.prototype_proj(prototypes)
        feature_norm = F.normalize(feature_flat, dim=1)
        proto_norm = F.normalize(projected, dim=-1)
        scores = torch.einsum('bcs,bnc->bns', feature_norm, proto_norm)
        weights = F.softmax(scores, dim=1)
        prototype_wise = torch.einsum('bns,bnc->bcs', weights, projected)

        proto_mean = projected.mean(dim=1)
        proto_max = projected.max(dim=1).values
        gate = self.prototype_gate(torch.cat([proto_mean, proto_max], dim=-1))
        prototype_aware = feature_flat * gate.view(b, c, 1)
        return prototype_wise, prototype_aware

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

    def forward(self, voxel_feature, prototypes, target_mask=None):
        task_feature = self._transform_feature(voxel_feature)
        proto_wise, proto_aware = self._prototype_context(task_feature, prototypes)
        enhanced_input = torch.cat([
            self._flatten_feature(task_feature),
            proto_wise,
            proto_aware,
        ], dim=1)
        enhanced_input = self._unflatten_feature(enhanced_input, task_feature)

        if self.with_cp and self.training:
            enhanced = checkpoint(self.fuse, enhanced_input)
        else:
            enhanced = self.fuse(enhanced_input)

        score_logits = self.score_predictor(enhanced)
        relevance = torch.sigmoid(score_logits)
        output = task_feature + self.residual_scale * (
            enhanced * relevance - task_feature * (1.0 - relevance))
        losses = self._suppression_loss(score_logits, target_mask)

        if self.is_bev_task:
            return output, losses

        output = output.permute(0, 1, 3, 4, 2).contiguous()
        return output, losses
