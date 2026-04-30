import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from mmdet3d.models import BACKBONES


@BACKBONES.register_module()
class MAESTROClasswisePrototypeGenerator(BaseModule):
    """Class-wise prototype generator for the 2-task MAESTRO variant.

    The module follows MAESTRO's CPG idea: predict semantic confidence on the
    shared voxel feature, pool class-wise features, then expose foreground and
    background prototype groups for downstream TSFG modules.
    """

    def __init__(self,
                 in_channels,
                 num_classes=18,
                 hidden_channels=None,
                 foreground_classes=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                 background_classes=(11, 12, 13, 14, 15, 16),
                 use_hard_masks=False,
                 ignore_index=255,
                 loss_weight=0.0,
                 init_cfg=None):
        super(MAESTROClasswisePrototypeGenerator, self).__init__(init_cfg)
        hidden_channels = hidden_channels or in_channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_hard_masks = use_hard_masks
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

        self.mask_classifier = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, num_classes, kernel_size=1),
        )
        self.class_embeddings = nn.Embedding(num_classes, in_channels)

        self.register_buffer(
            'foreground_indices',
            torch.as_tensor(foreground_classes, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            'background_indices',
            torch.as_tensor(background_classes, dtype=torch.long),
            persistent=False,
        )

    def _to_conv3d(self, voxel_feature):
        # ProtoOcc DBE returns (B, C, X, Y, Z); Conv3d expects (B, C, Z, X, Y).
        return voxel_feature.permute(0, 1, 4, 2, 3).contiguous()

    def _from_conv3d_logits(self, logits):
        # Return logits as (B, K, X, Y, Z), matching voxel labels.
        return logits.permute(0, 1, 3, 4, 2).contiguous()

    def _build_assignment_weights(self, logits):
        if not self.use_hard_masks:
            return F.softmax(logits, dim=1)

        assignment = logits.argmax(dim=1)
        weights = F.one_hot(assignment, num_classes=self.num_classes)
        return weights.permute(0, 4, 1, 2, 3).to(dtype=logits.dtype)

    def _gather_group(self, prototypes, indices):
        if indices.numel() == 0:
            return prototypes.new_zeros(prototypes.size(0), 0, prototypes.size(-1))
        indices = indices.to(prototypes.device)
        return prototypes.index_select(1, indices)

    def forward(self, voxel_feature):
        conv_feature = self._to_conv3d(voxel_feature)
        logits_3d = self.mask_classifier(conv_feature)
        weights_3d = self._build_assignment_weights(logits_3d)

        feature_flat = conv_feature.flatten(2)
        weight_flat = weights_3d.flatten(2)
        denom = weight_flat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        prototypes = torch.einsum('bks,bcs->bkc', weight_flat, feature_flat) / denom
        prototypes = prototypes + self.class_embeddings.weight.unsqueeze(0)

        logits = self._from_conv3d_logits(logits_3d)
        foreground = self._gather_group(prototypes, self.foreground_indices)
        background = self._gather_group(prototypes, self.background_indices)

        return dict(
            logits=logits,
            prototypes=prototypes,
            foreground=foreground,
            background=background,
            occupancy=torch.cat([foreground, background], dim=1),
        )

    def loss(self, logits, voxel_semantics, valid_mask=None):
        if self.loss_weight <= 0 or voxel_semantics is None:
            return {}

        target = voxel_semantics.long()
        if valid_mask is not None:
            target = target.clone()
            target[~valid_mask.bool()] = self.ignore_index

        loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        return {'loss_maestro_cpg_ce': loss * self.loss_weight}
