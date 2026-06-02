import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from mmdet3d.models import BACKBONES
from ..losses.lovasz_softmax import lovasz_softmax


@BACKBONES.register_module()
class MAESTROClasswisePrototypeGenerator(BaseModule):
    """Class-wise prototype generator for the 2-task MAESTRO variant.

    Paper mapping:
    - Shared Voxel Feature: F_s
    - Mask Classifier: S_v = h_mask(F_s)
    - Class-wise Masks: B_k from argmax(S_v) by default
    - Class-wise Prototypes: P_k = AvgPool(F_s * B_k) + E_k
    - Prototype Groups: P_fg, P_bg, and G_occ = [P_fg, P_bg]
    """

    def __init__(self,
                 in_channels,
                 num_classes=18,
                 hidden_channels=None,
                 foreground_classes=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                 background_classes=(11, 12, 13, 14, 15, 16),
                 use_hard_masks=True,
                 prototype_source='mask_classifier',
                 pqd_feature_channels=None,
                 uncertainty_threshold=0.5,
                 ignore_index=255,
                 loss_enabled=True,
                 dice_eps=1.0,
                 lovasz_classes='present',
                 init_cfg=None):
        super(MAESTROClasswisePrototypeGenerator, self).__init__(init_cfg)
        hidden_channels = hidden_channels or in_channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_hard_masks = use_hard_masks
        self.prototype_source = prototype_source
        self.uncertainty_threshold = uncertainty_threshold
        self.ignore_index = ignore_index
        self.loss_enabled = loss_enabled
        self.dice_eps = dice_eps
        self.lovasz_classes = lovasz_classes

        self.mask_classifier = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, num_classes, kernel_size=1),
        )
        if prototype_source == 'pqd':
            for param in self.mask_classifier.parameters():
                param.requires_grad = False
        self.class_embeddings = nn.Embedding(num_classes, in_channels)
        pqd_feature_channels = pqd_feature_channels or in_channels
        self.pqd_feature_channels = pqd_feature_channels
        if pqd_feature_channels != in_channels:
            self.pqd_feature_proj = nn.Linear(pqd_feature_channels, in_channels)
        else:
            self.pqd_feature_proj = None

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
        # MAESTRO2Task uses (B, C, X, Y, Z); Conv3d expects (B, C, Z, X, Y).
        return voxel_feature.permute(0, 1, 4, 2, 3).contiguous()

    def _from_conv3d_logits(self, logits):
        # Return logits as (B, K, X, Y, Z), matching voxel labels.
        return logits.permute(0, 1, 3, 4, 2).contiguous()

    def _build_classwise_masks(self, S_v_conv):
        if not self.use_hard_masks:
            return F.softmax(S_v_conv, dim=1)

        assignment = S_v_conv.argmax(dim=1)
        B_k = F.one_hot(assignment, num_classes=self.num_classes)
        return B_k.permute(0, 4, 1, 2, 3).to(dtype=S_v_conv.dtype)

    def _normalize_pqd_logits(self, occ_pred):
        if occ_pred.dim() != 5:
            raise ValueError(
                f'Expected 5D PQD occ_pred, got {tuple(occ_pred.shape)}.')
        if occ_pred.shape[-1] == self.num_classes:
            return occ_pred.permute(0, 4, 1, 2, 3).contiguous()
        if occ_pred.shape[1] == self.num_classes:
            return occ_pred.contiguous()
        raise ValueError(
            'Cannot infer PQD occ_pred class dimension: expected either '
            f'dim1 or dim4 to equal num_classes={self.num_classes}, got '
            f'{tuple(occ_pred.shape)}.')

    def _normalize_pqd_mask_feat(self, mask_feat):
        if mask_feat.dim() != 5:
            raise ValueError(
                f'Expected 5D PQD mask_feat, got {tuple(mask_feat.shape)}.')
        if mask_feat.shape[-1] == self.pqd_feature_channels:
            return mask_feat.contiguous()
        if mask_feat.shape[1] == self.pqd_feature_channels:
            return mask_feat.permute(0, 2, 3, 4, 1).contiguous()
        raise ValueError(
            'Cannot infer PQD mask feature channel dimension: expected either '
            f'dim1 or dim4 to equal pqd_feature_channels='
            f'{self.pqd_feature_channels}, got {tuple(mask_feat.shape)}.')

    def _build_pqd_prototypes(self, occ_pred, mask_feat):
        logits = self._normalize_pqd_logits(occ_pred)
        mask_feat = self._normalize_pqd_mask_feat(mask_feat)

        mask_probs = F.softmax(logits.detach().float(), dim=1)
        top2_values, _ = torch.topk(mask_probs, 2, dim=1)
        difference = top2_values[:, 0] - top2_values[:, 1]
        scores = 1.0 - difference
        uncertain_mask = scores >= self.uncertainty_threshold
        assignment = mask_probs.argmax(dim=1)

        b, x, y, z, c = mask_feat.shape
        mask_feat_flat = mask_feat.reshape(b, x * y * z, c)
        assignment_flat = assignment.reshape(b, x * y * z)
        valid_flat = (~uncertain_mask).reshape(b, x * y * z)

        prototypes = []
        for batch_idx in range(b):
            batch_prototypes = []
            for class_idx in range(self.num_classes):
                class_mask = (
                    (assignment_flat[batch_idx] == class_idx)
                    & valid_flat[batch_idx])
                if class_mask.any():
                    batch_prototypes.append(
                        mask_feat_flat[batch_idx, class_mask].mean(
                            dim=0, keepdim=True))
                else:
                    batch_prototypes.append(
                        mask_feat_flat.new_zeros(1, c))
            prototypes.append(torch.cat(batch_prototypes, dim=0))

        P_k = torch.stack(prototypes, dim=0)
        if self.pqd_feature_proj is not None:
            P_k = self.pqd_feature_proj(P_k)
        P_k = P_k + self.class_embeddings.weight.unsqueeze(0)
        return logits, P_k

    def _gather_group(self, prototypes, indices):
        if indices.numel() == 0:
            return prototypes.new_zeros(prototypes.size(0), 0, prototypes.size(-1))
        indices = indices.to(prototypes.device)
        return prototypes.index_select(1, indices)

    def forward(self, voxel_feature, occ_pred=None, mask_feat=None):
        # Shared Voxel Feature (F_s): ProtoOcc layout is (B, C, X, Y, Z).
        F_s = voxel_feature
        if self.prototype_source == 'pqd':
            if occ_pred is None or mask_feat is None:
                raise ValueError(
                    'prototype_source="pqd" requires occ_pred and mask_feat.')
            S_v, P_k = self._build_pqd_prototypes(occ_pred, mask_feat)
        elif self.prototype_source == 'mask_classifier':
            F_s_conv = self._to_conv3d(F_s)

            # Mask Classifier: S_v = h_mask(F_s).
            S_v_conv = self.mask_classifier(F_s_conv)

            # Class-wise Masks (B_k): hard argmax follows MAESTRO CPG.
            B_k_conv = self._build_classwise_masks(S_v_conv)

            # Class-wise Prototype Pooling: P_k = AvgPool(F_s * B_k).
            F_s_flat = F_s_conv.flatten(2)
            B_k_flat = B_k_conv.flatten(2)
            denom = B_k_flat.sum(dim=-1, keepdim=True).clamp_min(1.0)
            P_k = torch.einsum('bks,bcs->bkc', B_k_flat, F_s_flat) / denom

            # Learnable Class Embedding: P_k + E_k.
            P_k = P_k + self.class_embeddings.weight.unsqueeze(0)
            S_v = self._from_conv3d_logits(S_v_conv)
        else:
            raise ValueError(
                f'Unsupported prototype_source: {self.prototype_source}')

        # Foreground / Background Prototype Groups.
        P_fg = self._gather_group(P_k, self.foreground_indices)
        P_bg = self._gather_group(P_k, self.background_indices)
        G_occ = torch.cat([P_fg, P_bg], dim=1)

        return dict(
            logits=S_v,
            prototypes=P_k,
            foreground=P_fg,
            background=P_bg,
            occupancy=G_occ,
            S_v=S_v,
            P_k=P_k,
            P_fg=P_fg,
            P_bg=P_bg,
            G_occ=G_occ,
        )

    def _dice_loss(self, probs, target):
        valid = target != self.ignore_index
        if not valid.any():
            return probs.sum() * 0.0

        target_for_one_hot = target.clone()
        target_for_one_hot[~valid] = 0
        target_one_hot = F.one_hot(
            target_for_one_hot.clamp(min=0, max=self.num_classes - 1),
            num_classes=self.num_classes,
        )
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).to(probs.dtype)

        valid = valid.unsqueeze(1)
        probs = probs * valid
        target_one_hot = target_one_hot * valid

        reduce_dims = tuple(range(2, probs.dim()))
        intersection = (probs * target_one_hot).sum(dim=reduce_dims)
        cardinality = probs.sum(dim=reduce_dims) + target_one_hot.sum(
            dim=reduce_dims)
        dice = (2.0 * intersection + self.dice_eps) / (
            cardinality + self.dice_eps)

        present_classes = target_one_hot.sum(dim=reduce_dims) > 0
        if not present_classes.any():
            return probs.sum() * 0.0
        return (1.0 - dice)[present_classes].mean()

    def loss(self, logits, voxel_semantics, valid_mask=None):
        if not self.loss_enabled or voxel_semantics is None:
            return {}

        target = voxel_semantics.long()
        if valid_mask is not None:
            target = target.clone()
            target[~valid_mask.bool()] = self.ignore_index

        # MAESTRO supervises CPG mask classification with Dice + Lovasz.
        probs = F.softmax(logits.float(), dim=1)
        loss_dice = self._dice_loss(probs, target)
        loss_lovasz = lovasz_softmax(
            probs, target, classes=self.lovasz_classes, ignore=self.ignore_index)
        return {
            'loss_maestro_cpg_dice': loss_dice,
            'loss_maestro_cpg_lovasz': loss_lovasz,
        }
