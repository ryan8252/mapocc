import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from mmdet3d.models import BACKBONES


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

    def _gather_group(self, prototypes, indices):
        if indices.numel() == 0:
            return prototypes.new_zeros(prototypes.size(0), 0, prototypes.size(-1))
        indices = indices.to(prototypes.device)
        return prototypes.index_select(1, indices)

    def forward(self, voxel_feature):
        # Shared Voxel Feature (F_s): ProtoOcc layout is (B, C, X, Y, Z).
        F_s = voxel_feature
        F_s_conv = self._to_conv3d(F_s)

        # Mask Classifier: S_v = h_mask(F_s).
        S_v_conv = self.mask_classifier(F_s_conv)

        # Class-wise Masks (B_k): hard argmax follows the MAESTRO CPG design.
        B_k_conv = self._build_classwise_masks(S_v_conv)

        # Class-wise Prototype Pooling: P_k = AvgPool(F_s * B_k).
        F_s_flat = F_s_conv.flatten(2)
        B_k_flat = B_k_conv.flatten(2)
        denom = B_k_flat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        P_k = torch.einsum('bks,bcs->bkc', B_k_flat, F_s_flat) / denom

        # Learnable Class Embedding: P_k + E_k.
        E_k = self.class_embeddings.weight.unsqueeze(0)
        P_k = P_k + E_k

        # Foreground / Background Prototype Groups.
        S_v = self._from_conv3d_logits(S_v_conv)
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

    def loss(self, logits, voxel_semantics, valid_mask=None):
        if self.loss_weight <= 0 or voxel_semantics is None:
            return {}

        target = voxel_semantics.long()
        if valid_mask is not None:
            target = target.clone()
            target[~valid_mask.bool()] = self.ignore_index

        loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        return {'loss_maestro_cpg_ce': loss * self.loss_weight}
