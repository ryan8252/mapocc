import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS, build_loss


@HEADS.register_module()
class ProtoMapHead(BaseModule):
    """Prototype-based BEV map segmentation head.

    Mirrors ProtoOcc's prototype mechanism in 2D BEV space:
      1. CNN Proto Generator  : BEV feat -> coarse map pred + class prototypes
      2. Prototype Query Decoder 2D : prototypes cross-attend BEV feat -> final masks

    No Hungarian matching needed because map classes are fixed (class-fixed queries).
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=96,
                 num_classes=6,
                 num_convs=2,
                 attn_heads=8,
                 downsample_attn=True,
                 with_cp=False,
                 loss_coarse_bce=None,
                 loss_coarse_dice=None,
                 loss_mask_bce=None,
                 loss_mask_dice=None):
        super(ProtoMapHead, self).__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.with_cp = with_cp
        self.downsample_attn = downsample_attn

        # ── CNN layers → mask_feat ─────────────────────────────────────────
        blocks = []
        cur_ch = in_channels
        for _ in range(num_convs):
            blocks.append(ConvModule(
                cur_ch, hidden_channels,
                kernel_size=3, stride=1, padding=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True)))
            cur_ch = hidden_channels
        self.conv_layers = nn.Sequential(*blocks)

        # Coarse predictor (class logits before prototype decoding)
        self.coarse_predictor = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

        # ── Prototype Query Decoder 2D ─────────────────────────────────────
        self.self_attn = nn.MultiheadAttention(hidden_channels, attn_heads, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(hidden_channels)

        self.cross_attn = nn.MultiheadAttention(hidden_channels, attn_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hidden_channels)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 4, hidden_channels))
        self.ffn_norm = nn.LayerNorm(hidden_channels)

        # mask_embed MLP: refined query -> embedding for dot product
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))

        # ── Losses ─────────────────────────────────────────────────────────
        self.loss_coarse_bce  = build_loss(loss_coarse_bce)  if loss_coarse_bce  is not None else None
        self.loss_coarse_dice = build_loss(loss_coarse_dice) if loss_coarse_dice is not None else None
        self.loss_mask_bce    = build_loss(loss_mask_bce)    if loss_mask_bce    is not None else None
        self.loss_mask_dice   = build_loss(loss_mask_dice)   if loss_mask_dice   is not None else None

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _gen_prototypes(self, bev_feat):
        """CNN Proto Generator.

        Args:
            bev_feat: (B, C_in, H, W)
        Returns:
            coarse_pred : (B, num_classes, H, W)
            mask_feat   : (B, hidden_channels, H, W)
            prototypes  : (B, num_classes, hidden_channels)
        """
        if self.with_cp and self.training:
            mask_feat = checkpoint(self.conv_layers, bev_feat)
        else:
            mask_feat = self.conv_layers(bev_feat)

        coarse_pred = self.coarse_predictor(mask_feat)      # (B, K, H, W)

        # Masked average pooling: soft_mask-weighted mean of mask_feat per class
        soft_masks  = torch.sigmoid(coarse_pred)            # (B, K, H, W)
        feat_flat   = mask_feat.flatten(2)                  # (B, C', HW)
        masks_flat  = soft_masks.flatten(2)                 # (B, K, HW)
        # prototype_k = Σ(mask_k * feat) / Σ(mask_k)
        prototypes  = torch.bmm(masks_flat, feat_flat.permute(0, 2, 1))   # (B, K, C')
        prototypes  = prototypes / (masks_flat.sum(-1, keepdim=True) + 1e-6)

        return coarse_pred, mask_feat, prototypes

    def _decode(self, prototypes, mask_feat):
        """Prototype Query Decoder 2D.

        Args:
            prototypes : (B, num_classes, hidden_channels)
            mask_feat  : (B, hidden_channels, H, W)
        Returns:
            final_masks: (B, num_classes, H, W)
        """
        B, C, H, W = mask_feat.shape

        # K/V for cross-attention: optionally halve spatial resolution
        if self.downsample_attn:
            kv_feat = F.avg_pool2d(mask_feat, kernel_size=2, stride=2)  # (B, C, H/2, W/2)
        else:
            kv_feat = mask_feat
        kv_flat = kv_feat.flatten(2).permute(0, 2, 1)  # (B, HW', C)

        # Self-attention among prototype queries
        q = prototypes
        attn_out, _ = self.self_attn(q, q, q)
        q = self.self_attn_norm(q + attn_out)

        # Cross-attention: prototype queries attend to BEV feature
        attn_out, _ = self.cross_attn(q, kv_flat, kv_flat)
        q = self.cross_attn_norm(q + attn_out)

        # FFN
        q = self.ffn_norm(q + self.ffn(q))

        # Dot product with full-resolution mask_feat -> final masks
        mask_embed      = self.mask_embed(q)                    # (B, K, C)
        mask_feat_flat  = mask_feat.flatten(2)                  # (B, C, HW)
        final_masks     = torch.bmm(mask_embed, mask_feat_flat) # (B, K, HW)
        final_masks     = final_masks.view(B, self.num_classes, H, W)

        return final_masks

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, bev_feat):
        """
        Returns:
            coarse_pred : (B, num_classes, H, W)
            final_masks : (B, num_classes, H, W)
        """
        coarse_pred, mask_feat, prototypes = self._gen_prototypes(bev_feat)
        final_masks = self._decode(prototypes, mask_feat)
        return coarse_pred, final_masks

    def loss(self, coarse_pred, final_masks, gt_masks_bev):
        """
        Args:
            coarse_pred  : (B, num_classes, H, W)
            final_masks  : (B, num_classes, H, W)
            gt_masks_bev : (B, num_classes, H, W)  float binary masks
        """
        gt = gt_masks_bev.float()
        losses = {}

        if self.loss_coarse_bce is not None:
            losses['loss_map_coarse_bce'] = self.loss_coarse_bce(coarse_pred, gt)
        if self.loss_coarse_dice is not None:
            losses['loss_map_coarse_dice'] = self.loss_coarse_dice(
                coarse_pred.reshape(-1, *coarse_pred.shape[2:]),
                gt.reshape(-1, *gt.shape[2:]))
        if self.loss_mask_bce is not None:
            losses['loss_map_bce'] = self.loss_mask_bce(final_masks, gt)
        if self.loss_mask_dice is not None:
            losses['loss_map_dice'] = self.loss_mask_dice(
                final_masks.reshape(-1, *final_masks.shape[2:]),
                gt.reshape(-1, *gt.shape[2:]))
        return losses

    def predict(self, final_masks):
        return torch.sigmoid(final_masks)
