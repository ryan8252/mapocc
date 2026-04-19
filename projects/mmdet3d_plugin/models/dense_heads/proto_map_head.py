import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS, build_loss


@HEADS.register_module()
class ProtoMapHead(BaseModule):
    """Prototype-based decoder for BEV map segmentation.

    This head follows ProtoOcc's current query formation style:
      1. CNN proto generator produces `mask_feat` and `coarse_pred`
      2. AdaPG pools batch-shared local prototypes from confident map pixels
      3. AgnoPG maintains a class-wise EMA memory bank
      4. Scene-aware queries fuse learnable / global / local query sources
      5. Class-fixed queries interact with self-attention
      6. Final map masks come from `mask_embed(query) x mask_feat`

    Compared with the 3-D occupancy decoder, the main differences are:
      - The spatial lattice is 2-D BEV instead of 3-D voxels.
      - Map supervision is class-wise sigmoid, so AdaPG uses a confidence
        threshold on each class channel instead of softmax argmax.
      - RPL is intentionally omitted here.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=96,
                 num_classes=6,
                 num_convs=2,
                 attn_heads=8,
                 conf_thresh=0.5,
                 ema_weight=0.01,
                 use_scene_adaptive=True,
                 use_ema_bank=True,
                 use_learnable_query=True,
                 use_query_self_attn=True,
                 with_cp=False,
                 loss_coarse_bce=None,
                 loss_coarse_dice=None,
                 loss_mask_focal=None,
                 loss_mask_dice=None):
        super().__init__()
        if not any([use_scene_adaptive, use_ema_bank, use_learnable_query]):
            raise ValueError('At least one query source must be enabled for ProtoMapHead.')

        self.num_classes     = num_classes
        self.hidden_channels = hidden_channels
        self.conf_thresh     = conf_thresh
        self.ema_weight      = ema_weight
        self.use_scene_adaptive = use_scene_adaptive
        self.use_ema_bank       = use_ema_bank
        self.use_learnable_query = use_learnable_query
        self.use_query_self_attn = use_query_self_attn
        self.with_cp         = with_cp

        # ── 1. CNN Proto Generator ─────────────────────────────────────────
        blocks, cur_ch = [], in_channels
        for _ in range(num_convs):
            blocks.append(ConvModule(
                cur_ch, hidden_channels, 3, padding=1, bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True)))
            cur_ch = hidden_channels
        self.conv_layers      = nn.Sequential(*blocks)
        self.coarse_predictor = nn.Conv2d(hidden_channels, num_classes, 1)

        # ── 3. AgnoPG: EMA bank ────────────────────────────────────────────
        # Dim = hidden_channels, matching mask_feat dim (same convention as ProtoOcc
        # where prototype_EMA_feat dim == cnn3d_decoder out_dim == mask_feat last dim).
        self.prototype_EMA_feat = nn.Embedding(num_classes, hidden_channels)
        # Registered as buffer so it is saved / loaded with checkpoints.
        self.register_buffer('proto_first_flag', torch.ones(num_classes, dtype=torch.bool))

        # ── 4. Scene-Aware Query combination (mirrors ProtoOcc exactly) ───
        self.global_protoEMA_agg = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.local_protoEMA_agg = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.learnable_query  = nn.Embedding(num_classes, hidden_channels)
        self.for_query_embed  = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))

        # ── 5. Self-Attention ──────────────────────────────────────────────
        self.self_attn      = nn.MultiheadAttention(hidden_channels, attn_heads,
                                                     batch_first=False)
        self.self_attn_norm = nn.LayerNorm(hidden_channels)

        # ── 6. Mask embed (query -> embedding for dot product) ─────────────
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))

        # ── Losses ─────────────────────────────────────────────────────────
        self.loss_coarse_bce  = build_loss(loss_coarse_bce)  if loss_coarse_bce  else None
        self.loss_coarse_dice = build_loss(loss_coarse_dice) if loss_coarse_dice else None
        self.loss_mask_focal  = build_loss(loss_mask_focal)  if loss_mask_focal  else None
        self.loss_mask_dice   = build_loss(loss_mask_dice)   if loss_mask_dice   else None

    # ──────────────────────────────────────────────────────────────────────
    # Step 1 — CNN Proto Generator
    # ──────────────────────────────────────────────────────────────────────

    def _cnn_proto_generator(self, bev_feat):
        if self.with_cp and self.training:
            mask_feat = checkpoint(self.conv_layers, bev_feat)
        else:
            mask_feat = self.conv_layers(bev_feat)
        coarse_pred = self.coarse_predictor(mask_feat)   # (B, K, H, W)
        return mask_feat, coarse_pred

    # ──────────────────────────────────────────────────────────────────────
    # Step 2 — AdaPG  (Scene-Adaptive Prototype Generator)
    # ──────────────────────────────────────────────────────────────────────

    def _adaPG(self, coarse_pred, mask_feat):
        """Pool batch-shared local prototypes over confident map pixels.

        For occ (mutually exclusive classes), ProtoOcc uses softmax argmax +
        confidence filtering (exclude voxels where top-2 margin < 0.1).
        For map (binary-independent classes), we use sigmoid > conf_thresh,
        which is the direct equivalent for per-class binary prediction.

        Operates batch-level (all samples pooled together), following the
        same convention as ProtoOcc's Prototype_Query_Decoder_nuScenes.

        Returns:
            for_query: (num_classes, hidden_channels)
        """
        soft_masks     = torch.sigmoid(coarse_pred)        # (B, K, H, W)
        mask_feat_perm = mask_feat.permute(0, 2, 3, 1)    # (B, H, W, C)

        for_query = []
        for k in range(self.num_classes):
            conf_mask = soft_masks[:, k] > self.conf_thresh  # (B, H, W)
            if conf_mask.sum() == 0:
                proto = mask_feat.new_zeros(self.hidden_channels)
            else:
                proto = mask_feat_perm[conf_mask].mean(0)    # (C,)
            for_query.append(proto.unsqueeze(0))             # (1, C)
        return torch.cat(for_query, dim=0)                   # (K, C)

    # ──────────────────────────────────────────────────────────────────────
    # Step 3 — AgnoPG  (Scene-Agnostic Prototype Generator / EMA bank)
    # ──────────────────────────────────────────────────────────────────────

    def _agno_PG_update_and_get(self, for_query):
        """Update EMA bank during training; return global prototype.

        Mirrors ProtoOcc's AgnoPG logic exactly:
          - First time a class appears: copy for_query into EMA directly.
          - Subsequent times: exponential moving average.

        Returns:
            ema_query: (num_classes, hidden_channels)
        """
        if not self.use_ema_bank:
            return for_query.new_zeros(for_query.shape)

        if self.training:
            with torch.no_grad():
                cur_assign_flag = for_query.sum(1) != 0     # (K,) bool

                # First-time init
                init_flag = self.proto_first_flag & cur_assign_flag
                if init_flag.sum() > 0:
                    self.prototype_EMA_feat.weight.data[init_flag] = \
                        for_query[init_flag].detach()
                    self.proto_first_flag[init_flag] = False

                # EMA update for already-seen classes
                update_flag = (~self.proto_first_flag) & cur_assign_flag
                if update_flag.sum() > 0:
                    self.prototype_EMA_feat.weight.data[update_flag] = (
                        self.prototype_EMA_feat.weight[update_flag]
                        * (1 - self.ema_weight)
                        + for_query[update_flag].detach() * self.ema_weight)

        return self.prototype_EMA_feat.weight   # (K, hidden_channels)

    # ──────────────────────────────────────────────────────────────────────
    # Step 4 — Scene-Aware Query synthesis
    # ──────────────────────────────────────────────────────────────────────

    def _scene_aware_queries(self, for_query, ema_query):
        """Combine enabled query sources into batch-shared scene-aware queries.

        Exactly mirrors ProtoOcc:
            query_feat = learnable_query
                       + global_protoEMA_agg(ema_query)
                       + local_protoEMA_agg(for_query)
            query_feat = for_query_embed(query_feat)

        Returns:
            query_feat: (num_classes, hidden_channels)
        """
        query_feat = for_query.new_zeros(for_query.shape)

        if self.use_learnable_query:
            query_feat = query_feat + self.learnable_query.weight

        if self.use_ema_bank:
            if self.with_cp and self.training:
                query_feat = query_feat + checkpoint(self.global_protoEMA_agg, ema_query)
            else:
                query_feat = query_feat + self.global_protoEMA_agg(ema_query)

        if self.use_scene_adaptive:
            if self.with_cp and self.training:
                query_feat = query_feat + checkpoint(self.local_protoEMA_agg, for_query)
            else:
                query_feat = query_feat + self.local_protoEMA_agg(for_query)

        if self.with_cp and self.training:
            return checkpoint(self.for_query_embed, query_feat)
        return self.for_query_embed(query_feat)

    # ──────────────────────────────────────────────────────────────────────
    # Steps 5-6 — Self-Attention + Dot Product
    # ──────────────────────────────────────────────────────────────────────

    def _forward_head(self, query_feat, mask_feat):
        """Self-attention among class queries then dot product with mask_feat.

        Args:
            query_feat: (num_classes, hidden_channels)  — batch-shared query
            mask_feat:  (B, hidden_channels, H, W)
        Returns:
            final_masks: (B, num_classes, H, W)
        """
        B, C, H, W = mask_feat.shape

        # Expand to (K, B, C) — MultiheadAttention default (seq, batch, dim)
        q = query_feat.unsqueeze(1).expand(-1, B, -1)   # (K, B, C)
        if self.use_query_self_attn:
            attn_out, _ = self.self_attn(q, q, q)
            q = self.self_attn_norm(q + attn_out)        # (K, B, C)

        # Dot product: einsum mirrors ProtoOcc's forward_head
        # mask_embed: (K, B, C) -> (B, K, C)
        q = q.permute(1, 0, 2)                           # (B, K, C)
        mask_embed  = self.mask_embed(q)                 # (B, K, C)
        # mask_feat already (B, C, H, W); einsum equivalent to bmm
        final_masks = torch.einsum('bkc,bchw->bkhw', mask_embed, mask_feat)
        return final_masks                               # (B, K, H, W)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, bev_feat):
        """
        Args:
            bev_feat: (B, in_channels, H, W)
        Returns:
            coarse_pred : (B, num_classes, H, W)
            final_masks : (B, num_classes, H, W)
        """
        # 1. CNN → mask_feat + coarse_pred
        mask_feat, coarse_pred = self._cnn_proto_generator(bev_feat)

        # 2. AdaPG — local scene-adaptive prototype
        for_query = self._adaPG(coarse_pred, mask_feat)

        # 3. AgnoPG — update EMA, get global prototype
        ema_query = self._agno_PG_update_and_get(for_query)

        # 4. Scene-Aware Query: learnable + global + local
        query_feat = self._scene_aware_queries(for_query, ema_query)

        # 5-6. Self-attention + dot product → final masks
        final_masks = self._forward_head(query_feat, mask_feat)

        return coarse_pred, final_masks

    def loss(self, coarse_pred, final_masks, gt_masks_bev):
        gt = gt_masks_bev.float()
        losses = {}
        if self.loss_coarse_bce is not None:
            losses['loss_map_coarse_bce'] = self.loss_coarse_bce(coarse_pred, gt)
        if self.loss_coarse_dice is not None:
            losses['loss_map_coarse_dice'] = self.loss_coarse_dice(
                coarse_pred.reshape(-1, *coarse_pred.shape[2:]),
                gt.reshape(-1, *gt.shape[2:]))
        if self.loss_mask_focal is not None:
            losses['loss_map_focal'] = self.loss_mask_focal(final_masks, gt)
        if self.loss_mask_dice is not None:
            losses['loss_map_dice'] = self.loss_mask_dice(
                final_masks.reshape(-1, *final_masks.shape[2:]),
                gt.reshape(-1, *gt.shape[2:]))
        return losses

    def predict(self, final_masks):
        return torch.sigmoid(final_masks)
