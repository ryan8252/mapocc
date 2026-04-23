import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS, build_loss


class PrototypeGroundedBEVRefiner(nn.Module):
    """Prototype-grounded BEV feature refiner used by PGBR ablations.

    This module is intentionally separate from ProtoMapHead's main prototype
    decoder. Canonical ProtoMapHead does not instantiate it unless `pgbr_cfg`
    is provided.
    """

    def __init__(self,
                 hidden_channels,
                 temperature=1.0,
                 detach_query=True,
                 with_cp=False):
        super().__init__()
        self.temperature = temperature
        self.detach_query = detach_query
        self.with_cp = with_cp

        self.proto_value_proj = nn.Linear(hidden_channels, hidden_channels)
        self.suppress_proj = ConvModule(
            hidden_channels, hidden_channels, 3, padding=1, bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True))
        self.refine_fuse = nn.Sequential(
            ConvModule(
                hidden_channels * 3, hidden_channels, 1, padding=0, bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True)),
            ConvModule(
                hidden_channels, hidden_channels, 3, padding=1, bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True)))

    def forward(self, mask_feat, query_feat):
        """Refine BEV mask features with class-wise prototype grounding.

        Args:
            mask_feat:  (B, hidden_channels, H, W)
            query_feat: (num_classes, hidden_channels)
        Returns:
            refined_mask_feat: (B, hidden_channels, H, W)
        """
        refine_query = query_feat.detach() if self.detach_query else query_feat
        feat_norm = F.normalize(mask_feat, dim=1)
        query_norm = F.normalize(refine_query, dim=1)

        temperature = max(self.temperature, 1e-6)
        ground_logits = torch.einsum('kc,bchw->bkhw', query_norm, feat_norm)
        ground_logits = ground_logits / temperature

        # softmax handles class-wise prototype mixing; sigmoid preserves
        # absolute confidence so low-similarity pixels are not over-enhanced.
        ground_map = torch.sigmoid(ground_logits)
        ground_weight = F.softmax(ground_logits, dim=1)

        proto_value = self.proto_value_proj(refine_query)            # (K, C)
        enhance = torch.einsum('bkhw,kc->bchw', ground_weight, proto_value)
        enhance_gate = ground_map.max(dim=1, keepdim=True)[0]
        enhance = enhance * enhance_gate

        if self.with_cp and self.training:
            suppress_feat = checkpoint(self.suppress_proj, mask_feat)
        else:
            suppress_feat = self.suppress_proj(mask_feat)
        suppress = suppress_feat * (1.0 - enhance_gate)

        refine_input = torch.cat([mask_feat, enhance, suppress], dim=1)
        if self.with_cp and self.training:
            delta = checkpoint(self.refine_fuse, refine_input)
        else:
            delta = self.refine_fuse(refine_input)
        return mask_feat + delta


@HEADS.register_module()
class ProtoMapHead(BaseModule):
    """Prototype-based decoder for BEV map segmentation.

    This head follows ProtoOcc's current query formation style:
      1. CNN proto generator produces `mask_feat` and `coarse_pred`
      2. AdaPG pools batch-shared local prototypes from confident map pixels
      3. AgnoPG maintains a class-wise EMA memory bank
      4. Scene-aware queries fuse learnable / global / local query sources
      5. (Optional) Prototype-grounded BEV refinement, enabled only when
         `pgbr_cfg` is provided for PGBR ablations.
      6. Class-fixed queries interact with self-attention
      7. Final map masks come from `mask_embed(query) x refined_mask_feat`

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
                 pgbr_cfg=None,
                 use_bev_refinement=None,
                 refinement_temperature=1.0,
                 refinement_detach_query=True,
                 with_cp=False,
                 test_output='final',
                 loss_coarse_bce=None,
                 loss_coarse_dice=None,
                 loss_mask_focal=None,
                 loss_mask_dice=None):
        super().__init__()
        if not any([use_scene_adaptive, use_ema_bank, use_learnable_query]):
            raise ValueError('At least one query source must be enabled for ProtoMapHead.')
        if test_output not in ('final', 'coarse', 'coarse_final'):
            raise ValueError(
                'Unsupported ProtoMapHead test_output: '
                f'{test_output}. Expected one of final, coarse, coarse_final.')

        self.num_classes     = num_classes
        self.hidden_channels = hidden_channels
        self.conf_thresh     = conf_thresh
        self.ema_weight      = ema_weight
        self.use_scene_adaptive = use_scene_adaptive
        self.use_ema_bank       = use_ema_bank
        self.use_learnable_query = use_learnable_query
        self.use_query_self_attn = use_query_self_attn
        self.with_cp         = with_cp
        self.test_output     = test_output

        pgbr_cfg = self._normalize_pgbr_cfg(
            pgbr_cfg,
            use_bev_refinement,
            refinement_temperature,
            refinement_detach_query)

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

        # ── 5. Optional PGBR submodule ────────────────────────────────────
        self.pgbr_refiner = None
        if pgbr_cfg is not None:
            self.pgbr_refiner = PrototypeGroundedBEVRefiner(
                hidden_channels=hidden_channels,
                temperature=pgbr_cfg['temperature'],
                detach_query=pgbr_cfg['detach_query'],
                with_cp=with_cp)

        # ── 6. Self-Attention ──────────────────────────────────────────────
        self.self_attn      = nn.MultiheadAttention(hidden_channels, attn_heads,
                                                     batch_first=False)
        self.self_attn_norm = nn.LayerNorm(hidden_channels)

        # ── 7. Mask embed (query -> embedding for dot product) ─────────────
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))

        # ── Losses ─────────────────────────────────────────────────────────
        self.loss_coarse_bce  = build_loss(loss_coarse_bce)  if loss_coarse_bce  else None
        self.loss_coarse_dice = build_loss(loss_coarse_dice) if loss_coarse_dice else None
        self.loss_mask_focal  = build_loss(loss_mask_focal)  if loss_mask_focal  else None
        self.loss_mask_dice   = build_loss(loss_mask_dice)   if loss_mask_dice   else None

    def _normalize_pgbr_cfg(self,
                            pgbr_cfg,
                            use_bev_refinement,
                            refinement_temperature,
                            refinement_detach_query):
        """Normalize new `pgbr_cfg` and legacy PGBR config keys."""
        if use_bev_refinement is False and pgbr_cfg is not None:
            raise ValueError(
                'Conflicting ProtoMapHead config: `pgbr_cfg` enables PGBR, '
                'but legacy `use_bev_refinement=False` disables it.')

        # Backward compatibility for older configs / work_dirs.
        if use_bev_refinement is True and pgbr_cfg is None:
            pgbr_cfg = dict(
                temperature=refinement_temperature,
                detach_query=refinement_detach_query)

        if pgbr_cfg is None:
            return None

        pgbr_cfg = dict(pgbr_cfg)
        pgbr_type = pgbr_cfg.pop('type', 'pgbr')
        if pgbr_type not in ('pgbr', 'PrototypeGroundedBEVRefiner'):
            raise ValueError(f'Unsupported ProtoMapHead PGBR type: {pgbr_type}')

        normalized_cfg = dict(
            temperature=pgbr_cfg.pop('temperature', refinement_temperature),
            detach_query=pgbr_cfg.pop('detach_query', refinement_detach_query))
        if pgbr_cfg:
            unknown_keys = ', '.join(sorted(pgbr_cfg.keys()))
            raise ValueError(f'Unsupported ProtoMapHead pgbr_cfg keys: {unknown_keys}')
        return normalized_cfg

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Load new PGBR submodule keys and legacy inline PGBR keys."""
        legacy_prefixes = (
            'proto_value_proj.',
            'suppress_proj.',
            'refine_fuse.',
        )

        if self.pgbr_refiner is not None:
            for legacy_prefix in legacy_prefixes:
                old_prefix = prefix + legacy_prefix
                new_prefix = prefix + 'pgbr_refiner.' + legacy_prefix
                for key in list(state_dict.keys()):
                    if key.startswith(old_prefix):
                        new_key = new_prefix + key[len(old_prefix):]
                        if new_key not in state_dict:
                            state_dict[new_key] = state_dict[key]
                        state_dict.pop(key)
        else:
            for key in list(state_dict.keys()):
                if key.startswith(prefix + 'pgbr_refiner.'):
                    state_dict.pop(key)
                elif any(key.startswith(prefix + legacy_prefix)
                         for legacy_prefix in legacy_prefixes):
                    state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

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
    # Step 5 — Optional Prototype-Grounded BEV Refinement
    # ──────────────────────────────────────────────────────────────────────

    def _apply_pgbr(self, mask_feat, query_feat):
        if self.pgbr_refiner is None:
            return mask_feat
        return self.pgbr_refiner(mask_feat, query_feat)

    # ──────────────────────────────────────────────────────────────────────
    # Steps 6-7 — Self-Attention + Dot Product
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

        # 2. AdaPG — batch-shared local prototype
        for_query = self._adaPG(coarse_pred, mask_feat)

        # 3. AgnoPG — update EMA, get global prototype
        ema_query = self._agno_PG_update_and_get(for_query)

        # 4. Scene-Aware Query: learnable + global + local
        query_feat = self._scene_aware_queries(for_query, ema_query)

        # 5. Optional PGBR submodule. Canonical configs leave this as identity.
        refined_mask_feat = self._apply_pgbr(mask_feat, query_feat)

        # 6-7. Self-attention + dot product → final masks
        final_masks = self._forward_head(query_feat, refined_mask_feat)

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

    def predict(self, coarse_pred, final_masks=None):
        # Keep backward compatibility with older call sites that pass only
        # final mask logits.
        if final_masks is None:
            return torch.sigmoid(coarse_pred)

        if self.test_output == 'coarse':
            logits = coarse_pred
        elif self.test_output == 'coarse_final':
            logits = coarse_pred + final_masks
        else:
            logits = final_masks
        return torch.sigmoid(logits)
