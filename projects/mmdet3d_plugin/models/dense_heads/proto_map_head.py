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
      1. CNN proto generator produces `mask_feature` (hidden_channels, dot-
         product feature = PQD's `voxel_feats`) and `mask_feat` (proto_dim,
         prototype-pool feature = PQD's `mask_feat`) and `coarse_pred`
      2. AdaPG pools batch-shared local prototypes from confident map pixels
      3. AgnoPG maintains a class-wise EMA memory bank
      4. Scene-aware queries fuse learnable / global / local query sources
      5. (Optional) Prototype-grounded BEV refinement, enabled only when
         `pgbr_cfg` is provided for PGBR ablations.
      6. Class-fixed queries interact with self-attention
      7. Final map masks come from `mask_embed(query) x mask_feature`

    Compared with the 3-D occupancy decoder, the main differences are:
      - The spatial lattice is 2-D BEV instead of 3-D voxels.
      - Map supervision is class-wise sigmoid, so AdaPG uses a confidence
        threshold on each class channel instead of softmax argmax.
      - RPL is intentionally omitted here.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=96,
                 proto_dim=32,
                 num_classes=6,
                 num_convs=2,
                 attn_heads=8,
                 prototype_mining_thresh=0.7,
                 ema_weight=0.01,
                 prototype_mining_mode='pred_threshold',
                 prototype_mining_alpha=0.5,
                 prototype_pooling_scope='batch_local',
                 use_scene_adaptive=True,
                 use_ema_bank=True,
                 use_learnable_query=True,
                 use_query_self_attn=True,
                 pgbr_cfg=None,
                 use_bev_refinement=None,
                 refinement_temperature=1.0,
                 refinement_detach_query=True,
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
        self.proto_dim       = proto_dim
        self.prototype_mining_thresh = prototype_mining_thresh
        self.ema_weight      = ema_weight
        self.prototype_mining_mode = prototype_mining_mode
        self.prototype_mining_alpha = prototype_mining_alpha
        self.prototype_pooling_scope = prototype_pooling_scope
        self.prototype_support_eps = 1e-6
        self.use_scene_adaptive = use_scene_adaptive
        self.use_ema_bank       = use_ema_bank
        self.use_learnable_query = use_learnable_query
        self.use_query_self_attn = use_query_self_attn
        self.with_cp         = with_cp

        valid_mining_modes = ('pred_threshold', 'gt_hard', 'gt_soft')
        if self.prototype_mining_mode not in valid_mining_modes:
            raise ValueError(
                'Unsupported ProtoMapHead prototype_mining_mode: '
                f'{self.prototype_mining_mode}. '
                f'Expected one of {valid_mining_modes}.')
        if not 0.0 <= self.prototype_mining_alpha <= 1.0:
            raise ValueError(
                'ProtoMapHead prototype_mining_alpha must be in [0, 1], '
                f'but got {self.prototype_mining_alpha}.')
        if self.prototype_pooling_scope != 'batch_local':
            raise NotImplementedError(
                'ProtoMapHead currently supports only '
                "prototype_pooling_scope='batch_local' for Step 1.")

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
        # Prototype bottleneck. Mirrors ProtoOcc's split between cnn3d_decoder
        # out_dim=32 (low-dim `mask_feat` for AdaPG / EMA / scene-aware query)
        # and feat_channels=48 (higher-dim `mask_feature` / `voxel_feats` used
        # in the final dot product). Without this split, prototype refinement
        # gradients couple directly to the dot-product space and fail to
        # converge cleanly, which is the suspected cause of `coarse>final`
        # (39.44>39.09) on the 256ch ablation.
        self.proto_bottleneck = nn.Conv2d(
            hidden_channels, proto_dim, kernel_size=1)

        # ── 3. AgnoPG: EMA bank ────────────────────────────────────────────
        # Dim = proto_dim, mirroring ProtoOcc PQD where the EMA bank lives in
        # the cnn3d_decoder out_dim=32 prototype space, NOT in feat_channels.
        # This is the dim-bottleneck part of the P0 alignment fix.
        self.prototype_EMA_feat = nn.Embedding(num_classes, proto_dim)
        # Registered as buffer so it is saved / loaded with checkpoints.
        self.register_buffer('proto_first_flag', torch.ones(num_classes, dtype=torch.bool))

        # ── 4. Scene-Aware Query combination (mirrors ProtoOcc exactly) ───
        # Aggs project from proto_dim (low-dim prototype space) to
        # hidden_channels (query / dot-product space). This mirrors PQD's
        # `Linear(32, feat_channels=48)` first layer.
        self.global_protoEMA_agg = nn.Sequential(
            nn.Linear(proto_dim, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.local_protoEMA_agg = nn.Sequential(
            nn.Linear(proto_dim, hidden_channels), nn.ReLU(inplace=True),
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
        # Match PQD: MultiheadAttention(dropout=0.1) + residual + LayerNorm.
        # PQD's transformer uses `operation_order=['self_attn', 'norm']` (no
        # FFN), so we intentionally do NOT add an FFN block here.
        self.self_attn      = nn.MultiheadAttention(hidden_channels, attn_heads,
                                                     dropout=0.1,
                                                     batch_first=False)
        self.self_attn_norm = nn.LayerNorm(hidden_channels)
        # Final post_norm before the mask head, mirrors PQD `forward_head`'s
        # first line `decoder_out = self.post_norm(decoder_out)`.
        self.post_norm      = nn.LayerNorm(hidden_channels)

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
        """Produce both PQD-style features from a BEV input.

        Naming mirrors ProtoOcc PQD's two decoder inputs:
          - `mask_feature` (hidden_channels): high-dim feature for the final
            dot product. Equivalent to PQD's `voxel_feats` (48-dim CVF).
          - `mask_feat` (proto_dim): low-dim feature for AdaPG pooling and
            EMA bank. Equivalent to PQD's `mask_feat` input (32-dim from
            cnn3d_decoder).
        """
        if self.with_cp and self.training:
            mask_feature = checkpoint(self.conv_layers, bev_feat)
        else:
            mask_feature = self.conv_layers(bev_feat)
        coarse_pred = self.coarse_predictor(mask_feature)        # (B, K, H, W)
        mask_feat = self.proto_bottleneck(mask_feature)          # (B, proto_dim, H, W)
        return mask_feature, mask_feat, coarse_pred

    # ──────────────────────────────────────────────────────────────────────
    # Step 2 — AdaPG  (Scene-Adaptive Prototype Generator)
    # ──────────────────────────────────────────────────────────────────────

    def _adaPG(self, coarse_pred, mask_feat):
        """Pool batch-shared local prototypes over CONFIDENT map pixels.

        ProtoOcc PQD filters voxels by top-2 softmax margin
        (`scores = 1 - (top1 - top2); vis_mask = scores >= 0.9`), keeping
        only voxels where the model is confident about the assigned class
        AND no competing class is close.

        For map's per-class sigmoid (classes can overlap, so top-2 margin
        does not directly apply), we approximate the same design intent
        with a per-class threshold `prototype_mining_thresh` (default 0.7)
        — keeping only pixels well inside the positive side of the sigmoid
        decision boundary. This is NOT strictly equivalent (it cannot
        exclude pixels where multiple classes simultaneously fire high),
        but it carries the same intent of "use only high-confidence pixels
        for the prototype pool".

        Pools in proto_dim space (low-dim bottleneck `mask_feat`), NOT in
        hidden_channels (`mask_feature`). Mirrors PQD where AdaPG pools
        from the 32-dim `mask_feat` input.

        Returns:
            for_query: (num_classes, proto_dim)
            valid_mask: (num_classes,)
        """
        soft_masks     = torch.sigmoid(coarse_pred)            # (B, K, H, W)
        mask_feat_perm = mask_feat.permute(0, 2, 3, 1)         # (B, H, W, proto_dim)
        zero_proto = mask_feat.sum(dim=(0, 2, 3)) * 0.0

        for_query = []
        valid_mask = []
        for k in range(self.num_classes):
            # Stricter confidence filter, approximates PQD's `~vis_mask`
            # voxel filter under sigmoid supervision. Falls back to a
            # `valid=False` zero prototype when no pixels qualify (sparse
            # classes early in training).
            conf_mask = soft_masks[:, k] > self.prototype_mining_thresh
            if conf_mask.sum() == 0:
                proto = zero_proto
                valid = False
            else:
                proto = mask_feat_perm[conf_mask].mean(0)      # (proto_dim,)
                valid = True
            for_query.append(proto.unsqueeze(0))                # (1, proto_dim)
            valid_mask.append(valid)
        return (
            torch.cat(for_query, dim=0),
            torch.tensor(valid_mask, dtype=torch.bool, device=mask_feat.device))

    def _gt_guided_adaPG(self, coarse_pred, mask_feat, gt_masks_bev):
        """Pool prototypes with GT support and optional confidence weighting.

        Training-time GT guarantees sparse classes have positive support when
        they exist. Prediction confidence only reweights GT-positive pixels and
        is detached to avoid shortcut gradients through the pooling weights.

        Pools in proto_dim space (matches `_adaPG`).

        Returns:
            for_query: (num_classes, proto_dim)
            valid_mask: (num_classes,)
        """
        prob = torch.sigmoid(coarse_pred).detach()              # (B, K, H, W)
        feat = mask_feat.permute(0, 2, 3, 1)                    # (B, H, W, proto_dim)
        gt = gt_masks_bev.to(device=mask_feat.device, dtype=feat.dtype)
        zero_proto = mask_feat.sum(dim=(0, 2, 3)) * 0.0

        for_query = []
        valid_mask = []
        for k in range(self.num_classes):
            gt_mask = gt[:, k]
            if self.prototype_mining_mode == 'gt_hard':
                weight = gt_mask
            else:
                weight = gt_mask * (
                    self.prototype_mining_alpha
                    + (1.0 - self.prototype_mining_alpha) * prob[:, k])

            support = weight.sum()
            valid = bool(support.item() > self.prototype_support_eps)
            if valid:
                proto = (feat * weight.unsqueeze(-1)).sum(dim=(0, 1, 2)) / support
            else:
                proto = zero_proto

            for_query.append(proto.unsqueeze(0))
            valid_mask.append(valid)

        return (
            torch.cat(for_query, dim=0),
            torch.tensor(valid_mask, dtype=torch.bool, device=mask_feat.device))

    # ──────────────────────────────────────────────────────────────────────
    # Step 3 — AgnoPG  (Scene-Agnostic Prototype Generator / EMA bank)
    # ──────────────────────────────────────────────────────────────────────

    def _agno_PG_update_and_get(self, for_query, valid_mask=None):
        """Update EMA bank during training; return global prototype.

        Mirrors ProtoOcc's AgnoPG logic exactly:
          - First time a class appears: copy for_query into EMA directly.
          - Subsequent times: exponential moving average.

        Returns:
            ema_query: (num_classes, hidden_channels)
        """
        if not self.use_ema_bank:
            return for_query.new_zeros(for_query.shape)

        if valid_mask is None:
            valid_mask = for_query.sum(1) != 0
        else:
            valid_mask = valid_mask.to(
                device=self.proto_first_flag.device, dtype=torch.bool)

        if self.training:
            with torch.no_grad():
                cur_assign_flag = valid_mask                # (K,) bool

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

    def _scene_aware_queries(self, for_query, ema_query, valid_mask):
        """Combine enabled query sources into batch-shared scene-aware queries.

        Mirrors ProtoOcc PQD:
            query_feat = learnable_query
                       + global_protoEMA_agg(ema_query)
                       + local_protoEMA_agg(for_query)
            query_feat = for_query_embed(query_feat)

        `for_query` and `ema_query` live in proto_dim space; aggs project
        them up to hidden_channels, which is the query / dot-product space.

        Map-specific safeguards (NOT in PQD): under per-class sigmoid +
        a 0.7 threshold, sparse classes (ped_crossing, stop_line, divider)
        can go many iterations without any pixel passing the filter. Two
        contamination paths arise:
          - EMA bank rows whose class has never been initialized still hold
            random `nn.Embedding` init values; feeding them through
            `global_protoEMA_agg` injects pure noise into the query.
          - For classes that are uninitialized this iteration, `for_query`
            is zero, so `local_protoEMA_agg(0)` is just the layer's bias —
            also noise rather than signal.
        Mask both contributions per-class so uninitialized classes fall
        back cleanly to `learnable_query` only. PQD does not need this
        because softmax + dense occ classes virtually always produce
        confident voxels.

        Returns:
            query_feat: (num_classes, hidden_channels)
        """
        query_feat = for_query.new_zeros(
            (for_query.shape[0], self.hidden_channels))

        if self.use_learnable_query:
            query_feat = query_feat + self.learnable_query.weight

        if self.use_ema_bank:
            if self.with_cp and self.training:
                ema_contrib = checkpoint(self.global_protoEMA_agg, ema_query)
            else:
                ema_contrib = self.global_protoEMA_agg(ema_query)
            ema_init_mask = (~self.proto_first_flag).to(ema_contrib.dtype).unsqueeze(-1)
            query_feat = query_feat + ema_contrib * ema_init_mask

        if self.use_scene_adaptive:
            if self.with_cp and self.training:
                local_contrib = checkpoint(self.local_protoEMA_agg, for_query)
            else:
                local_contrib = self.local_protoEMA_agg(for_query)
            local_valid = valid_mask.to(
                device=local_contrib.device, dtype=local_contrib.dtype).unsqueeze(-1)
            query_feat = query_feat + local_contrib * local_valid

        if self.with_cp and self.training:
            return checkpoint(self.for_query_embed, query_feat)
        return self.for_query_embed(query_feat)

    # ──────────────────────────────────────────────────────────────────────
    # Step 5 — Optional Prototype-Grounded BEV Refinement
    # ──────────────────────────────────────────────────────────────────────

    def _apply_pgbr(self, mask_feature, query_feat):
        if self.pgbr_refiner is None:
            return mask_feature
        return self.pgbr_refiner(mask_feature, query_feat)

    # ──────────────────────────────────────────────────────────────────────
    # Steps 6-7 — Self-Attention + Dot Product
    # ──────────────────────────────────────────────────────────────────────

    def _forward_head(self, query_feat, mask_feature):
        """Self-attention among class queries then dot product with mask_feature.

        Mirrors PQD `forward_head(decoder_out, mask_feature)` exactly:
        post_norm -> mask_embed MLP -> einsum with the high-dim feature
        (analog of PQD's `voxel_feats`, called `mask_feature` inside its
        `forward_head`).

        Args:
            query_feat:   (num_classes, hidden_channels)  — batch-shared query
            mask_feature: (B, hidden_channels, H, W)
        Returns:
            final_masks: (B, num_classes, H, W)
        """
        B, C, H, W = mask_feature.shape

        # Expand to (K, B, C) — MultiheadAttention default (seq, batch, dim)
        q = query_feat.unsqueeze(1).expand(-1, B, -1)   # (K, B, C)
        if self.use_query_self_attn:
            attn_out, _ = self.self_attn(q, q, q)
            q = self.self_attn_norm(q + attn_out)        # (K, B, C)

        # Final post_norm before mask_embed, mirrors PQD `forward_head`'s
        # first line `decoder_out = self.post_norm(decoder_out)`.
        q = self.post_norm(q)

        # Dot product: einsum mirrors ProtoOcc's forward_head
        # mask_embed: (K, B, C) -> (B, K, C)
        q = q.permute(1, 0, 2)                           # (B, K, C)
        mask_embed  = self.mask_embed(q)                 # (B, K, C)
        final_masks = torch.einsum('bkc,bchw->bkhw', mask_embed, mask_feature)
        return final_masks                               # (B, K, H, W)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, bev_feat, gt_masks_bev=None):
        """
        Args:
            bev_feat: (B, in_channels, H, W)
            gt_masks_bev: optional training-time GT map masks, (B, K, H, W)
        Returns:
            coarse_pred : (B, num_classes, H, W)
            final_masks : (B, num_classes, H, W)
        """
        # 1. CNN → mask_feature (hidden, dot-product feat = PQD `voxel_feats`)
        #         + mask_feat (proto_dim, prototype-pool feat = PQD `mask_feat`)
        #         + coarse_pred
        mask_feature, mask_feat, coarse_pred = self._cnn_proto_generator(bev_feat)

        # 2. AdaPG — batch-shared local prototype, pooled in proto_dim space
        if (self.training and gt_masks_bev is not None
                and self.prototype_mining_mode in ('gt_hard', 'gt_soft')):
            for_query, valid_mask = self._gt_guided_adaPG(
                coarse_pred, mask_feat, gt_masks_bev)
        else:
            for_query, valid_mask = self._adaPG(coarse_pred, mask_feat)

        # 3. AgnoPG — update EMA, get global prototype
        ema_query = self._agno_PG_update_and_get(for_query, valid_mask)

        # 4. Scene-Aware Query: learnable + global + local (with per-class
        #    masking so uninitialized EMA / invalid local rows do not pollute
        #    the query for sparse classes).
        query_feat = self._scene_aware_queries(for_query, ema_query, valid_mask)

        # 5. Optional PGBR submodule. Canonical configs leave this as identity.
        refined_mask_feature = self._apply_pgbr(mask_feature, query_feat)

        # 6-7. Self-attention + dot product → final masks
        final_masks = self._forward_head(query_feat, refined_mask_feature)

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
