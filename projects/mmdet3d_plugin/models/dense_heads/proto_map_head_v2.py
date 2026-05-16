import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS, build_loss

from .cnn2d_decoder import cnn2d_decoder


@HEADS.register_module()
class ProtoMapHeadV2(BaseModule):
    """Clean PQD-style prototype decoder for BEV map segmentation.

    This is intentionally smaller than ``ProtoMapHead``:
      - no RPL
      - no GT-guided AdaPG
      - no PrototypeGroundedBEVRefiner/PGBR

    Shape convention:
      - input BEV feature:       (B, C, H, W)
      - coarse map logits:       (B, K, H, W)
      - prototype mining feature:(B, H, W, P)
      - query masks:             (B, K, H, W)
      - final map logits:        (B, K, H, W)
    """

    def __init__(self,
                 in_channels,
                 feat_channels=128,
                 out_channels=None,
                 proto_dim=64,
                 num_classes=6,
                 num_heads=4,
                 prototype_mining_thresh=0.45,
                 prototype_ema_weight=0.01,
                 use_internal_cnn2d_decoder=True,
                 cnn2d_decoder_cfg=None,
                 use_query_self_attn=True,
                 class_mixing_identity_bias=4.0,
                 with_cp=False,
                 loss_coarse_bce=None,
                 loss_coarse_dice=None,
                 loss_mask_focal=None,
                 loss_mask_dice=None):
        super(ProtoMapHeadV2, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.proto_dim = proto_dim
        self.num_classes = num_classes
        self.prototype_mining_thresh = float(prototype_mining_thresh)
        self.prototype_ema_weight = prototype_ema_weight
        self.use_internal_cnn2d_decoder = use_internal_cnn2d_decoder
        self.use_query_self_attn = use_query_self_attn
        self.class_mixing_identity_bias_value = float(class_mixing_identity_bias)
        self.with_cp = with_cp

        if self.out_channels != in_channels:
            raise ValueError(
                'ProtoMapHeadV2 expects out_channels to match in_channels '
                'because mask embeddings are dotted with the input BEV feature. '
                f'Got out_channels={self.out_channels}, '
                f'in_channels={in_channels}.')

        self.map_decoder = None
        if use_internal_cnn2d_decoder:
            decoder_cfg = dict(
                in_dim=in_channels,
                out_dim=proto_dim,
                num_classes=num_classes,
                with_cp=with_cp)
            if cnn2d_decoder_cfg is not None:
                decoder_cfg.update(cnn2d_decoder_cfg)
            self.map_decoder = cnn2d_decoder(**decoder_cfg)

        self.query_feat = nn.Embedding(num_classes, feat_channels)
        self.register_buffer(
            'proto_first_flag',
            torch.ones(num_classes, dtype=torch.bool))
        self.prototype_EMA_feat = nn.Embedding(num_classes, proto_dim)

        self.global_protoEMA_agg = nn.Sequential(
            nn.Linear(proto_dim, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels))
        self.local_protoEMA_agg = nn.Sequential(
            nn.Linear(proto_dim, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels))
        self.for_query_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels))

        self.query_self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, batch_first=False)
        self.self_attn_norm = nn.LayerNorm(feat_channels)
        self.post_norm = nn.LayerNorm(feat_channels)

        # Unlike occupancy PQD, map remains K independent sigmoid channels, so
        # this branch predicts K-way query-to-class correction without a
        # no-object class.
        self.cls_embed = nn.Linear(feat_channels, num_classes)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, self.out_channels))

        class_mixing_bias = (
            torch.eye(num_classes).unsqueeze(0)
            * self.class_mixing_identity_bias_value)
        self.register_buffer('class_mixing_bias', class_mixing_bias)

        self.loss_coarse_bce = (
            build_loss(loss_coarse_bce) if loss_coarse_bce is not None else None)
        self.loss_coarse_dice = (
            build_loss(loss_coarse_dice) if loss_coarse_dice is not None else None)
        self.loss_mask_focal = (
            build_loss(loss_mask_focal) if loss_mask_focal is not None else None)
        self.loss_mask_dice = (
            build_loss(loss_mask_dice) if loss_mask_dice is not None else None)

    def _normalize_mask_feat(self, mask_feat):
        if mask_feat.dim() != 4:
            raise ValueError(
                'ProtoMapHeadV2 expects mask_feat with 4 dims, got '
                f'{tuple(mask_feat.shape)}.')
        if mask_feat.shape[-1] == self.proto_dim:
            return mask_feat
        if mask_feat.shape[1] == self.proto_dim:
            return mask_feat.permute(0, 2, 3, 1).contiguous()
        raise ValueError(
            'ProtoMapHeadV2 expects mask_feat as (B,H,W,proto_dim) or '
            '(B,proto_dim,H,W), but got '
            f'{tuple(mask_feat.shape)} with proto_dim={self.proto_dim}.')

    def _adaPG(self, coarse_pred, mask_feat):
        """Scene-adaptive prototype generator for sigmoid map logits."""
        mask_feat = self._normalize_mask_feat(mask_feat)
        soft_masks = torch.sigmoid(coarse_pred.detach())
        zero_proto = mask_feat.sum(dim=(0, 1, 2)) * 0.0

        for_query = []
        valid_mask = []
        for class_idx in range(self.num_classes):
            conf_mask = soft_masks[:, class_idx] > self.prototype_mining_thresh
            if bool(conf_mask.any().item()):
                query = mask_feat[conf_mask].mean(dim=0)
                valid_mask.append(True)
            else:
                query = zero_proto
                valid_mask.append(False)
            for_query.append(query.unsqueeze(0))

        return (
            torch.cat(for_query, dim=0),
            torch.tensor(valid_mask, dtype=torch.bool, device=mask_feat.device))

    def _update_ema_and_get(self, for_query, valid_mask):
        if self.training:
            with torch.no_grad():
                init_flag = self.proto_first_flag & valid_mask
                if bool(init_flag.any().item()):
                    self.prototype_EMA_feat.weight.data[init_flag] = (
                        for_query[init_flag].detach())
                    self.proto_first_flag[init_flag] = False

                update_flag = (~self.proto_first_flag) & valid_mask
                if bool(update_flag.any().item()):
                    self.prototype_EMA_feat.weight.data[update_flag] = (
                        self.prototype_EMA_feat.weight.data[update_flag]
                        * (1.0 - self.prototype_ema_weight)
                        + for_query[update_flag].detach()
                        * self.prototype_ema_weight)

        return self.prototype_EMA_feat.weight

    def _scene_aware_queries(self, for_query, ema_query):
        learnable_query = self.query_feat.weight

        if self.with_cp and self.training:
            global_query = checkpoint(self.global_protoEMA_agg, ema_query)
            local_query = checkpoint(self.local_protoEMA_agg, for_query)
        else:
            global_query = self.global_protoEMA_agg(ema_query)
            local_query = self.local_protoEMA_agg(for_query)

        query_feat = learnable_query + global_query + local_query

        if self.with_cp and self.training:
            query_feat = checkpoint(self.for_query_embed, query_feat)
        else:
            query_feat = self.for_query_embed(query_feat)
        return query_feat

    def _query_self_attention(self, query_feat, batch_size):
        query_feat = query_feat.unsqueeze(1).repeat(1, batch_size, 1)
        if not self.use_query_self_attn:
            return query_feat

        attn_out, _ = self.query_self_attn(query_feat, query_feat, query_feat)
        return self.self_attn_norm(query_feat + attn_out)

    def forward_head(self, decoder_out, mask_feature):
        decoder_out = self.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1).contiguous()

        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        query_masks = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        final_masks = self.format_results(cls_pred, query_masks)
        return cls_pred, query_masks, final_masks

    def format_results(self, cls_pred, query_masks):
        class_prob = F.softmax(cls_pred + self.class_mixing_bias, dim=-1)
        query_probs = query_masks.sigmoid()
        final_probs = torch.einsum('bqk,bqhw->bkhw', class_prob, query_probs)
        final_probs = final_probs.clamp(min=1e-4, max=1.0 - 1e-4)
        final_logits = torch.logit(final_probs)
        return final_logits

    def forward(self,
                bev_feature,
                mask_feat=None,
                coarse_pred=None,
                gt_masks_bev=None,
                return_query_info=False):
        """Forward BEV map features.

        Args:
            bev_feature: (B, C, H, W), used as PQD-style mask feature canvas.
            mask_feat: optional (B, H, W, P) prototype mining feature.
            coarse_pred: optional (B, K, H, W) coarse map logits.
        """
        if coarse_pred is None or mask_feat is None:
            if self.map_decoder is None:
                raise ValueError(
                    'ProtoMapHeadV2 needs mask_feat and coarse_pred when '
                    'use_internal_cnn2d_decoder=False.')
            coarse_pred, mask_feat = self.map_decoder(bev_feature)

        for_query, valid_mask = self._adaPG(coarse_pred, mask_feat)
        ema_query = self._update_ema_and_get(for_query, valid_mask)
        query_feat = self._scene_aware_queries(for_query, ema_query)
        query_feat = self._query_self_attention(query_feat, bev_feature.shape[0])
        cls_pred, query_masks, final_masks = self.forward_head(
            query_feat, bev_feature)

        if return_query_info:
            query_info = dict(
                cls_pred=cls_pred,
                query_masks=query_masks,
                for_query=for_query,
                valid_mask=valid_mask,
                ema_query=ema_query)
            return coarse_pred, final_masks, query_info
        return coarse_pred, final_masks

    def loss(self, coarse_pred, final_masks, gt_masks_bev):
        gt_masks_bev = gt_masks_bev.float()
        losses = {}

        if self.loss_coarse_bce is not None:
            losses['loss_map_coarse_bce'] = self.loss_coarse_bce(
                coarse_pred, gt_masks_bev)

        if self.loss_coarse_dice is not None:
            losses['loss_map_coarse_dice'] = self.loss_coarse_dice(
                coarse_pred.reshape(-1, *coarse_pred.shape[2:]),
                gt_masks_bev.reshape(-1, *gt_masks_bev.shape[2:]))

        if self.loss_mask_focal is not None:
            losses['loss_map_focal'] = self.loss_mask_focal(
                final_masks, gt_masks_bev)

        if self.loss_mask_dice is not None:
            losses['loss_map_dice'] = self.loss_mask_dice(
                final_masks.reshape(-1, *final_masks.shape[2:]),
                gt_masks_bev.reshape(-1, *gt_masks_bev.shape[2:]))

        return losses

    def predict(self, final_masks):
        return torch.sigmoid(final_masks)
