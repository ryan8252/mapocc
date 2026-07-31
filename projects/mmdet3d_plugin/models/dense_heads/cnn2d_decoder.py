import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS, build_loss


@HEADS.register_module()
class cnn2d_decoder(BaseModule):
    """2D counterpart of ProtoOcc's ``cnn3d_decoder`` for BEV map masks.

    The shape convention mirrors ``cnn3d_decoder``:
      - ``map_pred`` is the coarse semantic prediction.
      - ``mask_feat`` is the low-dimensional feature used by prototype mining.

    For map segmentation, ``map_pred`` keeps BEVSegHead's channel-first layout
    ``(B, K, H, W)`` so existing map losses/evaluators can consume it directly.
    """

    def __init__(self,
                 in_dim=128,
                 hidden_dim=None,
                 out_dim=32,
                 num_classes=6,
                 num_convs=2,
                 norm_cfg=dict(type='BN'),
                 loss_bce=None,
                 loss_dice=None,
                 loss_weight=1.0,
                 with_cp=False):
        super(cnn2d_decoder, self).__init__()
        hidden_dim = in_dim if hidden_dim is None else hidden_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.with_cp = with_cp

        blocks = []
        current_dim = in_dim
        for _ in range(num_convs):
            blocks.append(
                ConvModule(
                    current_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU', inplace=True)))
            current_dim = hidden_dim
        self.decoder = nn.Sequential(*blocks)

        # Keep the coarse map classifier aligned with BEVSegHead: a dense
        # 1x1 convolution over the decoded BEV feature.
        self.predicter = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.proto_bottleneck = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

        self.loss_bce = build_loss(loss_bce) if loss_bce is not None else None
        self.loss_dice = build_loss(loss_dice) if loss_dice is not None else None

    def forward(self, img_feats):
        """Forward BEV features.

        Args:
            img_feats: (B, C, H, W)

        Returns:
            map_pred:  (B, K, H, W)
            mask_feat: (B, H, W, out_dim)
        """
        if self.with_cp and self.training:
            decoder_feat = checkpoint(self.decoder, img_feats)
        else:
            decoder_feat = self.decoder(img_feats)

        map_feat = self.proto_bottleneck(img_feats)
        mask_feat = map_feat.permute(0, 2, 3, 1).contiguous()

        if self.with_cp and self.training:
            map_pred = checkpoint(self.predicter, decoder_feat)
        else:
            map_pred = self.predicter(decoder_feat)
        return map_pred, mask_feat

    def loss(self, map_pred, gt_masks_bev):
        """Coarse map loss, using the same map target format as BEVSegHead."""
        gt_masks_bev = gt_masks_bev.float()
        losses = {}

        if self.loss_bce is not None:
            losses['loss_map_cnn2d_bce'] = (
                self.loss_bce(map_pred, gt_masks_bev) * self.loss_weight)

        if self.loss_dice is not None:
            losses['loss_map_cnn2d_dice'] = (
                self.loss_dice(
                    map_pred.reshape(-1, *map_pred.shape[2:]),
                    gt_masks_bev.reshape(-1, *gt_masks_bev.shape[2:]))
                * self.loss_weight)

        return losses

    def predict(self, map_pred):
        return torch.sigmoid(map_pred)
