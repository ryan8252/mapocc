import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmdet3d.models.builder import HEADS, build_loss


@HEADS.register_module()
class MAESTROBEVSegHead(BaseModule):
    """Lightweight BEV map segmentation head for MAESTRO 2-task ProtoOcc."""

    def __init__(self,
                 in_channels,
                 hidden_channels=128,
                 num_classes=6,
                 num_convs=2,
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 loss_bce=None,
                 loss_dice=None,
                 init_cfg=None):
        super(MAESTROBEVSegHead, self).__init__(init_cfg)
        self.with_cp = with_cp
        self.num_classes = num_classes

        blocks = []
        current_channels = in_channels
        for _ in range(num_convs):
            blocks.append(
                ConvModule(
                    current_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU', inplace=True),
                ))
            current_channels = hidden_channels

        self.decoder = nn.Sequential(*blocks)
        self.predictor = nn.Conv2d(current_channels, num_classes, kernel_size=1)
        self.loss_bce = build_loss(loss_bce) if loss_bce is not None else None
        self.loss_dice = build_loss(loss_dice) if loss_dice is not None else None

    def forward(self, bev_feature):
        if self.with_cp and self.training:
            bev_feature = checkpoint(self.decoder, bev_feature)
        else:
            bev_feature = self.decoder(bev_feature)
        return self.predictor(bev_feature)

    def loss(self, seg_logits, gt_masks_bev):
        gt_masks_bev = gt_masks_bev.float()
        losses = {}

        if self.loss_bce is not None:
            losses['loss_map_bce'] = self.loss_bce(seg_logits, gt_masks_bev)

        if self.loss_dice is not None:
            losses['loss_map_dice'] = self.loss_dice(
                seg_logits.reshape(-1, *seg_logits.shape[2:]),
                gt_masks_bev.reshape(-1, *gt_masks_bev.shape[2:]),
            )

        return losses

    def predict(self, seg_logits):
        return torch.sigmoid(seg_logits)
