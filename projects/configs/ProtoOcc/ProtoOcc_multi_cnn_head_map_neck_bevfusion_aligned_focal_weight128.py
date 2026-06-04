_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned.py']

# BEVFusion-aligned focal-only map loss baseline.
#
# Geometry is inherited from ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned:
#   shared feature canvas: [-51.2, 51.2] / 0.4m -> 256 x 256
#   occupancy supervision: [-40, 40] / 0.4m -> 200 x 200 x 16 crop
#   map supervision:       [-50, 50] / 0.5m -> 200 x 200
#   LSS depth:             [1.0, 45.0] / 0.5m -> 88 bins
#
# This formalizes the saved work-dir config used by the focal weight128 smoke
# runs. The high detector-level map_loss_weight compensates for focal-only map
# loss having a much smaller scalar than the original BCE + Dice map loss.
model = dict(
    map_loss_weight=128.0,
    bev_seg_head=dict(
        loss_bce=None,
        loss_dice=None,
        loss_focal=dict(
            type='BinaryMaskFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=-1.0,
            reduction='mean',
            loss_weight=1.0)))
