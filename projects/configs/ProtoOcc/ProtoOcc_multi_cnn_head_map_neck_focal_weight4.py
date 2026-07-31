_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_weight4.py']

# BEVFusion / MAESTRO-style map-loss ablation on the aligned map grid:
#   shared feature canvas: [-51.2, 51.2] x/y, 0.4m
#   occupancy supervision: [-40, 40] x/y, 0.4m
#   map supervision:       [-50, 50] x/y, 0.5m
#
# The base aligned_weight4 config keeps ProtoOcc's BCE + Dice map loss.
# This variant switches only the map segmentation loss to dense sigmoid focal
# loss with the BEVFusion / local-MAESTRO setting alpha=-1, gamma=2.
model = dict(
    map_loss_weight=4.0,
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
