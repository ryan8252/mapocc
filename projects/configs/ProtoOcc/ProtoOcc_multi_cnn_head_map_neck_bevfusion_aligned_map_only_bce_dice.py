_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only.py']

# Same BEVFusion-aligned map-only geometry as the focal128 upper-bound config:
#   shared BEV feature: +-51.2m / 0.4m -> 256 x 256
#   map supervision:    +-50m / 0.5m   -> 200 x 200
#
# This ablation restores the native map-only BCE+Dice objective so the effect of
# the aligned geometry can be measured without the focal-only weight128 confound.
model = dict(
    map_loss_weight=1.0,
    bev_seg_head=dict(
        loss_bce=dict(
            _delete_=True,
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            _delete_=True,
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            loss_weight=1.0),
        loss_focal=None))
