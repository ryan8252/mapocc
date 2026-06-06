_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only.py']

# BEVFusion-style classwise focal ablation for the aligned map-only ceiling.
#
# Inherited geometry:
#   shared BEV feature: +-51.2m / 0.4m -> 256 x 256
#   map supervision:    +-50m / 0.5m   -> 200 x 200
#
# Difference vs aligned_map_only:
#   - keep ProtoOcc depth/PV auxiliary losses unchanged
#   - keep focal recipe gamma=2, alpha=-1
#   - compute one focal loss per map class, matching BEVFusion's segmentation
#     head aggregation instead of one averaged 6-channel focal tensor
#
# NOTE: classwise_sum intentionally does not divide by num_classes. This is the
# exact BEVFusion-style aggregation and increases map pressure relative to the
# depth/PV auxiliary losses. Use classwise_mean only for a scale-controlled
# diagnostic.

map_classes = [
    'drivable_area',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
    'divider',
]

model = dict(
    map_loss_weight=128.0,
    bev_seg_head=dict(
        map_focal_loss_mode='classwise_sum',
        map_focal_loss_class_names=map_classes,
        loss_bce=None,
        loss_dice=None,
        loss_focal=dict(
            _delete_=True,
            type='BinaryMaskFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=-1.0,
            reduction='mean',
            loss_weight=1.0)))
