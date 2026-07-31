_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py']

# A2: joint training for the full run, but progressively increase the map loss
# multiplier from the validated 4.0 baseline to 8.0 during the first 12 epochs.
# The second half keeps map_loss_weight=8.0.
model = dict(map_loss_weight=4.0)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='MapLossWeightScheduleHook',
        start_epoch=0,
        end_epoch=12,
        start_value=4.0,
        end_value=8.0,
        priority='NORMAL',
    ),
]
