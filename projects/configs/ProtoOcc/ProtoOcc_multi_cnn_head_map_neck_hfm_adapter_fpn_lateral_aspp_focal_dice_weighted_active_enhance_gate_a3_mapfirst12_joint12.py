_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py']

# A3: same architecture as the current best config.
#
# Epoch 0-11: map-first warmup. The hook zeros OCC prototype/PQD loss weights
# while keeping map_loss_weight=4.0 and keeping depth supervision active.
# Epoch 12-23: restore the original OCC/PQD loss weights for joint training.
model = dict(map_loss_weight=4.0)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='MapFirstStageLossHook',
        map_first_epochs=12,
        map_loss_weight=4.0,
        disable_occ_loss=True,
        disable_depth_loss=False,
        priority='NORMAL',
    ),
]
