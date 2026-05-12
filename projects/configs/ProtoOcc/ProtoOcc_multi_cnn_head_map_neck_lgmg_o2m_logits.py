_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Stage 3: Occ-to-Map only map recovery.
# This keeps the strong map-neck baseline and adds only a detached coarse-OCC
# geometry side path that predicts a residual on top of map logits.
model = dict(
    map_loss_weight=4.0,
    bev_seg_head=dict(
        map_loss_balance_mode='none',
        map_balance_debug=False),
    occ_to_map_adapter=dict(
        type='OccToMapGeometryAdapter',
        num_occ_classes=18,
        num_map_classes=6,
        hidden_channels=64,
        num_convs=2,
        beta_o2m_max=1.0,
        zero_until_epoch=6,
        warmup_epochs=4,
        detach_occ=True,
        free_class_index=17,
        ground_layout_indices=(11, 12, 13, 14),
        structure_indices=(15, 16),
        dynamic_indices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        height_threshold=0.5,
        zero_init_output=True))

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(type='OccToMapWarmupHook', priority='LOW'),
]

# Start validation early enough to apply the Stage 3 epoch-11 stop criterion.
evaluation = dict(start=10)
