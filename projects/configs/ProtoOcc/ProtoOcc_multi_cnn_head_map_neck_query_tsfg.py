_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Map-neck Query-TSFG.
# Keep the 128-channel map neck as the primary map path and apply
# background PQD scene-aware queries directly in map feature space.

query_channels = 48
map_bev_channels = 128

model = dict(
    map_loss_weight=4.0,
    voxel_aware_map_ingest=dict(
        type='MapNeckQueryTSFG',
        query_channels=query_channels,
        map_channels=map_bev_channels,
        occ_num_classes=18,
        background_class_ids=(11, 12, 13, 14, 15, 16),
        prototype_source='query_norm_real_bqc',
        detach_query=True,
        prototype_hidden_channels=128,
        similarity_mode='cosine',
        use_prototype_wise=True,
        use_prototype_aware=True,
        residual=True,
        fixed_gamma=1.0,
        zero_init_delta=True,
        zero_init_gate=True,
        debug=False,
        debug_interval=100,
    ),
)

# This child config replaces the base list only explicitly here; keep EMA and
# do not add the CFV gamma warmup hook.
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# Bring evaluation forward enough to observe the epoch 11 stop criterion.
evaluation = dict(start=10)
