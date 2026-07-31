_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# V2: CFV Prototype-TSFG + scheduled residual.
# Keep the 128-channel map neck as the protected main path, then inject a
# background-query-guided CFV prior through a warmup-controlled residual branch.

voxel_out_channels = 48
query_channels = 48
map_bev_channels = 128

model = dict(
    map_loss_weight=4.0,
    voxel_aware_map_ingest=dict(
        type='CFVPrototypeTSFGMapFusion',
        voxel_in_channels=voxel_out_channels,
        query_channels=query_channels,
        map_channels=map_bev_channels,
        occ_num_classes=18,
        background_class_ids=(11, 12, 13, 14, 15, 16),
        z_collapse_mode='avg_max_concat',
        detach_cfv=True,
        detach_query=True,
        query_source='scene_aware',
        query_to_filter='pqd_mask_embed_detached',
        prototype_mlp_hidden_channels=128,
        mlp_proto_init='residual_zero',
        normalize_similarity=True,
        activation_mode='sigmoid_cosine',
        temperature=10.0,
        c_gate_mode='prototype',
        fusion_mode='residual_two_layer',
        residual=True,
        zero_init_delta=True,
        gamma_schedule=dict(
            type='linear',
            start_epoch=0,
            end_epoch=3,
            start_value=0.0,
            end_value=1.0,
        ),
        initial_gamma=1.0,
        refine_kernel_size=1,
        debug=False,
        debug_interval=100,
    ),
)

# This child config replaces the base list, so keep EMA explicitly.
custom_hooks = [
    dict(
        type='CFVProtoTSFGWarmupHook',
        start_epoch=0,
        end_epoch=3,
        start_value=0.0,
        end_value=1.0,
        priority='NORMAL'),
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL'),
]

# Bring evaluation forward enough to observe the epoch 11 stop criterion.
evaluation = dict(start=10)
