_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 7 ablation: auxiliary map supervision from the intermediate
# Custom_FPN_LSS map-neck features at 100x100 and 50x50. Targets are
# downsampled with per-class max pooling so thin labels do not disappear.
model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        map_bev_encoder_neck=dict(
            return_map_aux_features=True,
            map_aux_feature_levels=['100', '50'])),
    bev_seg_head=dict(
        use_map_aux_loss=True,
        map_aux_levels=['100', '50'],
        map_aux_weight_100=0.4,
        map_aux_weight_50=0.2,
        map_aux_downsample='maxpool',
        map_aux_in_channels={
            '100': 256,
            '50': 256,
        }))
