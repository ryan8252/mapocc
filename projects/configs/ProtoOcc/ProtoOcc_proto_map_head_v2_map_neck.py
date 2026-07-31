_base_ = ['./ProtoOcc_proto_map_head.py']

# Clean ProtoMapHeadV2 ablation.
# Keep the validated 128-channel map-specific BEV neck, but replace the old
# ProtoMapHead with the cleaner PQD-style V2 decoder.
numC_Trans = 80
map_bev_channels = 128
proto_dim = 64
num_map_classes = 6

model = dict(
    # V2 has both coarse and final map supervision, so start below the
    # BEVSegHead weight-4 baseline to avoid over-weighting the map branch.
    map_loss_weight=2.0,
    dual_branch_encoder=dict(
        return_bev_feature=True,
        return_map_feature=True,
        detach_map_feature=False,
        map_bev_encoder_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
            catconv_in_channels2=numC_Trans * 2 + map_bev_channels * 2,
            out_channels=map_bev_channels,
            input_feature_index=(0, 1, 2),
            with_cp=True)),
    proto_map_head=dict(
        _delete_=True,
        type='ProtoMapHeadV2',
        in_channels=map_bev_channels,
        feat_channels=map_bev_channels,
        proto_dim=proto_dim,
        num_classes=num_map_classes,
        num_heads=8,
        prototype_mining_thresh=0.45,
        prototype_ema_weight=0.01,
        use_internal_cnn2d_decoder=True,
        cnn2d_decoder_cfg=dict(
            hidden_dim=map_bev_channels,
            num_convs=2),
        use_query_self_attn=True,
        class_mixing_identity_bias=4.0,
        with_cp=True,
        loss_coarse_bce=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=2.5),
        loss_coarse_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            loss_weight=0.5),
        loss_mask_focal=dict(
            type='BinaryMaskFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=5.0),
        loss_mask_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            loss_weight=1.0)))
