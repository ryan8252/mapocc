_base_ = [
    './ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py'
]

# 方案1 on the NATIVE +-40 / 0.4 map grid (NOT the BEVFusion +-50/0.5 alignment).
#
# Same idea as the bevfusion-aligned `..._map_lss2d256_mapdepth.py`, but kept on
# the native grid so map-miou is comparable to the other +-40/0.4 runs
# (active_enhance_gate etc.). OCC is untouched (3D voxel + sharp supervised
# depth). The map head is fed by a separate 2D-LSS branch with its OWN soft,
# unsupervised depth (MapDepthNet); the DBE map path is disabled.
#
# Grids (all native, inherited from the base -- no pipeline re-declaration):
#   shared feature canvas: [-40, 40] x/y, 0.4m -> 200 x 200
#   occ supervision:       [-40, 40] / 0.4   -> 200 x 200 x 16 (no crop)
#   map supervision/eval:  [-40, 40] / 0.4   -> 200 x 200 (== shared, no resample)
#   map LSS2D output:      [B, 80, 200, 200]

numC_Trans = 80
map_bev_channels = 128
protoocc_decoder_channels = 160
depth_categories = 88

occ_point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
shared_feature_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
map_feature_range = [-40.0, -40.0, 40.0, 40.0]
map_feature_size = [200, 200]

# 2D-LSS map branch on the native +-40/0.4 canvas (single collapsed z bin).
map_lss2d_grid_config = {
    'x': [-40.0, 40.0, 0.4],
    'y': [-40.0, 40.0, 0.4],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 45.0, 0.5],
}

model = dict(
    shared_feature_range=shared_feature_range,
    occ_feature_range=occ_point_cloud_range,
    map_feature_range=map_feature_range,
    map_feature_size=map_feature_size,
    use_lss2d_as_bev_branch=False,
    map_img_view_transformer=dict(
        type='LSSViewTransformer_depthGT',
        grid_config=map_lss2d_grid_config,
        input_size=(256, 704),
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=16),
    # Separate soft, unsupervised depth for the map branch (方案1).
    map_depth_net=dict(
        type='MapDepthNet',
        in_channels=512,
        mid_channels=256,
        depth_channels=depth_categories,
        num_layers=2,
        with_cp=True,
        init_uniform=True),
    map_lss2d_encoder=dict(
        type='MapOnly_BEV2D_Encoder',
        input_projection=[numC_Trans, protoocc_decoder_channels],
        bev_encoder_backbone=dict(
            type='CustomBEVBackbone',
            stride=[2, 2, 2],
            numC_input=protoocc_decoder_channels,
            num_channels=[
                protoocc_decoder_channels,
                protoocc_decoder_channels * 2,
                protoocc_decoder_channels * 4,
            ]),
        map_bev_encoder_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=protoocc_decoder_channels * 4
            + protoocc_decoder_channels * 2,
            catconv_in_channels2=protoocc_decoder_channels
            + map_bev_channels * 2,
            out_channels=map_bev_channels,
            input_feature_index=(0, 1, 2),
            with_cp=True,
            use_fpn_lateral_projection=True,
            fpn_lateral_in_channels=[
                protoocc_decoder_channels,
                protoocc_decoder_channels * 2,
                protoocc_decoder_channels * 4,
            ],
            fpn_projection_channels=256,
            use_fpn_global_context=True,
            fpn_global_context_type='aspp')),
    # Disable the inherited DBE map neck/HFM/adapter path so this run isolates
    # the LSS2D map branch (mirrors the bevfusion-aligned map_lss2d256 config).
    dual_branch_encoder=dict(
        return_bev_feature=True,
        return_map_feature=False,
        use_map_hfm=False,
        map_residual_adapter=None,
        map_bev_encoder_neck=None,
        map_fpn_base_residual_neck=None,
        map_z_residual=None,
        map_highres_skip=False,
        use_map_z_aware_compression=False,
        use_map_pre_backbone_adapter=False))

# NOTE: do NOT set find_unused_parameters=True -- the DBE bev_encoder_neck is
# folded into the OCC voxel (not unused) and it clashes with the reentrant
# gradient checkpointing used throughout the model.
