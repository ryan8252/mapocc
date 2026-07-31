_base_ = [
    './ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned.py'
]

# Variant A: keep OCC on the original 3D LSS/DBE path, but feed the map head
# from a separate map-specific 2D LSS branch.
#
# Geometry:
#   OCC supervision:       [-40, 40] / 0.4m -> 200 x 200 x 16 crop
#   shared feature canvas: [-51.2, 51.2] / 0.4m -> 256 x 256
#   map supervision:       [-50, 50] / 0.5m -> 200 x 200
#   map LSS2D output:      [B, 80, 256, 256]

numC_Trans = 80
map_bev_channels = 128
protoocc_decoder_channels = 160

occ_point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
shared_feature_range = [-51.2, -51.2, -1.0, 51.2, 51.2, 5.4]
grid_size = [200, 200, 16]

map_lss2d_grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 45.0, 0.5],
}

map_xbound = [-50.0, 50.0, 0.5]
map_ybound = [-50.0, 50.0, 0.5]
map_feature_range = [
    map_xbound[0],
    map_ybound[0],
    map_xbound[1],
    map_ybound[1],
]
map_feature_size = [
    int((map_xbound[1] - map_xbound[0]) / map_xbound[2]),
    int((map_ybound[1] - map_ybound[0]) / map_ybound[2]),
]

model = dict(
    pc_range=occ_point_cloud_range,
    grid_size=grid_size,
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
    # The map feature comes from map_lss2d_encoder above. Disable the inherited
    # DBE map neck/HFM/adapter path so this run isolates the LSS2D map branch and
    # does not pay for an unused second map decoder.
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
