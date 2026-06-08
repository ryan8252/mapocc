_base_ = [
    '../ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8.py'
]

# Experiment (1): faithful lss2d map branch (the 0.491 map-only winner's recipe)
#   bolted onto the full multitask model, OCC kept on the original 3D DBE path.
# Goal: measure how much of lss2d's map (0.491 map-only) survives multitask when
#   the map lift and the OCC lift SHARE one depth_net (the coupling we diagnosed).
#
# This is map_lss2d256 reverted to lss2d's ORIGINAL recipe:
#   - map lift: 0.8m grid, z=[-10,10,20] -> single z-bin (flat), collapse_z=True
#   - map encoder: PLAIN MapOnly_BEV2D_Encoder (CustomBEVBackbone + Custom_FPN_LSS),
#                  WITHOUT the aspp / fpn_lateral enhancement map_lss2d256 added.
#   - inherited DBE map path disabled; map comes ONLY from the lss2d branch.
#   - inherits mapw8 (map_loss_weight=8), the best-tuned multitask map balance.
#
# Data flow (ProtoOccCnnSegHead.forward_train):
#   image_encoder (R50 + CustomFPN) -> x [B,512,h,w]                       SHARED
#   depth_net (CM_DepthNet)         -> pv_feat [B,80,h,w], depth           SHARED  <- coupling
#     OCC : img_view_transformer(depth, pv_feat)  [collapse_z=False, 0.4m, z16]
#             -> voxel_feat [B,80,Z,H,W] -> dual_branch_encoder -> cnn3d/PQD -> occ loss
#             depth_net.get_PV_loss -> depth loss
#     MAP : map_img_view_transformer(depth, pv_feat)  [collapse_z=True, 0.8m, flat]
#             -> map_lss2d_feat [B,80,128,128]
#             -> map_lss2d_encoder (CustomBEVBackbone + Custom_FPN_LSS) -> [B,128,.,.]
#             -> _align_map_feature -> 200x200 -> bev_seg_head -> map loss

numC_Trans = 80
map_bev_channels = 128
protoocc_decoder_channels = 160

occ_point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
shared_feature_range = [-51.2, -51.2, -1.0, 51.2, 51.2, 5.4]
grid_size = [200, 200, 16]

# lss2d original map lift grid: coarse 0.8m, single z-bin (flat BEV).
map_lss2d_grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 45.0, 0.5],
}

# Map supervision grid (BEVFusion / MAESTRO protocol): [-50, 50] / 0.5m -> 200 x 200.
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

# Two lifts share depth_net/pv_feat; the DBE map sub-modules are disabled (None),
# so some inherited params may not receive grad -> keep DDP happy.
find_unused_parameters = True

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
    # PLAIN lss2d encoder: no aspp, no fpn_lateral_projection (that is what
    # map_lss2d256 added and which regressed map to 0.428).
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
            with_cp=True)),
    # The map feature comes from map_lss2d_encoder above. Disable the inherited
    # DBE map neck/HFM/adapter path so map is driven ONLY by the lss2d branch.
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
