_base_ = [
    './ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned.py'
]

# Variant B: replace Dual_Branch_Encoder's BEV branch input with the 256 x 256
# LSS2D feature. This changes both consumers of the shared BEV branch:
#   1. OCC hierarchical BEV-to-voxel fusion
#   2. inherited map HFM/adapter/FPN-lateral-ASPP map path
#
# Geometry:
#   OCC supervision:       [-40, 40] / 0.4m -> 200 x 200 x 16 crop
#   shared feature canvas: [-51.2, 51.2] / 0.4m -> 256 x 256
#   map supervision:       [-50, 50] / 0.5m -> 200 x 200
#   shared LSS2D input:    [B, 80, 256, 256] -> 1x1 -> [B, 160, 256, 256]

numC_Trans = 80

occ_point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
shared_feature_range = [-51.2, -51.2, -1.0, 51.2, 51.2, 5.4]
grid_size = [200, 200, 16]

shared_lss2d_grid_config = {
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
    use_lss2d_as_bev_branch=True,
    map_img_view_transformer=dict(
        type='LSSViewTransformer_depthGT',
        grid_config=shared_lss2d_grid_config,
        input_size=(256, 704),
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=16),
    dual_branch_encoder=dict(
        external_bev_input_channels=numC_Trans))
