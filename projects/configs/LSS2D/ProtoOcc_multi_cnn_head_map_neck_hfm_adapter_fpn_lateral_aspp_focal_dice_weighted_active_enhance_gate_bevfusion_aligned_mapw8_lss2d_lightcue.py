_base_ = [
    '../ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8.py'
]

# First LSS2D cue ablation:
#   keep the DBE/mapw8 map feature as the base route
#   add a lightweight LSS2D residual cue before BEVSegHead
#
# map_lss2d_feat:
#   LSSViewTransformer_depthGT collapse_z=True
#   [B, 80, 256, 256] on the shared [-51.2, 51.2] / 0.4m canvas
#
# fusion path:
#   align to BEVFusion map grid -> [B, 80, 200, 200]
#   1x1/3x3 cue adapter -> [B, 128, 200, 200]
#   map_final = map_base + gate * zero_init_conv(cat(map_base, lss2d_cue))

numC_Trans = 80
map_bev_channels = 128

map_lss2d_grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 45.0, 0.5],
}

model = dict(
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
    map_lss2d_encoder=None,
    map_lss2d_fusion=dict(
        type='LSS2DMapCueFusion',
        base_channels=map_bev_channels,
        cue_channels=numC_Trans,
        hidden_channels=map_bev_channels,
        residual_gate_init=0.1,
        with_cp=True))
