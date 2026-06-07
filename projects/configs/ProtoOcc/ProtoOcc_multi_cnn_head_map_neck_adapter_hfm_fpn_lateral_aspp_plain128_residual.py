_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_adapter_hfm_fpn_lateral_aspp.py']

# Plain-128ch-FPN residual ablation.
#
# Refined path:
#   map_refined = FPN_Lateral_ASPP(F0, F1, F2)
#
# Base residual path:
#   map_base = FPN_plain128(B0, B1, B2)
#
# Output:
#   map_out = map_refined + map_base
numC_Trans = 80
map_bev_channels = 128

model = dict(
    dual_branch_encoder=dict(
        map_fpn_base_residual_scale=1.0,
        map_fpn_base_residual_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
            catconv_in_channels2=numC_Trans * 2 + map_bev_channels * 2,
            out_channels=map_bev_channels,
            input_feature_index=(0, 1, 2),
            with_cp=True)))
