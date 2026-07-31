_base_ = [
    './ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py'
]

# Simplified best-config composition.
#
# Inherited:
#   - FPN lateral projection + residual ASPP in the 128ch map neck
#   - focal + Dice weighted map loss
#   - map-active enhancement gate
#
# Changed here:
#   - map branch order becomes adapter-first additive Map-HFM:
#       B_i -> PerScaleMapResidualAdapter -> M_i
#       voxel_i -> mean/max-Z collapse -> V_i
#       F_i = M_i + V_i
#   - add a plain 128ch FPN residual from the original shared BEV pyramid:
#       map_out = FPN_Lateral_ASPP(F_i) + FPN_plain128(B_i)
numC_Trans = 80
map_bev_channels = 128

model = dict(
    dual_branch_encoder=dict(
        map_hfm_order='adapter_then_hfm',
        map_hfm_fusion_mode='voxel_add',
        map_hfm_voxel_residual_scale=1.0,
        map_fpn_base_residual_scale=1.0,
        map_fpn_base_residual_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
            catconv_in_channels2=numC_Trans * 2 + map_bev_channels * 2,
            out_channels=map_bev_channels,
            input_feature_index=(0, 1, 2),
            with_cp=True)))
