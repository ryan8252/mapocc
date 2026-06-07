_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp.py']

# Map-specific pyramid replacement ablation.
#
# Flow:
#   B_i -> PerScaleMapResidualAdapter -> M_i
#   voxel_i -> Mean/Max-Z collapse -> 1x1 projection -> V_i
#   F_i = M_i + V_i
#   map_out = FPN_Lateral_ASPP(F0, F1, F2)
#
# There is no extra FPN(B0, B1, B2) residual in this config.
model = dict(
    dual_branch_encoder=dict(
        use_map_hfm=True,
        map_hfm_lower_source='vox3',
        detach_map_feature=False,
        map_hfm_order='adapter_then_hfm',
        map_hfm_fusion_mode='voxel_add',
        map_hfm_voxel_residual_scale=1.0,
        map_residual_adapter=dict(
            type='PerScaleMapResidualAdapter',
            in_channels=[160, 320, 640],
            residual_scale=1.0,
            with_cp=True,
            detach_input=False)))
