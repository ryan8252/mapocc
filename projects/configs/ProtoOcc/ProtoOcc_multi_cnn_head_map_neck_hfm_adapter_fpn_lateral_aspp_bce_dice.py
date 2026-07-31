_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp.py']

# Clean cumulative-ablation row:
#   multi_scale_bev -> Map-HFM/VGMR -> PerScaleMapResidualAdapter
#   -> FPN lateral + ASPP map neck -> BEVSegHead.
#
# This keeps the original BCE(5.0)+Dice(1.0) map supervision inherited from
# ProtoOcc_multi_cnn_head.py. It intentionally does not enable weighted
# focal-Dice supervision or the active enhancement gate.
model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        use_map_hfm=True,
        map_hfm_lower_source='vox3',
        detach_map_feature=False,
        map_hfm_order='hfm_then_adapter',
        map_hfm_fusion_mode='residual_concat',
        map_hfm_voxel_residual_scale=1.0,
        map_residual_adapter=dict(
            type='PerScaleMapResidualAdapter',
            in_channels=[160, 320, 640],
            residual_scale=1.0,
            with_cp=True,
            detach_input=False)),
    bev_seg_head=dict(
        map_loss_type=None,
        use_map_class_weights=False,
        use_map_active_enhance_gate=False))
