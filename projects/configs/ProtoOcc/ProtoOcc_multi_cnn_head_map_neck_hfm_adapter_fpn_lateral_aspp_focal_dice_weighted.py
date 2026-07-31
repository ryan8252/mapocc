_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted.py']

# Composition config:
#   - map_hfm: voxel-branch geometry (mean+max over height) injected into each
#              map BEV pyramid level as a zero-init residual.
#   - PerScaleMapResidualAdapter: 2D zero-init per-scale map refinement.
#   - FPN lateral projection + residual ASPP remains inside the 128ch map neck.
#   - focal + Dice weighted map loss is inherited from the base config.
#
# Runtime order in Dual_Branch_Encoder:
#   multi_scale_bev -> map_hfm -> adapter -> FPN lateral ASPP map neck
#   -> BEVSegHead.
model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        use_map_hfm=True,
        map_hfm_lower_source='vox3',
        detach_map_feature=False,
        map_residual_adapter=dict(
            type='PerScaleMapResidualAdapter',
            in_channels=[160, 320, 640],
            residual_scale=1.0,
            with_cp=True,
            detach_input=False)))
