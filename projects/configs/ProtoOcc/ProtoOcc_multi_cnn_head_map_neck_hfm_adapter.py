_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Combine the two independently-validated map modules (both ~+0.9 Map mIoU at
# epoch 24, on different mechanisms, so they should be at least partly additive):
#   * map_hfm : inject voxel-branch geometry (mean+max over height) into the
#               map BEV path as a zero-init residual -> new information.
#   * PerScaleMapResidualAdapter : 2D zero-init refinement of the map BEV path.
# Encoder order: multi_scale_bev -> map_hfm -> adapter -> map neck.
# Both are zero-init residuals, so step-0 behaviour is identical to the
# map_loss_weight=4 baseline.
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
