_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 4 ablation: same-channel lateral projection inside Custom_FPN_LSS.
# Pyramid levels [160, 320, 640] are projected to 256 channels before the
# existing concat-conv fusion. The final map feature remains (B, 128, H, W).
numC_Trans = 80

model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        map_bev_encoder_neck=dict(
            use_fpn_lateral_projection=True,
            fpn_lateral_in_channels=[
                numC_Trans * 2,
                numC_Trans * 4,
                numC_Trans * 8,
            ],
            fpn_projection_channels=256)))
