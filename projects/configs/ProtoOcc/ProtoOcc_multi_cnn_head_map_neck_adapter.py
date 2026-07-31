_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Per-scale zero-init map residual adapter. This creates a lightweight
# map-specific BEV expert before the 128ch map neck. Zero initialization makes
# the initial model equivalent to the map-neck baseline.
model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        map_residual_adapter=dict(
            type='PerScaleMapResidualAdapter',
            in_channels=[160, 320, 640],
            residual_scale=1.0,
            with_cp=True,
            detach_input=False)))
