_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# A2: Z-aware residual with K=4 learned height pooling patterns.
#
# This keeps the shared 128ch map-neck path intact and adds only a zero-init
# residual. The residual branch learns four softmax-over-Z aggregation patterns
# from the full LSS voxel feature, e.g. ground / low obstacle / vertical /
# mixed structure, then projects 80*K channels back to 128 map channels.
numC_Trans = 80
map_bev_channels = 128

model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        map_z_residual=dict(
            type='MapZResidualLayer',
            in_channels=numC_Trans,
            z_channels=16,
            out_channels=map_bev_channels,
            hidden_channels=map_bev_channels,
            z_projection='learned_pool',
            num_height_patterns=4,
            residual_scale=1.0,
            detach_input=True,
            with_cp=True,
            zero_init_z_logits=True)))
