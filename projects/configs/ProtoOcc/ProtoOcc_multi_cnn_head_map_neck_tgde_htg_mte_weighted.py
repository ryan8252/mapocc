_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# TGDE 01+02:
#   01 Height Task Gate splits LSS voxel features into occ/map views.
#   02 Map Topology Encoder consumes the map view with weighted-Z pooling.
#
# The occupancy decoders and map head stay fixed.  The map path still uses the
# existing 128-channel map neck before BEVSegHead, so this tests encoder-side
# task-specific geometry decomposition rather than a new decoder.

numC_Trans = 80
grid_z = 16

model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        height_task_gate=dict(
            type='HeightTaskGate',
            in_channels=numC_Trans,
            reduction=4,
            gate_scale=0.1,
            zero_init_last=True),
        use_height_task_gate_for_legacy=True,
        map_topology_encoder=dict(
            type='MapTopologyEncoder',
            in_channels=numC_Trans,
            z_channels=grid_z,
            z_projection='weighted_pool',
            mid_channels=numC_Trans * 2,
            num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
            num_layer=[1, 1, 1],
            stride=[2, 2, 2],
            ConvNext_kernel_size=7,
            with_cp=True,
            zero_init_z_logits=True)))

# Bring evaluation forward enough to compare against the weight4 epoch-11
# guardrail before spending a full run.
evaluation = dict(start=10)
