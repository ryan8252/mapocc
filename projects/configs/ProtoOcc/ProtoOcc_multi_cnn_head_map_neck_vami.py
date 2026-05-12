_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# VAMI Stage 1: route the OCC-supervised comprehensive_voxel_feature back into
# the map branch via Z-collapse + residual fusion. Compared to the weight4
# baseline (which sets only ``map_loss_weight=4.0`` on top of the cnn-map-neck
# config), this config additionally inserts a VoxelAwareMapIngest module
# before BEVSegHead so that map can finally consume the voxel-level CE+Lovasz
# supervised feature it has been blind to.
#
# Detach defaults to True so map loss does not flow back into HFM/DBE; the
# VAMI projection / fusion conv themselves remain trainable. The final fusion
# conv is zero-initialised, so at step 0 the residual is exactly 0 and the
# map path behaves identically to the weight4 baseline.

voxel_out_channels = 48
map_bev_channels = 128

model = dict(
    map_loss_weight=4.0,
    voxel_aware_map_ingest=dict(
        type='VoxelAwareMapIngest',
        voxel_in_channels=voxel_out_channels,
        map_channels=map_bev_channels,
        z_collapse_mode='avg_max_concat',
        project_channels=map_bev_channels,
        fusion_hidden_channels=map_bev_channels,
        detach_voxel_source=True,
        residual=True,
        zero_init_output=True,
    ),
)

# Bring evaluation forward enough to observe the epoch 11 stop criterion
# (plan: epoch 11 EMA Map < 43.3 -> abort).
evaluation = dict(start=10)
