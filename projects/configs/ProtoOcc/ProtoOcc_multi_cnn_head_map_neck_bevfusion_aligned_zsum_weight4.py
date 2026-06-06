_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_weight4.py']

# BEVFusion-aligned MTL ablation: use a BEVFusion-like Z-collapsed map input
# while keeping the occupancy path unchanged.
#
# OCC path:
#   LSS voxel x [B, 80, 16, 256, 256] -> original Dual_Branch_Encoder
#
# Map path only:
#   LSS voxel x [B, 80, 16, 256, 256]
#     -> sum over Z [B, 80, 256, 256]
#     -> 1x1 projection [B, 160, 256, 256]
#     -> existing CustomBEVBackbone + map_bev_encoder_neck
#     -> aligned map feature / BEVSegHead
#
# This tests whether a BEVFusion-style height-collapsed map branch is enough
# under the same +-51.2m shared feature / +-50m map evaluation setup.
model = dict(
    dual_branch_encoder=dict(
        use_map_z_aware_compression=True,
        map_z_compression_type='sum',
        map_z_compression_detach=False))
