_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 2 ablation: replace the map branch's pre-backbone x0 with a
# map-specific Z-aware compression path.
#
# Route:
#   voxel x [B, 80, 16, H, W]
#     -> zero-init lightweight Conv3d residual refine
#     -> cat-Z
#     -> existing 1x1 1280->160 projection
#     -> shared CustomBEVBackbone weights for the map branch only
#
# The occupancy path still uses the original cat-Z + 1x1 pooled_x and original
# `multi_scale_bev`; only the map branch consumes this map-specific x0.
model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        use_map_z_aware_compression=True,
        map_z_compression_type='conv3d',
        map_z_compression_detach=False))
