_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 2 stronger ablation: K=4 height-attention map-specific Z compression.
#
# Route:
#   voxel x [B, 80, 16, H, W]
#     -> height attention logits [B, 4, 16, H, W]
#     -> softmax over Z
#     -> four weighted sums [B, 80, H, W]
#     -> concat [B, 320, H, W]
#     -> 1x1 projection to [B, 160, H, W]
#     -> shared CustomBEVBackbone weights for the map branch only
model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        use_map_z_aware_compression=True,
        map_z_compression_type='height_attention',
        map_height_attention_groups=4,
        map_z_compression_detach=False))
