_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 3 ablation: map-specific adapter before CustomBEVBackbone.
#
# Route:
#   shared pooled_x [B, 160, 200, 200]
#     -> zero-init res_bottleneck adapter
#     -> shared CustomBEVBackbone weights for the map branch only
#     -> existing 128ch map neck
#
# This differs from `ProtoOcc_multi_cnn_head_map_neck_adapter.py`, which adapts
# the post-backbone BEV pyramid `[160, 320, 640]` before the map neck.
model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        use_map_pre_backbone_adapter=True,
        map_pre_backbone_adapter_type='res_bottleneck',
        map_pre_backbone_adapter_hidden=64,
        map_pre_backbone_adapter_detach=False))
