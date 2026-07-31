_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Combined Sections 1+2+3 smoke/config check only.
#
# Keep the individual ablations separate for real comparison:
#   1. `..._highres_gated.py`
#   2. `..._zaware_compression_conv3d.py`
#   3. `..._pre_adapter.py`
#
# This config exists only to verify that the three new switches can compose.
model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        use_map_z_aware_compression=True,
        map_z_compression_type='conv3d',
        map_z_compression_detach=False,
        use_map_pre_backbone_adapter=True,
        map_pre_backbone_adapter_type='res_bottleneck',
        map_pre_backbone_adapter_hidden=64,
        map_pre_backbone_adapter_detach=False,
        map_highres_skip=True,
        map_highres_fusion='gated',
        map_highres_detach=True))
