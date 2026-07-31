_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter.py']

# Clean adapter-first additive Map-HFM ablation.
#
# This follows the same component set as `..._hfm_adapter.py`, but changes the
# map-branch order to the more diagram-friendly path:
#
#   multi_scale_bev -> PerScaleMapResidualAdapter -> additive Map-HFM
#   -> 128ch map neck -> BEVSegHead
#
# In additive Map-HFM, the collapsed voxel feature is projected by the HFM layer
# and added to the adapter-produced map BEV feature. This is closer to the
# original ProtoOcc HFM add-fusion style than the previous concat-conditioned
# zero-init residual.
model = dict(
    dual_branch_encoder=dict(
        map_hfm_order='adapter_then_hfm',
        map_hfm_fusion_mode='voxel_add',
        map_hfm_voxel_residual_scale=1.0))
