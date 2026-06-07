_base_ = [
    './ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned.py'
]

# Active-gate aligned MTL ablation with stronger map weighting.
#
# The previous map_loss_weight=4 full run had map loss around 14.3% of total.
# Doubling to map_loss_weight=8 targets roughly 25% map loss while preserving
# the same architecture and BEVFusion-aligned map protocol.
model = dict(
    map_loss_weight=8.0)

