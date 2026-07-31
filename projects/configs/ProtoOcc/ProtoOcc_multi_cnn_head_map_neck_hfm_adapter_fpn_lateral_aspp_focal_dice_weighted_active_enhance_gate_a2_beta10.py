_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py']

# A2: aggressive feature enhancement upper-bound smoke.
# F' = F * (1 + beta * gate), so beta=1.0 allows up to 2.0x feature scaling.
model = dict(
    bev_seg_head=dict(
        map_active_gate_beta=1.0))
