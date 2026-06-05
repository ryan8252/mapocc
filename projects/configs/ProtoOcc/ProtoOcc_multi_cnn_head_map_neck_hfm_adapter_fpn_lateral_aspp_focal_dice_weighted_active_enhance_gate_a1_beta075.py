_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py']

# A1: stronger feature enhancement.
# F' = F * (1 + beta * gate), so beta=0.75 allows up to 1.75x feature scaling.
model = dict(
    bev_seg_head=dict(
        map_active_gate_beta=0.75))
