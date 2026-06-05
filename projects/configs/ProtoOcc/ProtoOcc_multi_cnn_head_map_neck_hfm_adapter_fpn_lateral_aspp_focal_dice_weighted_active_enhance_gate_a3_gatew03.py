_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py']

# A3: stronger supervised gate auxiliary loss.
# This keeps feature enhancement beta fixed and only asks the gate branch to
# match active-region GT more strongly.
model = dict(
    bev_seg_head=dict(
        map_active_gate_loss_weight=0.3))
