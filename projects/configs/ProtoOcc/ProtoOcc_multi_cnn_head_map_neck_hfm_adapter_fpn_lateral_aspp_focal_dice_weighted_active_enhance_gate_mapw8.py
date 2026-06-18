_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py']

# Constant map-loss reweighting baseline for checking whether A2's gain comes
# from the final map emphasis itself or from the progressive schedule.
model = dict(map_loss_weight=8.0)
