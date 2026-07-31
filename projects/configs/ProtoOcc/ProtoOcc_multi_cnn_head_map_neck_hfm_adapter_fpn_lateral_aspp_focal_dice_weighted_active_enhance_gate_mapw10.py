_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py']

# Constant high map-loss reweighting to probe whether the useful range extends
# above the A2 schedule endpoint.
model = dict(map_loss_weight=10.0)
