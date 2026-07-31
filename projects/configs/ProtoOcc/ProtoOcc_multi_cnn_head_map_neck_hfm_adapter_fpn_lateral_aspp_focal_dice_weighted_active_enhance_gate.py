_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted.py']

# Map-active feature enhancement gate.
#
# This keeps the current strongest map feature path unchanged, then adds a
# supervised spatial gate on the decoded BEVSegHead feature before the final
# 1x1 map predictor. GT is used only as the gate loss target during training;
# inference predicts the gate from map features.
#
# Class order:
# [drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider].
model = dict(
    bev_seg_head=dict(
        use_map_active_enhance_gate=True,
        map_active_gate_channels=2,
        map_active_gate_class_groups=[
            [0, 2, 4],  # area/background regions
            [1, 3, 5],  # thin/overlay regions
        ],
        map_active_gate_dilations=[0, 2],
        map_active_gate_beta=0.5,
        map_active_gate_loss_weight=0.2,
        map_active_gate_use_focal=True,
        map_active_gate_use_dice=True,
        map_active_gate_init_bias=-4.0))
