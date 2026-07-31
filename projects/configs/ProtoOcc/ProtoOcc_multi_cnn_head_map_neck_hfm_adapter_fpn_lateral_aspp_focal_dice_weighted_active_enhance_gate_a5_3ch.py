_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py']

# A5: split the current area gate into broad-area and carpark-specific gates.
#
# Class order:
# [drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider].
# Groups:
#   0: broad area: drivable_area + walkway
#   1: carpark_area
#   2: thin / overlay: ped_crossing + stop_line + divider
model = dict(
    bev_seg_head=dict(
        map_active_gate_channels=3,
        map_active_gate_class_groups=[
            [0, 2],
            [4],
            [1, 3, 5],
        ],
        map_active_gate_dilations=[0, 1, 2]))
