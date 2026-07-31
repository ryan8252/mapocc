_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py']

# A4: wider thin-class active target.
# Current 2-channel gate uses [area, thin] groups. Increasing the thin group
# dilation from 2 to 3 changes the supervised thin gate target from 5x5 to 7x7.
model = dict(
    bev_seg_head=dict(
        map_active_gate_dilations=[0, 3]))
