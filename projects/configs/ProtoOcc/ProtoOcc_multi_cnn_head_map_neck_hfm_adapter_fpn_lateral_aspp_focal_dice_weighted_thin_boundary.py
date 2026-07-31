_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted.py']

# Section 10 ablation: keep the final map loss on the original GT, but add a
# thin-class auxiliary target where sparse line-like masks are dilated by two
# pixels. This teaches the model to find the nearby thin/boundary region while
# the inherited focal_dice loss still supervises final precision.
#
# Class order from LoadBEVSegmentation:
# [drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider]
model = dict(
    bev_seg_head=dict(
        use_thin_boundary_aux_loss=True,
        thin_boundary_class_indices=[1, 3, 5],
        thin_boundary_dilation=2,
        thin_boundary_loss_weight=0.2,
        thin_boundary_use_focal=True,
        thin_boundary_use_dice=True))
