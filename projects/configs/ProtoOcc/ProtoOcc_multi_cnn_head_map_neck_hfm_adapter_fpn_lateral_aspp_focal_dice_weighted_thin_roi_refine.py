_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted.py']

# Section 12 ablation: prediction-derived ROI residual refinement for thin
# classes. The ROI is built from detached thin-class logits, with a top-k
# fallback so empty early predictions still receive training signal. The final
# residual logits are zero-init, preserving the inherited baseline at step 0.
#
# Class order from LoadBEVSegmentation:
# [drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider]
model = dict(
    bev_seg_head=dict(
        use_thin_roi_refinement=True,
        thin_roi_class_indices=[1, 3, 5],
        thin_roi_threshold=0.35,
        thin_roi_min_pixels=64,
        thin_roi_dilation=2,
        thin_roi_hidden_channels=64))
