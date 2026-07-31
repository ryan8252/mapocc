_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 6 ablation: split map prediction into area-like and line-like
# decoders, then write logits back to the original map class order:
# [drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider].
model = dict(
    map_loss_weight=4.0,
    bev_seg_head=dict(
        use_dual_area_line_head=True,
        area_class_indices=[0, 1, 2, 4],
        line_class_indices=[3, 5],
        line_head_use_detail=False))
