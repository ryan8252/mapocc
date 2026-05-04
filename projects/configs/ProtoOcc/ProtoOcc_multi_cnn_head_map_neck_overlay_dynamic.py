_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# V3: MTL-aware batch-wise overlay-positive balancing.
#
# The ref ratios below are bootstrap values computed with:
#   conda run -n mapocc python tools/analysis_tools/compute_map_pos_ratio.py \
#       projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py \
#       --sample-count 400 --progress-interval 400 --progress-bar
#
# Before paper-level full training, recompute without --sample-count and update
# this list with the full train split ratios.
map_classes = [
    'drivable_area',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
    'divider',
]
overlay_class_indices = [1, 3, 5]
dynamic_overlay_ref_pos_ratio = [
    0.0166845,     # ped_crossing
    0.0213629375,  # stop_line
    0.033702125,   # divider
]

model = dict(
    map_loss_weight=4.0,
    bev_seg_head=dict(
        map_loss_balance_mode='overlay_dynamic',
        overlay_class_indices=overlay_class_indices,
        dynamic_overlay_ref_pos_ratio=dynamic_overlay_ref_pos_ratio,
        dynamic_overlay_gamma=0.5,
        dynamic_overlay_min_weight=1.0,
        dynamic_overlay_max_weight=5.0,
        dynamic_overlay_eps=1e-6,
        map_balance_debug=True,
        map_balance_debug_interval=50,
        map_balance_class_names=map_classes))
