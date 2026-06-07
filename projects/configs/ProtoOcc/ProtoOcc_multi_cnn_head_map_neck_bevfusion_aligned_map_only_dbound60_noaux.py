_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_focal_no_depth_no_seg.py']

# Diagnostic 5: depth discretization/range only.
#
# Keep the current ProtoOcc map-only architecture, but change dbound from
# [1, 45, 0.5] to [1, 60, 0.5]. This gives 118 depth bins and matches the
# available BEVDepth-style checkpoint's depth head shape. No OCC, no PV depth
# loss, no PV segmentation loss.

depth_categories = 118

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 60.0, 0.5],
}

grid_config_3dpool = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 60.0, 0.5],
}

model = dict(
    depth_net=dict(
        grid_config=grid_config,
        depth_channels=depth_categories),
    img_view_transformer=dict(grid_config=grid_config_3dpool))
