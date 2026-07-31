_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_focal_weight128.py']

# Small depth-range ablation for the BEVFusion-aligned focal weight128 baseline.
#
# This keeps the map protocol and focal loss identical to dbound45, and changes
# only the LSS lift depth range:
#   dbound45: [1.0, 45.0, 0.5] -> 88 bins
#   dbound50: [1.0, 50.0, 0.5] -> 98 bins
#
# The regular 88-bin load_from checkpoint cannot directly initialize the final
# depth prediction layer after this change. The local smoke runner creates a
# filtered warm-start checkpoint that drops depth_net.depth_conv.4.{weight,bias}
# and passes it through cfg-options. Keep this config's default load_from=None so
# direct runs do not crash on shape mismatch.
depth_categories = 98

grid_config_3dpool = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 50.0, 0.5],
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 50.0, 0.5],
}

model = dict(
    depth_net=dict(
        grid_config=grid_config,
        depth_channels=depth_categories,
    ),
    img_view_transformer=dict(
        grid_config=grid_config_3dpool,
    ),
)

load_from = None
