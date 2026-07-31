_base_ = ['./ProtoOcc_maestro_map_stl_candidate.py']

# BEVFusion/checkpoint-faithful variant of the local MAESTRO map-STL candidate.
#
# The original candidate inherits the local 45m depth discretization:
#   depth=[1.0, 45.0, 0.5] -> 88 bins
# but the warm-start checkpoint used in these runs has a 118-bin
# depth_net.depth_conv.4. This variant switches both the depth predictor and
# LSS lift grid to BEVFusion's 60m discretization:
#   depth=[1.0, 60.0, 0.5] -> 118 bins
#
# Everything else intentionally stays the same as the map-STL candidate:
#   - map-only objective
#   - train_depth=False
#   - map_loss_weight=1.0
#   - BEVFusion-style 2D map decoder/head
#   - BEVFusion map supervision grid [-50, 50] / 0.5m

depth_categories = 118

# Depth-net supervision/config range. train_depth=False keeps this out of the
# loss path, but CM_DepthNet still stores the range and emits 118 depth bins.
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 60.0, 0.5],
}

# BEVFusion-style LSS lift/pool grid:
#   x/y [-51.2, 51.2] at 0.4m -> 256 x 256
#   z [-10, 10] with a 20m step -> one z bin, collapsed to 2D
#   depth [1, 60] at 0.5m -> 118 bins
grid_config_3dpool = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 60.0, 0.5],
}

model = dict(
    depth_net=dict(
        grid_config=grid_config,
        depth_channels=depth_categories,
    ),
    img_view_transformer=dict(
        grid_config=grid_config_3dpool,
        collapse_z=True,
    ),
)
