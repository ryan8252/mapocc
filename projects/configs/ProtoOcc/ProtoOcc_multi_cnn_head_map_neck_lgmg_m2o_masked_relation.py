_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Stage 1: Map-to-Occ only.
# Uses the strong weight4 map-neck baseline and injects predicted map layout
# priors only into selected OCC background/layout semantic queries.
#
# Map class order:
#   0 drivable_area
#   1 ped_crossing
#   2 walkway
#   3 stop_line
#   4 carpark_area
#   5 divider
#
# OCC class indices follow the nuScenes occupancy order:
#   11 driveable_surface
#   13 sidewalk
#   14 terrain
#   15 manmade
layout_target_occ_indices = [11, 13, 14, 15]
map_to_occ_relation_mask = [
    # driveable_surface, sidewalk, terrain, manmade
    [1, 0, 0, 0],  # drivable_area
    [1, 1, 0, 0],  # ped_crossing
    [0, 1, 1, 1],  # walkway
    [1, 0, 0, 0],  # stop_line
    [1, 1, 0, 0],  # carpark_area
    [1, 0, 0, 0],  # divider
]

model = dict(
    map_loss_weight=4.0,
    map_to_occ_adapter=dict(
        type='MapToOccLayoutAdapter',
        in_channels=128,
        query_channels=48,
        num_map_classes=6,
        num_occ_queries=18,
        target_occ_indices=layout_target_occ_indices,
        relation_mask=map_to_occ_relation_mask,
        map_prior_detach=True,
        gate_mode='source_only',
        eps=1e-4,
        init_alpha=0.0,
        relation_init_std=0.02))
