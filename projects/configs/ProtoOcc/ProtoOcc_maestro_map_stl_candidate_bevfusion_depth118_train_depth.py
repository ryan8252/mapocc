_base_ = ['./ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118.py']

# MAESTRO-paper-style map-STL candidate with explicit depth supervision.
#
# This keeps the BEVFusion/checkpoint-faithful 60m depth discretization from
# ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118.py, but turns the local
# ProtoOccMapOnly depth/PV auxiliary loss back on:
#   train_depth=True -> add depth_net.get_PV_loss(...)
#
# The train pipeline therefore has to collect sa_gt_depth and sa_gt_semantic.
# Without those targets, simply setting train_depth=True would fail at loss
# computation time.

map_classes = [
    'drivable_area',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
    'divider',
]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

map_xbound = [-50.0, 50.0, 0.5]
map_ybound = [-50.0, 50.0, 0.5]

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 60.0, 0.5],
}

learning_map = {
    1: 0, 5: 0, 7: 0, 8: 0,
    10: 0, 11: 0, 13: 0, 19: 0,
    20: 0, 0: 0, 29: 0, 31: 0,
    9: 1, 14: 2, 15: 3, 16: 3,
    17: 4, 18: 5, 21: 6, 2: 7,
    3: 7, 4: 7, 6: 7, 12: 8,
    22: 9, 23: 10, 24: 11, 25: 12,
    26: 13, 27: 14, 28: 15, 30: 16,
}

model = dict(train_depth=True)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='LoadBEVSegmentation',
        dataset_root=data_root,
        xbound=map_xbound,
        ybound=map_ybound,
        classes=map_classes),
    dict(
        type='LoadLidarsegFromFile',
        grid_config=grid_config,
        occupancy_root='./data/nuscenes/pc_panoptic/',
        learning_map=learning_map,
        label_from='panoptic',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='MultitaskFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs',
            'sa_gt_depth',
            'sa_gt_semantic',
            'gt_masks_bev',
        ],
        meta_keys=['sample_idx', 'filename'])
]

data = dict(train=dict(pipeline=train_pipeline))
