_base_ = ['./ProtoOcc_1key.py']

# Python-level variables from _base_ are NOT inherited by mmcv config system.
# Re-declare everything used in this file's Python expressions.
voxel_out_channels = 48

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

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

learning_map = {
    1: 0,   5: 0,   7: 0,   8: 0,
    10: 0,  11: 0,  13: 0,  19: 0,
    20: 0,  0: 0,   29: 0,  31: 0,
    9: 1,   14: 2,  15: 3,  16: 3,
    17: 4,  18: 5,  21: 6,  2: 7,
    3: 7,   4: 7,   6: 7,   12: 8,
    22: 9,  23: 10, 24: 11, 25: 12,
    26: 13, 27: 14, 28: 15, 30: 16,
}

# BEVFusion-style semantic map classes.
map_classes = [
    'drivable_area',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
    'divider',
]

# Keep ProtoOcc's native BEV range so the future map head can supervise
# directly on the occupancy/BEV feature lattice without extra reprojection.
map_xbound = [-40.0, 40.0, 0.4]
map_ybound = [-40.0, 40.0, 0.4]

dataset_type = 'NuScenesDatasetMultitask'

model = dict(
    dual_branch_encoder=dict(
        return_bev_feature=True,
    ),
    bev_seg_head=dict(
        type='BEVSegHead',
        in_channels=voxel_out_channels,
        hidden_channels=voxel_out_channels * 2,
        num_classes=len(map_classes),
        num_convs=2,
        with_cp=True,
        loss_bce=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0,
        ),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            loss_weight=1.0,
        ),
    ),
)

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
    dict(type='LoadOccGTFromFile', ignore_nonvisible=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='MultitaskFormatBundle3D',
        class_names=class_names),
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
    dict(
        type='Collect3D',
        keys=[
            'img_inputs',
            'gt_depth',
            'voxel_semantics',
            'mask_lidar',
            'mask_camera',
            'sa_gt_depth',
            'sa_gt_semantic',
            'gt_masks_bev',
        ],
        meta_keys=['sample_idx', 'filename'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadBEVSegmentation',
        dataset_root=data_root,
        xbound=map_xbound,
        ybound=map_ybound,
        classes=map_classes),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='MultitaskFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=['points', 'img_inputs', 'gt_masks_bev'])
        ])
]

data = dict(
    train=dict(
        type=dataset_type,
        map_classes=map_classes,
        nusc_version='v1.0-trainval',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        map_classes=map_classes,
        nusc_version='v1.0-trainval',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        map_classes=map_classes,
        nusc_version='v1.0-trainval',
        pipeline=test_pipeline),
)

evaluation = dict(interval=1, start=23, pipeline=test_pipeline, metric=['miou', 'map-miou'])
