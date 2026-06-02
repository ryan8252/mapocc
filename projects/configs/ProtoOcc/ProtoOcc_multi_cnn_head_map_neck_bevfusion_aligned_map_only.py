_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_map_only.py']

# Map-only UPPER BOUND on the BEVFusion-aligned grid (map eval at +-50m / 0.5m).
#
# This is the comparable ceiling for the aligned-grid experiments. The old
# map-only 48.34 was measured at +-40m / 0.4m and is NOT comparable to
# BEVFusion 47.10 / MAESTRO 51.30. Geometry matches the MTL aligned config:
#   shared feature: +-51.2m / 0.4m -> 256 x 256
#   map supervision: +-50m / 0.5m  -> BEVFusion 200 x 200 (grid_sample aligned)
# No occupancy branch -> pure map ceiling.

dataset_type = 'NuScenesDatasetMultitask'
nusc_version = 'v1.0-trainval'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
map_classes = [
    'drivable_area',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
    'divider',
]

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

# Depth / lidarseg aux grid stays at +-40m (per-point supervision, unchanged).
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

# Shared BEV feature grid: +-51.2m / 0.4m -> 256 x 256.
grid_config_3dpool = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}
shared_feature_range = [-51.2, -51.2, -1.0, 51.2, 51.2, 5.4]

# Map eval grid: BEVFusion +-50m / 0.5m.
map_xbound = [-50.0, 50.0, 0.5]
map_ybound = [-50.0, 50.0, 0.5]
map_feature_range = [
    map_xbound[0], map_ybound[0], map_xbound[1], map_ybound[1]
]
map_feature_size = [
    int((map_xbound[1] - map_xbound[0]) / map_xbound[2]),
    int((map_ybound[1] - map_ybound[0]) / map_ybound[2]),
]

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

# Resample the map feature from the +-51.2m shared grid onto the +-50m / 0.5m
# map grid (grid_sample), and feed the view transformer the +-51.2m pool grid.
model = dict(
    shared_feature_range=shared_feature_range,
    map_feature_range=map_feature_range,
    map_feature_size=map_feature_size,
    img_view_transformer=dict(grid_config=grid_config_3dpool),
)

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# Re-declare the pipeline so the BEVFusion +-50m / 0.5m raster bounds are
# applied to LoadBEVSegmentation instead of silently inheriting the base
# +-40m / 0.4m map.
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
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='MultitaskFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs', 'gt_masks_bev'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    map_classes=map_classes,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet',
    nusc_version=nusc_version,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(test_data_config),
    test=dict(test_data_config))

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)
