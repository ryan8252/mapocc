_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Paper-parity benchmark for MAESTRO / BEVFusion map evaluation, depth-aligned.
#
# This is the dbound=60m variant of ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned.
# The only change vs that config is the LSS depth range: BEVFusion's camera-only
# seg model (configs/nuscenes/seg/camera-bev256d2.yaml) lifts to dbound=[1.0, 60.0, 0.5]
# on the same [-51.2, 51.2] / 0.4m (256x256) canvas. The original aligned config kept
# depth at 45m, leaving the 45-50m map ring camera-blind. Here depth goes to 60m so the
# camera actually reaches the full [-50, 50] map extent, matching BEVFusion exactly.
#
# Geometry:
#   shared feature canvas: [-51.2, 51.2] x/y, 0.4m -> 256 x 256
#   occupancy supervision: [-40, 40] x/y, 0.4m -> center crop 200 x 200 x 16
#   map supervision:       [-50, 50] x/y, 0.5m -> BEVFusion 200 x 200
#   LSS depth:             [1.0, 60.0] m, 0.5m  -> 118 bins (was 45m / 88 bins)
#
# NOTE: the pretrained depthnet in `load_from` was trained with 88 depth bins; its
# final depth-prediction layer will be skipped on load (shape mismatch) and relearned,
# so the depth loss will start higher for the first few epochs. This is expected.

occ_point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
shared_feature_range = [-51.2, -51.2, -1.0, 51.2, 51.2, 5.4]
grid_size = [200, 200, 16]

# Depth bins must match between depth_net (prediction) and the view transformer
# (placement). (60.0 - 1.0) / 0.5 = 118.
depth_categories = 118

grid_config_3dpool = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 60.0, 0.5],
}

# Depth supervision (PointToMultiViewDepth / depth loss) and the depth net use this.
# Bumped to 60m so the predicted depth distribution is supervised over the full range
# the lift now covers; otherwise far-depth bins would never get a training signal.
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 60.0, 0.5],
}

map_xbound = [-50.0, 50.0, 0.5]
map_ybound = [-50.0, 50.0, 0.5]
map_feature_range = [
    map_xbound[0],
    map_ybound[0],
    map_xbound[1],
    map_ybound[1],
]
map_feature_size = [
    int((map_xbound[1] - map_xbound[0]) / map_xbound[2]),
    int((map_ybound[1] - map_ybound[0]) / map_ybound[2]),
]

model = dict(
    # Match the comparable aligned baseline (bevfusion_aligned_weight4) so this is
    # a clean A/B against the 45m run with ONLY the LSS depth range changed.
    map_loss_weight=4.0,
    pc_range=occ_point_cloud_range,
    grid_size=grid_size,
    shared_feature_range=shared_feature_range,
    occ_feature_range=occ_point_cloud_range,
    map_feature_range=map_feature_range,
    map_feature_size=map_feature_size,
    # depth_net must emit the new 118-bin distribution over the 60m range.
    depth_net=dict(
        grid_config=grid_config,
        depth_channels=depth_categories,
    ),
    img_view_transformer=dict(
        grid_config=grid_config_3dpool,
    ),
)

# The pipeline is repeated here so the BEVFusion map raster bounds are applied
# to LoadBEVSegmentation instead of silently inheriting the base [-40, 40] map.
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

dataset_type = 'NuScenesDatasetMultitask'
nusc_version = 'v1.0-trainval'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

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
    dict(type='MultitaskFormatBundle3D', class_names=class_names),
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

evaluation = dict(
    interval=1,
    start=23,
    pipeline=test_pipeline,
    metric=['miou', 'map-miou'])
