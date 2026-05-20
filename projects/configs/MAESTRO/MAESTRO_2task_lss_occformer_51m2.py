plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -1, 51.2, 51.2, 5.4]
occ_point_cloud_range = [-40.0, -40.0, -1, 40.0, 40.0, 5.4]
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

grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

grid_config_3dpool = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
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

grid_size = [256, 256, 16]
occ_grid_size = [200, 200, 16]
numC_Trans = 80
depth_categories = 88
num_class = 18
maestro_channels = 96 # 96 is an implementation hidden width, not a paper-specified MAESTRO value.
occformer_num_queries = 100
occformer_num_layers = 9
map_xbound = [-51.2, 51.2, 0.4]
map_ybound = [-51.2, 51.2, 0.4]

model = dict(
    type='MAESTRO2Task',
    pc_range=point_cloud_range,
    grid_size=grid_size,
    occ_pc_range=occ_point_cloud_range,
    occ_grid_size=occ_grid_size,
    img_bev_encoder_backbone=None,
    img_bev_encoder_neck=None,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet',
        with_cp=True,
        in_channels=512,
        context_channels=numC_Trans,
        mid_channels=512,
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_categories,
        loss_depth_weight=3,
        use_dcn=False,
        aspp_mid_channels=96),
    img_view_transformer=dict(
        type='LSSViewTransformer_depthGT',
        grid_config=grid_config_3dpool,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        downsample=16),
    maestro_cpg=dict(
        type='MAESTROClasswisePrototypeGenerator',
        in_channels=numC_Trans,
        num_classes=num_class,
        hidden_channels=maestro_channels,
        foreground_classes=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        background_classes=(11, 12, 13, 14, 15, 16),
        use_hard_masks=True,
        ignore_index=255),
    maestro_map_tsfg=dict(
        type='MAESTROTaskSpecificFeatureGenerator',
        in_channels=numC_Trans,
        out_channels=maestro_channels,
        prototype_channels=numC_Trans,
        num_prototypes=6,
        task='map',
        voxel_z=grid_size[2],
        hidden_channels=maestro_channels * 2,
        with_cp=True,
        loss_name='loss_maestro_map_supp'),
    maestro_occ_tsfg=dict(
        type='MAESTROTaskSpecificFeatureGenerator',
        in_channels=numC_Trans,
        out_channels=maestro_channels,
        prototype_channels=numC_Trans,
        num_prototypes=17,
        task='occ',
        voxel_z=grid_size[2],
        hidden_channels=maestro_channels * 2,
        with_cp=True,
        loss_name='loss_maestro_occ_supp'),
    maestro_spa=dict(
        type='MAESTROOnewayScenePrototypeAggregator',
        prototype_channels=numC_Trans,
        map_feature_channels=maestro_channels,
        num_map_classes=len(map_classes),
        detach_source=True,
        aggregation_weight=1.0,
        semantic_aggregation_rules=(
            (0, (0, 1, 3, 5)),  # driveable_surface <- road layout labels
            (1, (4,)),          # other_flat <- carpark_area
            (2, (2,)),          # sidewalk <- walkway
        )),
    occ_head=dict(
        type='MAESTROOccFormerHead',
        in_channels=maestro_channels,
        feat_channels=maestro_channels,
        out_channels=maestro_channels,
        num_queries=occformer_num_queries,
        num_occupancy_classes=num_class,
        num_transformer_feat_level=3,
        pooling_attn_mask=True,
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=maestro_channels // 3,
            normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=occformer_num_layers,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=maestro_channels,
                    num_heads=maestro_channels // 32,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=maestro_channels,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=maestro_channels * 8,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_class + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544 * 4,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner_occ',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(
                    type='CrossEntropyLossCost',
                    weight=5.0,
                    use_sigmoid=True),
                dice_cost=dict(
                    type='DiceCost',
                    weight=5.0,
                    pred_act=True,
                    eps=1.0)),
            sampler=dict(type='MaskPseudoSampler')),
        test_cfg=dict(
            semantic_on=True,
            panoptic_on=False,
            instance_on=False)),
    bev_seg_head=dict(
        type='MAESTROBEVSegHead',
        in_channels=maestro_channels,
        hidden_channels=maestro_channels * 2,
        num_classes=len(map_classes),
        num_convs=2,
        with_cp=True,
        loss_focal=dict(
            gamma=2.0,
            alpha=-1.0,
            reduction='mean',
            loss_weight=1.0)),
    map_loss_weight=1.0,
    free_label=17)

dataset_type = 'NuScenesDatasetMultitask'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    # Occ3D GT is loaded on the 200x200 +/-40m grid after BDA. The 51.2m
    # variant keeps rotation/scale identity so center-cropping the 256x256
    # shared feature to 200x200 remains aligned under flip_dx/flip_dy.
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
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

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
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
    nusc_version='v1.0-trainval')

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl',
    test_mode=True)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[29])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL')
]

checkpoint_config = dict(interval=3, max_keep_ckpts=24)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'

load_from = 'ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth'
resume_from = None
evaluation = dict(
    interval=1,
    start=23,
    pipeline=test_pipeline,
    metric=['miou', 'map-miou'])
work_dir = './work_dirs/MAESTRO_2task_lss_occformer_51m2'
