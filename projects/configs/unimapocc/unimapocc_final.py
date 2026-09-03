_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']

# =============================================================================
# UniMapOcc final standalone model config
# =============================================================================
# This file intentionally does not inherit any ProtoOcc experiment config.
# Only the standard MMDetection3D dataset/runtime bases above are retained.
#
# Local implementation index (relative to the ProtoOcc repository root):
#   Multi-task detector / head wiring:
#     projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py
#   Shared OCC/Map encoder, Map-HFM, and map residual adapter:
#     projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py
#   128-channel map neck, FPN lateral projection, and residual ASPP:
#     projects/mmdet3d_plugin/models/necks/lss_fpn.py
#   OCC CNN head:
#     projects/mmdet3d_plugin/models/dense_heads/cnn3d_decoder.py
#   Occupancy prototype query decoder:
#     projects/mmdet3d_plugin/models/OccHead/Prototype_Query_Decoder_nuScenes.py
#   Map CNN head, weighted focal-Dice, and active enhancement gate:
#     projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py
#   Local binary mask focal loss:
#     projects/mmdet3d_plugin/models/losses/focal_loss.py
# =============================================================================

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-40.0, -40.0, -1, 40.0, 40.0, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# BEVFusion-style semantic map classes.
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

    # Augmentation
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

grid_config_3dpool = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

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

grid_size = [200, 200, 16]
numC_Trans = 80
depth_categories = 88
num_class = 18
voxel_out_channels = 48
map_bev_channels = 128

# Keep ProtoOcc's native BEV range so the future map head can supervise
# directly on the occupancy/BEV feature lattice without extra reprojection.
map_xbound = [-40.0, 40.0, 0.4]
map_ybound = [-40.0, 40.0, 0.4]

dataset_type = 'NuScenesDatasetMultitask'
nusc_version = 'v1.0-trainval'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# Multi-head detector:
#   OCC: cnn3d_decoder + Prototype_Query_Decoder_nuScenes
#   Map: BEVSegHead
# Implemented in:
#   projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py
model = dict(
    type='ProtoOccCnnSegHead',
    # Applied to every loss returned by BEVSegHead, including gate losses.
    map_loss_weight=4.0,  # λ_map in the paper, default = 4.0
    pc_range=point_cloud_range,
    grid_size=grid_size,
    img_bev_encoder_backbone=None,  # for avoiding error during init BEVDet
    img_bev_encoder_neck=None,  # for avoiding error during init BEVDet
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
    dual_branch_encoder=dict(
        type='Dual_Branch_Encoder',
        z_size=[4, 8, 16],
        vox_feat1=numC_Trans,
        vox_feat2=voxel_out_channels // 2,
        vox_feat3=voxel_out_channels * 4,
        vox_feat4=voxel_out_channels * 8,
        voxel_out_channels=voxel_out_channels,
        down_sample_for_3d_pooling=[numC_Trans * grid_size[2], numC_Trans * 2],
        return_bev_feature=True,
        return_map_feature=True,
        detach_map_feature=False,

        # Map-HFM / VGMR
        # Implementation:
        #   projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py
        #   class MapHFMFusionLayer and
        #   Dual_Branch_Encoder._build_map_hfm_features()
        # Three voxel sources [vox_res, vox1, vox3] are collapsed with
        # mean-Z + max-Z and injected into the three BEV pyramid levels.
        use_map_hfm=True,
        map_hfm_lower_source='vox3',
        map_hfm_order='hfm_then_adapter',
        map_hfm_fusion_mode='residual_concat',
        map_hfm_voxel_residual_scale=1.0,

        # Per-scale map residual adapter
        # Implementation:
        #   projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py
        #   class PerScaleMapResidualAdapter
        # Per level: 1x1 -> depthwise 3x3 -> 1x1 (last 1x1 is zero-init),
        # followed by an identity residual addition.
        map_residual_adapter=dict(
            type='PerScaleMapResidualAdapter',
            in_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
            residual_scale=1.0,
            with_cp=True,
            detach_input=False),

        bev_encoder_backbone=dict(
            type='CustomBEVBackbone',
            stride=[2, 2, 2],
            numC_input=numC_Trans * 2,
            num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
        bev_encoder_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
            catconv_in_channels2=numC_Trans * 2 + voxel_out_channels * 2,
            out_channels=voxel_out_channels,
            input_feature_index=(0, 1, 2)),

        # Dedicated 128-channel Map neck
        # Implementation:
        #   projects/mmdet3d_plugin/models/necks/lss_fpn.py
        #   class Custom_FPN_LSS
        #
        # FPN lateral:
        #   [160, 320, 640] -> three independent 1x1 projections to 256ch.
        # Residual ASPP:
        #   applied to the lowest-resolution 640ch / 25x25 level before
        #   lateral projection and top-down fusion.
        map_bev_encoder_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
            catconv_in_channels2=numC_Trans * 2 + map_bev_channels * 2,
            out_channels=map_bev_channels,
            input_feature_index=(0, 1, 2),
            with_cp=True,
            use_fpn_lateral_projection=True,
            fpn_lateral_in_channels=[
                numC_Trans * 2,
                numC_Trans * 4,
                numC_Trans * 8,
            ],
            fpn_projection_channels=256,
            use_fpn_global_context=True,
            fpn_global_context_type='aspp')),

    # OCC CNN head
    # Implementation:
    #   projects/mmdet3d_plugin/models/dense_heads/cnn3d_decoder.py
    cnn3d_decoder=dict(
        type='cnn3d_decoder',
        in_dim=voxel_out_channels,
        out_dim=32,
        use_mask=True,
        num_classes=18,
        class_wise=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0),
        loss_weight=10.),
    # Occupancy prototype query decoder
    # Implementation:
    #   projects/mmdet3d_plugin/models/OccHead/
    #   Prototype_Query_Decoder_nuScenes.py
    prototype_query_decoder=dict(
        type='Prototype_Query_Decoder_nuScenes',
        with_cp=True,  # for reducing GPU memory usage
        feat_channels=voxel_out_channels,
        out_channels=voxel_out_channels,
        num_queries=18,
        num_occupancy_classes=num_class,
        prototpye_EMA_weight=0.01,
        RPL_Groups=3,
        RPL_label_noise_ratio=0.0,
        RPL_mask_noise_scale=0.2,
        shift_noise=[10, 10, 3],
        mask_noise_scale=0.1,
        mask_size=[200, 200, 16],
        query_self_attn_cfg=dict(
            type='Query_Transformer_RPL',
            decoder=dict(
                type='Query_TransformerDecoder_RPL',
                num_layers=1,
                transformerlayers=dict(
                    type='Query_Transformer_DecoderLayer_self_RPL',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=voxel_out_channels,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=voxel_out_channels,
                        feedforward_channels=voxel_out_channels * 4,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    batch_first=True,
                    with_cp=True,
                    operation_order=['self_attn', 'norm',]))),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_class + [0.1]),
        loss_mask=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=20.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0),
        train_cfg=dict(
            num_points=12544 * 3,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner',
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
    # Map CNN head
    # Implementation:
    #   projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py
    bev_seg_head=dict(
        type='BEVSegHead',
        in_channels=map_bev_channels,
        hidden_channels=map_bev_channels,
        num_classes=len(map_classes),
        num_convs=2,
        with_cp=True,

        # Retained from the original merged config for exact config parity.
        # With map_loss_type='focal_dice', BEVSegHead._resolve_map_loss_cfgs()
        # replaces these runtime loss objects with focal + Dice definitions.
        loss_bce=dict(
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
            loss_weight=1.0),

        # Static-class-weighted focal + Dice map loss
        # Loss selection, weighting, and aggregation:
        #   projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py
        # Local BinaryMaskFocalLoss implementation:
        #   projects/mmdet3d_plugin/models/losses/focal_loss.py
        # DiceLoss itself is provided by the MMDetection loss registry.
        map_loss_type='focal_dice',
        map_dice_weight=1.0,
        map_focal_gamma=2.0,
        map_focal_alpha=0.25,
        use_map_class_weights=True,
        # Class order:
        # [drivable_area, ped_crossing, walkway,
        #  stop_line, carpark_area, divider]
        map_class_weights=[1.0, 2.0, 2.0, 2.0, 4.0, 4.0],

        # Group-aware map-active feature enhancement gate
        # Network, feature enhancement, GT target, and gate losses:
        #   projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py
        # The gate is predicted from decoded map features at both training and
        # inference. GT is used only to supervise its focal/Dice gate losses.
        use_map_active_enhance_gate=True,
        map_active_gate_channels=2,
        map_active_gate_class_groups=[
            [0, 2, 4],  # area/background regions
            [1, 3, 5],  # thin/overlay regions
        ],
        map_active_gate_dilations=[0, 2],
        map_active_gate_beta=0.5,
        map_active_gate_loss_weight=0.2,
        map_active_gate_use_focal=True,
        map_active_gate_use_dice=True,
        map_active_gate_init_bias=-4.0))

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

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[29, ])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# fp16 = dict(loss_scale='dynamic')
load_from = 'ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth'

evaluation = dict(
    interval=1,
    start=23,
    pipeline=test_pipeline,
    metric=['miou', 'map-miou'])
checkpoint_config = dict(interval=3, max_keep_ckpts=24)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
