_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_bevfusion_decoder.py']

# Candidate reconstruction of MAESTRO's map-only Baseline-STL.
#
# This is not an official MAESTRO config. It is the closest runnable local
# approximation based on the paper description:
#   - single-task map training only
#   - R50/FPN + LSS camera path from the local ProtoOcc/MAESTRO family
#   - MAESTRO/BEVFusion map protocol: feature canvas +-51.2m, map eval +-50m
#   - BEVFusion-style 2D BEV decoder and 256ch BEV segmentation head
#   - per-class sigmoid focal loss, matching BEVFusion's map-head aggregation
#
# Deliberate differences from the current aligned map-only upper-bound config:
#   - no ProtoOcc PV/depth auxiliary loss in the training objective
#   - no large map_loss_weight used to compete with auxiliary losses
#   - AdamW lr=1e-4, matching the MAESTRO paper/local MAESTRO config

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

model = dict(
    # Pure map STL objective. The depth network still produces the LSS depth
    # distribution and receives gradients through map loss, but no PV/depth
    # auxiliary terms are added to the loss dict.
    train_depth=False,
    map_loss_weight=1.0,
    bev_seg_head=dict(
        map_focal_loss_mode='classwise_sum',
        map_focal_loss_class_names=map_classes,
        loss_bce=None,
        loss_dice=None,
        loss_focal=dict(
            _delete_=True,
            type='BinaryMaskFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=-1.0,
            reduction='mean',
            loss_weight=1.0)))

# Remove ProtoOcc's PV semantic/depth supervision from the data path. The
# map-only detector does not need sa_gt_depth or sa_gt_semantic when
# train_depth=False.
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
    dict(type='MultitaskFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs',
            'gt_masks_bev',
        ],
        meta_keys=['sample_idx', 'filename'])
]

data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)

# ProtoOccMapOnly freezes CM_DepthNet.class_predictor when train_depth=False,
# so DDP can keep the default path without conflicting with checkpointed blocks.
find_unused_parameters = False
