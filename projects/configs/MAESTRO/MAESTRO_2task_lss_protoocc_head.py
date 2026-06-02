_base_ = ['./MAESTRO_2task_lss_occformer.py']

# MAESTRO 2-task front-end with ProtoOcc occupancy head.
# Map and occupancy supervision both stay on the same +/-40m, 0.4m grid.
point_cloud_range = [-40.0, -40.0, -1, 40.0, 40.0, 5.4]

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

grid_size = [200, 200, 16]
numC_Trans = 80
depth_categories = 88
num_class = 18
voxel_out_channels = 48
maestro_channels = voxel_out_channels

model = dict(
    _delete_=True,
    type='ProtoOccMAESTRO2Task',
    pc_range=point_cloud_range,
    grid_size=grid_size,
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
    dual_branch_encoder=dict(
        type='Dual_Branch_Encoder',
        z_size=[4, 8, 16],
        vox_feat1=numC_Trans,
        vox_feat2=voxel_out_channels // 2,
        vox_feat3=voxel_out_channels * 4,
        vox_feat4=voxel_out_channels * 8,
        voxel_out_channels=voxel_out_channels,
        down_sample_for_3d_pooling=[
            numC_Trans * grid_size[2], numC_Trans * 2
        ],
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
            input_feature_index=(0, 1, 2))),
    maestro_cpg=dict(
        type='MAESTROClasswisePrototypeGenerator',
        in_channels=maestro_channels,
        num_classes=num_class,
        hidden_channels=maestro_channels,
        foreground_classes=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        background_classes=(11, 12, 13, 14, 15, 16),
        use_hard_masks=True,
        ignore_index=255),
    maestro_map_tsfg=dict(
        type='MAESTROTaskSpecificFeatureGenerator',
        in_channels=maestro_channels,
        out_channels=maestro_channels,
        prototype_channels=maestro_channels,
        num_prototypes=6,
        task='map',
        voxel_z=grid_size[2],
        hidden_channels=maestro_channels * 2,
        with_cp=True,
        loss_name='loss_maestro_map_supp'),
    maestro_occ_tsfg=dict(
        type='MAESTROTaskSpecificFeatureGenerator',
        in_channels=maestro_channels,
        out_channels=maestro_channels,
        prototype_channels=maestro_channels,
        num_prototypes=17,
        task='occ',
        voxel_z=grid_size[2],
        hidden_channels=maestro_channels * 2,
        with_cp=True,
        loss_name='loss_maestro_occ_supp'),
    maestro_spa=dict(
        type='MAESTROOnewayScenePrototypeAggregator',
        prototype_channels=maestro_channels,
        map_feature_channels=maestro_channels,
        num_map_classes=len(map_classes),
        detach_source=True,
        aggregation_weight=1.0,
        semantic_aggregation_rules=(
            (0, (0, 1, 3, 5)),
            (1, (4,)),
            (2, (2,)),
        )),
    cnn3d_decoder=dict(
        type='cnn3d_decoder',
        in_dim=maestro_channels,
        out_dim=32,
        use_mask=True,
        num_classes=num_class,
        class_wise=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0),
        loss_weight=10.0),
    prototype_query_decoder=dict(
        type='Prototype_Query_Decoder_nuScenes',
        with_cp=True,
        feat_channels=maestro_channels,
        out_channels=maestro_channels,
        num_queries=num_class,
        num_occupancy_classes=num_class,
        prototpye_EMA_weight=0.01,
        RPL_Groups=3,
        RPL_label_noise_ratio=0.0,
        RPL_mask_noise_scale=0.2,
        shift_noise=[10, 10, 3],
        mask_noise_scale=0.1,
        mask_size=grid_size,
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
                            embed_dims=maestro_channels,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=maestro_channels,
                        feedforward_channels=maestro_channels * 4,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    batch_first=True,
                    with_cp=True,
                    operation_order=['self_attn', 'norm']))),
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

work_dir = './work_dirs/MAESTRO_2task_lss_protoocc_head'
