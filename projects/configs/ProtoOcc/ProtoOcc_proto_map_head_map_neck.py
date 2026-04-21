_base_ = ['./ProtoOcc_proto_map_head.py']

# First-stage map-specific BEV neck experiment.
# Keep the occupancy branch at 48 channels, and branch a 128-channel map
# feature from the shared multi-scale BEV representations.
numC_Trans = 80
map_bev_channels = 128

model = dict(
    map_loss_weight=1.0,
    dual_branch_encoder=dict(
        return_bev_feature=True,
        return_map_feature=True,
        detach_map_feature=False,
        map_bev_encoder_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
            catconv_in_channels2=numC_Trans * 2 + map_bev_channels * 2,
            out_channels=map_bev_channels,
            input_feature_index=(0, 1, 2),
            with_cp=True)),
    proto_map_head=dict(
        in_channels=map_bev_channels,
        hidden_channels=map_bev_channels))

# 2026-04-22 00:41:11,802 - mmdet - INFO - Epoch [1][200/7033]    
# lr: 1.990e-04, eta: 2 days, 4:44:02, time: 1.076, 
# data_time: 0.009, memory: 15569, loss_cls: 1.8846, 
# loss_mask: 0.8636, loss_dice: 3.2453, loss_cls_RPL: 5.6713, 
# loss_mask_RPL: 2.5823, loss_dice_RPL: 9.7515, 
# loss_CE_prototype: 6.1128, lovasz_softmax_loss_prototype: 8.7014, 
# loss_segmentation: 1.4631, loss_depth: 8.9559, 
# loss_map_coarse_bce: 0.7199, loss_map_coarse_dice: 0.4426, 
# loss_map_focal: 0.1034, loss_map_dice: 0.8854, 
# loss: 51.3831, grad_norm: 29.3275