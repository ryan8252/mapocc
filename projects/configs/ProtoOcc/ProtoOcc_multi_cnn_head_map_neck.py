_base_ = ['./ProtoOcc_multi_cnn_head.py']

# CNN-head ablation for the map-specific BEV neck.
# This keeps the same occupancy branch and replaces only the map feature input:
# shared multi-scale BEV -> 128-channel map-specific neck -> BEVSegHead.
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
    bev_seg_head=dict(
        in_channels=map_bev_channels,
        hidden_channels=map_bev_channels))

# 2026-04-22 02:03:16,149 - mmdet - INFO - 
# Epoch [1][450/7033]	lr: 2.000e-04, eta: 2 days, 3:11:39, 
# time: 1.104, data_time: 0.010, memory: 15594, 
# loss_cls: 0.6038, loss_mask: 0.7212, loss_dice: 2.9029, 
# loss_cls_RPL: 1.8059, loss_mask_RPL: 2.1579, 
# loss_dice_RPL: 8.7082, loss_CE_prototype: 4.5556, 
# lovasz_softmax_loss_prototype: 7.9040, 
# loss_segmentation: 1.1129, loss_depth: 8.4076, 
# loss_map_bce: 0.9784, loss_map_dice: 0.8697, 
# loss: 40.7282, grad_norm: 30.1793

# epoch_3 
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 7.19
# ===> barrier - IoU = 39.01
# ===> bicycle - IoU = 20.3
# ===> bus - IoU = 38.48
# ===> car - IoU = 46.28
# ===> construction_vehicle - IoU = 16.25
# ===> motorcycle - IoU = 21.46
# ===> pedestrian - IoU = 23.87
# ===> traffic_cone - IoU = 24.6
# ===> trailer - IoU = 23.95
# ===> truck - IoU = 31.86
# ===> driveable_surface - IoU = 78.2
# ===> other_flat - IoU = 40.91
# ===> sidewalk - IoU = 47.83
# ===> terrain - IoU = 50.11
# ===> manmade - IoU = 35.37
# ===> vegetation - IoU = 32.71
# ===> mIoU of 6019 samples: 34.02

# drivable_area    69.87
# ped_crossing     16.73
# walkway          35.85
# stop_line        9.78
# carpark_area     20.91
# divider          19.78
# mean             28.82