_base_ = ['./ProtoOcc_proto_map_head_map_neck.py']

# Higher-capacity map-specific BEV neck. Run after the 128-channel version
# verifies that the added branch fits memory and improves map segmentation.
numC_Trans = 80
map_bev_channels = 256

model = dict(
    dual_branch_encoder=dict(
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


# 2026-04-22 00:34:57,365 - mmdet - INFO - Epoch [1][100/7033]    
# lr: 9.910e-05, eta: 2 days, 14:10:15, time: 1.181, 
# data_time: 0.009, memory: 17116, loss_cls: 4.8499, 
# loss_mask: 1.2418, loss_dice: 3.5501, loss_cls_RPL: 14.5811, 
# loss_mask_RPL: 3.7315, loss_dice_RPL: 10.6538, 
# loss_CE_prototype: 15.9096, lovasz_softmax_loss_prototype: 9.2859, 
# loss_segmentation: 2.5730, loss_depth: 12.3225, loss_map_coarse_bce: 0.9892, 
# loss_map_coarse_dice: 0.4438, loss_map_focal: 0.1160, loss_map_dice: 0.8914, 
# loss: 81.1395, grad_norm: 61.4598

# ===> per class IoU of 6019 samples:
# ===> others - IoU = 12.05
# ===> barrier - IoU = 48.24
# ===> bicycle - IoU = 24.88
# ===> bus - IoU = 44.2
# ===> car - IoU = 51.84
# ===> construction_vehicle - IoU = 23.48
# ===> motorcycle - IoU = 26.4
# ===> pedestrian - IoU = 27.85
# ===> traffic_cone - IoU = 28.09
# ===> trailer - IoU = 32.98
# ===> truck - IoU = 37.2
# ===> driveable_surface - IoU = 82.18
# ===> other_flat - IoU = 46.45
# ===> sidewalk - IoU = 53.46
# ===> terrain - IoU = 56.63
# ===> manmade - IoU = 42.97
# ===> vegetation - IoU = 37.06
# ===> mIoU of 6019 samples: 39.76

# Class	        128ch EMA	 256ch EMA	差異
# drivable_area	 75.43	      74.52	   -0.91
# ped_crossing	 30.50	      32.10	   +1.60
# walkway	     44.55	      45.03	   +0.48
# stop_line	     20.54	      21.30	   +0.76
# carpark_area	 35.47	      33.51	   -1.96
# divider	     27.62	      28.05	   +0.43
# mean	         39.02	      39.09	   +0.07