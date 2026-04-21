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