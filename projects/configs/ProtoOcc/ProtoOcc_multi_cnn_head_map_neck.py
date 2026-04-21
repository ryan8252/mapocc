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
