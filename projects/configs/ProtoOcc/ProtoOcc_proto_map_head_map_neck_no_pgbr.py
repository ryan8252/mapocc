_base_ = ['./ProtoOcc_proto_map_head_map_neck.py']

# Ablation: map-specific BEV neck without prototype-grounded BEV refinement.
model = dict(
    proto_map_head=dict(
        use_bev_refinement=False))

# 2026-04-22 01:28:18,945 - mmdet - INFO - 
# Epoch [1][250/7033]	lr: 2.000e-04, eta: 2 days, 4:14:14, 
# time: 1.080, data_time: 0.009, memory: 15420, 
# loss_cls: 1.2073, loss_mask: 0.8232, loss_dice: 3.1040, 
# loss_cls_RPL: 3.6148, loss_mask_RPL: 2.4571, 
# loss_dice_RPL: 9.3259, loss_CE_prototype: 5.6110, 
# lovasz_softmax_loss_prototype: 8.4812, 
# loss_segmentation: 1.3139, loss_depth: 8.7947, 
# loss_map_coarse_bce: 0.6354, loss_map_coarse_dice: 0.4403, 
# loss_map_focal: 0.1101, loss_map_dice: 0.8813, 
# loss: 46.8002, grad_norm: 31.4298
