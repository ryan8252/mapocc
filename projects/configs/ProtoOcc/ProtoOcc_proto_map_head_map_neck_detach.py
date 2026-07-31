_base_ = ['./ProtoOcc_proto_map_head_map_neck.py']

# Detach ablation: map branch still uses the map-specific neck, but does not
# backpropagate into the shared multi-scale BEV backbone.
# 這個 config 是在做 detach ablation：測「map branch 的 loss 要不要反向影響共享 BEV backbone」。
model = dict(
    dual_branch_encoder=dict(
        detach_map_feature=True,
        map_bev_encoder_neck=dict(
            with_cp=False)))

# 2026-04-22 01:44:52,334 - mmdet - INFO - 
# Epoch [1][300/7033]    lr: 2.000e-04, eta: 2 days, 4:07:22, 
# time: 1.097, data_time: 0.010, memory: 15864, 
# loss_cls: 0.8689, loss_mask: 0.8055, 
# loss_dice: 3.0943, loss_cls_RPL: 2.6037, 
# loss_mask_RPL: 2.4212, loss_dice_RPL: 9.2816, 
# loss_CE_prototype: 5.3157, lovasz_softmax_loss_prototype: 8.2924,
# loss_segmentation: 1.2428, loss_depth: 8.9024, 
# loss_map_coarse_bce: 0.5824, loss_map_coarse_dice: 0.4477, 
# loss_map_focal: 0.1070, loss_map_dice: 0.8914, 
# loss: 44.8569, grad_norm: 27.7542