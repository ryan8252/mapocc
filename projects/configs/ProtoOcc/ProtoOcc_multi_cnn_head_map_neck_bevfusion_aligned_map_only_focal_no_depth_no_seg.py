_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only.py']

# Aligned map-only focal ablation with no PV depth/segmentation auxiliary loss.
#
# The base config already uses:
#   - ProtoOccMapOnly
#   - BEVFusion-aligned map grid: [-50, 50] / 0.5m -> 200 x 200
#   - focal-only BEVSegHead
#   - map_loss_weight=128.0
#
# Setting train_depth=False removes both loss_depth and loss_segmentation from
# ProtoOccMapOnly.forward_train. ProtoOccMapOnly also freezes the unused
# depth_net.class_predictor when train_depth=False to avoid DDP unused-parameter
# failures.
model = dict(
    train_depth=False,
    map_loss_weight=128.0)

