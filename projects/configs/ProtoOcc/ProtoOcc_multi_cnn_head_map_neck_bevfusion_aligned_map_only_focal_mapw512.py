_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only.py']

# Aligned map-only focal ablation with PV depth/segmentation auxiliary loss kept.
#
# Previous full run at map_loss_weight=128 ended around:
#   map ~= 1.6369, depth+seg ~= 6.5922, map ratio ~= 19.9%.
# Raising the post-scale focal map loss by 4x targets roughly:
#   map ~= depth+seg -> map ratio ~= 50%.
model = dict(
    train_depth=True,
    map_loss_weight=512.0)

