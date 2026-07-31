_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only.py']

# Aligned map-only focal ablation with PV depth/segmentation auxiliary loss kept.
#
# map_loss_weight=512 was intended to target about 50% map loss late in training,
# but it is too aggressive early and can make the map focal term dominate.
# This replacement doubles the previous stable map_loss_weight=128 run and
# targets roughly 33% late-stage map loss while keeping the same recipe.
model = dict(
    train_depth=True,
    map_loss_weight=256.0)
