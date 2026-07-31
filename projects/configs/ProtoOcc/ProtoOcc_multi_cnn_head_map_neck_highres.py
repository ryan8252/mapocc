_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Thin-class resolution test (Path B, experiment A).
#
# Diagnosis: the map is predicted from 100x100 features. The BEV backbone
# strides 200 -> 100 -> 50 -> 25, the map neck fuses those and the final 200x200
# is a bilinear upsample of 100-res content. A divider / stop_line rasterized at
# 0.4m is ~1 px, i.e. sub-feature-resolution, so thin classes cannot be
# predicted sharply regardless of model capacity.
#
# Fix: a zero-init high-res skip reads the *real* 200x200 pre-backbone BEV
# (pooled_x) through a shallow stride-1 conv stack and fuses it into the 200-res
# map feature before BEVSegHead. The map head finally sees genuine high-res
# spatial detail. Source detached by default so map gradients do not reshape the
# occ-shared pooling (occ-safe); the high-res block itself stays trainable.
#
# How to read the result:
#   * divider / stop_line jump while drivable / big classes barely move
#       -> resolution WAS the thin-class bottleneck (hypothesis confirmed).
#   * no thin-class gain -> the thin-line signal is already lost upstream
#       (view transform / depth on the ground plane), not at this stage.
model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        map_highres_skip=True,
        map_highres_fusion='add',
        map_highres_detach=True))
