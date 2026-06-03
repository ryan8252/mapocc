_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_dbound60.py']

# HENet-style coarse internal map feature ablation.
#
# Keep the final BEVFusion-aligned map supervision unchanged:
#   map GT/output: [-50, 50] / 0.5m -> 200 x 200
#
# Change only the map-neck feature before `_align_map_feature`:
#   baseline dbound60 map feature: [B, 128, 256, 256] over [-51.2, 51.2]
#   this config map feature:       [B, 128, 128, 128] over [-51.2, 51.2]
#
# `_align_map_feature` then resamples 128 x 128 -> 200 x 200, similar to
# BEVFusion's BEVGridTransform. Use `extra_upsample=1` instead of None so
# Custom_FPN_LSS still returns a valid projected feature tensor.

model = dict(
    dual_branch_encoder=dict(
        map_bev_encoder_neck=dict(
            extra_upsample=1)))
