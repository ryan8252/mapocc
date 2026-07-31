_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 1 ablation: 200x200 high-resolution detail skip with gated fusion.
#
# Compared with `ProtoOcc_multi_cnn_head_map_neck_highres.py`, this keeps the
# same full-resolution detail branch but uses sigmoid gating:
#   map_feature = fpn_out + sigmoid(gate([fpn_out, detail])) * detail
#
# The detail branch final projection is zero-initialized, so step-0 behavior is
# still equivalent to the clean 128ch map-neck weight4 baseline.
model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        map_highres_skip=True,
        map_highres_fusion='gated',
        map_highres_detach=True))
