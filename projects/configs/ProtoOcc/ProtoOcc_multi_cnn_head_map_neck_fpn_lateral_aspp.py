_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_fpn_lateral.py']

# Section 4 ablation: lateral-projected Custom_FPN_LSS plus residual ASPP
# context on the 25x25 level before top-down fusion.
model = dict(
    dual_branch_encoder=dict(
        map_bev_encoder_neck=dict(
            use_fpn_global_context=True,
            fpn_global_context_type='aspp')))
