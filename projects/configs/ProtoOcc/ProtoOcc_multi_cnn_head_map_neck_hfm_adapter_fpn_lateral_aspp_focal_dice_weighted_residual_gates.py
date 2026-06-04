_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted.py']

# Section 11 ablation: add learnable residual gates to the additive map
# add-ons. The underlying HFM/adapter/ASPP residual projections stay zero-init,
# so step-0 behavior remains identical to the ungated combined candidate.
model = dict(
    dual_branch_encoder=dict(
        map_hfm_learnable_residual_gate=True,
        map_hfm_residual_gate_init=0.1,
        map_residual_adapter=dict(
            learnable_residual_gate=True,
            residual_gate_init=0.1),
        map_bev_encoder_neck=dict(
            fpn_context_learnable_gate=True,
            fpn_context_gate_init=0.1)))
