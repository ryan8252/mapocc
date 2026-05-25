_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_adapter.py']

# Safer variant for checking whether map-branch gradients through the adapter
# hurt occupancy. The adapter still learns, but its input identity path is
# detached from the shared BEV backbone.
model = dict(
    dual_branch_encoder=dict(
        map_residual_adapter=dict(
            detach_input=True)))
