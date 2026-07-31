_base_ = ['./ProtoOcc_proto_map_head_map_neck.py']

# Step 1: map-only GT-soft prototype mining.
# Inference still falls back to prediction-threshold local mining + EMA bank.
model = dict(
    proto_map_head=dict(
        prototype_mining_mode='gt_soft',
        prototype_mining_alpha=0.5,
        prototype_pooling_scope='batch_local'))
