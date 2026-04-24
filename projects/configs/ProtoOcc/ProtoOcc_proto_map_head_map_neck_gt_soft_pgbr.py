_base_ = ['./ProtoOcc_proto_map_head_map_neck_gt_soft.py']

# GT-soft map prototype mining + PGBR ablation.
model = dict(
    proto_map_head=dict(
        pgbr_cfg=dict(
            temperature=1.0,
            detach_query=True)))
