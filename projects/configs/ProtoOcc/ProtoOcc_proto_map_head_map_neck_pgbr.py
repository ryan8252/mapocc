_base_ = ['./ProtoOcc_proto_map_head_map_neck.py']

# PGBR ablation: map-specific BEV neck with prototype-grounded BEV refinement.
# Kept for reference; canonical map-neck experiment uses PGBR=False (base).
model = dict(
    proto_map_head=dict(
        pgbr_cfg=dict(
            temperature=1.0,
            detach_query=True)))
