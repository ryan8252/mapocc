_base_ = ['./ProtoOcc_proto_map_head.py']

# PGBR ablation: enable prototype-grounded BEV refinement on top of the
# canonical ProtoMapHead base. Ablation result showed PGBR hurts both metrics
# (Occ 36.96 vs 37.61, Map 31.46 vs 32.95) — kept for reference only.
model = dict(
    proto_map_head=dict(
        pgbr_cfg=dict(
            temperature=1.0,
            detach_query=True),
    ))

# Eval results (ProtoOcc_proto_map_head_PGBR_TWCC / epoch 24):
# Occ mIoU: 36.96   Map mIoU: 31.46
#
# class        IoU
# drivable_area  71.83
# walkway        37.81
# carpark_area   23.33
# ped_crossing   21.86
# divider        19.01
# stop_line      14.92
