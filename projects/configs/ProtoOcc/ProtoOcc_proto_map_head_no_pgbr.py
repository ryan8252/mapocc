_base_ = ['./ProtoOcc_proto_map_head.py']

model = dict(
    proto_map_head=dict(
        use_bev_refinement=False,
    ))
