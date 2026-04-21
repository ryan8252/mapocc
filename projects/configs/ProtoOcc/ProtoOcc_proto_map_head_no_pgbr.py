_base_ = ['./ProtoOcc_proto_map_head.py']

model = dict(
    proto_map_head=dict(
        use_bev_refinement=False,
    ))

# ProtoMapHead no PGBR, epoch 24	72.14	24.52	38.44	15.22	25.88	21.50	32.95

# Occupancy 分類結果
#              epoch 17	 epoch 24
# others		9.91	10.65
# barrier		43.87	43.87
# bicycle		21.36	21.52
# bus		42.05	43.31
# car		49.63	50.54
# construction_vehicle		20.65	20.87
# motorcycle		22.58	26.05
# pedestrian		26.59	25.29
# traffic_cone		25.84	26.66
# trailer		28.32	31.72
# truck		34.81	35.60
# driveable_surface		80.10	81.01
# other_flat		41.83	42.78
# sidewalk		50.24	51.17
# terrain		53.41	53.05
# manmade		40.27	40.87
# vegetation		33.63	34.37
# mIoU		36.77	37.61
