_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Detach ablation for CNN map head + 128-channel map-specific neck.
#
# `detach_map_feature=True` makes the map neck inputs not require gradients.
# Keep checkpointing disabled inside the map neck; otherwise PyTorch checkpoint
# will not build gradients for the checkpointed map-neck parameters.
model = dict(
    dual_branch_encoder=dict(
        detach_map_feature=True,
        map_bev_encoder_neck=dict(
            with_cp=False)))

#Starting Evaluation...
# 100%|█████████████| 6019/6019 [00:55<00:00, 108.29it/s]
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 11.03
# ===> barrier - IoU = 44.56
# ===> bicycle - IoU = 25.33
# ===> bus - IoU = 44.62
# ===> car - IoU = 50.87
# ===> construction_vehicle - IoU = 23.93
# ===> motorcycle - IoU = 25.43
# ===> pedestrian - IoU = 27.41
# ===> traffic_cone - IoU = 26.55
# ===> trailer - IoU = 30.88
# ===> truck - IoU = 36.9
# ===> driveable_surface - IoU = 80.8
# ===> other_flat - IoU = 42.71
# ===> sidewalk - IoU = 51.13
# ===> terrain - IoU = 54.35
# ===> manmade - IoU = 40.26
# ===> vegetation - IoU = 34.83
# ===> mIoU of 6019 samples: 38.33
# {'mIoU': array([0.11 , 0.446, 0.253, 0.446, 0.509, 0.239, 0.254, 0.274, 0.266,
#        0.309, 0.369, 0.808, 0.427, 0.511, 0.543, 0.403, 0.348, 0.896]), 'TP': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0.]), 'FP': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0.]), 'FN': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0.]), 'GT_counts': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0.]), '
# map/drivable_area/iou@max': 0.5328726768493652, 'map/drivable_area/iou@0.35': 0.5328726768493652, 'map/drivable_area/iou@0.40': 0.5320510864257812, 'map/drivable_area/iou@0.45': 0.5254619717597961, 'map/drivable_area/iou@0.50': 0.512442946434021, 'map/drivable_area/iou@0.55': 0.4929153025150299, 'map/drivable_area/iou@0.60': 0.4672273099422455, 'map/drivable_area/iou@0.65': 0.4360702931880951, 
# 'map/ped_crossing/iou@max': 0.07825203984975815, 'map/ped_crossing/iou@0.35': 0.07825203984975815, 'map/ped_crossing/iou@0.40': 0.06823921948671341, 'map/ped_crossing/iou@0.45': 0.05836346372961998, 'map/ped_crossing/iou@0.50': 0.04889089986681938, 'map/ped_crossing/iou@0.55': 0.04040521755814552, 'map/ped_crossing/iou@0.60': 0.03311922401189804, 'map/ped_crossing/iou@0.65': 0.026751257479190826, 
# 'map/walkway/iou@max': 0.179426908493042, 'map/walkway/iou@0.35': 0.179426908493042, 'map/walkway/iou@0.40': 0.16273270547389984, 'map/walkway/iou@0.45': 0.14550431072711945, 'map/walkway/iou@0.50': 0.12832196056842804, 'map/walkway/iou@0.55': 0.11119061708450317, 'map/walkway/iou@0.60': 0.09420661628246307, 'map/walkway/iou@0.65': 0.07731270790100098, 
# 'map/stop_line/iou@max': 0.0559050627052784, 'map/stop_line/iou@0.35': 0.0559050627052784, 'map/stop_line/iou@0.40': 0.045173775404691696, 'map/stop_line/iou@0.45': 0.0376407653093338, 'map/stop_line/iou@0.50': 0.031707942485809326, 'map/stop_line/iou@0.55': 0.026884200051426888, 'map/stop_line/iou@0.60': 0.02344037964940071, 'map/stop_line/iou@0.65': 0.020913027226924896, 
# 'map/carpark_area/iou@max': 0.06663017719984055, 'map/carpark_area/iou@0.35': 0.06663017719984055, 'map/carpark_area/iou@0.40': 0.0584561750292778, 'map/carpark_area/iou@0.45': 0.0504816472530365, 'map/carpark_area/iou@0.50': 0.04326716810464859, 'map/carpark_area/iou@0.55': 0.03715633228421211, 'map/carpark_area/iou@0.60': 0.03243260085582733, 'map/carpark_area/iou@0.65': 0.028149347752332687, 
# 'map/divider/iou@max': 0.07609163224697113, 'map/divider/iou@0.35': 0.07609163224697113, 'map/divider/iou@0.40': 0.06238357722759247, 'map/divider/iou@0.45': 0.05232413113117218, 'map/divider/iou@0.50': 0.045304033905267715, 'map/divider/iou@0.55': 0.04002993553876877, 'map/divider/iou@0.60': 0.035597484558820724, 'map/divider/iou@0.65': 0.03156773000955582, 
# 'map/mean/iou@max': 0.16486307978630066}