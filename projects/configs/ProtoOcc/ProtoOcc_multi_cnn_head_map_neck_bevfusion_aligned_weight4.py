_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned.py']

# Strong MTL baseline on the BEVFusion-aligned grid (map eval at +-50m / 0.5m).
#
# This is the *comparable* baseline for every aligned-grid experiment. The old
# strong baseline (Occ 39.72 / Map 45.79) was measured at +-40m / 0.4m and is
# NOT comparable to BEVFusion's 47.10 or MAESTRO's 51.30 map numbers; this one
# is. Occupancy still runs on the +-40m / 0.4m crop, so the occ number stays
# comparable to MAESTRO's occ (which also crops to +-40m).
model = dict(map_loss_weight=4.0)
