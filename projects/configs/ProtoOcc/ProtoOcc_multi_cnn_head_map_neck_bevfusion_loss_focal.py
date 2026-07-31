_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# BEVFusion / local-MAESTRO-style map loss ablation on the native ProtoOcc
# map/occ lattice. This keeps the 128ch map neck, occupancy branch, and
# [-40m, 40m] / 0.4m map raster unchanged. The map loss itself uses the
# BEVFusion / local-MAESTRO focal recipe (alpha=-1, gamma=2). The detector-level
# map_loss_weight is intentionally higher than weight4 because focal-only
# produced a much smaller scalar than the original BCE + Dice map loss.
#
# Use ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_focal_weight4.py
# when the experiment also needs the BEVFusion map evaluation grid
# ([-50m, 50m] / 0.5m).
model = dict(
    map_loss_weight=128.0,
    bev_seg_head=dict(
        loss_bce=None,
        loss_dice=None,
        loss_focal=dict(
            type='BinaryMaskFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=-1.0,
            reduction='mean',
            loss_weight=1.0)))
