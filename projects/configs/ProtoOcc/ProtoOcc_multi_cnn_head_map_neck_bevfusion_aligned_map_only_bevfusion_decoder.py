_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only.py']

# BEVFusion-style map-only upper bound.
#
# Purpose:
#   Isolate whether the low BEVFusion-aligned map score comes from the
#   ProtoOcc map feature path or from the map GT / protocol itself.
#
# Differences from ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only:
#   - no OCC / prototype branch (inherited ProtoOccMapOnly)
#   - LSS view transform produces a 2D BEV feature like BEVFusion:
#       zbound [-10, 10, 20] -> one z bin, collapse_z=True
#   - map encoder is BEVFusion-style:
#       GeneralizedResNet -> LSSFPN -> 256ch BEVSegHead
#   - map GT/output remains BEVFusion protocol:
#       [-50, 50] / 0.5m -> 200 x 200
#
# Keep dbound at 45m first. MAESTRO does not report dbound, and the local
# dbound60 runs were worse, so this config isolates decoder architecture before
# changing depth discretization again.

numC_Trans = 80
bevfusion_map_channels = 256

grid_config_3dpool = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 45.0, 0.5],
}

model = dict(
    img_view_transformer=dict(
        grid_config=grid_config_3dpool,
        collapse_z=True),
    map_bev_encoder=dict(
        _delete_=True,
        type='MapOnly_BEVFusion_Encoder',
        bev_decoder_backbone=dict(
            type='GeneralizedResNet',
            in_channels=numC_Trans,
            blocks=[
                [2, 160, 2],
                [2, 320, 2],
                [2, 640, 1],
            ],
            with_cp=True),
        bev_decoder_neck=dict(
            type='LSSFPN',
            in_indices=[-1, 0],
            in_channels=[640, 160],
            out_channels=bevfusion_map_channels,
            scale_factor=2)),
    bev_seg_head=dict(
        in_channels=bevfusion_map_channels,
        hidden_channels=bevfusion_map_channels,
        num_convs=2,
        with_cp=True,
        loss_bce=None,
        loss_dice=None,
        loss_focal=dict(
            type='BinaryMaskFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=-1.0,
            reduction='mean',
            loss_weight=1.0)),
    # This follows the existing aligned focal map-only upper-bound config so the
    # map objective is not drowned by the optional depth auxiliary loss.
    map_loss_weight=128.0,
    train_depth=True)
