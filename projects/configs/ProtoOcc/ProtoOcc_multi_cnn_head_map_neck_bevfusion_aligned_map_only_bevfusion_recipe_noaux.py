_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_focal_no_depth_no_seg.py']

# Diagnostic 6: combined BEVFusion-style map STL architecture inside ProtoOcc.
#
# Combines diagnostics 1-4:
#   - SECONDFPN C2-C5 image neck
#   - 2D LSS BEV feature over [-51.2, 51.2] / 0.8m
#   - GeneralizedResNet + LSSFPN BEV decoder
#   - 256-channel BEVSegHead
#
# dbound remains [1, 45, 0.5] here so the combined run isolates architecture
# before adding the dbound60 change. No OCC, no PV depth loss, no PV
# segmentation loss.

numC_Trans = 64
bevfusion_map_channels = 256

grid_config_3dpool = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 45.0, 0.5],
}

model = dict(
    img_backbone=dict(pretrained=None, out_indices=(0, 1, 2, 3)),
    img_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=[128, 128, 128, 128],
        upsample_strides=[0.25, 0.5, 1, 2]),
    depth_net=dict(context_channels=numC_Trans),
    img_view_transformer=dict(
        grid_config=grid_config_3dpool,
        out_channels=numC_Trans,
        collapse_z=True),
    map_bev_encoder=dict(
        _delete_=True,
        type='MapOnly_BEVFusion_Encoder',
        bev_decoder_backbone=dict(
            type='GeneralizedResNet',
            in_channels=numC_Trans,
            blocks=[
                [2, 128, 2],
                [2, 256, 2],
                [2, 512, 1],
            ],
            with_cp=True),
        bev_decoder_neck=dict(
            type='LSSFPN',
            in_indices=[-1, 0],
            in_channels=[512, 128],
            out_channels=bevfusion_map_channels,
            scale_factor=2)),
    bev_seg_head=dict(
        in_channels=bevfusion_map_channels,
        hidden_channels=bevfusion_map_channels))
