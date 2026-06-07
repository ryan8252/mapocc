_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_focal_no_depth_no_seg.py']

# Diagnostic 3: BEVFusion-style BEV decoder topology only.
#
# Keep the current 5D ProtoOcc LSS voxel feature, but replace
# CustomBEVBackbone + Custom_FPN_LSS with GeneralizedResNet + LSSFPN. A 1x1
# projection keeps the cat-Z input manageable while preserving the current
# z-bin source. No OCC, no PV depth loss, no PV segmentation loss.

numC_Trans = 80
bevfusion_decoder_in_channels = 64
map_bev_channels = 128

model = dict(
    map_bev_encoder=dict(
        _delete_=True,
        type='MapOnly_BEVFusion_Encoder',
        z_collapse='cat',
        input_projection=[numC_Trans * 16, bevfusion_decoder_in_channels],
        bev_decoder_backbone=dict(
            type='GeneralizedResNet',
            in_channels=bevfusion_decoder_in_channels,
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
            out_channels=map_bev_channels,
            scale_factor=2)),
    bev_seg_head=dict(
        in_channels=map_bev_channels,
        hidden_channels=map_bev_channels))
