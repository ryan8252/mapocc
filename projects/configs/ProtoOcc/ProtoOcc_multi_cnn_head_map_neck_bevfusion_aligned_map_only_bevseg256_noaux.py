_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_focal_no_depth_no_seg.py']

# Diagnostic 4: 256-channel map feature/head width only.
#
# Keep the current ProtoOcc map-only representation and decoder topology, but
# widen the final map feature and BEVSegHead from 128 channels to 256 channels.
# No OCC, no PV depth loss, no PV segmentation loss.

numC_Trans = 80
map_bev_channels = 256

model = dict(
    img_backbone=dict(pretrained=None),
    map_bev_encoder=dict(
        map_bev_encoder_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
            catconv_in_channels2=numC_Trans * 2 + map_bev_channels * 2,
            out_channels=map_bev_channels,
            input_feature_index=(0, 1, 2),
            with_cp=True)),
    bev_seg_head=dict(
        in_channels=map_bev_channels,
        hidden_channels=map_bev_channels))
