_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_focal_no_depth_no_seg.py']

# Diagnostic 2: BEVFusion-style 2D LSS representation only.
#
# Produce a 2D BEV feature with one z bin and collapse_z=True, then feed it
# through the existing ProtoOcc CustomBEVBackbone + Custom_FPN_LSS map decoder.
# This isolates the view-transform representation from the BEVFusion decoder.
# No OCC, no PV depth loss, no PV segmentation loss.

numC_Trans = 64
map_bev_channels = 128
protoocc_decoder_channels = 160

grid_config_3dpool = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-10.0, 10.0, 20.0],
    'depth': [1.0, 45.0, 0.5],
}

model = dict(
    depth_net=dict(context_channels=numC_Trans),
    img_view_transformer=dict(
        grid_config=grid_config_3dpool,
        out_channels=numC_Trans,
        collapse_z=True),
    map_bev_encoder=dict(
        _delete_=True,
        type='MapOnly_BEV2D_Encoder',
        input_projection=[numC_Trans, protoocc_decoder_channels],
        bev_encoder_backbone=dict(
            type='CustomBEVBackbone',
            stride=[2, 2, 2],
            numC_input=protoocc_decoder_channels,
            num_channels=[
                protoocc_decoder_channels,
                protoocc_decoder_channels * 2,
                protoocc_decoder_channels * 4,
            ]),
        map_bev_encoder_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=protoocc_decoder_channels * 4
            + protoocc_decoder_channels * 2,
            catconv_in_channels2=protoocc_decoder_channels
            + map_bev_channels * 2,
            out_channels=map_bev_channels,
            input_feature_index=(0, 1, 2),
            with_cp=True)),
    bev_seg_head=dict(
        in_channels=map_bev_channels,
        hidden_channels=map_bev_channels))
