_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_map_only.py']

numC_Trans = 80
map_bev_channels = 128

# Path B map-only diagnostic.
#
# Swap MapOnly_BEV_Encoder (fixed cat-Z flatten 80*16=1280 -> 1x1 -> 160 ->
# shared CustomBEVBackbone) for a map-private MapTopologyEncoder with *learned*
# softmax-Z pooling (80 -> weighted-sum over Z -> 80 -> 1x1 -> 160 -> same
# CustomBEVBackbone). Backbone depth/width/stride are kept identical to the
# baseline (num_layer=2,2,2, channels=[160,320,640], stride=2,2,2), so the ONLY
# difference is learned-Z pooling vs cat-Z.
#
# Question this answers:
#   * If map-only Map mIoU > 48.34 -> a learned-Z map-private backbone raises the
#     ceiling, so a map-private branch is worth building in the 2-task model.
#   * If <= 48.34 -> learned single weighted-Z projection discards too much
#     height detail relative to cat-Z; stay with residual injection (map_hfm).
#
# Note: cat-Z keeps all 16 height slices (1280 ch) while weighted_pool keeps one
# learned weighted combination (80 ch), so this is a genuine information test.
# For a parity sanity check, duplicate this config with z_projection='cat_z'
# and z_channels=16 (should reproduce ~48.34).
model = dict(
    map_bev_encoder=dict(
        type='MapOnly_MTE_Encoder',
        _delete_=True,
        map_topology_encoder=dict(
            type='MapTopologyEncoder',
            in_channels=numC_Trans,
            z_projection='weighted_pool',
            mid_channels=numC_Trans * 2,
            num_channels=(numC_Trans * 2, numC_Trans * 4, numC_Trans * 8),
            num_layer=(2, 2, 2),
            stride=(2, 2, 2),
            with_cp=True),
        map_bev_encoder_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
            catconv_in_channels2=numC_Trans * 2 + map_bev_channels * 2,
            out_channels=map_bev_channels,
            input_feature_index=(0, 1, 2),
            with_cp=True)))
