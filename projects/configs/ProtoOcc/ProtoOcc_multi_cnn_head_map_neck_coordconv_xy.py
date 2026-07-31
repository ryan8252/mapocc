_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 9 ablation: add normalized BEV x/y coordinate channels before the
# BEVSegHead decoder, then project back to the protected 128-channel map path.
model = dict(
    map_loss_weight=4.0,
    bev_seg_head=dict(
        use_bev_coordconv=True,
        bev_coord_type='xy'))
