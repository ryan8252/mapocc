_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 5 ablation: strengthen BEVSegHead with two zero-init residual
# refinement blocks while keeping the same 128-channel map-neck input.
model = dict(
    map_loss_weight=4.0,
    bev_seg_head=dict(
        bevseg_head_type='res_refine',
        bevseg_num_refine_blocks=2))
