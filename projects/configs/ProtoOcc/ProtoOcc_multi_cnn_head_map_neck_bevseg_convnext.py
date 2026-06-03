_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 5 ablation: ConvNeXt-like BEVSegHead refinement with depthwise 7x7
# blocks. The final block projection is zero-initialized inside each block.
model = dict(
    map_loss_weight=4.0,
    bev_seg_head=dict(
        bevseg_head_type='convnext',
        bevseg_num_refine_blocks=2))
