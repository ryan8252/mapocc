_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp.py']

# Composition smoke config for Sections 4-6:
#   Section 4: lateral projection + residual ASPP in Custom_FPN_LSS
#   Section 5: residual BEVSegHead refinement
#   Section 6: area/line dual map decoders
#
# Use individual configs first for clean attribution; this file is only for
# verifying the combined path after the separate ablations are understood.
model = dict(
    bev_seg_head=dict(
        bevseg_head_type='res_refine',
        bevseg_num_refine_blocks=2,
        use_dual_area_line_head=True,
        area_class_indices=[0, 1, 2, 4],
        line_class_indices=[3, 5],
        line_head_use_detail=False))
