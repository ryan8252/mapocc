_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted.py']

# M3Net-like map query refinement without iterative transformer decoding:
# CNN logits select 5 x 6 spatial/class queries, one self-attn+MLP refines them,
# and query logits are added back as a learnable residual.
model = dict(
    bev_seg_head=dict(
        use_query_refinement=True,
        query_refine_num_bins=5,
        query_refine_heads=8,
        query_refine_ffn_ratio=2,
        query_refine_attn_dropout=0.0,
        query_refine_alpha_init=0.0,
        query_refine_detach_logits=True,
        query_refine_scale_dot=True))
