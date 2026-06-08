_base_ = [
    '../ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8.py'
]

# K=1 learned height-pooling residual ablation.
#
# Goal:
#   keep the aligned mapw8 DBE/map-neck route as the base feature, then add a
#   zero-init map residual distilled from the shared 16-bin LSS voxel feature.
#
# Route:
#   voxel_feat [B, 80, 16, 256, 256]
#     -> one learned softmax-over-Z pooling pattern
#     -> pooled BEV cue [B, 80, 256, 256]
#     -> residual projection [B, 128, 200, 200]
#     -> map_final = mapw8_base + residual
#
# This is the closest shared-voxel counterpart to a one-bin LSS2D map cue while
# preserving the original MTL shared encoder and mapw8 map route.

numC_Trans = 80
map_bev_channels = 128

model = dict(
    dual_branch_encoder=dict(
        map_z_residual=dict(
            type='MapZResidualLayer',
            in_channels=numC_Trans,
            z_channels=16,
            out_channels=map_bev_channels,
            hidden_channels=map_bev_channels,
            z_projection='learned_pool',
            num_height_patterns=1,
            residual_scale=1.0,
            detach_input=True,
            with_cp=True,
            zero_init_z_logits=True)))
