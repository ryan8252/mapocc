_base_ = [
    './ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_map_lss2d256.py'
]

# Variant A + separate unsupervised map depth (方案 1 / map depth decoupling).
#
# The base config already routes the map head through a separate 2D-LSS branch
# (map_img_view_transformer + map_lss2d_encoder), but that branch still consumes
# the OCC depth distribution, which is sharpened by loss_depth. Sharp depth makes
# the BEV splat sparse and starves flat map classes.
#
# This variant gives the map LSS2D branch its OWN soft depth (MapDepthNet),
# shaped only by the map segmentation gradient -- there is no depth loss on it.
# OCC keeps its sharp, supervised CM_DepthNet depth unchanged. depth_channels
# must match the map_img_view_transformer frustum: depth=[1.0, 45.0, 0.5] -> 88.

depth_categories = 88

model = dict(
    map_depth_net=dict(
        type='MapDepthNet',
        in_channels=512,
        mid_channels=256,
        depth_channels=depth_categories,
        num_layers=2,
        with_cp=True,
        init_uniform=True))

# NOTE: do NOT set find_unused_parameters=True here. The DBE's bev_encoder_neck
# output is *not* unused -- it is folded back into the OCC voxel feature
# (comprehensive_voxel_feature = vox + vox_raw), and map_depth_net is trained by
# the map seg gradient. All params receive gradients, so the DDP default
# (find_unused_parameters=False) works and stays compatible with the reentrant
# gradient checkpointing used throughout the model.
