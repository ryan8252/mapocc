_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted.py']

# M3Net-style task-aware channel scaling ablation.
# Zero-init residual gates make the initial forward exactly match the baseline.
voxel_out_channels = 48
map_bev_channels = 128

model = dict(
    task_channel_scaling=dict(
        occ_channels=voxel_out_channels,
        map_channels=map_bev_channels,
        reduction=4,
        min_hidden_channels=16,
        max_residual=0.5,
        detach_context=False))
