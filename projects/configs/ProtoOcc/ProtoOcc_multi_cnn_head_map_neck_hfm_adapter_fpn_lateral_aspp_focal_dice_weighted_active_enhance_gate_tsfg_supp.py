_base_ = [
    './ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py'
]

# Native-grid A ablation: supervised multiplicative suppression at the
# 128-channel map-neck output, before BEVSegHead. Keep the current strongest
# map path, map_loss_weight=4.0, and active_enhance_gate unchanged.
model = dict(
    voxel_aware_map_ingest=dict(
        type='MapTSFGSuppression',
        mode='suppress_only',
        map_channels=128,
        score_init_bias=2.0,
        supp_loss_weight=1.0,
        focal_gamma=2.0,
        gt_dilation=0))
