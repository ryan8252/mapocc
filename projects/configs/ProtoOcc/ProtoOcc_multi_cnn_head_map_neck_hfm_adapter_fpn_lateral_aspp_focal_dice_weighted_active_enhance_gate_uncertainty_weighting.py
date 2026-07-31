_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py']

# Pure homoscedastic uncertainty weighting over four post-scale loss groups.
# Map losses first keep the validated baseline multiplier of 4.0, then the
# uncertainty module learns a per-group precision term.
model = dict(
    map_loss_weight=4.0,
    loss_uncertainty_weighting=dict(
        min_log_var=-2.0,
        max_log_var=2.0,
        init_log_vars=dict(
            map=0.0,
            depth_img=0.0,
            occ_proto=0.0,
            pqd=0.0),
        groups=[
            dict(name='map', prefixes=['loss_map_']),
            dict(
                name='depth_img',
                keys=['loss_depth', 'loss_segmentation']),
            dict(
                name='occ_proto',
                keys=[
                    'loss_CE_prototype',
                    'lovasz_softmax_loss_prototype',
                ]),
            dict(
                name='pqd',
                keys=[
                    'loss_cls',
                    'loss_mask',
                    'loss_dice',
                    'loss_cls_RPL',
                    'loss_mask_RPL',
                    'loss_dice_RPL',
                ]),
        ]))
