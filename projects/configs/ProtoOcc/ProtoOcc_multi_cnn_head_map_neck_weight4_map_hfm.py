_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_weight4.py']

model = dict(
    dual_branch_encoder=dict(
        use_map_hfm=True,
        map_hfm_lower_source='vox3',
        detach_map_feature=False))
