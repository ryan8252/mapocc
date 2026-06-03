_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# Section 8 ablation: multi-label sigmoid focal + Dice map loss with static
# per-class weights for rare/thin map classes. Class order follows
# LoadBEVSegmentation in the base config:
# [drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider].
model = dict(
    map_loss_weight=4.0,
    bev_seg_head=dict(
        map_loss_type='focal_dice',
        map_dice_weight=1.0,
        map_focal_gamma=2.0,
        map_focal_alpha=0.25,
        use_map_class_weights=True,
        map_class_weights=[1.0, 2.0, 2.0, 2.0, 4.0, 4.0]))
