_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_aux_loss.py']

# Composition smoke config for Sections 7-9:
#   Section 7: 100x100 and 50x50 map auxiliary losses
#   Section 8: focal + Dice map loss with static class weights
#   Section 9: BEV x/y CoordConv before BEVSegHead
#
# Use the single-section configs first for attribution; this file is only for
# combined-path sanity checks.
model = dict(
    bev_seg_head=dict(
        map_loss_type='focal_dice',
        map_dice_weight=1.0,
        map_focal_gamma=2.0,
        map_focal_alpha=0.25,
        use_map_class_weights=True,
        map_class_weights=[1.0, 2.0, 2.0, 2.0, 4.0, 4.0],
        use_bev_coordconv=True,
        bev_coord_type='xy'))
