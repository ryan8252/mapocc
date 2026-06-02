_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

# A1: Z-lite residual before map prediction.
#
# Keep the strong shared path unchanged:
#   LSS voxel x -> cat-Z -> 1x1 1280->160 -> shared BEV backbone
#       -> 128ch map neck -> BEVSegHead
#
# Add a detached, zero-init map-only residual from the original 3D LSS voxel
# feature. This tests whether lightweight local 3D context before Z compression
# helps the map branch without replacing the protected 128ch map-neck path.
numC_Trans = 80
map_bev_channels = 128

model = dict(
    map_loss_weight=4.0,
    dual_branch_encoder=dict(
        map_z_residual=dict(
            type='MapZResidualLayer',
            in_channels=numC_Trans,
            z_channels=16,
            out_channels=map_bev_channels,
            hidden_channels=map_bev_channels,
            z_projection='cat_z',
            residual_scale=1.0,
            detach_input=True,
            with_cp=True)))
