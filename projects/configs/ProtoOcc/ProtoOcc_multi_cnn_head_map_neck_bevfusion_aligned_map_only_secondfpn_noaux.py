_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_focal_no_depth_no_seg.py']

# Diagnostic 1: image-neck parity only.
#
# Keep the ProtoOcc map-only 3D voxel -> map decoder path unchanged, but replace
# the C4/C5 CustomFPN image neck with BEVFusion camera-r50's C2-C5 SECONDFPN.
# No OCC, no PV depth loss, no PV segmentation loss.
model = dict(
    img_backbone=dict(out_indices=(0, 1, 2, 3)),
    img_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=[128, 128, 128, 128],
        upsample_strides=[0.25, 0.5, 1, 2]))
