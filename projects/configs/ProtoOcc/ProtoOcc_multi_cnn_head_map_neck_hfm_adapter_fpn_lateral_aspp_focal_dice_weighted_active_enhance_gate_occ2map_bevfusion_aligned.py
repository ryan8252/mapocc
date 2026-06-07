_base_ = [
    './ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned.py'
]

# OCC2Map prior-guided active enhancement gate.
#
# Keep the current strongest aligned map path unchanged. This config adds an
# auxiliary low-Z OCC prior only to the BEVSegHead active-gate logits:
#   shared CVF [-50, 50], z context 0..2 -> CNN3D -> road/sidewalk soft prior
#   -> resample to map 200 x 200 -> small zero-init 1x1 area-gate prior.
#
# OCC/PQD supervision remains the original center crop [-40, 40] x/y x 16 z.
# The thin gate is intentionally not driven by OCC2Map here. A dense
# driveable-surface prior is too broad for ped_crossing/stop_line/divider and
# can destabilize the shared map/depth/OCC trunk.

model = dict(
    use_occ2map_prior=True,
    occ2map_prior_detach=True,
    occ2map_prior_feature_range=[
        -50.0, -50.0, -1.0,
        50.0, 50.0, 5.4,
    ],
    # Use three low-z slices for CNN3D context, then consume z=0..1 for prior.
    occ2map_prior_context_z=3,
    occ2map_prior_output_z_indices=[0, 1],
    # NuScenes OCC ids used by ProtoOcc: 11=driveable_surface, 13=sidewalk.
    occ2map_prior_class_ids=[11, 13],
    bev_seg_head=dict(
        use_occ2map_active_gate_prior=True,
        occ2map_prior_channels=2,
        occ2map_prior_zero_init=True,
        occ2map_prior_gate_scale=0.1,
        # Active Enhance Gate groups are [area/background, thin/overlay].
        occ2map_prior_gate_channel_mask=[1.0, 0.0]))
