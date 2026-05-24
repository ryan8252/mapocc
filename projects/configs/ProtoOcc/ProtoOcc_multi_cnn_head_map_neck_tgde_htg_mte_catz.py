_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_tgde_htg_mte_weighted.py']

# TGDE 01+02 cat-Z parity:
# Keep HeightTaskGate enabled, but make MTE preserve all height bins with the
# same fixed Z flattening used by the original DBE BEV path.  This isolates
# whether the first TGDE improvement should come from HTG + map-specific BEV
# topology encoding before relying on learned weighted-Z pooling.

numC_Trans = 80
grid_z = 16

model = dict(
    dual_branch_encoder=dict(
        map_topology_encoder=dict(
            type='MapTopologyEncoder',
            in_channels=numC_Trans,
            z_channels=grid_z,
            z_projection='cat_z',
            mid_channels=numC_Trans * 2,
            num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
            num_layer=[1, 1, 1],
            stride=[2, 2, 2],
            ConvNext_kernel_size=7,
            with_cp=True)))
