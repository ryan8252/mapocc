# MAESTRO 2-task reproduction

## Goal

This version is a clean MAESTRO-style occupancy + BEV map reproduction. It no longer routes the MAESTRO branch through ProtoOcc's Dual_Branch_Encoder, CNN3D decoder, or Prototype Query Decoder.

The active training config is:

```bash
projects/configs/MAESTRO/MAESTRO_2task_lss_occformer.py
```

## Forward Path

1. Camera images go through the existing ResNet-50 image backbone, `CustomFPN`, `CM_DepthNet`, and `LSSViewTransformer_depthGT`.
2. The LSS voxel feature is converted from `(B, C, Z, H, W)` to `(B, C, H, W, Z)`.
3. `MAESTROClasswisePrototypeGenerator` runs directly on that LSS voxel feature.
4. Map TSFG uses MAESTRO background prototypes to generate a BEV map feature.
5. `MAESTROBEVSegHead` predicts six map masks, with BEVFusion-style sigmoid focal loss.
6. SPA uses map prototypes to enrich the background prototype group for occupancy.
7. Occupancy TSFG generates an occupancy-specific voxel feature.
8. `MAESTROOccFormerHead` predicts occupancy with an OccFormer-style 3D Mask2Former query decoder.

## Paper Mapping

- CPG: `projects/mmdet3d_plugin/models/model_utils/maestro_cpg.py`
  - Input is now the LSS voxel feature, not DBE output.
  - The config sets `in_channels=80`, matching `LSSViewTransformer_depthGT.out_channels`.
  - The forward path follows `F_s -> S_v -> B_k -> P_k`, uses hard class masks by default, and exposes `P_fg`, `P_bg`, and `G_occ`.
- TSFG: `projects/mmdet3d_plugin/models/model_utils/maestro_tsfg.py`
  - Map branch collapses height into BEV.
  - Occupancy branch keeps the 3D feature.
  - The implementation now uses MAESTRO-style dot-product `F_wise`, prototype-gated `F_aware`, and suppression `F_TS = F_tilde * S_t_supp`.
- SPA: `projects/mmdet3d_plugin/models/model_utils/maestro_spa.py`
  - Kept one-way map-to-occ because this is a two-task reproduction.
  - The code explicitly documents this as a 2-task adaptation of MAESTRO SPA: map prototypes are pooled from `map_feature` and `map_logits`, then aggregated into `P_bg`.
  - Prototype aggregation is rule-based like the paper, not similarity attention: multiple fine-grained map labels are averaged before being summed into the corresponding background prototype.
- Map head: `MAESTROBEVSegHead`
  - Same simple BEV segmentation head.
  - Loss changed from BCE/Dice to BEVFusion-style sigmoid focal.
- Occupancy head: `MAESTROOccFormerHead`
  - Uses multi-scale 3D masked cross-attention and query-mask prediction.
  - Supervision uses existing ProtoOcc `voxel_semantics` and `mask_camera`, so no OccFormer lidarseg-point dataset conversion is required.

## Files Added

- `projects/mmdet3d_plugin/models/detectors/MAESTRO2Task.py`
  - New pure MAESTRO detector.
- `projects/mmdet3d_plugin/models/OccHead/maestro_occformer_head.py`
  - OccFormer-style occupancy head adapted to ProtoOcc occupancy labels.

## Files Modified

- `projects/configs/MAESTRO/MAESTRO_2task_lss_occformer.py`
  - Switched model type to `MAESTRO2Task`.
  - Removed `dual_branch_encoder`, `cnn3d_decoder`, and `prototype_query_decoder`.
  - Added `MAESTROOccFormerHead`.
  - Set CPG to hard class masks and set TSFG prototype counts to 6 for map and 17 for occupancy.
  - Moved from `projects/configs/ProtoOcc/` to `projects/configs/MAESTRO/` to avoid implying this path still uses ProtoOcc's core DBE/PQD modules.
  - Changed work dir to `work_dirs/MAESTRO_2task_lss_occformer`.
- `projects/mmdet3d_plugin/models/model_utils/maestro_cpg.py`
  - Fixed the CPG voxel layout bug and aligned variable names/comments with the MAESTRO notation.
- `projects/mmdet3d_plugin/models/model_utils/maestro_tsfg.py`
  - Replaced the previous cosine/softmax context with MAESTRO's dot-product prototype-wise activation and suppression-gated task feature.
- `projects/mmdet3d_plugin/models/model_utils/maestro_spa.py`
  - Replaced the zero-initialized learned fusion scale and similarity fallback with explicit semantic prototype aggregation rules.
- `projects/mmdet3d_plugin/models/dense_heads/maestro_bev_seg_head.py`
  - Added BEVFusion-style sigmoid focal map loss.
- `projects/mmdet3d_plugin/models/detectors/__init__.py`
  - Registers `MAESTRO2Task`.
- `projects/mmdet3d_plugin/models/OccHead/__init__.py`
  - Registers `MAESTROOccFormerHead`.

## Verification

Done in the `mapocc` environment:

```bash
python -m py_compile \
  projects/mmdet3d_plugin/models/model_utils/maestro_cpg.py \
  projects/mmdet3d_plugin/models/model_utils/maestro_tsfg.py \
  projects/mmdet3d_plugin/models/model_utils/maestro_spa.py \
  projects/mmdet3d_plugin/models/detectors/MAESTRO2Task.py \
  projects/mmdet3d_plugin/models/OccHead/maestro_occformer_head.py \
  projects/mmdet3d_plugin/models/dense_heads/maestro_bev_seg_head.py \
  projects/configs/MAESTRO/MAESTRO_2task_lss_occformer.py
```

Model registry/build passed:

```text
MAESTRO2Task
MAESTROClasswisePrototypeGenerator MAESTROOccFormerHead MAESTROBEVSegHead
```

Dataset build and first training sample pipeline passed:

```text
NuScenesDatasetMultitask 28130
['gt_depth', 'gt_masks_bev', 'img_inputs', 'img_metas', 'mask_camera', 'mask_lidar', 'sa_gt_depth', 'sa_gt_semantic', 'voxel_semantics']
```

CPU smoke for the train-time MAESTRO path passed on a reduced grid:

```text
smoke-ok (1, 96, 16, 16, 8) (1, 96, 16, 16) (1, 6, 16, 16)
['d0.loss_occformer_cls', 'd0.loss_occformer_dice', 'd0.loss_occformer_mask', 'loss_maestro_cpg_ce', 'loss_maestro_map_supp', 'loss_maestro_occ_supp', 'loss_map_focal', 'loss_occformer_cls', 'loss_occformer_dice', 'loss_occformer_mask']
```

This environment reports no CUDA runtime, so I verified the model/dataset build and CPU train-graph path instead of launching full GPU training here.

## Training Command

Single-GPU training:

```bash
conda run -n mapocc bash tools/dist_train.sh \
  projects/configs/MAESTRO/MAESTRO_2task_lss_occformer.py 1
```

If memory is tight, reduce the OccFormer query/head cost first:

```bash
conda run -n mapocc bash tools/dist_train.sh \
  projects/configs/MAESTRO/MAESTRO_2task_lss_occformer.py 1 \
  --cfg-options model.occ_head.transformer_decoder.num_layers=6 \
  model.occ_head.train_cfg.num_points=25088
```
