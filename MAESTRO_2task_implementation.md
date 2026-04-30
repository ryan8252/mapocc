# MAESTRO 2-task implementation for ProtoOcc

## Goal

This implementation adapts MAESTRO to the current ProtoOcc codebase for two tasks:

- 3D occupancy prediction
- BEV map segmentation

The original MAESTRO paper uses three tasks: detection, map segmentation, and occupancy. Since this ProtoOcc branch already has the occupancy pipeline and a multitask map-GT data path, this version keeps ProtoOcc's DBE/PQD occupancy stack and adds MAESTRO-style task-specific feature generation for occupancy and map segmentation.

## Paper Mapping

MAESTRO components are mapped as follows:

- CPG: `MAESTROClasswisePrototypeGenerator` predicts class confidence on the shared DBE voxel feature and pools class-wise prototypes.
- TSFG: `MAESTROTaskSpecificFeatureGenerator` creates separate task features.
  - map TSFG uses background prototypes and collapses voxel height into a BEV feature.
  - occupancy TSFG uses foreground plus background/scene prototypes and keeps the voxel feature layout.
- SPA-oneway: `MAESTROOnewayScenePrototypeAggregator` lets map prototypes enrich occupancy prototypes only. `detach_source=True` keeps occupancy gradients from rewriting the map branch through SPA.

Foreground/background groups follow the nuScenes occupancy class ids:

- foreground/occupancy object side: `0..10`, including `others` so occupancy still has a prototype route for class 0.
- background/map side: `11..16`, matching drivable surface, other flat, sidewalk, terrain, manmade, vegetation.
- free class `17` is excluded from prototype grouping and used as the occupancy non-object label.

## Design

The new detector is `ProtoOccMAESTRO2Task`.

The forward path is:

1. Existing ProtoOcc image encoder, depth net, LSS view transformer.
2. Existing `Dual_Branch_Encoder` produces the shared voxel feature.
3. CPG generates foreground/background prototypes from the shared feature.
4. Map TSFG generates a BEV map feature from background prototypes.
5. `MAESTROBEVSegHead` predicts six BEV map masks.
6. SPA-oneway aggregates map-task prototypes into the background prototype group.
7. Occupancy TSFG generates an occupancy-specific voxel feature from foreground + SPA-enhanced background prototypes.
8. Existing `cnn3d_decoder` and `Prototype_Query_Decoder_nuScenes` run on the occupancy-specific feature.

This avoids changing the existing ProtoOcc detector, DBE, CNN decoder, and PQD source files.

## Files Added

- `projects/mmdet3d_plugin/models/model_utils/maestro_cpg.py`
  - CPG module and optional CPG cross-entropy auxiliary loss.
- `projects/mmdet3d_plugin/models/model_utils/maestro_tsfg.py`
  - Shared TSFG implementation for map and occupancy.
  - Includes prototype-wise features, prototype-aware channel gating, and optional feature-suppression loss.
- `projects/mmdet3d_plugin/models/model_utils/maestro_spa.py`
  - One-way map-to-occupancy SPA.
- `projects/mmdet3d_plugin/models/dense_heads/maestro_bev_seg_head.py`
  - Lightweight BEV map segmentation head.
- `projects/mmdet3d_plugin/models/detectors/ProtoOccMAESTRO2Task.py`
  - Detector wrapper that inserts MAESTRO modules after DBE and before existing occupancy decoders.
- `projects/configs/ProtoOcc/ProtoOcc_maestro_2task.py`
  - Standalone config. It does not inherit from `ProtoOcc_bevfusion_mapgt_protoocc_range.py` or `ProtoOcc_1key.py`.
  - Defines the ProtoOcc base model, multitask occupancy/map dataset pipeline, MAESTRO CPG/TSFG/SPA modules, map head, optimizer, runtime, and evaluation settings in one file.

## Files Modified

- `projects/mmdet3d_plugin/models/__init__.py`
  - Imports `model_utils` so MAESTRO modules are registered.
- `projects/mmdet3d_plugin/models/model_utils/__init__.py`
  - Exposes CPG, TSFG, and SPA.
- `projects/mmdet3d_plugin/models/dense_heads/__init__.py`
  - Registers `MAESTROBEVSegHead`.
- `projects/mmdet3d_plugin/models/detectors/__init__.py`
  - Registers `ProtoOccMAESTRO2Task`.

## Why New Files

The user request was to avoid modifying existing files when possible. The implementation therefore:

- does not edit `ProtoOcc.py`;
- does not edit `dual_branch_encoder.py`;
- does not edit `Prototype_Query_Decoder_nuScenes.py`;
- keeps the existing occupancy decoder behavior intact;
- uses small registry-only edits in `__init__.py` files.

## Verification

Completed:

- `python -m py_compile` on all new Python files.
- `python -m py_compile` on `projects/configs/ProtoOcc/ProtoOcc_maestro_2task.py`.

Blocked in this shell:

- Full registry/model import test failed because the active Python environment does not have `mmcv` installed:
  - `ModuleNotFoundError: No module named 'mmcv'`

This is an environment dependency issue, not a syntax error in the new files.

## Suggested Run Command

Use the normal ProtoOcc training entrypoint with:

```bash
projects/configs/ProtoOcc/ProtoOcc_maestro_2task.py
```

The config writes to:

```bash
work_dirs/ProtoOcc_maestro_2task
```
