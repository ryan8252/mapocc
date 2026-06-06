ˊProtoOcc BEV Map Segmentation Status Brief for GPT-5.5 Pro

Last updated: 2026-06-06

This document is meant to be pasted into an online GPT-5.5 Pro session.
Please use it as the experiment evidence summary, then inspect the GitHub repo
for exact code and config details.

## Repository

- GitHub repo to inspect: https://github.com/ryan8252/mapocc
- Local git remote: `git@github-ryan:ryan8252/mapocc.git`
- Local checkout root used for these results: `/home/robot/Desktop/Ryan/ProtoOcc`
- Current local branch: `aaaaaaaaaaaaaaa`
- Current local commit: `98351d3`

Important caveat: many `work_dirs/` result files may not be committed to
GitHub. Treat this markdown, `experiment_results_summary.md`, and
`bev_map_segmentation_improvement_plan_for_codex.md` as the experiment evidence.
Use GitHub mainly to inspect model code, config inheritance, loaders, losses,
and launch scripts.

If the GitHub repo is private or the branch is not pushed, ask the user to
grant access or paste the relevant files.

## What We Are Trying To Do

We are modifying ProtoOcc for camera-only nuScenes 3D occupancy plus BEV HD map
segmentation. The key target is to improve BEV map segmentation without
changing the image backbone and without adding temporal modeling.

The important BEV map classes are:

- `drivable_area`
- `ped_crossing`
- `walkway`
- `stop_line`
- `carpark_area`
- `divider`

The hard classes are mainly thin or sparse map classes:

- `ped_crossing`
- `stop_line`
- `divider`

We need to compare against BEVFusion and MAESTRO, so there are two different
protocols that must not be mixed.

## Protocols

### Native ProtoOcc Map Protocol

Native ProtoOcc map experiments use approximately:

- map range: `[-40, 40] x [-40, 40]`
- map resolution: `0.4m`
- map raster size: `200 x 200`
- occupancy supervision: also centered on the `[-40, 40]` region

This is the protocol where the current local branch is strongest.

### BEVFusion / MAESTRO Aligned Map Protocol

The aligned branch tries to match BEVFusion / MAESTRO map evaluation:

- shared BEV feature canvas: `[-51.2, 51.2] x/y`, `0.4m`, so `256 x 256`
- occupancy supervision crop: `[-40, 40] x/y`, `0.4m`, so `200 x 200 x 16`
- map supervision/evaluation: `[-50, 50] x/y`, `0.5m`, so `200 x 200`

In our aligned config, occupancy is intentionally kept on the Occ3D-style
`[-40, 40]` crop while the map grid is widened to BEVFusion's `[-50, 50] / 0.5m`
protocol.

Key aligned config:

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned.py
```

Important warning: native `48.x` map mIoU and aligned `44.x` map mIoU are not
directly comparable as identical metrics. The aligned protocol is harder or at
least materially different because the physical map extent and cell size differ.

## BEVFusion Map GT Parity Audit Result

A dedicated audit script was added:

```text
ProtoOcc/tools/debug_compare_bevfusion_map_gt.py
```

It compares 100 val samples under test-mode/no-augmentation assumptions:

```bash
conda run -n ProtoOcc python tools/debug_compare_bevfusion_map_gt.py --num-samples 100
```

Report:

```text
ProtoOcc/work_dirs/debug_bevfusion_map_gt_parity/result.md
```

Main finding:

```text
ProtoOcc label vs BEVFusion-exact label:
identity mean IoU: 0.023211
best transform: transpose_xy_flip_x
best transform mean IoU: 0.984422
```

When using BEVFusion axes but ProtoOcc's drivable-area layer mapping, the same
transform reaches exact agreement:

```text
ProtoOcc label vs BEVFusion axes + ProtoOcc layers:
identity mean IoU: 0.022847
best transform: transpose_xy_flip_x
best transform mean IoU: 1.000000
```

This means the current ProtoOcc aligned GT tensor is not identity-equal to
BEVFusion's original GT tensor convention. The difference is mostly axis
orientation:

```text
bevfusion_axes_proto_layers == protoocc_label.transpose(0, 2, 1)[:, ::-1, :]
```

There is also a smaller semantic-layer mismatch for `drivable_area`:

- BEVFusion default config uses `drivable_area`, and BEVFusion's loader maps it
  to the nuScenes `drivable_area` layer.
- ProtoOcc's loader maps both `drivable_area` and `drivable_area*` to
  `road_segment + lane`.

This is now a real parity concern to investigate before spending more runs on
loss/module tuning. However, do not blindly flip the GT without checking the
model's BEV feature coordinate convention. ProtoOcc may be internally
self-consistent with its occupancy/grid convention, even if it is not
BEVFusion-tensor-identical.

## Feature Convention Probe Result

A second probe was added to check whether a trained aligned map-only model's
prediction follows the current ProtoOcc GT convention or the BEVFusion-exact GT
convention:

```text
ProtoOcc/tools/debug_feature_convention_probe.py
```

Command:

```bash
conda run -n ProtoOcc python tools/debug_feature_convention_probe.py --sample-index 0
```

Report and images:

```text
ProtoOcc/work_dirs/debug_feature_convention_probe/result.md
ProtoOcc/work_dirs/debug_feature_convention_probe/drivable_area.png
ProtoOcc/work_dirs/debug_feature_convention_probe/walkway.png
ProtoOcc/work_dirs/debug_feature_convention_probe/divider.png
ProtoOcc/work_dirs/debug_feature_convention_probe/feature_norm.png
```

Model/checkpoint used:

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only.py
work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_nano4_h200_2/epoch_24_ema.pth
```

For sample index `0`, the prediction clearly follows the current ProtoOcc GT
orientation, not the BEVFusion-exact orientation:

| class | label | pearson | best_iou | iou@0.5 |
| --- | --- | ---: | ---: | ---: |
| drivable_area | current_protoocc | 0.853963 | 0.788725 | 0.769418 |
| drivable_area | bevfusion_exact | -0.209551 | 0.074806 | 0.066455 |
| walkway | current_protoocc | 0.439409 | 0.339565 | 0.238691 |
| walkway | bevfusion_exact | -0.098281 | 0.027693 | 0.019276 |
| divider | current_protoocc | 0.448043 | 0.305188 | 0.238671 |
| divider | bevfusion_exact | -0.024772 | 0.031850 | 0.026283 |

Interpretation:

- Current ProtoOcc loader and current model output are internally aligned.
- Directly changing the GT tensor to BEVFusion axes without changing the model
  output coordinate convention would likely break the feature-target alignment.
- The right next step is not a blind loader flip. It is a controlled BEVFusion
  parity ablation, ideally with both the GT convention and the map feature/output
  convention handled together.

## External Reference Numbers

From the user's MAESTRO paper/supplement notes:

| Method | Backbone | Occ mIoU | Map mIoU | Notes |
| --- | ---: | ---: | ---: | --- |
| BEVFusion | R50 | n/a | 47.10 | MAESTRO supplemental Table 2, user-recorded |
| MAESTRO | R50 | 38.60 | 51.30 | MAESTRO supplemental Table 2 / Table 4, user-recorded |
| MAESTRO Baseline-STL | R50 | n/a | 47.5 | user says paper reports this |
| MAESTRO Baseline-MTL | R50 | n/a | 43.5 | user says paper reports this |

Please verify the MAESTRO Baseline-STL / Baseline-MTL recipe in the paper if
the PDFs are uploaded to the online GPT session. The local files are:

```text
/home/robot/Desktop/Ryan/MAESTRO.pdf
/home/robot/Desktop/Ryan/MAESTRO_supplemental.pdf
```

The online GPT session cannot see these local PDFs unless the user uploads them.

## Strong Local Native Results

Source summary file:

```text
experiment_results_summary.md
```

The key native-protocol results are:

| Experiment | Config | Occ mIoU | Map mIoU |
| --- | --- | ---: | ---: |
| Naive MTL CNN map head | `ProtoOcc_multi_cnn_head.py` | 32.54 | 16.13 |
| CNN map head + 128ch map neck, weight1 | `ProtoOcc_multi_cnn_head_map_neck.py` | 39.82 | 39.94 |
| CNN map head + 128ch map neck, `map_loss_weight=4` | `ProtoOcc_multi_cnn_head_map_neck.py` | 39.72 | 45.79 |
| Overlay dynamic V3-lite | `ProtoOcc_multi_cnn_head_map_neck_overlay_dynamic.py` | 39.60 | 46.38 |
| Map-only upper bound | `ProtoOcc_multi_cnn_head_map_neck_map_only.py` | n/a | 48.34 |
| Current strong native MTL stack | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py` | 39.58 | 48.03 |

The most important observation: native MTL with the current strong stack reaches
`Map 48.03`, very close to native map-only `48.34`. This means the map branch
is not fundamentally broken on the native grid. The remaining problem is mainly
how to preserve that strength under the BEVFusion / MAESTRO aligned protocol.

Classwise for the current strong native MTL stack:

```text
work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200/result.md
Occ mIoU: 39.58
Map mean iou@max: 0.480266
ped_crossing: 0.440598
stop_line: 0.292826
divider: 0.372603
drivable_area: 0.788812
walkway: 0.515463
carpark_area: 0.471296
```

## Aligned Results So Far

The aligned protocol is the main current concern.

| Experiment | Work dir / result | Occ mIoU | Map mIoU |
| --- | --- | ---: | ---: |
| Aligned map-only plain, full Nano4 | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_nano4_h200_2/result.md` | n/a | 44.42 |
| Aligned map-only BCE+Dice, full Nano4 | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_bce_dice_nano4_h200/result.md` | n/a | 43.49 |
| Aligned MTL focal weight128, full Nano4 | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_focal_weight128_nano4_h200/result.md` | 39.75 | 41.70 |
| Aligned map-only classwise BEVFusion focal, 1epoch smoke | `work_dirs/smoke_bevfusion_aligned_map_only_classwise_bevfusion_focal_1quarter_4090/result.md` | n/a | 13.43 |

Classwise for aligned map-only full Nano4:

```text
drivable_area: 0.767489
ped_crossing: 0.383056
walkway: 0.478610
stop_line: 0.276680
carpark_area: 0.426001
divider: 0.333527
mean: 0.444227
```

Classwise for aligned MTL focal weight128 full Nano4:

```text
Occ mIoU: 39.75
drivable_area: 0.757009
ped_crossing: 0.339792
walkway: 0.456835
stop_line: 0.228363
carpark_area: 0.401865
divider: 0.317959
mean: 0.416970
```

The aligned full-run MTL gap is roughly:

```text
aligned map-only: 44.42
aligned MTL focal weight128: 41.70
gap: -2.72 map mIoU
```

This is similar in spirit to the native map-only vs MTL gap after strong map
loss balancing, but the aligned map-only ceiling itself is lower than expected.

## What Has Clearly Helped

The local evidence supports these components:

1. `128ch map neck`
   - This is the biggest architectural improvement over naive map head.
   - It is the protected baseline.

2. `map_loss_weight=4`
   - Native MTL improves from `Map 39.94` to `Map 45.79` with only a small Occ
     drop.
   - This suggests the first bottleneck was task-level map suppression.

3. FPN lateral projection + ASPP
   - Useful feature-path improvement in 1epoch smoke.

4. `focal_dice` map loss with static class weights
   - Useful loss-side improvement, especially for thin/rare map classes.

5. HFM adapter + per-scale residual adapter + FPN lateral ASPP + focal-Dice
   weighted loss + active enhance gate
   - Best current native full result: `Occ 39.58 / Map 48.03`.

## What Has Not Been Worth Continuing

These have either failed, only produced noise-level gains, or are dominated by
the current strong branch:

- ProtoMapHead / prototype map head
- PQD-align map prototype variants
- GT-soft prototype mining
- 256ch map neck
- high-resolution skip variants
- Z-aware compression variants
- pre-backbone adapter
- BEV CoordConv XY
- dual area-line map head
- auxiliary map supervision alone
- stronger BEVSegHead variants such as `res_refine` and `convnext`
- residual gates as currently implemented
- thin ROI residual refinement as currently implemented
- BEVFusion decoder clone
- classwise BEVFusion focal in 1epoch smoke

The active-gate A0-A5 Nano4 smoke batch also did not reveal a big new lever.
Only the thin boundary dilation variant had a very small directionally positive
signal. Do not overfit to these 1epoch smoke differences.

## Current Hypothesis

The main problem is no longer "we need another map module."

The stronger hypothesis is:

1. On the native `[-40,40]/0.4` protocol, the current MTL branch almost reaches
   the native map-only upper bound.
2. The BEVFusion / MAESTRO aligned protocol introduces a meaningful penalty:
   wider physical extent, different map cell size, and possibly different
   rasterization / loss / training-recipe behavior.
3. 1epoch 1quarter smoke tests are poor predictors for aligned full-run map
   quality. They are useful mainly for detecting collapse or runtime errors.
4. To reach MAESTRO Baseline-STL `47.5`, we probably need to lift the aligned
   map-only ceiling first, not add more MTL fusion modules.

## Current Run To Watch

The user is considering or running:

```bash
LR=4e-4 sbatch --export=ALL TWCC_nano4/train_eval_record_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_nano4.sh
```

Launcher:

```text
TWCC_nano4/train_eval_record_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_nano4.sh
```

Config:

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned.py
```

My rough expectation before seeing the result:

```text
Occ: 39.3 to 39.7
Map: 44.0 to 45.5
```

If this run gets `Map >= 46`, that is a strong positive result. If it gets
`43.x`, do not immediately abandon the architecture. First test the same aligned
strong config with `LR=2e-4`, because the launcher default was intentionally
`2e-4` for aligned-grid stability.

## Suggested Next Experiments

Please prioritize experiments that can plausibly raise the aligned map-only or
aligned strong-MTL ceiling. Do not propose changing the image backbone or adding
temporal modeling.

Recommended order:

1. Finish the aligned strong MTL full run above.
2. If `LR=4e-4` is weak, repeat with `LR=2e-4`.
3. Build an aligned map-only version of the current native strong stack:
   HFM adapter + per-scale adapter + FPN lateral ASPP + focal-Dice weighted loss
   + active enhance gate, but map-only.
4. Try warm-starting aligned training from the best native strong checkpoint,
   then fine-tune on the aligned map grid.
5. Try a two-stage schedule:
   - stage 1: map-only or map-heavy warmup
   - stage 2: MTL fine-tuning with map loss kept strong
6. Investigate map-aware sampling or class-aware scene sampling for
   `ped_crossing`, `stop_line`, and `divider`.
7. Investigate probability calibration / threshold behavior per class. Several
   failures show thin classes surviving only at low thresholds.

## Questions For GPT-5.5 Pro

Please inspect the GitHub repo and answer these with concrete code/config
references:

1. Is the aligned config really matching BEVFusion / MAESTRO map protocol, or is
   there a subtle mismatch in `LoadBEVSegmentation`, x/y bounds, resizing,
   feature alignment, augmentation, or evaluation?
2. How exactly does BEVFusion train camera-only map segmentation in this repo?
   Compare its loss, decoder, `BEVGridTransform`, depth supervision, optimizer,
   batch size, and dataset wrapping against our aligned ProtoOcc config.
3. Is MAESTRO Baseline-STL `47.5` likely using special MAESTRO modules, or is it
   closer to a plain single-task BEV map segmentation baseline with a specific
   recipe?
4. Is the current aligned map-only `44.42` unexpectedly low given the model
   capacity, or is it explainable by protocol and training recipe?
5. What is the highest-value non-backbone, non-temporal ablation to try next?
   Please avoid generic suggestions. Give a short ranked list with expected
   effect, risk, and exact config/code touch points.
6. Should depth loss be changed to match BEVFusion/MAESTRO more closely, or is
   the current depth loss unlikely to explain the map-only aligned gap?

## Files To Inspect In The Repo

Experiment summaries:

```text
ProtoOcc/experiment_results_summary.md
ProtoOcc/bev_map_segmentation_improvement_plan_for_codex.md
```

Key configs:

```text
ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py
ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_map_only.py
ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only.py
ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_focal_weight128.py
ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted.py
ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted.py
ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py
ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned.py
```

Key code paths:

```text
ProtoOcc/projects/mmdet3d_plugin/datasets/pipelines/loading_bev_seg.py
ProtoOcc/projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py
ProtoOcc/projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py
ProtoOcc/projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py
ProtoOcc/projects/mmdet3d_plugin/models/detectors/ProtoOccMapOnly.py
```

Launcher:

```text
ProtoOcc/TWCC_nano4/train_eval_record_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_nano4.sh
```

## Bottom Line

Do not treat this project as doomed because many small modules failed. The
native strong MTL branch already reached `Map 48.03`, almost equal to the native
map-only upper bound `48.34`.

The real unsolved problem is narrower:

```text
How do we transfer the native strong map branch to the BEVFusion / MAESTRO
aligned map protocol and lift aligned map-only from 44.42 toward 47.5?
```

Any useful suggestion should focus on that question.
