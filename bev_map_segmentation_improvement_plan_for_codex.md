# BEV Map Segmentation Improvement Plan for Codex

## 0. Context

Current map segmentation pipeline:

```text
LSS voxel feature
(B, 80, 16, 200, 200)
  -> Z flatten / concat
(B, 1280, 200, 200)
  -> down_sample_for_3d_pooling: 1x1 Conv2d 1280 -> 160
(B, 160, 200, 200)
  -> CustomBEVBackbone: 2D BEV pyramid
[
  level 0: (B, 160, 100, 100),
  level 1: (B, 320, 50, 50),
  level 2: (B, 640, 25, 25)
]
  -> map_bev_encoder_neck / Custom_FPN_LSS
(B, 128, 200, 200)
  -> BEVSegHead
(B, 6, 200, 200)
```

Current map-specific neck:

```text
x3 = level 0: (B, 160, 100, 100)
x2 = level 1: (B, 320, 50, 50)
x1 = level 2: (B, 640, 25, 25)

x1: 640, 25, 25
  -> upsample to 50, 50
  -> concat x2
  => (B, 960, 50, 50)
  -> cat_conv1
  => (B, 256, 50, 50)

  -> upsample to 100, 100
  -> concat x3
  => (B, 416, 100, 100)
  -> cat_conv2
  => (B, 256, 100, 100)

  -> upsample to 200, 200
  -> output conv
  => (B, 128, 200, 200)
```

Current map head:

```text
BEVSegHead:
(B, 128, 200, 200)
  -> hidden 128
  -> 6 logits
(B, 6, 200, 200)
```

The goal is to improve raster BEV map segmentation, especially small and thin classes such as:

```text
pedestrian crossing
lane divider
road divider / divider
stop line
boundary
```

This plan contains 9 possible improvements. Implement them in ablation-friendly fashion with config flags. Do not implement all changes as one inseparable block. Each change should be individually switchable.

## Codex Pre-Implementation Overlap Check - 2026-06-01

Checked before implementing Sections 1-3:

- Section 1 had a partial prior implementation: `map_highres_skip=True` in `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_highres.py`. It was add-only, not `gated` or `concat`. Recorded smoke result: `work_dirs/smoke_map_neck_highres_weight4_1quarter_4090/epoch_1_ema.pth`, Occ mIoU `26.52`, Map mIoU `16.30`, `stop_line 0.30`, `divider 9.28`. No 24-epoch full result was found locally.
- Section 2 exact implementation was not found. Existing `MapZResidualLayer` and configs `ProtoOcc_multi_cnn_head_map_neck_zlite_residual_catz.py` / `ProtoOcc_multi_cnn_head_map_neck_zaware_residual_k4.py` add a zero-init residual after the 128ch map neck, so they are related but not the same as replacing `cat-Z + 1x1` before `CustomBEVBackbone`. No local `result.md` was found for those two configs.
- Section 3 exact implementation was not found. Existing `PerScaleMapResidualAdapter` in `ProtoOcc_multi_cnn_head_map_neck_adapter.py` runs after `CustomBEVBackbone` on the BEV pyramid before the 128ch map neck, not on the `200x200` `x0` feature before `CustomBEVBackbone`. Prior full result: `work_dirs/ProtoOcc_multi_cnn_head_map_neck_adapter/epoch_24_ema.pth`, Occ mIoU `39.64`, Map mIoU `46.74`, `ped_crossing 42.88`, `stop_line 28.62`, `divider 34.19`.
- Clean reference baseline: `work_dirs/ProtoOcc_multi_cnn_head_map_neck_TWCC/epoch_24_ema.pth` with `map_loss_weight=4`, Occ mIoU `39.72`, Map mIoU `45.79`.

Implemented after the check:

- Section 1: `Dual_Branch_Encoder` now supports `map_highres_fusion='add'|'gated'|'concat'`. Existing `ProtoOcc_multi_cnn_head_map_neck_highres.py` remains add-fusion; new gated ablation is `ProtoOcc_multi_cnn_head_map_neck_highres_gated.py`.
- Section 2: `Dual_Branch_Encoder` now supports `use_map_z_aware_compression=True` with `map_z_compression_type='conv3d'` or `'height_attention'`. New configs: `ProtoOcc_multi_cnn_head_map_neck_zaware_compression_conv3d.py` and `ProtoOcc_multi_cnn_head_map_neck_zaware_compression_heightattn_k4.py`.
- Section 3: `Dual_Branch_Encoder` now supports a `200x200` pre-backbone map adapter via `use_map_pre_backbone_adapter=True` or the plan alias `use_map_adapter=True`. New config: `ProtoOcc_multi_cnn_head_map_neck_pre_adapter.py`.
- Composition smoke config: `ProtoOcc_multi_cnn_head_map_neck_stage123.py`.
- Validation: `python -m py_compile` passed for the modified encoder and new configs; `conda run -n mapocc` encoder forward smoke passed for all five new configs with output shapes `(1,48,16,16,16)`, `(1,48,16,16)`, `(1,128,16,16)` on synthetic `[1,80,16,16,16]` input.

## Codex Pre-Implementation Overlap Check for Sections 4-6 - 2026-06-02

Checked before implementing Sections 4-6:

- Section 4 exact implementation was not found. Existing `Custom_FPN_LSS` still directly concatenates raw pyramid levels (`640+320`, then `256+160`) without same-channel lateral projections, and no `use_fpn_lateral_projection`, `use_fpn_global_context`, ASPP, or PPM flag was found. Related but different prior modules exist: `MapHFMFusionLayer` injects Z-collapsed voxel context before the map neck, and `PerScaleMapResidualAdapter` refines each BEV scale before the map neck. They do not replace the internal FPN scale-fusion rule. Related result: `work_dirs/ProtoOcc_multi_cnn_head_map_neck_adapter/epoch_24_ema.pth`, Occ mIoU `39.64`, Map mIoU `46.74`, `ped_crossing 42.88`, `stop_line 28.62`, `divider 34.19`.
- Section 5 exact implementation was not found. `BEVSegHead` was still the shallow `num_convs` stack plus one `1x1` predictor, without `bevseg_head_type='res_refine'|'convnext'` or residual refinement blocks. Existing `ProtoMapHead` / `ProtoMapHeadV2` configs are related map-head replacements, not the same as strengthening the current `BEVSegHead`.
- Section 6 exact implementation was not found. No current `BEVSegHead` split area-like and line-like map classes into separate decoders with configurable `area_class_indices` / `line_class_indices`, and no result was found for an area-line dual-head ablation.
- Clean reference baseline for Sections 4-6 remains `work_dirs/ProtoOcc_multi_cnn_head_map_neck_TWCC/epoch_24_ema.pth` with `map_loss_weight=4`, Occ mIoU `39.72`, Map mIoU `45.79`.

Implemented after the check:

- Section 4: `Custom_FPN_LSS` now supports `use_fpn_lateral_projection=True`, `fpn_lateral_in_channels`, `fpn_projection_channels`, `use_fpn_global_context=True`, and `fpn_global_context_type='aspp'|'ppm'`. New configs: `ProtoOcc_multi_cnn_head_map_neck_fpn_lateral.py` and `ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp.py`.
- Section 5: `BEVSegHead` now supports `bevseg_head_type='simple'|'res_refine'|'convnext'` and `bevseg_num_refine_blocks`. The default `simple` path keeps the old `decoder + predictor` structure. New configs: `ProtoOcc_multi_cnn_head_map_neck_bevseg_res_refine.py` and `ProtoOcc_multi_cnn_head_map_neck_bevseg_convnext.py`.
- Section 6: `BEVSegHead` now supports `use_dual_area_line_head=True` with configurable `area_class_indices`, `line_class_indices`, and `line_head_use_detail`. The final output remains `(B, num_map_classes, 200, 200)` with logits restored to original class order. New config: `ProtoOcc_multi_cnn_head_map_neck_dual_area_line_head.py`.
- Composition smoke config: `ProtoOcc_multi_cnn_head_map_neck_stage456.py`. Use the single-section configs first for attribution; stage456 is only for combined-path sanity checks.
- Validation: `python -m py_compile` passed for the modified neck/head and six new configs; `git diff --check` passed; `conda run -n mapocc` config/build/forward smoke passed for all six new configs with neck output `(1,128,200,200)` and logits `(1,6,200,200)` on synthetic inputs. The original `ProtoOcc_multi_cnn_head_map_neck.py` baseline config also passed the same shape smoke.

## Codex Pre-Implementation Overlap Check for Sections 7-9 - 2026-06-02

Checked before implementing Sections 7-9:

- Section 7 exact implementation was not found. `Custom_FPN_LSS` returned the final `128ch` map feature only and did not expose the internal `100x100` / `50x50` map features for auxiliary map supervision.
- Section 8 had partial prior support: `BEVSegHead` already accepted explicit `loss_bce`, `loss_dice`, and `loss_focal` configs, and also had a `class_static` balance mode. Existing focal configs were focal-only variants, not the requested `map_loss_type='focal_dice'` interface with optional static class weights.
- Section 9 exact implementation was not found. No `use_bev_coordconv` / `bev_coord_type` flag or BEV coordinate projection was present in `BEVSegHead`.

Implemented after the check:

- Section 7: `Custom_FPN_LSS` now supports `return_map_aux_features=True` and returns `aux_features['100']` / `aux_features['50']` from the internal FPN fusion stages. `BEVSegHead` now supports `use_map_aux_loss=True`, `map_aux_levels`, `map_aux_weight_100`, `map_aux_weight_50`, and max-pool GT downsampling. New config: `ProtoOcc_multi_cnn_head_map_neck_aux_loss.py`.
- Section 8: `BEVSegHead` now supports `map_loss_type='bce'|'bce_dice'|'focal'|'focal_dice'|'focal_lovasz'`, `map_dice_weight`, `map_focal_gamma`, `map_focal_alpha`, `map_lovasz_weight`, `use_map_class_weights`, and `map_class_weights`. Class weights apply to BCE/Focal per-pixel losses and Dice/Lovasz per-class losses. New config: `ProtoOcc_multi_cnn_head_map_neck_focal_dice_weighted.py`.
- Section 9: `BEVSegHead` now supports `use_bev_coordconv=True` with `bev_coord_type='xy'|'xyr'|'xyrtheta'`, concatenates dynamic BEV coordinate channels, and projects back to the original map feature channel count before the decoder. New config: `ProtoOcc_multi_cnn_head_map_neck_coordconv_xy.py`.
- Composition smoke config: `ProtoOcc_multi_cnn_head_map_neck_stage789.py`. Use the single-section configs first for clean attribution; stage789 is only for combined-path sanity checks.
- Validation: `python -m py_compile` passed for the modified encoder/neck/head/detector files and four new configs; `conda run -n mapocc` neck/head synthetic forward + loss smoke passed for all four new configs with final logits `(1,6,200,200)`; `conda run -n mapocc` full `Dual_Branch_Encoder` small-input forward passed with outputs `(1,48,16,16,16)`, `(1,48,16,16)`, `(1,128,16,16)`.

## Empirical Usefulness Summary - 2026-06-03

Usefulness here means there is local evidence under the same 1-epoch /
1-quarter nuScenes smoke protocol unless explicitly marked as a 24-epoch
result. All smoke evals use the full val set (`6019` samples) and the EMA
checkpoint when available.

### Methods that are useful for our current branch

| Plan section | Method / config | Evidence | Decision |
| --- | --- | --- | --- |
| Section 8 | `focal_dice` map loss with static rare/thin class weights, `ProtoOcc_multi_cnn_head_map_neck_focal_dice_weighted.py` | Smoke rank 2: Map mean iou@max `0.191608`, OCC mIoU `27.49`, thin classes became non-zero: `ped_crossing=0.062106`, `stop_line=0.078230`, `divider=0.146944`. | Useful. Keep this as the default loss-side change for thin map classes. |
| Section 4 | FPN lateral projection + ASPP global context, `ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp.py` | Smoke rank 3 from the raw eval dict: Map mean iou@max `0.187836`, OCC mIoU `25.56`. This improved the feature path, but by itself still left `ped_crossing=0.0` and weak stop-line IoU. | Useful as a feature-side change, especially when paired with Section 8. Not sufficient alone for thin classes. |
| Sections 4 + 8 | `ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted.py` | Smoke rank 1: Map mean iou@max `0.220636`, OCC mIoU `26.61`, `ped_crossing=0.068804`, `stop_line=0.090866`, `divider=0.173588`. It beats Section 8 alone (`0.191608`) and Section 4 alone (`0.187836`). | Strongest current candidate. Use this for the next longer run / TWCC nano4 run. |
| Related to Section 3 | Existing per-scale map residual adapter, `ProtoOcc_multi_cnn_head_map_neck_adapter.py` | 24-epoch full run was better than the clean TWCC baseline: Map mIoU `46.74` vs baseline `45.79`, with `ped_crossing=42.88`, `stop_line=28.62`, `divider=34.19`. The 1-epoch smoke was only `0.167216`, so this seems training-length dependent. | Useful long-run signal, but do not treat the 1-epoch smoke as proof. Consider as a later add-on after the Section 4 + 8 candidate is validated for more epochs. |

### Methods not yet useful as standalone smoke changes

| Plan section | Method / config | Smoke result | Decision |
| --- | --- | --- | --- |
| Section 7 | Auxiliary map supervision, `ProtoOcc_multi_cnn_head_map_neck_aux_loss.py` | Map mean iou@max `0.171397`; thin classes mostly remained weak (`ped_crossing=0.0`, `stop_line=0.001920`, `divider=0.103686`). | Not a current mainline change. Revisit only in combination with stronger loss/feature settings. |
| Section 6 | Dual area/line head, `ProtoOcc_multi_cnn_head_map_neck_dual_area_line_head.py` | Map mean iou@max `0.167525`; `ped_crossing=0.0`, `stop_line=0.015571`, `divider=0.097174`. | Not useful standalone. |
| Section 9 | BEV CoordConv XY, `ProtoOcc_multi_cnn_head_map_neck_coordconv_xy.py` | Map mean iou@max `0.166692`; `ped_crossing=0.0`, `stop_line=0.007365`, `divider=0.089257`. | Not useful standalone. |
| Section 5 | Stronger BEVSegHead (`res_refine`, `convnext`) | `res_refine`: `0.161818`; `convnext`: `0.154764`; thin classes remained weak. | Not useful standalone in this branch. |
| Section 1 | High-resolution gated skip, `ProtoOcc_multi_cnn_head_map_neck_highres_gated.py` | Map mean iou@max `0.161200`; thin classes weak. | Not useful standalone. |
| Section 2 | Z-aware compression (`conv3d`, `height_attention`) | `conv3d`: `0.131500`; `heightattn_k4`: `0.076000`. | Avoid for now; it likely damages the map/OCC feature path under this setup. |
| Section 3 exact variant | Pre-backbone 200x200 adapter, `ProtoOcc_multi_cnn_head_map_neck_pre_adapter.py` | Map mean iou@max `0.154400`. | Not useful standalone, despite the related per-scale adapter having long-run evidence. |

### Current recommendation

Do not stack all nine plan ideas. The most defensible current path is:

```text
128ch map neck + FPN lateral projection + ASPP + focal_dice weighted map loss
```

Use `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted.py`
as the next main candidate. If this survives a longer run, the next add-on to
test should be the related per-scale adapter from `ProtoOcc_multi_cnn_head_map_neck_adapter.py`,
not the weaker standalone smoke variants.

---

# 1. Add 200x200 High-Resolution Detail Skip

## Motivation

The current `CustomBEVBackbone` immediately downsamples:

```text
(B, 160, 200, 200) -> (B, 160, 100, 100)
```

Although `Custom_FPN_LSS` upsamples back to `200x200`, the final feature does not directly preserve the original high-resolution `200x200` BEV detail.

This is harmful for thin map classes, because lane dividers, stop lines, and boundaries may occupy only 1 to 3 pixels.

## Required Implementation

Preserve the feature right after `down_sample_for_3d_pooling`:

```python
bev_x0 = down_sample_for_3d_pooling(flattened_voxel)
# bev_x0: (B, 160, 200, 200)
```

Add a map detail branch:

```text
bev_x0: (B, 160, 200, 200)
  -> detail_branch
detail_feat: (B, 128, 200, 200)
```

Then fuse with the FPN output:

```text
fpn_out:     (B, 128, 200, 200)
detail_feat: (B, 128, 200, 200)
  -> fusion
map_bev_feature: (B, 128, 200, 200)
```

## Recommended Design

Use gated fusion:

```python
detail = self.map_detail_branch(bev_x0)
gate = torch.sigmoid(self.map_detail_gate(torch.cat([fpn_out, detail], dim=1)))
map_bev_feature = fpn_out + gate * detail
map_bev_feature = self.map_detail_refine(map_bev_feature)
```

Recommended module:

```text
map_detail_branch:
  Conv2d(160, 128, kernel_size=3, padding=1)
  BN/GN
  ReLU
  Conv2d(128, 128, kernel_size=3, padding=1)
  BN/GN
  ReLU

map_detail_gate:
  Conv2d(256, 128, kernel_size=1)
  Sigmoid

map_detail_refine:
  Conv2d(128, 128, kernel_size=3, padding=1)
  BN/GN
  ReLU
```

## Config Flag

```python
use_map_highres_skip = True
map_highres_fusion = "gated"  # options: "add", "concat", "gated"
```

## Expected Output Shape

```text
map_bev_feature: (B, 128, 200, 200)
```

## Ablation

Compare:

```text
Baseline
Baseline + high-res skip add
Baseline + high-res skip gated
```

---

# 2. Replace Naive Z Flatten + 1x1 Conv with Map-Specific Z-Aware Compression

## Motivation

Current compression:

```text
(B, 80, 16, 200, 200)
  -> flatten Z
(B, 1280, 200, 200)
  -> 1x1 Conv2d 1280 -> 160
(B, 160, 200, 200)
```

This performs aggressive compression from 1280 channels to 160 channels. It also treats height bins as channels and does not explicitly learn which height regions are important for map segmentation.

Map segmentation mainly cares about ground-level and road-surface-related information, while occupancy prediction needs full 3D structure. Therefore, the map branch should have a map-specific Z aggregation module.

## Required Implementation

Add an optional map-specific Z-aware compression module before the 2D BEV backbone.

Input:

```text
voxel_feature: (B, 80, 16, 200, 200)
```

Output:

```text
map_bev_x0: (B, 160, 200, 200)
```

## Recommended Simple Version

Add lightweight 3D refinement before flattening:

```text
voxel_feature
  -> depthwise/separable Conv3d or normal lightweight Conv3d
  -> flatten Z
  -> 1x1 Conv2d 1280 -> 160
```

Pseudo-code:

```python
x = self.map_z_refine_3d(voxel_feature)
x = x.flatten(start_dim=1, end_dim=2)  # B, C*Z, H, W
map_bev_x0 = self.map_z_project_2d(x)  # B, 160, H, W
```

## Recommended Stronger Version

Learn height attention:

```text
voxel_feature: (B, 80, 16, 200, 200)
  -> height_attention
z_weight: (B, K, 16, 200, 200)
  -> softmax over Z
  -> weighted sum over Z
(B, 80*K, 200, 200)
  -> 1x1 Conv2d
(B, 160, 200, 200)
```

Suggested:

```python
K = 2 or 4
```

Pseudo-code:

```python
attn = self.height_attn(voxel_feature)     # B, K, Z, H, W
attn = torch.softmax(attn, dim=2)

pooled = []
for k in range(K):
    pooled_k = (voxel_feature * attn[:, k:k+1]).sum(dim=2)  # B, C, H, W
    pooled.append(pooled_k)

pooled = torch.cat(pooled, dim=1)           # B, C*K, H, W
map_bev_x0 = self.height_project(pooled)    # B, 160, H, W
```

## Config Flag

```python
use_map_z_aware_compression = False
map_z_compression_type = "conv3d"  # options: "conv3d", "height_attention"
map_height_attention_groups = 4
```

## Ablation

Compare:

```text
Baseline flatten + 1x1
Conv3D refine + flatten + 1x1
Height attention pooling
```

---

# 3. Add Map-Specific Adapter Before CustomBEVBackbone

## Motivation

Occupancy and map segmentation need different feature preferences.

Occupancy prefers:

```text
3D volume semantics
object geometry
occupied / free / unknown structure
height-aware representation
```

Map segmentation prefers:

```text
ground-plane layout
road topology
thin line continuity
surface semantics
```

If both tasks share the same BEV feature without adaptation, map segmentation may be suppressed by occupancy-oriented representation.

## Required Implementation

Add a residual map adapter after `down_sample_for_3d_pooling` or after map-specific Z-aware compression:

```text
shared_bev_x0: (B, 160, 200, 200)
  -> map_adapter
map_bev_x0: (B, 160, 200, 200)
```

Recommended residual design:

```python
map_bev_x0 = shared_bev_x0 + self.map_adapter(shared_bev_x0)
```

Adapter:

```text
Conv2d(160, 64, kernel_size=1)
BN/GN
ReLU
Conv2d(64, 64, kernel_size=3, padding=1)
BN/GN
ReLU
Conv2d(64, 160, kernel_size=1)
```

Alternative ConvNeXt-like adapter:

```text
Depthwise Conv2d(160, 160, kernel_size=7, padding=3, groups=160)
LayerNorm / BatchNorm
1x1 Conv 160 -> 4*160
GELU
1x1 Conv 4*160 -> 160
Residual add
```

## Config Flag

```python
use_map_adapter = True
map_adapter_type = "res_bottleneck"  # options: "res_bottleneck", "convnext"
```

## Ablation

Compare:

```text
Baseline
Baseline + map adapter
Baseline + map adapter + high-res skip
```

---

# 4. Improve Custom_FPN_LSS Scale Fusion

## Motivation

Current `Custom_FPN_LSS` directly concatenates features from different scales:

```text
upsampled level 2 + level 1
upsampled fused feature + level 0
```

However, each level has different channel dimensions and feature distributions:

```text
level 0: 160 channels
level 1: 320 channels
level 2: 640 channels
```

This may make scale fusion unstable or inefficient.

## Required Implementation 4.1: Add Lateral Projections

Before fusion, project each pyramid level to the same channel dimension:

```text
level 2: 640 -> 256
level 1: 320 -> 256
level 0: 160 -> 256
```

Then perform FPN fusion:

```text
p2_up + p1
p1_up + p0
```

Or concatenate after projection:

```text
cat([upsampled p2, p1]) -> conv
cat([upsampled fused, p0]) -> conv
```

## Required Implementation 4.2: Add Global Context at 25x25 Level

Add optional global context on the lowest-resolution feature:

```text
level 2: (B, 640, 25, 25)
  -> global context module
level 2 enhanced: (B, 640, 25, 25)
```

Recommended modules:

```text
ASPP
PPM
lightweight axial attention
```

For initial implementation, use ASPP or PPM because they are simpler and stable.

ASPP design:

```text
3x3 dilation=1
3x3 dilation=2
3x3 dilation=4
global pooling branch
concat
1x1 projection
```

## Config Flags

```python
use_fpn_lateral_projection = True
use_fpn_global_context = False
fpn_global_context_type = "aspp"  # options: "aspp", "ppm"
```

## Expected Output Shape

The final neck output should remain:

```text
(B, 128, 200, 200)
```

## Ablation

Compare:

```text
Original Custom_FPN_LSS
+ lateral projection
+ lateral projection + ASPP
```

---

# 5. Strengthen BEVSegHead

## Motivation

Current BEVSegHead is shallow:

```text
128 -> 128 hidden -> 6 logits
```

This may not be enough to refine thin classes and high-resolution boundaries after FPN upsampling.

## Required Implementation

Replace or optionally extend the head with residual refinement blocks.

Input:

```text
map_bev_feature: (B, 128, 200, 200)
```

Output:

```text
bev_seg_logits: (B, 6, 200, 200)
```

Recommended design:

```text
ResidualBlock(128)
ResidualBlock(128)
Conv2d(128, 6, kernel_size=1)
```

ResidualBlock:

```text
Conv2d(128, 128, kernel_size=3, padding=1)
BN/GN
ReLU
Conv2d(128, 128, kernel_size=3, padding=1)
BN/GN
Residual add
ReLU
```

Alternative ConvNeXt-like refinement:

```text
7x7 depthwise conv
1x1 expansion
GELU
1x1 projection
residual add
```

## Config Flag

```python
bevseg_head_type = "res_refine"  # options: "simple", "res_refine", "convnext"
bevseg_num_refine_blocks = 2
```

## Ablation

Compare:

```text
Original BEVSegHead
Residual refinement BEVSegHead
ConvNeXt-like BEVSegHead
```

---

# 6. Split Area Classes and Line Classes into Dual Segmentation Heads

## Motivation

Map classes have different geometric properties.

Area-like classes:

```text
drivable_area
ped_crossing
walkway / sidewalk
carpark_area
```

Line-like classes:

```text
lane_divider
road_divider / divider
stop_line
boundary
```

Area classes need semantic context. Line classes need high-resolution detail and boundary continuity.

A single shared segmentation head may bias toward large area classes and hurt thin classes.

## Required Implementation

Create two branches:

```text
map_bev_feature: (B, 128, 200, 200)

area_branch -> area_logits
line_branch -> line_logits

concat -> bev_seg_logits
```

Pseudo-code:

```python
area_feat = self.area_decoder(map_bev_feature)
line_feat = self.line_decoder(line_input)

area_logits = self.area_cls(area_feat)
line_logits = self.line_cls(line_feat)

bev_seg_logits = merge_logits(area_logits, line_logits)
```

For line branch, optionally include high-resolution detail feature:

```python
line_input = torch.cat([map_bev_feature, detail_feat], dim=1)
```

## Important

Make class index mapping configurable.

Example:

```python
area_class_indices = [0, 1, 2, 3]
line_class_indices = [4, 5]
```

Do not hard-code this unless the dataset class order is already fixed in the config.

## Config Flags

```python
use_dual_area_line_head = False
area_class_indices = [...]
line_class_indices = [...]
line_head_use_detail = True
```

## Output Requirement

The final output must still be:

```text
bev_seg_logits: (B, num_map_classes, 200, 200)
```

The logits must be placed back into the original class order.

## Ablation

Compare:

```text
Single head
Dual area-line head
Dual area-line head + high-res detail to line branch
```

---

# 7. Add Auxiliary Supervision at 100x100 and 50x50

## Motivation

The FPN intermediate features contain useful semantic information:

```text
after cat_conv1: (B, 256, 50, 50)
after cat_conv2: (B, 256, 100, 100)
final output:    (B, 128, 200, 200)
```

Auxiliary losses can force intermediate FPN levels to learn map semantics and stabilize training.

## Required Implementation

Add optional auxiliary segmentation heads:

```text
50x50 aux head
100x100 aux head
200x200 final head
```

Loss:

```text
L_map = L_200 + aux_weight_100 * L_100 + aux_weight_50 * L_50
```

Recommended:

```python
aux_weight_100 = 0.4
aux_weight_50 = 0.2
```

## GT Downsampling Requirement

For thin classes, do not use bilinear interpolation. Prefer per-class max pooling so thin labels do not disappear.

If GT is multi-hot shape:

```text
gt_map: (B, C, 200, 200)
```

Use:

```python
gt_100 = F.max_pool2d(gt_map.float(), kernel_size=2, stride=2)
gt_50 = F.max_pool2d(gt_map.float(), kernel_size=4, stride=4)
```

If GT is class-index map, convert to one-hot first if the task is multi-label.

## Config Flags

```python
use_map_aux_loss = True
map_aux_levels = ["100", "50"]
map_aux_weight_100 = 0.4
map_aux_weight_50 = 0.2
map_aux_downsample = "maxpool"
```

## Output

During training, return:

```python
{
    "bev_seg_logits": logits_200,
    "aux_logits_100": logits_100,
    "aux_logits_50": logits_50,
}
```

During inference, only `bev_seg_logits` is required.

## Ablation

Compare:

```text
No aux loss
+ 100x100 aux
+ 100x100 and 50x50 aux
```

---

# 8. Improve Map Loss for Class Imbalance and Thin Structures

## Motivation

BEV map segmentation classes are highly imbalanced. Large classes dominate:

```text
drivable_area
walkway
carpark_area
```

Thin or rare classes are easy to suppress:

```text
lane_divider
stop_line
ped_crossing
boundary
```

Also, many BEV map labels may be multi-label rather than mutually exclusive. For example, crossing or divider can overlap with drivable area.

## Required Checks

Check the current target format:

1. Multi-label target:
   ```text
   gt_map: (B, C, H, W), each channel binary
   ```
   Use:
   ```text
   sigmoid + BCE/Focal + Dice
   ```

2. Mutually exclusive target:
   ```text
   gt_map: (B, H, W), class index
   ```
   Use:
   ```text
   softmax + CE
   ```

For BEVFusion-style map segmentation, prefer multi-label formulation if labels overlap.

## Recommended Loss

Use:

```text
L_map = L_focal + lambda_dice * L_dice
```

Suggested:

```python
lambda_dice = 1.0
```

Optional:

```text
L_map = L_bce + L_dice
L_map = L_focal + L_dice
L_map = L_focal + L_lovasz
```

## Class Weights

Add class weights for rare/thin classes.

Example config:

```python
map_class_weights = [1.0, 2.0, 2.0, 2.0, 4.0, 4.0]
```

Do not hard-code this; read from config.

## Config Flags

```python
map_loss_type = "focal_dice"  # options: "bce", "bce_dice", "focal", "focal_dice", "focal_lovasz"
map_dice_weight = 1.0
map_focal_gamma = 2.0
map_focal_alpha = 0.25
use_map_class_weights = True
map_class_weights = [...]
```

## Implementation Notes

For multi-label segmentation:

```python
pred = bev_seg_logits  # raw logits
target = gt_map.float()
loss = focal_loss_with_logits(pred, target) + dice_loss_with_logits(pred, target)
```

Dice should be computed per class and averaged, optionally weighted.

## Ablation

Compare:

```text
Original map loss
BCE + Dice
Focal + Dice
Focal + Dice + class weights
```

---

# 9. Add BEV Positional Encoding / CoordConv

## Motivation

Ego-centric BEV map segmentation has strong spatial priors.

Examples:

```text
front area has more visible lane structures
near ego has stronger image evidence
far range has weaker camera evidence
road layout is location-dependent in ego coordinates
```

Adding explicit BEV coordinates can help the segmentation head learn these spatial priors.

## Required Implementation

Generate coordinate channels for BEV grid:

```text
x coordinate normalized to [-1, 1]
y coordinate normalized to [-1, 1]
optional r = sqrt(x^2 + y^2)
optional theta = atan2(y, x)
```

Then concatenate with map BEV feature:

```text
map_bev_feature: (B, 128, 200, 200)
coord:           (B, 2 or 4, 200, 200)
concat:          (B, 130 or 132, 200, 200)
  -> coord projection
(B, 128, 200, 200)
```

Pseudo-code:

```python
coord = self.build_bev_coord(batch_size=B, height=H, width=W, device=x.device)
x = torch.cat([x, coord], dim=1)
x = self.coord_proj(x)
```

## Config Flags

```python
use_bev_coordconv = True
bev_coord_type = "xy"  # options: "xy", "xyr", "xyrtheta"
```

## Implementation Notes

Coordinate channels should be registered or generated dynamically. Avoid creating CPU tensors inside every forward pass without moving to the correct device.

Use `register_buffer` if grid size is fixed:

```python
self.register_buffer("bev_coord", coord, persistent=False)
```

## Ablation

Compare:

```text
No CoordConv
+ x,y CoordConv
+ x,y,r CoordConv
+ x,y,r,theta CoordConv
```

---

# Recommended Implementation Order

Implement in this order to keep ablation clean:

## Stage A: Low-risk, high-impact changes

1. High-resolution 200x200 detail skip
2. Auxiliary map losses at 100x100 and 50x50
3. Improved map loss: Focal + Dice

## Stage B: Task-specific feature improvements

4. Map-specific adapter before CustomBEVBackbone
5. BEV CoordConv
6. Stronger BEVSegHead

## Stage C: Larger architectural changes

7. Improved Custom_FPN_LSS with lateral projection
8. Global context at 25x25 level
9. Map-specific Z-aware compression
10. Dual area-line segmentation head

---



# Acceptance Criteria

Codex should ensure the following:

## Shape Consistency

All variants must keep final map logits shape:

```text
(B, num_map_classes, 200, 200)
```

Expected intermediate shapes:

```text
bev_x0:          (B, 160, 200, 200)
level 0:         (B, 160, 100, 100)
level 1:         (B, 320, 50, 50)
level 2:         (B, 640, 25, 25)
fpn_out:         (B, 128, 200, 200)
map_bev_feature: (B, 128, 200, 200)
```

## Config Compatibility

All new modules must be controlled by config flags. The default config should reproduce the original baseline when all new flags are disabled.

## Training Compatibility

Training should support:

```python
loss_depth
loss_occ
loss_map
optional loss_map_aux_100
optional loss_map_aux_50
```

The final total loss may remain:

```text
L = L_depth + L_occ + map_loss_weight * L_map
```

where current working value may be:

```python
map_loss_weight = 4.0
```

If aux losses are enabled:

```text
L_map_total = L_200 + 0.4 * L_100 + 0.2 * L_50
L_total = L_depth + L_occ + map_loss_weight * L_map_total
```

## Inference Compatibility

During inference, only final logits are required:

```python
bev_seg_logits
```

Auxiliary logits should not be used for final evaluation.

## No Hard-Coded Class Order

For area-line dual head, class index mapping must be configurable.

Example:

```python
area_class_indices = [...]
line_class_indices = [...]
```

## No Breaking Existing Occupancy Branch

The occupancy branch should remain unchanged unless the config explicitly enables shared changes. Map-specific modules should not degrade or alter occupancy outputs unintentionally.

---



# Notes for Codex

Before coding, inspect the repository and identify:

1. Where `down_sample_for_3d_pooling` is defined and called.
2. Where `CustomBEVBackbone` is defined.
3. Where `Custom_FPN_LSS` is defined.
4. Where `BEVSegHead` is defined.
5. How map loss is currently computed.
6. Whether map labels are multi-label `(B, C, H, W)` or class-index `(B, H, W)`.
7. How configs are structured.
8. How training outputs are returned.
9. Whether occupancy and map branches share the same BEV feature.

Do not assume all class names. Search the codebase first and make the smallest clean modifications necessary.

---

# 10. Thin-Class Boundary Auxiliary Target

## Motivation

The useful current path already improves map features and map loss, but the
remaining decision target is still concentrated in thin classes:

```text
ped_crossing
stop_line
divider
```

These classes can occupy only 1 to 3 pixels. A normal final-mask loss asks the
model to learn exact thin geometry immediately, which is brittle early in
training and can leave thin logits near zero.

## Required Implementation

Keep the final `bev_seg_logits` supervised by the original map GT. Add only an
auxiliary thin-region target:

```text
gt_masks_bev:            (B, 6, 200, 200)
thin_gt = gt[:, [1,3,5]]
thin_region = maxpool(thin_gt, kernel=2*d+1, stride=1, padding=d)
```

Use `thin_region` as an auxiliary target on the same thin-class logits. The
main target remains the original, non-dilated GT.

## Config Flags

```python
use_thin_boundary_aux_loss = True
thin_boundary_class_indices = [1, 3, 5]
thin_boundary_dilation = 2
thin_boundary_loss_weight = 0.2
thin_boundary_use_focal = True
thin_boundary_use_dice = True
```

## Expected Loss Terms

```text
loss_map_focal
loss_map_dice
loss_map_thin_boundary_focal
loss_map_thin_boundary_dice
```

## Ablation

Compare:

```text
combined candidate
combined candidate + thin boundary auxiliary target
```

Judge mainly on `ped_crossing`, `stop_line`, and `divider`, including fixed
thresholds, not only optimistic `iou@max`.

---

# 11. Learnable Residual Gates for Map Add-ons

## Motivation

The useful map add-ons are residual or additive:

```text
MapHFMFusionLayer:          bev + delta_hfm
PerScaleMapResidualAdapter: feat + delta_adapter
FPN ASPP context:           x + delta_aspp
```

The residual shape is safe, but each add-on currently has a fixed residual
strength. A learnable gate lets the model decide how much geometry/context/refine
signal to use per module.

## Required Implementation

Add optional learnable gates at the residual add sites:

```python
out = source + sigmoid(gate) * delta
```

Initialize gates to a small nonzero value such as `0.1`. Keep the residual
projection itself zero-initialized so step-0 behavior stays identical to the
ungated combined candidate.

## Config Flags

```python
map_hfm_learnable_residual_gate = True
map_hfm_residual_gate_init = 0.1

map_residual_adapter = dict(
    learnable_residual_gate=True,
    residual_gate_init=0.1)

map_bev_encoder_neck = dict(
    fpn_context_learnable_gate=True,
    fpn_context_gate_init=0.1)
```

## Ablation

Compare:

```text
combined candidate
combined candidate + gates on HFM / adapter / ASPP
```

Debug by logging or inspecting the learned sigmoid gate values after training.

---

# 12. Thin-Class ROI Residual Refinement

## Motivation

A full query decoder or heavy segmentation head risks replacing the protected
`128ch map neck -> BEVSegHead` path. A smaller option is to refine only the
regions where thin classes are likely to exist.

## Required Implementation

Use the current thin-class logits to build an ROI:

```text
thin_score = max(sigmoid(logits[:, thin_indices]), dim=class)
roi = thin_score >= threshold
roi = roi OR topk(thin_score, min_pixels)
roi = maxpool(roi, dilation)
```

The ROI must be derived from predictions, not GT, so train and test behavior
match. Use detached logits to build the ROI, then apply a zero-init residual
refiner on the decoded map feature:

```python
delta_thin_logits = thin_roi_refiner(decoded_feature) * roi
logits[:, thin_indices] = logits[:, thin_indices] + delta_thin_logits
```

## Config Flags

```python
use_thin_roi_refinement = True
thin_roi_class_indices = [1, 3, 5]
thin_roi_threshold = 0.35
thin_roi_min_pixels = 64
thin_roi_dilation = 2
thin_roi_hidden_channels = 64
```

## Ablation

Compare:

```text
combined candidate
combined candidate + thin ROI residual refinement
```

Main failure mode: the ROI becomes empty or too sparse early in training. Keep
the top-k fallback enabled so every sample has at least a small refinement
region.

---

# 13. Smoke Result Record for Directions 10-12

Date: 2026-06-05

Comparison ruler:

```text
1 epoch
1/4 nuScenes train split
single local 4090 smoke run
EMA checkpoint eval
metric: miou map-miou
```

Reference baseline for this comparison:

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted.py
work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_1quarter_4090/result.md
```

Baseline smoke result:

| Metric | Value |
| --- | ---: |
| OCC mIoU | 27.54 |
| map mean IoU@max | 0.219697 |
| thin avg: ped_crossing / stop_line / divider | 0.111283 |
| ped_crossing | 0.069360 |
| stop_line | 0.090233 |
| divider | 0.174255 |

## Direction 10: Thin-Class Boundary Auxiliary Target

Result file:

```text
work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_thin_boundary_1quarter_4090/result.md
```

| Metric | Value | Delta vs baseline |
| --- | ---: | ---: |
| OCC mIoU | 27.79 | +0.25 |
| map mean IoU@max | 0.220669 | +0.000972 |
| thin avg: ped_crossing / stop_line / divider | 0.112686 | +0.001404 |
| ped_crossing | 0.070159 | +0.000799 |
| stop_line | 0.089970 | -0.000263 |
| divider | 0.177930 | +0.003675 |

Judgment: keep. This is the only one of the three new directions that improves
same-ruler map mean, thin-class average, divider, and OCC mIoU at the same time.
The gain is small in 1-epoch smoke scale, but it is directionally useful and
does not show a stability regression.

Next check: run a longer Nano4/H200 job before treating it as a real result.

## Direction 11: Learnable Residual Gates for Map Add-ons

Result file:

```text
work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_residual_gates_1quarter_4090/result.md
```

| Metric | Value | Delta vs baseline |
| --- | ---: | ---: |
| OCC mIoU | 26.53 | -1.01 |
| map mean IoU@max | 0.206450 | -0.013248 |
| thin avg: ped_crossing / stop_line / divider | 0.100369 | -0.010914 |
| ped_crossing | 0.066097 | -0.003263 |
| stop_line | 0.080010 | -0.010223 |
| divider | 0.154999 | -0.019256 |

Judgment: do not adopt in this form. The 0.1 gate initialization suppresses
useful residual paths too strongly for a 1-epoch smoke run. If this direction is
retried, test larger initial gates such as 0.5 or gate only the riskiest add-on
instead of gating HFM, adapter, and ASPP together.

## Direction 12: Thin-Class ROI Residual Refinement

Result file:

```text
work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_thin_roi_refine_1quarter_4090/result.md
```

| Metric | Value | Delta vs baseline |
| --- | ---: | ---: |
| OCC mIoU | 25.19 | -2.35 |
| map mean IoU@max | 0.218582 | -0.001116 |
| thin avg: ped_crossing / stop_line / divider | 0.108898 | -0.002384 |
| ped_crossing | 0.071407 | +0.002047 |
| stop_line | 0.090007 | -0.000226 |
| divider | 0.165281 | -0.008974 |

Judgment: do not adopt as-is. It improves ped_crossing slightly, but loses
divider enough to make thin-class average and map mean worse. The likely issue
is that prediction-derived ROI is not reliable enough early in training, so the
refiner helps isolated crossings but disturbs long thin lines.

Potential retry: build the ROI from a softer score map or add a boundary-guided
ROI warmup instead of hard threshold plus top-k from the current thin logits.
