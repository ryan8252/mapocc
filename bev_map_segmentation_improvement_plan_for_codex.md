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


