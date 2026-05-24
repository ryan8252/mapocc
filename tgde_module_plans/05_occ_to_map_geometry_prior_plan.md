# O2M Geometry Prior 模組計劃

日期：2026-05-25

## 一句話定位

> 將 occupancy-specific volumetric feature 壓成 map 可用的 geometry prior，幫助 BEV map segmentation，但不直接把 OCC raw feature 灌進 map head。

模組名稱：

```text
Occupancy-to-Map Geometry Prior (O2M)
```

## 動機

OCC feature 包含 map branch 缺少的 3D geometry：

- occupied / free structure
- vertical-to-ground projection
- occlusion-aware object support
- static / dynamic spatial context

Map segmentation 需要 topology，但 topology 常受遮擋和影像深度不確定影響。OCC volumetric feature 可以提供幾何支撐，尤其對：

```text
drivable area
walkway
ped crossing
divider
stop line
```

但不能直接 raw fusion，否則 OCC noise 會傷害 map thin classes。

## 挑戰

1. OCC feature 是 3D，Map feature 是 2D BEV，需要合理 height pooling。
2. O2M prior 不能直接變成 map logits。
3. 需要控制融合強度，避免 map branch 被 OCC 主導。
4. 要和先前 O2M-only negative evidence 切開：這次只傳 prior，不傳 prediction / raw logits。

## 目標

輸入：

```text
F_occ_zyx [B, 48, Z, H, W]
F_map [B, 128, H, W]
```

輸出：

```text
O2M_prior [B, 128, H, W]
F_map_out [B, 128, H, W]
```

融合：

```text
F_map_out = F_map + alpha_map * O2M_prior
```

## 要做什麼

### V1：HeightPool O2M

```text
F_occ_zyx
  -> Z attention / weighted sum
  -> F_occ_bev [B, 48, H, W]
  -> Conv2d(48 -> 128)
  -> O2M_prior
```

Z attention：

```text
w_z = softmax(Conv3d(F_occ_zyx), dim=Z)
F_occ_bev = sum_z w_z * F_occ_z
```

### V2：Map-conditioned O2M

如果 V1 有效果，再用 map feature condition pooling：

```text
w_z = softmax(Conv([F_occ_z, F_map]), dim=Z)
```

這讓 map branch 自己決定要從 OCC volume 看哪個 height。

## 預計改動哪些檔案

新增：

```text
projects/mmdet3d_plugin/models/backbones/task_modules/o2m_geometry_prior.py
```

config：

```python
o2m_geometry_prior=dict(
    type='O2MGeometryPrior',
    occ_channels=48,
    map_channels=128,
    use_map_condition=False,
    init_alpha=0.0,
)
```

## 實驗設計

| ID | Setting | 目的 |
| --- | --- | --- |
| O2M0 | HTG + MTE baseline | 對照 |
| O2M1 | + O2M HeightPool | 測 OCC geometry prior |
| O2M2 | + map-conditioned O2M | 測 map-conditioned height pooling |
| O2M3 | direct raw fusion | 證明不是 raw fusion |
| O2M4 | detached OCC prior | 測是否需要梯度回 OCC |

## Debug 指標

```text
O2M_prior_norm / F_map_norm
alpha_map_mean/std/min/max
z_attention_entropy
z_attention_distribution
cosine(F_map, F_map_out)
```

健康跡象：

- `O2M_prior_norm` 不應為 0。
- `alpha_map` 初期小，中後期有 spatial variation。
- `z_attention` 不應永遠平均，也不應全部 collapse 到單一 z。

## 預期效果

最低成功標準：

```text
Map mIoU > MTE-only
Occ mIoU 不明顯下降
```

強成功標準：

```text
Map 接近 48.34 map-only upper bound
thin classes 有穩定改善
```

如果 O2M 有效，可以獨立寫成：

```text
Occupancy-specific volume provides reliable geometry priors for BEV map
topology when projected through learned height-aware pooling.
```
