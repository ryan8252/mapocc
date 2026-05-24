# M2O Topology Lift 模組計劃

日期：2026-05-25

## 一句話定位

> 將 map-specific topological BEV feature lift 回 voxel space，作為 OCC branch 的 ground / layout prior，但不直接改 PQD decoder。

模組名稱：

```text
Map-to-Occupancy Topology Lift (M2O)
```

## 動機

Map feature 可以提供 OCC branch 缺少的 layout prior：

- drivable surface
- walkway
- divider / boundary
- static ground topology
- road structure constraint

OCC 預測不只要 object volume，也要 free space 和 ground layout。Map topology 可以幫助 OCC 在地面附近建立更穩定的 semantic structure。

## 挑戰

1. Map feature 是 2D，OCC feature 是 3D，需要合理 lift 到 `Z`。
2. 不能把 map logits 或 GT map 當 guidance。
3. M2O 太強可能傷害 OCC vertical object feature。
4. 需要先做 M2O-only ablation，不能直接放進 full bidirectional。

## 目標

輸入：

```text
F_map [B, 128, H, W]
F_occ_zyx [B, 48, Z, H, W]
```

輸出：

```text
M2O_prior [B, 48, Z, H, W]
F_occ_out_zyx [B, 48, Z, H, W]
```

融合：

```text
F_occ_out_zyx = F_occ_zyx + alpha_occ * M2O_prior
```

## 要做什麼

### V1：Linear Geometry Lift

最小版：

```text
F_map
  -> Conv2d(128 -> 48 * Z)
  -> reshape [B, 48, Z, H, W]
  -> M2O_prior
```

### V2：Height-basis Lift

如果 V1 有效果，再用 height basis：

```text
F_map -> K basis features [B, K, 48, H, W]
height_embedding -> K weights per Z
M2O_prior = sum_k basis_k * weight_k(z)
```

這比直接 `Conv2d(128 -> 48*Z)` 更有幾何解釋性。

### V3：Ground-focused Lift

Map topology 多數在地面附近，因此可以限制 prior 主要影響低 z：

```text
M2O_prior = M2O_prior * learnable_z_mask
```

`z_mask` 初始化為低 z 較大、高 z 較小。

## 預計改動哪些檔案

新增：

```text
projects/mmdet3d_plugin/models/backbones/task_modules/m2o_topology_lift.py
```

config：

```python
m2o_topology_lift=dict(
    type='M2OTopologyLift',
    map_channels=128,
    occ_channels=48,
    z_size=16,
    lift_mode='linear',
    init_alpha=0.0,
)
```

## 實驗設計

| ID | Setting | 目的 |
| --- | --- | --- |
| M2O0 | HTG + OVE baseline | 對照 |
| M2O1 | + linear M2O | 測 map prior 對 OCC |
| M2O2 | + height-basis M2O | 測 z-aware lift |
| M2O3 | + ground-focused z mask | 測地面 topology prior |
| M2O4 | detached map prior | 測梯度是否該回 map branch |

## Debug 指標

```text
M2O_prior_norm / F_occ_norm
alpha_occ_mean/std/min/max
z_mask_distribution
free_ratio
non_free_recall
PQD losses
```

健康跡象：

- `alpha_occ` 不應一開始過大。
- `M2O_prior` 主要影響 ground / low-height region，不應壓掉高處 object volume。
- OCC free ratio 不應異常升高。

## 預期效果

最低成功標準：

```text
Occ mIoU >= OVE-only
Map mIoU 不明顯下降
```

強成功標準：

```text
Occ mIoU > 39.72
Map mIoU >= 45.79
```

如果 M2O 有效，可以主張：

```text
Map topology can regularize occupancy volume through learned geometry lift,
without changing the occupancy decoder.
```
