# Geometry Register Bank 模組計劃

日期：2026-05-25

## 一句話定位

> 用 scene-level geometry registers 作為 encoder memory，取代直接照搬 MAESTRO semantic prototype grouping。

模組名稱：

```text
Geometry Register Bank (GRB)
```

## 動機

MAESTRO 的 CPG / TSFG / SPA 證明 prototype-guided filtering 可以幫助多任務 feature。但 OCC + Map 的互補不只來自 semantic class，更多來自 geometry / topology：

```text
ground
boundary
drivable layout
static structure
dynamic volume
free space
occlusion
```

這些資訊不適合直接用 MAESTRO 的 foreground / background semantic grouping 表達。GRB 的目標是提供一組 encoder-level geometry memory，讓 map / occ branch 用它做 feature modulation。

## 挑戰

1. Registers 不能變成 PQD query，否則會和 decoder contribution 混在一起。
2. Registers 不能依賴 GT 或 final prediction。
3. 如果一開始上 cross-attention，失敗時難歸因。
4. Register collapse 需要監控。

## 目標

輸入：

```text
X
F_map_base
F_occ_base
```

輸出：

```text
R_geo [B, K, C_reg]
gamma_map
gamma_occ
```

用途：

```text
F_map_reg = F_map_base + gamma_map * MapRegisterAdapter(F_map_base)
F_occ_reg = F_occ_base + gamma_occ * OccRegisterAdapter(F_occ_base)
```

## 要做什麼

### V1：Pooling-based registers

先不用 transformer：

```text
global_x   = GlobalPool(X)
global_map = GlobalPool(F_map_base)
global_occ = GlobalPool(F_occ_base)

R_geo = MLP([global_x, global_map, global_occ])
```

`K` 建議：

```text
K = 6 或 8
```

register 不需要手動指定 class name，但論文可以解釋成 geometry memory slots。

### V2：Register-to-feature modulation

用 registers 產生 channel gate：

```text
gamma_map = MLP_map(mean(R_geo))
gamma_occ = MLP_occ(mean(R_geo))
```

第一版只做 channel gate，不做 spatial attention。

### V3：Spatial register attention

如果 V2 有效果，再做：

```text
R_geo query
F_map / F_occ key-value
-> spatial modulation
```

這是後續加強版，不建議第一版做。

## 預計改動哪些檔案

新增：

```text
projects/mmdet3d_plugin/models/backbones/task_modules/geometry_register_bank.py
```

config：

```python
geometry_register_bank=dict(
    type='GeometryRegisterBank',
    num_registers=8,
    register_channels=128,
    use_spatial_attention=False,
)
```

## 實驗設計

| ID | Setting | 目的 |
| --- | --- | --- |
| R0 | HTG + MTE + OVE baseline | 對照 |
| R1 | + GRB channel gate | 測 register memory |
| R2 | + GRB spatial gate | 測 spatial modulation |
| R3 | random registers | 排除只是 noise regularization |
| R4 | semantic prototype style grouping | 和 MAESTRO-like 設計對照 |

## Debug 指標

```text
register_norm
register_cosine_offdiag
register_channel_std
gamma_map_mean/std
gamma_occ_mean/std
cosine(gamma_map, gamma_occ)
```

健康跡象：

- register 不應全部相同。
- `gamma_map` 和 `gamma_occ` 不應完全一致。
- 如果 `register_cosine_offdiag` 長期接近 1，代表 collapse。

## 預期效果

最低成功標準：

```text
Map mIoU 或 Occ mIoU 至少一個提升，另一個不明顯下降
```

強成功標準：

```text
Map thin classes 改善
Occ free / object / static classes 穩定或改善
```

如果 GRB 有效，論文 contribution 可以寫成：

```text
We replace semantic prototype grouping with geometry registers that encode
scene-level layout and volumetric priors for task-aware feature modulation.
```

