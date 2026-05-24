# Confidence-Gated Fusion Controller 模組計劃

日期：2026-05-25

## 一句話定位

> 控制 O2M 與 M2O prior 的融合強度，讓兩個任務只在可信區域互相幫助，避免 SOGDet-style 固定比例融合造成 negative transfer。

模組名稱：

```text
Confidence-Gated Fusion Controller (CGFC)
```

## 動機

SOGDet 的 fusion 可抽象成固定比例：

```text
F_task = lambda * F_task + (1 - lambda) * G(F_other)
```

這個想法簡單，但對 OCC + Map 不夠細。OCC feature 不是所有位置都可信，Map feature 也不是所有區域都能幫 OCC。需要一個 learned confidence gate 判斷：

```text
where to fuse
how much to fuse
which task should dominate
```

## 挑戰

1. Gate 太強會變成 raw fusion，造成 negative transfer。
2. Gate 太弱會讓 O2M / M2O 沒效果。
3. Gate 若沒有 spatial variation，就退化成 SOGDet fixed lambda。
4. 必須和 O2M / M2O 分開 ablation。

## 目標

對 O2M：

```text
alpha_map = CGFC_map(F_map, O2M_prior, optional R_geo)
F_map_out = F_map + alpha_map * O2M_prior
```

對 M2O：

```text
alpha_occ = CGFC_occ(F_occ, M2O_prior, optional R_geo)
F_occ_out = F_occ + alpha_occ * M2O_prior
```

輸出 gate：

```text
alpha_map [B, 1, H, W]
alpha_occ [B, 1, H, W, Z]
```

## 要做什麼

### V1：Feature agreement gate

Map side：

```text
concat(F_map, O2M_prior, abs(F_map - O2M_prior))
  -> Conv2d
  -> Sigmoid
  -> alpha_map
```

Occ side：

```text
concat(F_occ, M2O_prior, abs(F_occ - M2O_prior))
  -> Conv3d
  -> Sigmoid
  -> alpha_occ
```

### V2：Low-init residual gate

用 learnable scalar 控制 gate：

```text
F_out = F + beta * alpha * prior
beta initialized at 0
```

這比直接把 `alpha` 初始化小更穩，因為可以保證 step 0 接近 baseline。

### V3：Register-conditioned gate

如果 `Geometry Register Bank` 有效果，再加：

```text
alpha = Gate(F_task, prior, R_geo)
```

讓 scene-level geometry memory 控制 fusion。

## 預計改動哪些檔案

新增：

```text
projects/mmdet3d_plugin/models/backbones/task_modules/confidence_gated_fusion.py
```

config：

```python
confidence_gated_fusion=dict(
    type='ConfidenceGatedFusionController',
    map_channels=128,
    occ_channels=48,
    use_register_condition=False,
    beta_init=0.0,
)
```

## 實驗設計

| ID | Setting | 目的 |
| --- | --- | --- |
| C0 | O2M / M2O without gate, direct add | 對照 |
| C1 | fixed lambda fusion | SOGDet-style baseline |
| C2 | CGFC map only | 測 O2M gate |
| C3 | CGFC occ only | 測 M2O gate |
| C4 | CGFC bidirectional | full controller |
| C5 | CGFC + register condition | 測 GRB 是否能幫 gate |

## Debug 指標

```text
alpha_map_mean/std/min/max
alpha_occ_mean/std/min/max
beta_map
beta_occ
spatial_entropy(alpha_map)
spatial_entropy(alpha_occ)
prior_norm / feature_norm
```

健康跡象：

- `beta` 從 0 慢慢變大。
- `alpha` 有 spatial variation。
- fixed lambda fusion 若輸給 CGFC，才能主張 confidence gate 有意義。
- direct add 若不穩，CGFC 能穩住，才代表它真的避免 negative transfer。

## 預期效果

最低成功標準：

```text
CGFC >= fixed lambda fusion
CGFC >= direct add
```

強成功標準：

```text
O2M + CGFC improves map
M2O + CGFC improves OCC
bidirectional CGFC does not create task regression
```

如果 CGFC 有效，可以寫成：

```text
Instead of fixed-ratio cross-task fusion, we introduce confidence-gated
geometry exchange that activates only where cross-task priors are reliable.
```

