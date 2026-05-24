# Height Task Gate 模組計劃

日期：2026-05-25

## 一句話定位

> 從 LSS voxel feature 中學出 map-specific 與 occ-specific 的 height selection，取代 ProtoOcc DBE 中固定的 `torch.cat(x.unbind(dim=2), dim=1)`。

模組名稱：

```text
Height Task Gate (HTG)
```

## 動機

LSS voxel feature 是：

```text
X [B, C, Z, H, W]
```

ProtoOcc DBE 目前直接把 `Z` 維攤成 channel：

```python
pooled_x = torch.cat(x.unbind(dim=2), dim=1)
```

這等於假設所有 height bins 對 OCC 和 Map 都一樣重要。但兩個任務需求不同：

| Task | height cue |
| --- | --- |
| Map | ground plane, lane topology, sidewalk / divider / stop line, low-height static layout |
| OCC | object volume, vertical surface, free space, occlusion, height distribution |

如果要把 DBE 改成 task-specific encoder，第一個最小模組就是先把 `X` 分成：

```text
X_map
X_occ
```

## 挑戰

1. 不能一開始大幅破壞 LSS feature，否則 OCC / Map 都可能掉。
2. Gate 如果太自由，可能只學成 attention noise。
3. Gate 如果太弱，又會退化成原本 fixed z concat。
4. 需要能單獨 ablation，證明 task-specific height selection 有用。

## 目標

HTG 的輸出：

```text
G_map [B, 1, Z, H, W]
G_occ [B, 1, Z, H, W]
X_map = X * (1 + scale * G_map)
X_occ = X * (1 + scale * G_occ)
```

第一版要避免 `init_scale=0.0` 的 cold start。若 `scale=0`，gate branch 的梯度會被乘掉，前幾步只有 scale 自己在動，gate conv 幾乎學不到。

建議使用：

```text
scale = 0.1
last gate conv zero-init
```

這樣 step 0 仍然是 identity，因為 `G_task = 0`，但 gate conv 有非零梯度，能從第一步開始學。

## 要做什麼

### V1：共享 gate backbone，雙 task output

```text
X
  -> Conv3d(C -> C/4, k=3)
  -> BN / ReLU
  -> Conv3d(C/4 -> 2, k=1)
  -> Tanh
  -> G_occ, G_map
```

用 `Tanh` 而不是 `Sigmoid` 的理由：

```text
X_task = X * (1 + scale * G_task)
G_task in [-1, 1]
```

這讓 gate 可以增強或抑制某些 height bins，而不是只能增強。

### V2：加全域 height prior

如果 V1 有效果，再加入全域 z distribution：

```text
z_context = GlobalPool_HW(X)  # [B, C, Z]
z_gate = MLP(z_context)       # [B, 2, Z]
```

再和 spatial gate 相乘：

```text
G_task = G_spatial_task * z_gate_task
```

## 預計改動哪些檔案

新增：

```text
projects/mmdet3d_plugin/models/backbones/task_modules/height_task_gate.py
```

或若先做最小版，可直接放在：

```text
projects/mmdet3d_plugin/models/backbones/task_specific_geometry_decomposition_encoder.py
```

新增 config 開關：

```python
height_task_gate=dict(
    type='HeightTaskGate',
    in_channels=80,
    reduction=4,
    gate_scale=0.1,
    zero_init_last=True,
)
```

## 實驗設計

| ID | Setting | 目的 |
| --- | --- | --- |
| H0 | weight4 map-neck baseline | 對照 |
| H1 | HTG gradient sanity, no metric claim | 確認 zero-init identity 但 gate 有梯度 |
| H2 | HTG + map-specific branch only | 測 map 是否受益 |
| H3 | HTG + occ-specific branch only | 測 OCC 是否受益 |
| H4 | random / shared gate | 排除只是多參數 |

H1 不應當成正式模型結果，因為 HTG 只產生 `X_map/X_occ`，若 downstream 仍然共用同一條 DBE，task split 並沒有真的被消費。H1 只檢查：

```text
step 0: X_task == X
after a few iterations: G_map / G_occ 不再全為 0
gate conv gradients are non-zero
```

## Debug 指標

```text
G_map_mean/std/min/max
G_occ_mean/std/min/max
mean_abs(G_map - G_occ)
height_gate_map_z_distribution
height_gate_occ_z_distribution
```

健康跡象：

- `G_map` 和 `G_occ` 不應完全相同。
- `G_map` 應更集中在 low-height / ground-related bins。
- `G_occ` 應保留較完整的 height distribution。
- 初期 gate magnitude 應小，避免破壞 baseline。

## 預期效果

最低成功標準：

```text
Map mIoU > 45.79
Occ mIoU >= 39.5
```

強成功標準：

```text
Map mIoU >= 46.5
Occ mIoU >= 39.72
```

如果 HTG 單獨有效，代表「LSS voxel height decomposition」可以成為整篇方法的第一個 contribution。
