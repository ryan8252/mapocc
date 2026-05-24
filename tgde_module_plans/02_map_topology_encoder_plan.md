# Map Topology Encoder 模組計劃

日期：2026-05-25

## 一句話定位

> 為 BEV map segmentation 建立 map-specific topology feature，讓 map head 不再吃 ProtoOcc DBE 的 shared BEV feature。

模組名稱：

```text
Map Topology Encoder (MTE)
```

## 動機

目前 map branch 最強 baseline 是：

```text
shared multi_scale_bev
  -> map_bev_encoder_neck
  -> BEVSegHead
```

這已經比直接把 BEVSegHead 接 shared `bev_feature` 好，但 source 還是 shared `multi_scale_bev`。Map 的弱點集中在：

- divider
- stop line
- pedestrian crossing
- walkway
- carpark area

這些 class 需要 topology / thin structure，不只是 generic BEV semantic feature。因此 map branch 應該有自己的 topology encoder。

## 挑戰

1. Map encoder 不能變成新 decoder，否則 contribution 會從 encoder 變成 head design。
2. 要維持 BEVSegHead 不變，才能歸因到 feature。
3. 必須避免只是加深 map neck，否則故事會變成 capacity 增加。
4. 需要能和 `Height Task Gate` 單獨或一起測。

## 目標

MTE 輸入：

```text
X_map [B, C, Z, H, W]
```

MTE 輸出：

```text
F_map [B, 128, H, W]
```

然後固定：

```text
F_map -> BEVSegHead
```

## 要做什麼

### V1：Task-aware z projection + BEV topology backbone

```text
X_map
  -> weighted_z_pool
  -> 1x1 Conv(C -> 160)
  -> shallow map-specific CustomBEVBackbone
  -> multi_scale_map
  -> map_bev_encoder_neck  # existing neck, not MTE novelty
  -> F_map
```

V1 固定選 `weighted_z_pool`，不要同時保留 `cat_z` 與 `weighted_z_pool` 兩條主線。原因是 MTE 的目的不是複製 ProtoOcc fixed z concat，而是讓 map branch 自己選高度。

建議 weighted z pooling：

```text
w_map = softmax(Conv3d(X_map), dim=Z)     # [B,1,Z,H,W]
X_map_bev = sum_z w_map * X_map_z         # [B,C,H,W]
```

第一版不要用 transformer。`shallow map-specific CustomBEVBackbone` 使用現有 ProtoOcc large-kernel / depthwise BEV block，但權重與 OCC branch 分開：

```text
BasicBlock downsample
ConvNeXt-style Block:
  DWConv k=7
  LayerNorm
  1x1 expansion / GELU / GRN / 1x1 reduction
  residual
```

建議第一版：

```text
num_channels=[160, 320, 640]
stride=[2, 2, 2]
num_layer=[1, 1, 1]
ConvNext_kernel_size=7
```

`num_layer=[2,2,2]` 可以作為 capacity ablation，不作 V1 default。

### MTE / neck 邊界

MTE proper 到 `multi_scale_map` 為止：

```text
MTE = weighted_z_pool + 1x1 projection + shallow map-specific CustomBEVBackbone
```

`map_bev_encoder_neck` 是沿用 baseline 的既有 FPN neck：

```text
multi_scale_map -> map_bev_encoder_neck -> F_map [B,128,H,W]
```

因此論文與 ablation 要分清：

- 改 MTE：改 map-specific source feature。
- 改 neck：改 map feature aggregation capacity，不是 MTE 主貢獻。

### V2：Boundary-aware local context

如果 V1 有效果，再加 thin-structure branch：

```text
F_map
  -> local edge / boundary conv branch
  -> residual add
```

注意這裡仍然是 feature，不輸出 boundary mask，不額外加 GT。

## 預計改動哪些檔案

新增：

```text
projects/mmdet3d_plugin/models/backbones/task_modules/map_topology_encoder.py
```

或整合進：

```text
projects/mmdet3d_plugin/models/backbones/task_specific_geometry_decomposition_encoder.py
```

新增 config：

```python
map_topology_encoder=dict(
    type='MapTopologyEncoder',
    in_channels=80,
    z_channels=16,
    z_projection='weighted_pool',
    mid_channels=160,
    num_channels=[160, 320, 640],
    num_layer=[1, 1, 1],
    out_channels=128,
    use_large_kernel=True,
)
```

固定不改：

```text
projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py
```

## 實驗設計

| ID | Setting | 目的 |
| --- | --- | --- |
| M0 | weight4 map-neck baseline | 對照 |
| M1 | MTE with weighted z pool, no HTG | 測 map topology encoder |
| M2 | HTG + MTE | 測 height selection + map topology |
| M3 | same params shared BEV encoder | 排除只是 capacity |
| M4 | MTE without large-kernel block | 測 topology context 是否必要 |
| M5 | cat_z projection ablation | 確認 weighted z pool 是否優於 fixed z concat |

## Debug 指標

```text
F_map_norm
cosine(F_map, baseline_map_feature)
map_feature_channel_std
thin_class_logits_norm
```

若可以加 per-class early eval，重點看：

```text
divider
stop_line
ped_crossing
walkway
carpark_area
```

## 預期效果

最低成功標準：

```text
Map mIoU > 45.79
Occ mIoU 不應因 map branch 改動而明顯下降
```

強成功標準：

```text
Map mIoU >= 47.0
thin classes 明顯改善
```

如果 MTE 有效，論文可以主張：

```text
Map segmentation benefits from topological BEV features decomposed from LSS
voxels, rather than shared occupancy-oriented BEV features.
```
