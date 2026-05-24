# TGDE 01+02 Implementation Note

日期：2026-05-25

## 實作範圍

本次完成 `01_height_task_gate_plan.md` + `02_map_topology_encoder_plan.md`
的第一版實作：

```text
LSS voxel X [B,80,Z,H,W]
  -> HeightTaskGate
       -> X_occ: legacy DBE / OCC path
       -> X_map: MapTopologyEncoder
  -> MapTopologyEncoder weighted_z_pool
  -> existing 128ch map_bev_encoder_neck
  -> fixed BEVSegHead
```

沒有實作 `03_occ_volumetric_encoder_plan.md`，因此 OCC decoder 邊界仍維持
現有 ProtoOcc：

```text
comprehensive_voxel_feature [B,48,H,W,Z]
  -> existing cnn3d_decoder + PQD
```

## 新增檔案

```text
projects/mmdet3d_plugin/models/backbones/task_modules/__init__.py
projects/mmdet3d_plugin/models/backbones/task_modules/height_task_gate.py
projects/mmdet3d_plugin/models/backbones/task_modules/map_topology_encoder.py
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_tgde_htg_mte_weighted.py
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_tgde_htg_mte_catz.py
tgde_module_plans/implementation_01_02_htg_mte_weighted.md
```

## 修改檔案

```text
projects/mmdet3d_plugin/models/backbones/__init__.py
projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py
```

## HeightTaskGate

`HeightTaskGate` 註冊為 backbone module，輸入 shape 固定為：

```text
X [B,C,Z,H,W]
```

輸出：

```text
X_occ = X * (1 + 0.1 * tanh(G_occ))
X_map = X * (1 + 0.1 * tanh(G_map))
```

最後一層 gate conv 使用 zero-init，所以 step 0 時 `X_occ == X_map == X`。
`gate_scale=0.1` 是固定非零值，因此 gate conv 從第一步就能收到梯度，不會出現
`init_scale=0` 的 cold-start 問題。

## MapTopologyEncoder

`MapTopologyEncoder` 目前支援兩種 `z_projection`：

```text
weighted_pool
cat_z
```

`weighted_pool` 流程：

```text
X_map [B,80,16,H,W]
  -> Conv3d(80 -> 1)
  -> softmax over Z
  -> sum_z weighted X_map
  -> 1x1 Conv2d(80 -> 160)
  -> CustomBEVBackbone(num_channels=[160,320,640], num_layer=[1,1,1])
  -> multi_scale_map
```

`z_logit` 使用 zero-init，因此第一步是 uniform weighted-Z pooling。MTE 只輸出
multi-scale feature，不輸出 mask / logits，後面仍接既有 `map_bev_encoder_neck`
和 `BEVSegHead`。

`cat_z` 流程：

```text
X_map [B,80,16,H,W]
  -> torch.cat(X_map.unbind(dim=2), dim=1)
  -> X_map_bev [B,1280,H,W]
  -> 1x1 Conv2d(1280 -> 160)
  -> CustomBEVBackbone(num_channels=[160,320,640], num_layer=[1,1,1])
  -> multi_scale_map
```

`cat_z` 是 02 MTE 的 parity / ablation 入口：它保留原本 DBE 的 fixed-Z
flattening，不讓 MTE 自己先做 learned Z pooling。這樣可以先測：

```text
HTG 是否能在 concat 前提供有效的 task-specific height reweight
MTE map-specific BEV backbone 是否比 shared multi_scale_bev 更適合 map
```

## Dual_Branch_Encoder 接線

新增 optional config：

```python
height_task_gate=dict(...)
map_topology_encoder=dict(...)
use_height_task_gate_for_legacy=True
```

當這些 config 不存在時，`Dual_Branch_Encoder` 行為維持原本 baseline。

當打開 TGDE 01+02 時：

- legacy DBE/OCC path 使用 `X_occ`
- map path 使用 `X_map -> MapTopologyEncoder -> map_bev_encoder_neck`
- `cnn3d_decoder`、PQD、`BEVSegHead` 都不改
- `map_loss_weight=4.0` 維持 detector-level strong baseline 設定

## 實驗入口

新增 config：

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_tgde_htg_mte_weighted.py
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_tgde_htg_mte_catz.py
```

它繼承：

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py
```

並只打開：

```text
map_loss_weight=4.0
height_task_gate
map_topology_encoder(weighted_pool)
```

`tgde_htg_mte_catz.py` 則只把 MTE 的 projection 換成：

```text
map_topology_encoder(cat_z)
```

建議第一個正式比較對照：

```text
CNN head + 128ch map neck + map_loss_weight=4
vs.
TGDE 01+02 HTG + MTE cat_z + map_loss_weight=4
vs.
TGDE 01+02 HTG + MTE weighted_z_pool + map_loss_weight=4
```
