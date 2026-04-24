# Step 1: Map-only GT-soft Mining 更新紀錄

日期：2026-04-24

## 這次完成了什麼

### 1. `ProtoMapHead` 加入 training-time GT-guided prototype mining

已在 `projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py` 實作：

- `prototype_mining_mode`
  - `pred_threshold`
  - `gt_hard`
  - `gt_soft`
- `prototype_mining_alpha`
- `prototype_pooling_scope='batch_local'`

目前行為：

- training 且 `gt_masks_bev` 有提供時：
  - `gt_hard` 會用 GT mask 做 prototype pooling
  - `gt_soft` 會用 `GT * (alpha + (1 - alpha) * pred_prob.detach())` 做 weighted pooling
- inference 時：
  - 不使用 GT
  - 自動退回原本的 prediction-threshold mining

### 2. `ProtoMapHead` 的 AdaPG / EMA update 改成顯式 `valid_mask`

已實作：

- `_adaPG()` 回傳 `(for_query, valid_mask)`
- 新增 `_gt_guided_adaPG()`，同樣回傳 `(for_query, valid_mask)`
- `_agno_PG_update_and_get()` 改為使用 `valid_mask`

目的：

- 某個 class 在當前 batch 沒有 GT support 時，不會用 zero prototype 去覆蓋 EMA bank
- EMA 只更新真正有 prototype support 的類別

### 3. `ProtoOccMultitask` 訓練流程會把 `gt_masks_bev` 傳進 `ProtoMapHead`

已在 `projects/mmdet3d_plugin/models/detectors/ProtoOccMultitask.py` 修改：

- training 時改成：

```python
coarse_pred, final_masks = self.proto_map_head(
    map_feature, gt_masks_bev=gt_masks_bev)
```

這樣 `gt_soft` / `gt_hard` 路徑才會真的生效。

### 4. 新增可直接跑的 Step 1 config

新增：

- `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_gt_soft.py`
  - map neck baseline + GT-soft mining
- `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_gt_soft_pgbr.py`
  - GT-soft mining + PGBR ablation

另外也把 baseline config `projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py` 顯式補上：

- `prototype_mining_mode='pred_threshold'`
- `prototype_mining_alpha=0.5`
- `prototype_pooling_scope='batch_local'`

## 更新了哪些檔案

### 修改

- `projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py`
- `projects/mmdet3d_plugin/models/detectors/ProtoOccMultitask.py`
- `projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py`

### 新增

- `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_gt_soft.py`
- `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_gt_soft_pgbr.py`
- `step1_map_only_gt_soft_mining_update.md`

## 建議怎麼用

### Step 1 canonical GT-soft map-only 實驗

使用：

```text
projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_gt_soft.py
```

### GT-soft + PGBR ablation

使用：

```text
projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_gt_soft_pgbr.py
```

## 已做的驗證

已通過：

- `python -m py_compile` 檢查修改過的 Python / config 檔
- `git diff --check` 檢查 patch 格式

## 目前限制

這一版只實作：

- `prototype_pooling_scope='batch_local'`

還沒有做：

- `prototype_pooling_scope='sample'`
- occupancy branch 的 GT-soft mining
- alpha sweep / curriculum
