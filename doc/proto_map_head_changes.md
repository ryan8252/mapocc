# Prototype-based Map Head — 改動說明

## 背景：為什麼要改

### 我們在做什麼

把 BEV map segmentation 的預測 head，從 **Naive CNN head（BEVSegHead）** 升級為 **Scene-Aware Prototype Map Decoder（ProtoMapHead）**，讓 map 任務也採用 prototype query + dot-product mask decoding 的設計哲學。

---

## 對標論文：MAESTRO（arXiv 2509.17462）

MAESTRO 的整體架構：
- **CPG（Class-wise Prototype Generator）**：從 shared voxel feature 產生 foreground/background prototype groups
- **TSFG（Task-Specific Feature Generator）**：用 prototype 對 shared feature 做 Adaptive Feature Enhancement（dot-product activation map）+ Feature Suppression，輸出 task-specific BEV features（F^TS_Map）
- **Map Head**：接收 F^TS_Map → 經 downstream BEV map segmentation head 輸出 class-wise masks
- **SPA（Scene Prototype Aggregator）**：從 map head 和 det head 的輸出做 masked average pooling，得到 P_Map 和 P_Det，送進 occupancy decoder 強化 occ 任務

### MAESTRO map branch 的實際限制

MAESTRO 的 map branch 走的是：

```
shared voxel feature
    → CPG prototype（background group）
    → TSFG: feature enhancement + suppression
    → F^TS_Map（task-specific BEV feature）
    → Map Head（CNN dense predictor）
    → predicted masks
```

**Prototype 的角色**：在 TSFG 裡主要作為 task-specific feature refinement 的語意先驗（prototype 和 feature 做 dot product 產生 activation map，再配合 suppression module 過濾無關訊號）。Prototype 本身不是 final map mask 的直接解碼器；最終 map mask 仍由 downstream segmentation head 產生。

**具體限制**：
1. **Map 最終預測仍依賴 task head**：prototype 主要在 TSFG 中引導 feature refinement，但 final mask 並不是由 class-fixed prototype query 直接解碼
2. **無跨場景 prototype 記憶**：prototype 由當前 shared feature 即時計算，論文中沒有引入 EMA-style memory bank 去累積跨 iteration / 跨場景統計
3. **Map task 的 per-class prototype 沒被直接落成 final decoder queries**：MAESTRO 的 CPG 雖然先產生 per-class prototype `P_k`，但 map task 端實際使用的是 background prototype group `G_Map` 來做 feature enhancement，沒有把每個 map class 顯式對應到最終的 class-fixed query decoder

---

## 我們攻擊的點（Step 1 本身即為貢獻）

**攻擊點 1：Map 最終解碼機制**
MAESTRO 的 map branch 仍走 task-specific feature → CNN dense predictor 這條路。Prototype 引導 feature transformation（TSFG），但 final mask 由 CNN 輸出，prototype 不是 mask 的直接解碼器。
我們：每個 map class 有獨立的 prototype query，直接和 mask feature 做 dot product → final mask。Prototype 是預測本身，不是引導 feature 的工具。

**攻擊點 2：跨場景 prototype 記憶**
MAESTRO 的 prototype 每次都從當前場景的 voxel feature 重新算，沒有跨場景的累積統計。
我們：引入 EMA bank（AgnoPG），在訓練過程中持續積累每個 map class 的全局語意統計，讓 prototype 不只反映當前場景，也包含歷史場景的語意分布。

**攻擊點 3：per-class prototype 的粒度**
MAESTRO 的 CPG 雖然會先生成 per-class prototype，但在 map branch 中主要是以 background prototype group 的形式參與 TSFG，沒有把每個 map class 顯式落成 final class-fixed decoder query。
我們：6 個 map class 各有獨立的 prototype query 和獨立的 EMA，語意粒度更細，且直接對應最終 mask decoding。

> **Step 1 的貢獻定位**：上述三個攻擊點在 Step 1 就已成立，不需要等到 Step 2（Shared Prototype Bank）。Step 1 本身是一個獨立的貢獻——「把 Scene-Aware Prototype Decoding 從 3D occupancy 延伸到 2D BEV map segmentation」。Step 2 是在此基礎上進一步挑戰 MAESTRO 的跨任務 prototype 獨立性。

---

## 改動的檔案清單

### 1. 新增 `projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py`

**實作 `ProtoMapHead`，保留 ProtoOcc `Prototype_Query_Decoder_nuScenes` 的核心 query formation / prototype decoding 主幹，並針對 2D BEV map segmentation 重新落地：**

#### Step 1：CNN Proto Generator（對應 `cnn3d_decoder`）

```
bev_feature (B, 48, 200, 200)
    │
    ▼
2× Conv2D(48→96) + BN + ReLU  →  mask_feat (B, 96, 200, 200)
    │
    └─ coarse_predictor: Conv2D(96→6)
           → coarse_map_pred (B, 6, 200, 200)   ← 粗預測，直接算 loss
```

#### Step 2：AdaPG（Scene-Adaptive Prototype Generator）

對應 ProtoOcc 的 argmax + confidence filtering。  
Map supervision 採用的是 **class-wise binary masks** `(B, C, H, W)`，因此這裡對每個 map class 獨立使用 **sigmoid > conf_thresh** 篩選 confident positive pixels，而不是在 6 個 class 之間做 softmax 互斥分類：

```
soft_mask = sigmoid(coarse_map_pred)
conf_mask_k = soft_mask[:, k] > 0.5          ← confident positive pixels
prototype_k = mean(mask_feat[conf_mask_k])   ← masked avg pooling

for_query: (6, 96)  ← per-class local scene prototype
                       （概念上對應 scene-adaptive prototype；實作沿用 ProtoOcc released code 的 batch-level pooling 寫法）
```

ProtoOcc 用 softmax argmax，是因為 occ supervision 是互斥類別。  
這裡用 sigmoid，較準確的理由不是「每個 pixel 一定應該同時屬於多個 class」，而是「目前的 map GT / loss / evaluation 都採 per-class binary mask 形式，不強制一個 `0.4m × 0.4m` 的 BEV cell 在 6 類之間互斥」。當不同 HD map layers 在 rasterization 後落進同一個 cell 時，該 cell 可以在多個 class channel 同時為 positive。

進一步地，若某個 cell 對多個 class 都是 confident positive，它會同時參與那些 class 的 prototype pooling；這裡不是為每個 cell 從多個 class 中只選一個 prototype，而是每個 class 各自估計自己的 prototype query。

**Empty-class fallback**：若某個 class 在當前 batch 完全沒有 confident pixel（`conf_mask.sum() == 0`），`for_query[k]` 退化為 zero vector，和 ProtoOcc released code 的處理方式完全一致（見 `Prototype_Query_Decoder_nuScenes.py` 第 362–367 行）。Zero vector 不更新 EMA（`cur_assign_flag[k] = False`），self-attn 後的影響由其他兩路 query（learnable + EMA）補償。

#### Step 3：AgnoPG（Scene-Agnostic Prototype Generator / EMA bank）

```
prototype_EMA_feat: nn.Embedding(6, 96)   ← 跨場景全局 prototype 記憶
proto_first_flag: register_buffer         ← 可正確儲存進 checkpoint

訓練時 EMA 更新：
  - 第一次看到此 class：直接複製 for_query
  - 之後：EMA = EMA × (1 - 0.01) + for_query × 0.01
  （EMA weight = 0.01 對齊 ProtoOcc 原版預設值 prototpye_EMA_weight=0.01，
   等同 momentum=0.99，保證跨場景積累時更新穩定不震盪）

推論時：EMA update 在 `if self.training:` block 內，推論不更新 bank，
直接使用凍結的 `prototype_EMA_feat.weight`；和 ProtoOcc released code 行為一致。

ema_query: (6, 96)
```

#### Step 4：Scene-Aware Query 合成（三合一，延續 ProtoOcc 的核心邏輯）

```
query_feat = learnable_query.weight                   (6, 96)
           + global_protoEMA_agg(ema_query)           (6, 96)  ← 全局 EMA 分支
           + local_protoEMA_agg(for_query)            (6, 96)  ← 局部場景分支
query_feat = for_query_embed(query_feat)              (6, 96)  ← Scene-Aware Query
```

當 `use_scene_adaptive / use_ema_bank / use_learnable_query` 都啟用時，這裡保留了 ProtoOcc released code 的主幹思路：learnable query、global EMA prototype、local adaptive prototype 三路相加，再送入 `for_query_embed`。

#### Step 5-6：Self-Attention + Dot Product（保留核心解碼，簡化 self-attn 模組）

```
query: (6, B, 96)  ← expand batch
    │
    └─ MultiheadAttention (self-attn, no bipartite matching)
    │
    └─ mask_embed MLP
    │
    └─ einsum('bkc,bchw->bkhw', mask_embed, mask_feat)
           → final_masks (B, 6, 200, 200)
```

這裡用 `nn.MultiheadAttention` 實作 class query 間的 self-attn；相較 ProtoOcc released code 的 `Query_Transformer_RPL`，我們沒有搬入 RPL 專用的 pad / self-attn mask 機制，但最核心的 `mask_embed(query) × mask_feat` 解碼邏輯保持一致。

**注意**：目前沿用 ProtoOcc released code 的 class-fixed query 設計，**不額外導入 bipartite matching**（與 Mask2Former 不同）。因為 6 個 map class 各自對應固定的一個 prototype query，預測結果直接對應 class label。

**RPL（Robust Prototype Learning）刻意略過**：ProtoOcc 的 RPL 是訓練期對 binary mask 加 noise/shift 生成額外 augmented queries，是一種 query-level data augmentation。這部分不改變核心的 prototype query formation 與 `mask_embed × mask_feat` 解碼機制，但會引入額外的 query augmentation、padding 與對應 loss；目前先保留 AdaPG + AgnoPG + Scene-Aware Query 三合一作為 Step 1 的主要主體，RPL 留待後續補入。

**Loss 設計：**

| Loss key | 對象 | 意義 |
|---|---|---|
| `loss_map_coarse_bce` + `loss_map_coarse_dice` | `coarse_map_pred` | 粗預測監督，幫助 AdaPG 生成有意義的 prototype |
| `loss_map_focal` + `loss_map_dice` | `final_masks` | 最終 mask 監督（其中 `loss_map_focal` 採用 **multi-label binary mask focal loss**，對每個 map class 的 channel 分別做 sigmoid focal；γ=2, α=0.25，用來處理 sparse map class 的 foreground-background imbalance） |

**重要實作註記（2026-04-19 補充）**：

- `gt_masks_bev` 的 supervision 形式是 **class-wise binary mask**，shape 為 `(B, C, H, W)`，不是單一 `class index map`
- 因此 `loss_map_focal` **不能直接使用** MMCV / MMDetection 內建的 stock `FocalLoss`（其 CUDA 路徑要求 `target` 為 `long` 且語意上是 class index）
- 目前實作改為 project-local 的 `BinaryMaskFocalLoss`，本質上是 **sigmoid focal loss for dense multi-label masks**，和 BEVFusion 的 map loss 設計哲學一致，只是我們保留 ProtoMapHead 的 prototype query decoder 架構
- 換句話說，`loss_map_focal` 監督的是 `final_masks` 的 **per-class binary occupancy over BEV cells**，而不是像 Mask2Map / Mask2Former 那樣做 query-to-instance mask matching

---

### 2. 新增 `projects/mmdet3d_plugin/models/detectors/ProtoOccMultitask.py`

把 `bev_seg_head`（CNN head）換成 `proto_map_head`（Scene-Aware Prototype Decoder）。

**不修改** `ProtoOccCnnSegHead.py`（Naive MTL baseline，保留作為 ablation）

改動的三個地方：

```python
# __init__
self.proto_map_head = build_head(proto_map_head)

# forward_train
coarse_pred, final_masks = self.proto_map_head(bev_feature)
losses.update(self.proto_map_head.loss(coarse_pred, final_masks, gt_masks_bev))

# simple_test
_, final_masks = self.proto_map_head(bev_feature)
map_probs = self.proto_map_head.predict(final_masks)
```

---

### 3. 修改 `projects/mmdet3d_plugin/models/dense_heads/__init__.py`

新增 `ProtoMapHead` 的 import / export。

### 4. 修改 `projects/mmdet3d_plugin/models/detectors/__init__.py`

新增 `ProtoOccMultitask` 的 import / export。

### 5. 新增 `projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py`

與 `ProtoOcc_multi_cnn_head.py` 的差異：

| 項目 | Naive MTL | Proto Map Head |
|---|---|---|
| `model.type` | `'ProtoOccCnnSegHead'` | `'ProtoOccMultitask'` |
| Map head key | `bev_seg_head` | `proto_map_head` |
| Map head type | `'BEVSegHead'` | `'ProtoMapHead'` |
| Loss | `loss_bce` + `loss_dice` | `loss_coarse_bce` + `loss_coarse_dice` + `loss_mask_focal` + `loss_mask_dice`（四項，其中 `loss_mask_focal` 為 `BinaryMaskFocalLoss`，不是 stock `FocalLoss`） |

---

### 6. 推論與評估行為

- `simple_test` 階段只使用 `final_masks`，再經 `sigmoid` 輸出 `map_probs`
- `predict()` 回傳的是連續機率圖，不在 head 內硬閾值化
- `NuScenesDatasetMultitask.evaluate_map()` 會 sweep 多個 threshold（0.35 到 0.65），回報各類別 `iou@max` 與 `map/mean/iou@max`
- 若需要輸出單張彩色 BEV 視覺化，必須另外定義 **display priority**；這個規則只用於顯示，不代表訓練 supervision 互斥，也不參與 loss / evaluator

這代表 ProtoMapHead 的訓練和推論 protocol，和原本 `BEVSegHead` 一樣維持 **probability map first, threshold in evaluator** 的評估方式，只是 logits 的來源從 naive CNN predictor 改成 prototype decoder。

**目前使用的 display priority（對齊 `vis_occ_map_gt.py` 的 `MAP_DRAW_ORDER`）**：

```
drivable_area
    → carpark_area
    → walkway
    → ped_crossing
    → divider
    → stop_line
```

這個順序代表繪圖時採「後畫覆蓋前畫」的 layer-style rendering；因此單張彩色圖上最終可見的優先權為：

```
stop_line > divider > ped_crossing > walkway > carpark_area > drivable_area
```

---

## Ablation Table（預期）

| Method | Occ mIoU | Map mIoU | 說明 |
|---|---|---|---|
| ProtoOcc single-task | ~39.6 | — | 原始 ProtoOcc，僅 occ |
| Naive MTL（CNN head） | TBD | TBD | `ProtoOccCnnSegHead` |
| **Scene-Aware Proto Map（本次）** | TBD | TBD | `ProtoOccMultitask`，Step 1 |
| Shared Prototype Bank（下一步） | TBD | TBD | Step 2，occ ↔ map prototype 共享 |

---

## 對 MAESTRO 的 challenge 總結

| 面向 | MAESTRO | 我們（Step 1）|
|---|---|---|
| Map 最終預測 | downstream task head 產生 final mask（prototype 主要引導 feature） | Prototype 直接 dot product → mask |
| 跨場景 prototype 記憶 | 無（每場景重算） | EMA bank（AgnoPG） |
| Per-class prototype 落地方式 | per-class prototype 先形成 group，再做 task-specific feature refinement | 每個 map class 獨立 prototype query + EMA，直接參與最終 decoding |
| 下一步可擴展 | prototype 獨立於各任務 | 天然支援 Shared Prototype Bank（Step 2）|
