# Prototype-based Map Head — 改動說明

## 背景：為什麼要改

### 我們在做什麼

這次改動的目標是把 BEV map segmentation 的預測 head，從 **Naive CNN head（BEVSegHead）** 升級為 **Prototype-based Map Head（ProtoMapHead）**，使其和 ProtoOcc 的 3D occupancy 預測採用相同的 prototype 機制。

### 對標論文：MAESTRO

MAESTRO（arXiv 2509.17462）是目前 nuScenes 上 3-task MTL（Detection + Map Seg + Occupancy）的 SOTA。其核心設計是：

- **CPG（Class-wise Prototype Generator）**：從 shared voxel feature 產生 foreground/background prototype groups
- **TSFG（Task-Specific Feature Generator）**：用 prototypes 做 feature enhancement / suppression，產生 task-specific features
- **Map Head**：接收 TSFG 產生的 task-specific BEV feature，用**普通 CNN** 做 map 預測

**MAESTRO 的關鍵弱點**：prototype 只被用來「引導 feature」（TSFG 的 attention gate），而非直接驅動 map 的最終預測。Map head 本身仍然是一個 class-agnostic 的 CNN，沒有明確的 class-level prototype 語意。

### 我們攻擊的點

> MAESTRO 的 BEV map segmentation head 是 CNN-based，prototype 僅作為 feature enhancement 的引導信號，不參與 mask 預測本身。
>
> 我們提出 **ProtoMapHead**：每個 map class 各有一個 prototype vector，這個 prototype 經過 self-attention 和 cross-attention 後，直接與 BEV feature 做 dot product 生成最終 mask。Prototype 不再只是調味料，而是 map 預測的核心驅動力。

---

## 改動的檔案清單

### 1. 新增 `projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py`

**做什麼：** 實作 `ProtoMapHead`，包含兩個子模組：

**Module 1 — CNN Prototype Generator（對應 ProtoOcc 的 `cnn3d_decoder`）**

```
bev_feature (B, 48, 200, 200)
    │
    ▼
2× Conv2D(48→96) + BN + ReLU  →  mask_feat (B, 96, 200, 200)
    │
    ├─ coarse_predictor: Conv2D(96→6)
    │       → coarse_map_pred (B, 6, 200, 200)   ← 粗預測，直接算 loss
    │
    └─ Masked Average Pooling:
            soft_mask = sigmoid(coarse_map_pred)
            prototype_k = Σ(mask_k × feat) / Σ(mask_k)
            → prototypes (B, 6, 96)               ← 每個 map class 一個向量
```

**Module 2 — Prototype Query Decoder 2D（對應 ProtoOcc 的 `Prototype_Query_Decoder`，大幅簡化）**

```
prototypes (B, 6, 96)   +   mask_feat (B, 96, 200, 200)
    │
    ├─ Self-Attention: 6 prototype queries 互看
    ├─ Cross-Attention: prototype queries attend to BEV feature（100×100）
    ├─ FFN
    │
    └─ Dot Product:
            mask_embed = MLP(queries)               # (B, 6, 96)
            final_masks = mask_embed × mask_feat    # (B, 6, 200×200)
            → reshape → (B, 6, 200, 200)
```

**為什麼不需要 Hungarian matching：**
ProtoOcc 的 occ decoder 需要 Hungarian matching 因為 query 數量（18）> class 數量（18），且是 instance-level 任務。Map seg 是 class-fixed 的（6 個 class 各對應一個 prototype），query 和 class 一對一，不需要 matching。

**Loss 設計：**

| Loss key | 對象 | 意義 |
|---|---|---|
| `loss_map_coarse_bce` | `coarse_map_pred` | 粗預測監督，幫助 prototype 生成 |
| `loss_map_coarse_dice` | `coarse_map_pred` | 同上 |
| `loss_map_bce` | `final_masks` | 最終 mask 監督 |
| `loss_map_dice` | `final_masks` | 同上 |

---

### 2. 新增 `projects/mmdet3d_plugin/models/detectors/ProtoOccMultitask.py`

**做什麼：** 新的 detector class，把 `bev_seg_head`（CNN head）換成 `proto_map_head`（prototype head）。

**不修改**：`ProtoOccCnnSegHead.py`（Naive MTL baseline，保留作為 ablation）

**改動的三個地方：**

1. `__init__`：`build_head(proto_map_head)` 代替 `build_head(bev_seg_head)`
2. `forward_train`：
   ```python
   # 舊（CNN head）
   bev_seg_logits = self.bev_seg_head(bev_feature)
   losses.update(self.bev_seg_head.loss(bev_seg_logits, gt_masks_bev))

   # 新（Prototype head）
   coarse_pred, final_masks = self.proto_map_head(bev_feature)
   losses.update(self.proto_map_head.loss(coarse_pred, final_masks, gt_masks_bev))
   ```
3. `simple_test`：
   ```python
   # 舊
   bev_seg_probs = self.bev_seg_head.predict(self.bev_seg_head(bev_feature))

   # 新
   _, final_masks = self.proto_map_head(bev_feature)
   map_probs = self.proto_map_head.predict(final_masks)
   ```

---

### 3. 修改 `projects/mmdet3d_plugin/models/dense_heads/__init__.py`

新增 `ProtoMapHead` 的 import 和 export。

---

### 4. 修改 `projects/mmdet3d_plugin/models/detectors/__init__.py`

新增 `ProtoOccMultitask` 的 import 和 export。

---

### 5. 新增 `projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py`

**和 `ProtoOcc_multi_cnn_head.py` 的差異：**

| 項目 | Naive MTL（CNN head） | Proto Map Head |
|---|---|---|
| `model.type` | `'ProtoOccCnnSegHead'` | `'ProtoOccMultitask'` |
| Map head key | `bev_seg_head` | `proto_map_head` |
| Map head type | `'BEVSegHead'` | `'ProtoMapHead'` |
| Loss | `loss_bce` + `loss_dice` | `loss_coarse_bce` + `loss_coarse_dice` + `loss_mask_bce` + `loss_mask_dice` |

---

## Ablation Table（預期）

| Method | Occ mIoU | Map mIoU | 說明 |
|---|---|---|---|
| ProtoOcc single-task | ~39.6 | — | 原始 ProtoOcc |
| Naive MTL（CNN head） | TBD | TBD | `ProtoOccCnnSegHead` |
| **Proto Map Head（本次）** | TBD | TBD | `ProtoOccMultitask` |
| Shared Prototype Bank（下一步） | TBD | TBD | Step 2 |

---

## 對 MAESTRO 的 challenge 總結

| 面向 | MAESTRO | 我們 |
|---|---|---|
| Map head 類型 | CNN（task-specific feature 後接普通 conv） | Prototype-based（prototype 直接驅動 mask 預測） |
| Prototype 的角色 | Feature enhancement 的 gate（調味料） | Mask 預測的核心（class-level semantic query） |
| Class-level 語意 | FG/BG hardcoded 分組 | 每個 map class 獨立 prototype vector |
| 下一步可擴展 | 無法直接共享 occ/map prototype | 天然支援 Shared Prototype Bank（Step 2）|
