# Task-Specific Geometry Decomposition Encoder 計劃

日期：2026-05-25

## 一句話定位

> 從 LSS voxel feature 直接分解出 occupancy-specific volumetric feature 與 map-specific topological BEV feature，並用 confidence-gated geometry exchange 讓兩個任務只交換可信的幾何先驗；OCC decoder 固定使用 ProtoOcc `cnn3d_decoder + PQD`，Map decoder 固定使用 BEVFusion-style `BEVSegHead`。

暫定方法名稱：

```text
Task-Specific Geometry Decomposition Encoder (TGDE)
```

這個方法不是「ProtoOcc 加 map head」，也不是「在 ProtoOcc DBE 後面補 adapter」。更準確的定位是：

```text
We replace ProtoOcc's shared Dual Branch Encoder with a task-specific
geometry decomposition encoder that produces separate task features before
fixed occupancy and map decoders.
```

論文主張：

```text
Existing multi-task BEV perception methods often rely on shared BEV features
or semantic prototype filtering. However, occupancy prediction and BEV map
segmentation require different geometric abstractions from the same LSS voxel
feature. We therefore decompose LSS voxel features into task-specific geometry
representations and exchange only reliable geometry priors between tasks.
```

## 動機

目前架構是 ProtoOcc 主幹加上一個 map head：

```text
image backbone / depth / LSS
  -> LSS voxel feature
  -> ProtoOcc Dual Branch Encoder
  -> comprehensive voxel feature
  -> cnn3d_decoder + PQD

same encoder output / BEV feature
  -> map_bev_encoder_neck
  -> BEVSegHead
```

問題是 map 表現明顯落後 map-only upper bound，代表 map head 本身不是唯一瓶頸。現有結果顯示：

| Setting | Occ mIoU | Map mIoU | 意義 |
| --- | ---: | ---: | --- |
| CNN head + 128ch map neck, `map_loss_weight=1` | 39.82 | 39.94 | naive multitask baseline |
| CNN head + 128ch map neck, `map_loss_weight=4` | 39.72 | 45.79 | strongest clean multitask baseline |
| CNN head + 128ch map neck, map-only | - | 48.34 | map upper bound diagnostic |

`map_loss_weight=4` 可以把 map 從 39.94 拉到 45.79，而且 OCC 幾乎不掉，說明問題很大一部分來自任務競爭與 shared representation suppression。若要寫成論文，主線應該放在：

```text
How to generate task-specific geometry features for occupancy and map
segmentation from the same LSS voxel feature.
```

這樣可以對齊 MAESTRO / SOGDet 類型的論文邏輯：

- MAESTRO：主要提出 encoder-side CPG + TSFG + SPA，decoder 使用既有任務 head。
- SOGDet：提出 semantic-occupancy guided fusion，detection / occupancy head 不是主要創新。
- Ours：提出 task-specific geometry decomposition encoder，map head 用 BEVFusion-style head，occ head 用 ProtoOcc PQD。

## 挑戰

### 1. ProtoOcc DBE 不是 task-specific encoder

ProtoOcc 的 Dual Branch Encoder 是：

```text
LSS voxel feature X [B, C, Z, H, W]
  |
  +-- Small-kernel voxel branch
  |
  +-- Large-kernel BEV branch
        -> torch.cat(X.unbind(dim=2), dim=1)
        -> shared multi-scale BEV feature
  |
  +-- fixed hierarchical fusion
        -> comprehensive voxel feature
```

這個設計有 voxel branch 和 BEV branch，但它不是為 OCC / Map 分開學 feature。主要問題：

- `Z` 維度用 fixed concat 壓成 BEV channel，沒有 task-aware height selection。
- OCC 和 Map 共用 BEV trunk，map loss 和 occ loss 會拉同一份 feature。
- HFM 使用固定 addition，沒有判斷 BEV context 對哪個任務有幫助。
- Map neck 只是從 shared BEV 後面補救，source feature 仍然不是 map-specific。

### 2. MAESTRO 的 prototype grouping 不適合直接照搬

MAESTRO 的 CPG / TSFG / SPA 有價值，因為它證明 task-specific feature filtering 可以提升多任務感知。但它的核心是 semantic prototype grouping。對 OCC + BEV map segmentation 來說，map 的關鍵不是只有 semantic category：

- divider / stop line / crossing 是 thin structure。
- walkway / drivable area 需要 topology。
- carpark / boundary 常需要 scene layout。
- OCC 需要 vertical volume、free space、occlusion relation。

因此我們應該借 MAESTRO 的「task-specific filtering」概念，但不要直接使用 foreground/background grouping 或 semantic prototype 取代整個方法。

### 3. SOGDet 的 fusion 概念有用，但固定比例融合太弱

SOGDet 的 task interaction 可以抽象成：

```text
F_task = lambda * F_task + (1 - lambda) * G(F_other)
```

它證明另一個任務的 representation 可以作為 context。但對我們來說，OCC 和 Map 不能直接粗暴互灌 feature，否則容易把 noise 也帶過來。更合理的是：

```text
exchange reliable geometry priors, not raw task features
```

所以 TGDE 要做 confidence-gated bidirectional geometry exchange，而不是固定 `lambda` 加權融合。

## 目標

### 核心目標

取代 ProtoOcc `Dual_Branch_Encoder`，讓 encoder 直接輸出兩個 task-specific feature：

```text
F_occ: occupancy-specific volumetric feature [B, 48, H, W, Z]
F_map: map-specific topological BEV feature [B, 128, H, W]
```

然後固定 task decoders：

```text
F_occ -> ProtoOcc cnn3d_decoder + PQD
F_map -> BEVFusion-style BEVSegHead
```

### 邊界原則

- 不修改 PQD 的 RPL、query matching、mask decoding、loss。
- 不修改 `BEVSegHead` 的 BCE / Dice loss。
- 不把 OCC prediction logits 當 teacher 或 pseudo label。
- 不使用 GT map / GT occ 做 inference-time guidance。
- 方法貢獻集中在 encoder-side task-specific geometry representation。

## 要做什麼

整體架構：

```text
LSS voxel feature X [B, 80, 16, 200, 200]
  |
  +-- Task-Specific Height Projection
  |      +-- X_occ -> Volumetric Occ Encoder
  |      +-- X_map -> Topological Map Encoder
  |
  +-- Geometry Register Bank
  |      +-- ground / boundary / drivable / static-layout
  |      +-- dynamic-volume / free-space / occlusion
  |
  +-- Confidence-Gated Bidirectional Geometry Exchange
  |      +-- O2M: occ volume -> map topology prior
  |      +-- M2O: map topology -> occ geometry prior
  |
  +-- F_occ -> ProtoOcc cnn3d_decoder + PQD
  +-- F_map -> BEVFusion-style BEVSegHead
```

## 模組 A：Task-Specific Height Projection

### 動機

LSS output 是 voxel feature：

```text
X [B, C, Z, H, W]
```

Map 和 OCC 對高度資訊的需求不同：

| Task | 需要的 height cue |
| --- | --- |
| Map | ground plane, lane topology, boundary, low-height static layout |
| OCC | object volume, vertical structure, free space, occlusion |

ProtoOcc 直接使用：

```python
pooled_x = torch.cat(x.unbind(dim=2), dim=1)
```

這會把所有 height bins 當 channel 攤平，沒有任務選擇性。

### 設計

TGDE 先預測兩個 task-specific height gates：

```text
G_occ = HeightGate_occ(X)  # [B, 1, Z, H, W]
G_map = HeightGate_map(X)  # [B, 1, Z, H, W]

X_occ = X * (1 + G_occ)
X_map = X * (1 + G_map)
```

然後分別產生 two task streams：

```text
X_occ -> Volumetric Occ Encoder -> F_occ_base [B, 48, H, W, Z]
X_map -> Topological Map Encoder -> F_map_base [B, 128, H, W]
```

### 實作策略

第一版先使用輕量 gate：

```text
Conv3d(C -> C/4)
BN/ReLU
Conv3d(C/4 -> 2)
Sigmoid
```

輸出兩個 channel：

```text
gate[:, 0] -> G_occ
gate[:, 1] -> G_map
```

初始化時讓 gate 接近 0，保證初期不大幅破壞 pretrained / baseline feature。

## 模組 B：Volumetric Occ Encoder

### 動機

OCC 需要保留 3D volume，而不是過早壓成 BEV。ProtoOcc DBE 的 voxel branch 可以參考，但不應維持原本 shared HFM。

### 設計

```text
X_occ
  -> local 3D conv branch
  -> multi-scale voxel features
  -> geometry-conditioned voxel fusion
  -> F_occ_base [B, 48, H, W, Z]
```

與 ProtoOcc DBE 的差異：

- input 是 `X_occ`，不是原始 shared `X`。
- BEV context 不再從 shared `multi_scale_bev` 固定加回來。
- 後續 M2O prior 只作為 gated residual，不直接覆蓋 voxel feature。

### 建議第一版

為了降低風險，可保留 ProtoOcc voxel branch 的大部分 3D conv 結構，但把融合來源換成 TGDE 的 task-specific branch：

```text
vox_occ = OccVoxelBranch(X_occ)
bev_occ = OccBEVContext(X_occ)
F_occ_base = GeometryFusion(vox_occ, bev_occ)
```

## 模組 C：Topological Map Encoder

### 動機

Map segmentation 的 bottleneck 是 ground topology / thin structure，不是 3D volume reconstruction。Map branch 應該從 `X_map` 產生 map-specific BEV feature，而不是吃 shared BEV。

### 設計

```text
X_map
  -> task-aware height projection
  -> BEV topology backbone
  -> map_bev_encoder_neck
  -> F_map_base [B, 128, H, W]
```

其中 height projection 可先採用：

```text
weighted_z_pool(X_map) 或 cat_z(X_map) + 1x1 Conv
```

第一版建議保留與 baseline 相同的 `map_bev_encoder_neck` 與 `BEVSegHead`，只替換 map source feature，保持比較邊界乾淨。

## 模組 D：Geometry Register Bank

### 動機

MAESTRO 使用 semantic prototypes 做 task-specific filtering。但 OCC + Map 的互補關係更接近 geometry / topology，而不是純 class prototype。

TGDE 使用 geometry registers：

```text
R_ground
R_boundary
R_drivable
R_static_layout
R_dynamic_volume
R_free_space
R_occlusion
```

這些 register 是 encoder memory，不是 final class query，也不是 PQD query。

### 設計

```text
R_geo = RegisterEncoder(X, F_occ_base, F_map_base)
```

第一版不用 transformer cross-attention，先用 pooling + MLP：

```text
global_occ = GlobalPool(F_occ_base)
global_map = GlobalPool(F_map_base)
global_x   = GlobalPool(X)

R_geo = MLP([global_x, global_occ, global_map])
```

再產生 task-specific gates：

```text
gamma_occ = MLP_occ(R_geo)
gamma_map = MLP_map(R_geo)
```

作用方式：

```text
F_occ_reg = F_occ_base + gamma_occ * OccRegisterAdapter(F_occ_base)
F_map_reg = F_map_base + gamma_map * MapRegisterAdapter(F_map_base)
```

### 為什麼這不是 MAESTRO copy

MAESTRO 的 prototype 偏 semantic grouping，TGDE 的 register 偏幾何分解：

| Method | 中間表徵 | 目的 |
| --- | --- | --- |
| MAESTRO | CPG semantic prototypes | task-aware semantic filtering |
| ProtoOcc | DBE comprehensive voxel feature | occupancy feature aggregation |
| TGDE | geometry registers | task-specific geometry decomposition and exchange |

## 模組 E：Confidence-Gated Bidirectional Geometry Exchange

### 動機

OCC 和 Map 應該互相幫助，但不能直接 raw feature fusion。需要只交換可信的 geometry prior。

### O2M：Occupancy to Map

OCC feature 可以提供：

- occupied / free structure
- vertical-to-ground projection
- object / static context
- occlusion-aware support

路徑：

```text
F_occ_reg [B, 48, H, W, Z]
  -> HeightPool / Z-attention
  -> O2M_prior [B, 128, H, W]
  -> alpha_map [B, 1, H, W]
  -> F_map_out = F_map_reg + alpha_map * O2M_prior
```

`alpha_map` 由 map feature、O2M prior、geometry registers 共同預測：

```text
alpha_map = sigmoid(Conv([F_map_reg, O2M_prior, R_geo_map]))
```

### M2O：Map to Occupancy

Map feature 可以提供：

- drivable topology
- walkway / boundary layout
- static ground prior
- map-consistent occupancy support

路徑：

```text
F_map_reg [B, 128, H, W]
  -> GeometryLift
  -> M2O_prior [B, 48, H, W, Z]
  -> alpha_occ [B, 1, H, W, Z]
  -> F_occ_out = F_occ_reg + alpha_occ * M2O_prior
```

第一版 `GeometryLift` 可以用簡單 linear lift：

```text
Conv2d(128 -> 48 * Z)
reshape -> [B, 48, H, W, Z]
```

後續再考慮 z-aware learned basis：

```text
F_map -> K basis maps
height_basis -> Z weights
sum_k basis_k * z_weight_k
```

### Confidence Gate 健康條件

訓練時應記錄：

```text
alpha_map_mean/std/min/max
alpha_occ_mean/std/min/max
O2M_prior_norm / F_map_norm
M2O_prior_norm / F_occ_norm
cosine(F_map_base, F_map_out)
cosine(F_occ_base, F_occ_out)
```

健康現象：

- 初期 `alpha` 不應接近 1，避免直接破壞 baseline。
- 中後期 `alpha` 應該在 boundary / occupied / drivable region 形成差異。
- O2M / M2O prior norm 不應長期接近 0，否則 exchange 沒有學到。

## 預計改動哪些檔案

### 新增檔案

```text
projects/mmdet3d_plugin/models/backbones/task_specific_geometry_decomposition_encoder.py
```

放 TGDE 主體：

```text
TaskSpecificGeometryDecompositionEncoder
HeightTaskGate
VolumetricOccEncoder
TopologicalMapEncoder
GeometryRegisterBank
BidirectionalGeometryExchange
```

### 修改檔案

```text
projects/mmdet3d_plugin/models/backbones/__init__.py
```

註冊新 backbone。

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py
```

新增 TGDE variant 或另開 config：

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_tgde.py
```

建議另開新 config，不直接覆蓋 baseline。

### 可能新增訓練腳本

```text
TWCC/train_multi_cnn_head_tgde.sh
```

保持與 weight4 baseline 相同的 global batch / LR，避免 optimization regime 混淆。

### 不應修改

第一版不要改：

```text
projects/mmdet3d_plugin/models/dense_heads/Prototype_Query_Decoder_nuScenes.py
projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py
projects/mmdet3d_plugin/models/dense_heads/cnn3d_decoder.py
```

原因：保持 decoder 固定，才能把 gain 歸因到 TGDE encoder。

## 實作階段

### Stage 0：建立 baseline 對齊

目標：確保比較對象是 strongest clean baseline。

設定：

```text
config: ProtoOcc_multi_cnn_head_map_neck.py
map_loss_weight=4.0
decoder: ProtoOcc PQD + BEVFusion-style BEVSegHead
```

記錄：

```text
Occ mIoU = 39.72
Map mIoU = 45.79
```

### Stage 1：只做 Task-Specific Height Projection

目標：驗證「從 LSS voxel 分出 X_occ / X_map」是否有效。

```text
X -> HeightTaskGate -> X_occ / X_map
X_occ -> Occ branch -> F_occ
X_map -> Map branch -> F_map
```

不開 geometry register，不開 O2M / M2O。

判斷：

- Map 是否超過 45.79。
- OCC 是否接近或超過 39.72。
- gate 是否學出不同 z distribution。

### Stage 2：加入 Geometry Register Bank

目標：驗證 scene-level geometry memory 是否比 local conv 更有用。

```text
F_occ_base / F_map_base / X -> R_geo
R_geo -> task-specific channel/spatial gates
```

判斷：

- divider / stop_line / walkway 是否改善。
- OCC 的 free / static / object classes 是否穩定。

### Stage 3：O2M only

目標：測 OCC geometry 對 map topology 的幫助。

```text
F_occ -> HeightPool -> O2M prior
F_map = F_map + alpha_map * O2M
```

不開 M2O。

判斷：

- Map 是否接近 map-only upper bound 48.34。
- OCC 是否不受影響。

### Stage 4：M2O only

目標：測 map topology 對 OCC volume 的幫助。

```text
F_map -> GeometryLift -> M2O prior
F_occ = F_occ + alpha_occ * M2O
```

不開 O2M。

判斷：

- OCC 是否超過 baseline。
- Map 是否不受影響。

### Stage 5：Bidirectional TGDE full model

目標：完整方法。

```text
Task-specific height projection
+ geometry registers
+ O2M confidence-gated prior
+ M2O confidence-gated prior
```

這是 paper main result。

## Ablation 設計

| ID | Setting | 目的 |
| --- | --- | --- |
| A0 | weight4 map-neck baseline | strongest clean baseline |
| A1 | TGDE height projection only | 測 task-specific decomposition |
| A2 | A1 + geometry registers | 測 scene-level geometry memory |
| A3 | A2 + O2M only | 測 OCC -> Map |
| A4 | A2 + M2O only | 測 Map -> OCC |
| A5 | A2 + bidirectional O2M/M2O | main method |
| A6 | same params, shared feature no task split | 排除只是 capacity 增加 |
| A7 | fixed lambda fusion like SOGDet | 證明 confidence gate 優於固定融合 |
| A8 | no geometry registers | 證明 registers 的價值 |
| A9 | no confidence gate, direct addition | 證明不是 raw fusion 就有效 |

關鍵 comparison：

```text
A1 vs A0: 是否 task-specific height decomposition 本身有效
A5 vs A0: TGDE 是否超越 ProtoOcc DBE baseline
A5 vs A6: 是否真的需要 task split，而不是只加參數
A5 vs A7: 是否比 SOGDet-style fixed lambda fusion 更好
A5 vs MAESTRO-2T: 是否比 prototype-guided task filtering 更適合 OCC+Map
```

## Debug 指標

### Height gate

```text
height_gate_occ_z_distribution
height_gate_map_z_distribution
mean_abs(G_occ - G_map)
```

期待：

- `G_map` 對低高度 / ground 相關 bin 更敏感。
- `G_occ` 對 vertical bins 更分散。
- 兩個 gate 不應完全相同。

### Geometry register

```text
register_norm
register_cosine_offdiag
gamma_occ_mean/std
gamma_map_mean/std
```

期待：

- registers 不 collapse。
- task gates 有差異。

### Bidirectional exchange

```text
alpha_map_mean/std/min/max
alpha_occ_mean/std/min/max
O2M_prior_norm / F_map_norm
M2O_prior_norm / F_occ_norm
```

期待：

- `alpha` 初期偏小，中後期逐漸學出空間差異。
- O2M 對 map thin classes 有幫助。
- M2O 不應讓 OCC 大幅掉點。

### 任務梯度

```text
map loss gradient norm on shared / task-specific modules
occ loss gradient norm on shared / task-specific modules
cosine gradient(map, occ)
```

期待：

- map gradient 不再被 OCC 完全壓制。
- task-specific branch 的 gradient conflict 低於 shared DBE baseline。

## 預期效果

### 最低成功標準

```text
Map mIoU > 45.79
Occ mIoU >= 39.5
```

代表 TGDE 至少比 weight4 baseline 有 map gain，且 OCC 沒有明顯犧牲。

### 強成功標準

```text
Map mIoU >= 47.0
Occ mIoU >= 39.72
```

代表接近 map-only upper bound，且 OCC 不輸 baseline。

### Paper-level 成功標準

```text
Map mIoU close to or above 48.34 map-only upper bound trend
Occ mIoU improves over 39.72
TGDE > MAESTRO-2T under same decoder / protocol boundary
```

若能達到，論文敘事可以是：

```text
TGDE closes the gap between naive OCC+Map multitask learning and the
map-only upper bound while preserving or improving occupancy performance.
```

## 論文貢獻寫法

### Contribution 1：Task-Specific Geometry Decomposition

```text
We propose a task-specific geometry decomposition encoder that replaces the
shared DBE representation with occupancy-specific volumetric features and
map-specific topological BEV features directly from LSS voxel features.
```

### Contribution 2：Geometry Register Bank

```text
Instead of semantic prototype grouping, we introduce geometry registers that
capture scene-level ground, boundary, free-space, and volume priors for
task-aware feature modulation.
```

### Contribution 3：Confidence-Gated Bidirectional Geometry Exchange

```text
We exchange reliable geometry priors between occupancy and map branches using
learned confidence gates, avoiding negative transfer caused by raw feature
sharing or fixed-ratio fusion.
```

### Contribution 4：Fixed Decoder Verification

```text
With ProtoOcc PQD and BEVFusion-style BEVSegHead unchanged, improvements can
be attributed to the proposed encoder rather than task-specific decoder
redesign.
```

## 與相關方法的差異

| Method | 核心 | Decoder 是否主要貢獻 | 與 TGDE 差異 |
| --- | --- | --- | --- |
| ProtoOcc DBE | voxel + BEV dual branch, fixed HFM | no | shared feature, not task-specific |
| MAESTRO | CPG + TSFG + SPA semantic prototype filtering | no | prototype grouping，不是 voxel geometry decomposition |
| SOGDet | semantic occupancy guided detection fusion | no | fixed / simple branch fusion，沒有 confidence-gated geometry exchange |
| TGDE | task-specific geometry decomposition + gated exchange | no | 直接從 LSS voxel 產生 OCC / Map 專屬 feature |

## 風險與處理

### 風險 1：完整替換 DBE 後 OCC 掉太多

處理：

- Stage 1 先保留 ProtoOcc voxel branch 結構，只替換 input split 與 fusion source。
- `F_occ_out` channel / shape 完全對齊原 PQD input。
- 可加入 residual fallback：

```text
F_occ_out = F_occ_new + beta * F_occ_protoocc_like
```

初期 `beta` 可設為 learnable small value 或 zero-init。

### 風險 2：Map gain 只是因為參數變多

處理：

- 做 A6：same params, no task split。
- 做 A7：SOGDet-style fixed lambda fusion。
- 如果 A5 明顯優於 A6 / A7，才能主張 task-specific decomposition 有效。

### 風險 3：Bidirectional fusion 造成 negative transfer

處理：

- O2M / M2O 分開 ablation。
- confidence gate zero-init / low-init。
- 先讓 task-specific branches 自己穩定，再開 exchange。

### 風險 4：Geometry registers 沒有學到東西

處理：

- 先用 pooling + MLP，不用重 transformer。
- 加 register diversity monitoring。
- 若 registers collapse，再考慮 entropy / orthogonality regularization，但不要第一版就加。

## 最終判斷

TGDE 比「改良 ProtoOcc DBE」更適合作為論文主方法。原因是：

1. 方法邊界清楚：replace DBE, keep decoders fixed。
2. 對比清楚：ProtoOcc 是 shared dual branch，TGDE 是 task-specific geometry decomposition。
3. 可以自然對齊 MAESTRO / SOGDet 的論文模式：提出中間表徵與融合方法，而不是重新發明 decoder。
4. 實驗可歸因：只要固定 PQD 和 BEVSegHead，gain 就能歸到 encoder。

推薦主線：

```text
不是「ProtoOcc + Map Head」
而是「LSS voxel feature -> TGDE -> fixed OCC / Map decoders」
```

