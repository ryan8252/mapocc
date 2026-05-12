# Voxel-aware Map Ingest (VAMI) 計劃

日期：2026-05-12  
更新：2026-05-12（codex code review 後）。前一版以 AVSS（Auxiliary Voxel Semantic Supervision）為核心論點，但對 code 的診斷錯誤，已修正。

## 一句話定位

> ProtoOcc 並**不缺** voxel-level class supervision：`cnn3d_decoder` 已經對 `comprehensive_voxel_feature` 做 18-class CE + Lovász 監督。真正的缺口是 **map branch 在 HFM 之前就從 `multi_scale_bev` 分岔出去，從未使用 voxel-supervised feature**。VAMI 補的是這條使用路徑，不是補 supervision 本身。

```text
PGBR / C2：改 prototype consumption（feature 沒變強）
LGMG O2M：在 logits 加 residual（沒結構化監督）
舊 AVSS plan：補 voxel-level supervision（但 supervision 早就有）
VAMI（本 plan）：直接讓 map 使用既有的 voxel-supervised feature
```

AVMC（auxiliary voxel mask classifier）從前一版主線降為**可選 ablation**，因為 ProtoOcc 已經有 18-class CE+Lovász，再疊一個淺 Dice/Lovász head 是「再加 OCC regularizer」，不是「補缺失監督」，論點站不穩。

## Code-level 診斷

### A. `comprehensive_voxel_feature` 已經被 voxel-level supervise

`projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py:162-172`:

```python
prototoype_occ_pred, mask_feat = self.cnn3d_decoder(
    comprehensive_voxel_feature.permute(0,1,4,2,3))
...
loss_prototype = self.cnn3d_decoder.loss(
    prototoype_occ_pred, voxel_semantics, mask_camera)
```

`projects/mmdet3d_plugin/models/dense_heads/cnn3d_decoder.py:144-147`:

```python
voxel_loss = self.cross_entropy_loss(occ_pred, voxel_semantics.long())
lovasz_softmax_loss = self.lovasz_softmax_loss(
    F.softmax(occ_pred, dim=1), voxel_semantics, ignore=255)
loss['loss_CE_prototype'] = voxel_loss * self.loss_weight
loss['lovasz_softmax_loss_prototype'] = lovasz_softmax_loss * self.loss_weight
```

→ `comprehensive_voxel_feature` 經由 `cnn3d_decoder.final_conv` 和 `predicter` 已被 **18-class CE + Lovász** 監督，這是 MAESTRO 那種 voxel-level multi-class mask supervision 的對應物，**ProtoOcc 早就有**。

### B. Map branch 完全沒接觸這個 voxel-supervised feature

`projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py:171-194`:

```python
# 在 HFM 之前
map_bev_feature = None
if self.map_bev_encoder_neck is not None:
    map_source = multi_scale_bev           # ← 從 HFM 上游分岔
    map_bev = self.map_bev_encoder_neck(map_source)
    map_bev_feature = map_bev[0]

# Hierarchical Fusion Module 從 line 180 才開始
vox3 = vox3 + vox2 + self.bev_ch1(multi_scale_bev[2])...
...
comprehensive_voxel_feature = vox + vox_raw   # ← HFM 之後才生
```

→ `map_bev_feature` 是從 `multi_scale_bev` 跑 `map_bev_encoder_neck` 出來的，整個流程**和 HFM 平行**，**從未使用** `comprehensive_voxel_feature`。

### 結論

> Map 看的是 HFM 上游的 BEV feature，OCC supervision 雕過的 `comprehensive_voxel_feature` 從沒進 map 路徑。VAMI 要做的就是補這條使用路徑。

## 方法設計

### 主模組：VAMI（Voxel-aware Map Ingest）

目的：把 `comprehensive_voxel_feature` 收 Z 進 BEV，融合到 `map_bev_feature`，再給 `BEVSegHead`。

實作位置：**detector 端**（`ProtoOccCnnSegHead.py`），**不改 `Dual_Branch_Encoder`、不改 `BEVSegHead`、不改 `cnn3d_decoder`、不改 `PQD`**。

流程：

```text
comprehensive_voxel_feature [B, C_v, X, Y, Z]              (已存在，已被 OCC voxel loss 監督)
  -> optional detach on voxel source                         (只切 map loss -> HFM/DBE，不切 VAMI 自身參數)
  -> Z-collapse: avg_pool + max_pool over Z, channel concat
  -> [B, 2*C_v, X, Y]
  -> Conv2d(2*C_v -> C_map_in, kernel=1) + BN + ReLU      (channel align)
  -> voxel_aware_feat [B, C_map_in, X, Y]

map_bev_feature [B, C_map, H, W]                           (已存在，目前直接進 BEVSegHead)
voxel_aware_feat 經 bilinear resize 對齊 (H, W)            (避免 shape 不匹配)
  -> concat with map_bev_feature  -> [B, C_map + C_map_in, H, W]
  -> FusionConv: Conv2d(... -> C_map) + BN + ReLU
  -> Conv2d(C_map -> C_map)
  -> map_bev_feature_fused [B, C_map, H, W]                (residual: + map_bev_feature)
  -> BEVSegHead (既有，不動)
```

設計細節：

- **detach 預設 True，但只 detach source**：先做 `voxel_source = comprehensive_voxel_feature.detach()`，再把 `voxel_source` 丟進 VAMI 的 Z-collapse / projection / fusion。原因是 map loss 不應透過 `comprehensive_voxel_feature` 反推回 HFM/DBE，避免拉壞 OCC 主路徑；但 VAMI 自己的 projection / fusion weights 必須仍可由 map loss 訓練，不能 detach 已經投影後的 `voxel_aware_feat`。
- **residual fusion**：`out = map_bev_feature + fusion_conv(concat(...))`。如果 voxel feature 沒帶來新訊息，模型可以自然把 fusion branch 權重壓低，不會傷既有 map performance。
- **Channel 設計**：`C_v` 是 `Dual_Branch_Encoder.voxel_out_channels`（預設 48）；Z-collapse 後 2×48=96；`C_map_in` 對齊 map neck 輸出通道（128）。Fusion conv 把 concat 後的 256 通道降回 128。
- **Spatial align**：`comprehensive_voxel_feature` 是 `(X, Y) = grid_size[:2]`（例如 200×200）；`map_bev_feature` 也應該是 200×200（從 `multi_scale_bev` 經 map neck 上採樣）。若兩者大小不一致，VAMI 內部用 bilinear resize 把 voxel side 對齊 map side。
- **不改 DBE return**：`Dual_Branch_Encoder` 已經回傳 `(comprehensive_voxel_feature, bev_feature, map_bev_feature)`，detector 直接拿這三個就夠了，不需要動 backbone。

### 可選模組（ablation）：AVMC（Auxiliary Voxel Mask Classifier）

降為 ablation，**不是 Stage 1 主線**。

目的：在 `comprehensive_voxel_feature` 上額外加一個 18-class（**對齊現有 OCC 而非 17**）Dice + Lovász head，補強 sparse class supervision。Class index 與 ignore 規則：和 `cnn3d_decoder.loss` 完全一致（`ignore_index=255`, 18 類）。

設定如果做：
```python
aux_voxel_mask_cfg = dict(
    enabled=True,
    aux_channels=64,
    aux_loss_weight=0.5,              # 比 cnn3d_decoder 主 loss 小
    num_classes=18,                   # 對齊 OCC
    ignore_index=255,
    dice_weight=1.0,
    lovasz_weight=1.0,
)
```

但前提是 VAMI 已驗證有效；如果 VAMI 都沒效，AVMC 大概率只是再多一個 OCC regularizer，不會救 Map。

### 可選模組（後續 ablation）：Spatial RoI Suppression

對應 MAESTRO Table 3 的 +0.7 Feature Suppression。

注意 codex 提醒的設計問題：
- 若 RoI = 「任一 map class 出現的位置」，會被 `drivable_area` 主導，學成粗糙 active-region mask
- **必須用 balanced focal BCE 或 Dice loss**，不是 plain BCE
- **接受條件**：mean map 持平或微升 + `stop_line / divider / ped_crossing` 至少 2 類有 +0.3 以上的提升；否則丟掉

僅在 VAMI 通過後測。

## 預計改動哪些檔案

### 新增

| 檔案 | 用途 |
| --- | --- |
| `projects/mmdet3d_plugin/models/task_modules/voxel_aware_map_ingest.py` | VAMI 模組：Z-collapse + channel align + fusion conv + residual |
| `projects/mmdet3d_plugin/models/task_modules/__init__.py`（更新） | 註冊 VAMI |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_vami.py` | Stage 1 主 config |
| `TWCC/train_multi_cnn_head_map_neck_vami.sh` | TWCC launcher |

後續才考慮：
| 檔案 | 用途 |
| --- | --- |
| `aux_voxel_mask_classifier.py` | AVMC ablation |
| `spatial_roi_suppression.py` | L_Sup ablation |
| `*_vami_no_detach.py` | gradient flow ablation |
| `*_vami_avmc.py` | VAMI + AVMC 組合 |

### 修改

| 檔案 | 預計改動 |
| --- | --- |
| `projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py` | 新增 optional `voxel_aware_map_ingest`。在 `_forward_map_logits` 之前對 `map_feature` 做 VAMI fusion；`forward_train` 和 `simple_test` 都走同一條 helper。Adapter 關閉時 baseline path 完全不變 |

### 嚴格不改

- `dual_branch_encoder.py`（DBE/HFM 主結構）
- `cnn3d_decoder.py`（已存在的 18-class CE+Lovász 監督就是我們要使用的訊號源）
- `bev_seg_head.py`
- `Prototype_Query_Decoder_nuScenes.py`

## Detector hook 怎麼接

`ProtoOccCnnSegHead.forward_train` 既有結構：

```python
encoder_output = self.dual_branch_encoder(voxel_feat)
comprehensive_voxel_feature, bev_feature, map_bev_feature = \
    self._split_encoder_output(encoder_output)

prototoype_occ_pred, mask_feat = self.cnn3d_decoder(
    comprehensive_voxel_feature.permute(0,1,4,2,3))
...
if self.bev_seg_head is not None:
    map_feature = self._select_map_feature(bev_feature, map_bev_feature)
    bev_seg_logits = self._forward_map_logits(map_feature, prototoype_occ_pred)
```

VAMI 接入：

```python
# 在 _forward_map_logits 之前
if self.voxel_aware_map_ingest is not None:
    map_feature = self.voxel_aware_map_ingest(
        map_feature=map_feature,
        voxel_feature=comprehensive_voxel_feature)
bev_seg_logits = self._forward_map_logits(map_feature, prototoype_occ_pred)
```

`simple_test` 同樣處理一遍，確保 eval 用同一條路徑。

## 實驗階段

### Stage 0：Baseline（已完成）

| Setting | Occ mIoU | Map mIoU |
| --- | ---: | ---: |
| weight4 strong baseline | 39.72 | 45.79 |
| overlay_dynamic | 39.52 | 46.00 |
| map-only upper bound | - | 48.34 |

### Stage 1：VAMI（detached）— 主實驗

目的：直接測「讓 map 使用 voxel-supervised feature」是否有效。**這是 plan 的核心**，不必先做 AVMC。

設定：
```python
voxel_aware_map_ingest_cfg = dict(
    enabled=True,
    z_collapse_mode='avg_max_concat',
    voxel_in_channels=48,           # = dual_branch_encoder.voxel_out_channels
    project_channels=128,           # 對齊 map neck 輸出
    fusion_hidden_channels=128,
    detach_voxel_for_map=True,      # 第一版 detach
    residual=True,                  # voxel-aware feat 走 residual fusion
)
map_loss_weight = 4.0               # 沿用 weight4 baseline
```

預期：
- Map mIoU：epoch 24 EMA ≥ 46.0；理想 46.5-47.0
- Occ mIoU：和 weight4 持平（39.72 ± 0.2）。detach 切掉 map → voxel 的 gradient，所以 OCC 不會被 map loss 拉
- Class-level：`stop_line / divider / ped_crossing / carpark_area` 至少 2 類 +0.5

Stop criterion：
- epoch 11 EMA Map < 43.3（比 weight4 epoch 11 ~43.82 低 0.5）→ 停
- epoch 24 EMA Map < 45.6 → 放棄整個方向
- 45.6 ≤ Map < 46.0 → weak positive，記錄結果不進下一階段
- Map ≥ 46.0 但只有 `drivable_area` 拉的，thin class 沒動 → 視為失敗

### Stage 2：必要 ablations（VAMI 通過後）

| Ablation | 目的 |
| --- | --- |
| `detach_voxel_for_map=False` | 看放開 gradient 後 Map 能不能再升，OCC 會不會崩 |
| Z-collapse mode（avg / max / avg+max / learnable）| 哪種 height aggregation 最有效 |
| `residual=False`（純取代而非殘差） | 證明殘差設計是必要保險 |
| voxel feature source ablation：用 `bev_feature` 而不是 `comprehensive_voxel_feature` | 證明關鍵在 voxel-supervised feature，不是「多塞 feature」 |
| VAMI 接在 `map_bev_feature` 還是 `bev_seg_head` 內部 | 找最佳注入點 |

最重要的是 **"voxel feature source ablation"**：把 VAMI 改成使用 `bev_feature`（HFM 後但 OCC 沒監督的 BEV residual）而不是 `comprehensive_voxel_feature`，看 Map 還會不會升。如果這個 ablation 和主實驗差不多，論點變成「給 map 多塞 feature 就有效」；如果有顯著差距，才能宣稱「voxel-supervised feature 是關鍵」。

### Stage 3（可選）：加 AVMC

只有 Stage 1 通過、且 Stage 2 的 source ablation 證明 voxel supervision 是關鍵時才做。

目的：看在既有 OCC supervision 上再加一個淺 Dice+Lovász head，能否進一步推高 Map。

設定：見上面「AVMC 可選模組」。

接受條件：Map 在 Stage 1 結果基礎上 ≥ +0.3，且 OCC 不掉超過 0.2。

### Stage 4（可選）：Spatial RoI Suppression

只有 Stage 3 也通過才做。

設計重點（修正 codex 提醒）：
- Loss：focal BCE 或 Dice，不是 plain BCE
- RoI 不要用「全部 map class union」直接做 binary target，可能被 drivable 主導；改為 multi-binary（6 class）並加上 class-balanced weighting
- 接受條件：mean Map ≥ Stage 3 結果，且 `stop_line / divider / ped_crossing` 至少 2 類有 +0.3

### Stage 5（可選）：VAMI + overlay_dynamic

確認 architecture-level VAMI 和 loss-level overlay 能否疊加。

## 為什麼這個方向跟前三次不一樣

| 維度 | PGBR | C2 GT-soft | LGMG O2M | VAMI |
| --- | --- | --- | --- | --- |
| 改 prototype 機制 | ✓ | ✓ | ✗ | **✗** |
| 在 logits 層加 residual | ✗ | ✗ | ✓ | **✗** |
| 改 map 使用的 feature 來源 | ✗ | ✗ | ✗ | **✓**（從 `multi_scale_bev` 衍生 → 加 `comprehensive_voxel_feature` 衍生） |
| 補新 supervision | ✗ | ✗ | ✗ | ✗（VAMI 主線不補；AVMC 是 ablation） |
| 對應 MAESTRO 的哪個元件 | 沒有 | 沒有 | 沒有 | **TSFG-Map 的 Z-collapse + voxel feature 使用部分** |

VAMI 是「我們有 `comprehensive_voxel_feature` 已被監督，但 map 沒看；那就接給 map 看」。最小、最直接、最 falsifiable。

## 風險與緩解

### 1. 「VAMI 有效但只是因為 map 多看到 feature」

Reviewer / codex 都會問：「會不會只是 channel 變多、capacity 變大？」

**緩解**：Stage 2 source ablation 直接回答。改用 `bev_feature`（沒 OCC supervision 的 BEV residual）做 VAMI；如果 Map 升幅一樣，方法只是「給 map 加 capacity」，不是「使用 voxel-supervised feature」。如果有顯著差距才能宣稱主張。

### 2. detach 之下 fusion conv 可能 dead

如果 voxel feature 對 map 完全沒幫助，detach 下 fusion conv 會學成 zero。

**這不是問題**：誠實的 negative result。比 no-detach 安全（後者會反推回 OCC 主路徑）。

### 3. Shape align 風險

`comprehensive_voxel_feature` 是 `(B, C, X, Y, Z)`；DBE 內部會把 voxel feature 排成 `permute(0,1,3,4,2)`（line 188），所以最終是 `(B, C, X, Y, Z)`。`map_bev_feature` 大小應該是 `(B, C_map, H, W)`，視 map neck 設計可能不是 200×200。

**緩解**：VAMI 內部對 voxel side 做 bilinear resize 對齊 map side；不在 map side resize 因為 map_bev_feature 馬上要進 BEVSegHead。

### 4. AVMC 論點不成立的風險

如果 Stage 3 加 AVMC 沒有額外提升，我們就不能宣稱「voxel-level multi-class supervision 是關鍵」。但 VAMI 本身的論點不依賴 AVMC，所以 paper 主線仍站得住：「voxel-supervised feature 使用路徑是關鍵」。

### 5. 跟 `overlay_dynamic` 的 OCC 衝突

`overlay_dynamic` baseline 是 Occ 39.52 / Map 46.00。VAMI 主線目標 Map ≥ 46.0，如果只能達到這個水準就和 overlay_dynamic 一樣。但 VAMI 是 architecture 改動，overlay 是 loss 改動，兩者可疊加（Stage 5）。

## 預期效果

理想：
- Stage 1：Occ 39.6 ± 0.2, **Map 46.5+**，thin class 至少 2 類 +0.5
- Stage 2 source ablation 顯示 `comprehensive_voxel_feature` 顯著優於 `bev_feature`
- Stage 5（VAMI + overlay）：Map 接近 47.0，OCC 持平

部分勝：
- Stage 1：Map 46.0-46.3，thin class 動少。Stage 2 source ablation 看是否仍有意義；如果是「給 map 加 capacity」就降為 weak ablation

失敗（不接受）：
- Stage 1：Map < 45.6，或 Occ < 39.0
- 收手回 weight4 + overlay_dynamic 主線，把 VAMI 寫成 negative ablation：「我們嘗試讓 map 使用 OCC-supervised voxel feature，detached residual fusion 下未觀察到顯著提升；這暗示 map 任務在 BEV 平面上的關鍵訊號和 voxel-level height structure 重疊有限」

## 論文敘事修正

之前版本（錯）：
> ProtoOcc lacks voxel-level multi-class supervision on shared feature; we propose AVSS to add it.

修正版（對）：
> ProtoOcc already supervises `comprehensive_voxel_feature` with voxel-level 18-class CE + Lovász loss via its 3D CNN prototype decoder. However, the map branch is forked from `multi_scale_bev` upstream of the hierarchical fusion module, and therefore never consumes this voxel-supervised feature. We propose **VAMI**, a minimal voxel-aware feature ingest module that performs Z-axis collapse on the already-supervised voxel feature and residually fuses it into the map BEV feature before the segmentation head. We analyse via source ablation whether the gain (if any) is attributable to voxel-level supervision or to mere capacity increase.

## 相關 work

### MAESTRO

仍 reference，但敘述要小心：
- MAESTRO 把 `F_s` 直接給所有 task 使用，map 路徑透過 TSFG-Map 收 Z 進 BEV
- ProtoOcc 設計上 map 從 HFM 上游分岔，所以等於缺了「TSFG-Map 的 Z-collapse 運用」這一塊
- VAMI 補的就是這個 architectural gap

不要再宣稱「補 L_CPG-level supervision」，因為 ProtoOcc 已經有等價物。

### ProtoOcc

- VAMI 不改 PQD、不改 DBE/HFM、不改 cnn3d_decoder
- 純粹在 detector 端做 cross-feature ingest

### PGBR / C2 / LGMG（負面對照）

明確切割：
- 都是 consumption-side 機制（prototype, logits residual）
- 都沒改 map 使用的**feature 本身**從哪來
- VAMI 是第一個從 feature source 著手的方案

## 建議實作順序

### Step 1：寫 VAMI 模組（最小）

1. 寫 `voxel_aware_map_ingest.py`：optional source detach → avg+max Z-collapse → 1×1 conv → optional resize → fusion conv → residual add
2. 註冊到 task_modules
3. py_compile

### Step 2：detector 接入

1. 在 `ProtoOccCnnSegHead.__init__` 新增 optional `voxel_aware_map_ingest`
2. 在 `forward_train` 把 VAMI 插在 `_forward_map_logits` 前
3. `simple_test` 同步
4. CPU dummy forward smoke

### Step 3：config + smoke

1. 寫 Stage 1 config 繼承 weight4 baseline
2. 1-GPU 10-iter smoke
3. 全 24 epoch on TWCC

### Step 4：Stage 2 source ablation

主實驗通過才做。改 VAMI 的 input source，量化 voxel supervision 的貢獻。

## 一句話總結

VAMI = **「ProtoOcc 已經有 voxel-supervised feature，但 map 從來沒使用；補上這條使用路徑」**。

不補 supervision（早就有），不做 prototype（試過沒用），不做 logits residual（試過失敗）；只改 map 看的 feature 從哪來。

## 2026-05-12 Stage 1 實作紀錄

已完成 VAMI Stage 1 全部 code，所有 smoke test 通過。

### 新增的檔案

| 檔案 | 用途 |
| --- | --- |
| `projects/mmdet3d_plugin/models/task_modules/voxel_aware_map_ingest.py` | **VAMI 模組本體**。`VoxelAwareMapIngest` 註冊在 `HEADS` registry，可用 `build_head` 建立。`forward(map_feature, voxel_feature)` 內部流程：optional `voxel_feature.detach()` → Z-collapse（avg / max / avg_max_concat）→ 1×1 Conv channel projection（含 BN+ReLU）→ optional bilinear resize 對齊 `map_feature` 的 (H, W) → concat → 兩層 fusion conv（3×3 → 1×1）→ residual add 回 `map_feature`。最後一層 conv 採 zero-init，所以 step 0 的 residual 恰為 0，初始 forward 等價 baseline。 |
| `projects/mmdet3d_plugin/models/task_modules/__init__.py` | 新建立的 sub-package，`from .voxel_aware_map_ingest import VoxelAwareMapIngest` 並 export。 |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_vami.py` | **Stage 1 主 config**。繼承 `ProtoOcc_multi_cnn_head_map_neck.py`，覆寫 `model.map_loss_weight=4.0`（沿用 weight4 baseline 訓練條件）並注入 `model.voxel_aware_map_ingest` dict。預設 `detach_voxel_source=True`、`residual=True`、`zero_init_output=True`、`z_collapse_mode='avg_max_concat'`、`voxel_in_channels=48`、`map_channels=128`。`evaluation.start=10`，方便觀察 epoch 11 stop criterion。 |
| `TWCC/train_multi_cnn_head_map_neck_vami.sh` | **Stage 1 TWCC launcher**。沿用 `train_multi_cnn_head_map_neck_overlay_dynamic.sh` 的 sbatch 設定（4 GPU、samples_per_gpu=4、LR=2e-4、24 epoch、`ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth` 為 pretrain），只把 CONFIG / WORK_DIR / job name 改為 VAMI 對應路徑。`work_dir=work_dirs/ProtoOcc_multi_cnn_head_map_neck_vami`。 |

### 修改的檔案

| 檔案 | 改了什麼 |
| --- | --- |
| `projects/mmdet3d_plugin/models/__init__.py` | 加一行 `from .task_modules import *`，讓 plugin 載入時 VAMI 會註冊到 `HEADS`。 |
| `projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py` | (1) `__init__` 新增 optional `voxel_aware_map_ingest=None` 參數，用 `build_head` 建立；(2) 新增 helper `_apply_voxel_aware_map_ingest(map_feature, voxel_feature)`，VAMI 沒設定時 short-circuit 回傳原 `map_feature`；(3) `forward_train` 在 `_select_map_feature` 後、`bev_seg_head` 前插入 `map_feature = self._apply_voxel_aware_map_ingest(map_feature, comprehensive_voxel_feature)`；(4) `simple_test` 對應位置同樣插入。VAMI 關掉時 baseline path 一個指令都不變。 |

### 嚴格沒改的檔案

- `projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py`
- `projects/mmdet3d_plugin/models/dense_heads/cnn3d_decoder.py`
- `projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py`
- `projects/mmdet3d_plugin/models/OccHead/Prototype_Query_Decoder_nuScenes.py`

VAMI 完全只在 detector edge 注入，不動 DBE/HFM/PQD/BEVSegHead/cnn3d_decoder 任何主結構。

### Smoke test 結果

於 `ProtoOcc` conda env 跑：

1. **`py_compile`**：5 個檔案（VAMI、`task_modules/__init__.py`、`models/__init__.py`、`ProtoOccCnnSegHead.py`、Stage 1 config）全過。Shell launcher `bash -n` 通過。
2. **VAMI tensor smoke**：
   - Zero-init residual 確認：初始 forward `out == map_feature`，max diff = 0
   - 擾動 fusion 最後一層後 output 變動 ✓
   - `detach_voxel_source=True` 時 `voxel_feature.grad is None`、`map_feature.grad` 正常累積 ✓
   - 空間不匹配時 `(16,16) → (32,32)` 自動 bilinear resize ✓
   - `avg` / `max` / `avg_max_concat` 三種 Z-collapse mode 輸出 shape 正確（96 vs 48）
   - Channel mismatch 會 raise `ValueError`
3. **Config + registry smoke**：
   - `VoxelAwareMapIngest` 已在 `HEADS._module_dict`
   - `Config.fromfile()` 載入 Stage 1 config，所有欄位（`map_loss_weight=4.0`、`detach_voxel_source=True`、`zero_init_output=True`、`z_collapse_mode='avg_max_concat'`、`evaluation.start=10`、`metric=['miou','map-miou']`）正確
   - `build_detector(cfg.model)` 成功，`det.voxel_aware_map_ingest` 是 `VoxelAwareMapIngest` instance
4. **Baseline 等價性**：
   - 不帶 VAMI 的 baseline detector：`det.voxel_aware_map_ingest is None`，helper 直接 passthrough 同一個 tensor 對象
   - 帶 VAMI 的 detector：在 zero-init 下，VAMI helper 第一次 forward 的輸出和輸入 `map_feature` 完全相等（max diff = 0），證明 step 0 等價 weight4 baseline

本機沒 GPU 環境跑完整 distributed training smoke；下一步應在 TWCC 上跑全 24 epoch。建議第一個 epoch 11 EMA 出來就先比 weight4 baseline 的 epoch 11 ~43.82，低於 43.3 就停（plan stop criterion）。
