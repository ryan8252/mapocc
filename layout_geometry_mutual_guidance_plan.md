# Layout-Geometry Mutual Guidance 計劃

日期：2026-05-05

## 動機

目前 Occ+Map 多任務線已經從「map 明顯被壓制」推進到「map 接近 map-only upper bound」：

| Setting | Checkpoint | Occ mIoU | Map mIoU | 備註 |
| --- | --- | ---: | ---: | --- |
| CNN head + 128ch map neck, `map_loss_weight=1` | epoch 24 EMA | 39.82 | 39.94 | map-specific neck 的基本 MTL baseline |
| CNN head + 128ch map neck, `map_loss_weight=4` | epoch 24 EMA | 39.72 | 45.79 | 目前 strong MTL baseline |
| CNN head + 128ch map neck, map-only | epoch 24 EMA | - | 48.34 | map-only upper bound |

`map_loss_weight=4` 已經把 MTL map gap 從 `48.34 - 39.94 = 8.40` 縮到 `48.34 - 45.79 = 2.55`，而 OCC 只從 39.82 降到 39.72。這代表 map branch 的主要退化已經不是單純 map head capacity，而是多任務訓練中 map priority 與 task interaction 的問題。

下一階段不能再只寫「把 map 拉起來」。因為 map 已經接近單任務結果，我們需要一個真正的多任務架構，讓 OCC 和 Map 在不互相污染主幹特徵的前提下交換有用資訊，目標是共同提升：

- Map 提供 road layout / static scene structure，幫助 OCC 的地面、道路、sidewalk、manmade、terrain 等類別。
- OCC 提供 3D geometry / height / free-space / objectness cue，幫助 Map 的可行駛區域、邊界、遮擋區域與拓撲判斷。
- Cross-task interaction 必須是可控的 side path，不能破壞已經有效的 DBE / HFM / map neck 主路徑。

因此本計劃提出 **Layout-Geometry Mutual Guidance (LGMG)**：以 `CNN head + 128ch map neck + map_loss_weight=4` 為強 baseline，新增兩個輕量、可關閉、zero-init 的 cross-task guidance adapter：

1. **Map-to-Occ Layout Query Adapter**：把 map 預測轉成 layout prior，透過 semantic-masked learnable relation 只注入 OCC background semantic queries。
2. **Occ-to-Map Geometry Adapter**：把 OCC 預測轉成 BEV geometry prior，作為 optional second-stage；第一版優先做 map logits residual / gate conditioning，不直接改 map feature。

這不是 teacher distillation，也不是 GT-guided mining。兩個任務仍然 end-to-end 一起訓練；只是 cross-task guidance 走受控 side path，而不是把 raw feature 直接混進 shared DBE/HFM。

## 要挑戰的是什麼

### 1. 現有 Occ+Map MTL 缺少受控的 task interaction

Naive MTL 把 OCC 和 Map 接在 shared feature 上，容易讓任務 loss 互相拉扯。過去結果顯示：

- `map_loss_weight=1` 時，Map 只有 39.94，和 map-only 48.34 差 8.40。
- `map_loss_weight=4` 後，Map 到 45.79，OCC 只小掉 0.10。

這表示 Map 並不是沒有能力學，而是在 MTL 內需要 task-level protection。現在問題進一步變成：在已經保護 Map 的前提下，怎麼讓 Map 和 OCC 互相幫助，而不是只避免互相傷害。

### 2. ProtoOcc 的 DBE/HFM 不應被改成 map-interaction 入口

ProtoOcc 的 Dual Branch Encoder / Hierarchical Fusion Module 原本就是 OCC encoder：voxel branch 抓 fine-grained 3D structure，BEV branch 抓 large receptive field semantic context，HFM 把兩者融合成 Comprehensive Voxel Feature。原論文 ablation 顯示 DBE/HFM 對 OCC 是正貢獻。

所以不能把「BEV branch」誤解成 map feature，也不應該把 map interaction 直接塞進 HFM 裡。真正容易污染 OCC 的不是 ProtoOcc 原本的 BEV+voxel 融合，而是 map loss 或 map feature 透過未受控路徑反向影響 OCC 主路徑。

本計劃的原則是：

- 不改 DBE/HFM 的主幹 fused feature 生成。
- 不把 Map raw feature 直接 concat 到 OCC voxel feature。
- 只在 task-specific head 前後加入 side-path guidance。

### 3. 前人 work 的不足

**ProtoOcc**  
ProtoOcc 專注 3D occupancy。DBE 和 PQD 對 OCC 有效，但它不是為 Occ+Map multi-task 設計，沒有處理 map task priority、map-only upper bound gap，也沒有設計 map-to-occ / occ-to-map 的互助機制。

**MAESTRO**  
MAESTRO 的價值是指出 shared backbone MTL 會有 feature interference，並用 prototype-guided task-specific feature generation 去 enhance / suppress task-relevant features。這個方向值得借鑑，但直接搬到我們的 Occ+Map 設定有幾個問題：

- 原方法是 Det+Map+Occ 三任務，前景/背景 grouping 對 Occ+Map 兩任務太粗，對 `ped_crossing / stop_line / divider` 這種 thin map class 不夠細。
- SPA 主要是 Det/Map -> Occ，一方向強化 OCC；我們現在需要的是 OCC 和 Map 都能受益。
- TSFG / SPA 較重，之前 MAESTRO-2T 實作已顯示資源成本高，不適合直接作為 ProtoOcc 主方法。
- MAESTRO 的 map prototype 主要是 background semantic prior；我們不能把它解讀成 raw map feature 可以直接灌入 OCC。

**BEVFusion / BEVerse**  
這些 work 支持 task-specific BEV feature/head 的設計，但它們主要處理 detection/map 或多任務 heads，沒有針對 camera-only OCC + BEV map segmentation 的 mutual guidance 問題。它們比較像證明「task-specific branch 合理」，不是我們最終互助模組的完整答案。

**HintOcc / dynamic class weighting 類方法**  
這類方法提供 batch-wise dynamic weighting 的 loss-level 思路，但它解決的是 class imbalance，不是 architecture-level task interaction。本計劃保留 overlay-aware loss 作為 map protection component，但真正的新架構要放在 layout-geometry mutual guidance。

## 目標

### 主要目標

建立一個 end-to-end Occ+Map MTL 架構，使 Map 和 OCC 透過受控 side-path guidance 互相補充：

- Map branch 保持目前最強 CNN head + 128ch map neck，不回到 ProtoMapHead 主線。
- OCC branch 保持 ProtoOcc DBE/HFM/PQD 主體，不破壞原始 OCC representation。
- Cross-task guidance 用 low-dimensional prior / gate / residual，不做 raw feature full fusion。

### 指標目標

以目前 strong baseline 作為正式比較：

| Baseline | Occ mIoU | Map mIoU |
| --- | ---: | ---: |
| CNN head + 128ch map neck + `map_loss_weight=4` | 39.72 | 45.79 |

目標：

- **OCC**：超過 39.72，至少希望 +0.3 到 +0.8；不能低於 39.5。
- **Map**：維持或超過 45.79，理想上推到 46.x；Occ-to-Map 若加入，epoch 24 低於 45.6 就應放棄該方向。
- **Class-level**：Map 優先看 `stop_line / divider / ped_crossing / carpark_area`；OCC 優先看 `driveable_surface / sidewalk / other_flat / terrain / manmade` 和容易受道路拓撲影響的類別。
- **Ablation**：單向 Map->Occ、單向 Occ->Map、雙向、no-gate、no-stop-gradient、raw concat 必須能說明模組不是偶然調參。

## 要增加什麼模組

### Module A: Map-to-Occ Layout Query Adapter

目的：讓 Map 提供 static layout prior 給 OCC，但不改 DBE/HFM 主幹 feature，也不把 map feature raw concat 到 OCC voxel feature。

第一版建議用 **semantic-masked learnable relation**。也就是 relation 不是純手工固定，也不是完全自由學；它只能在合理的 map-OCC background 對應內學權重。

核心流程用橫向寫法：

```text
map_logits -> sigmoid confidence -> masked pooling on map_feature -> 6 layout tokens
          -> semantic-masked learnable relation -> background OCC query offsets
          -> Q_occ_bg + alpha_m2o * gate * Delta_Q_bg -> PQD
```

輸入：

- `map_feature`: map neck 輸出的 BEV feature，shape 約為 `[B, 128, 200, 200]`
- `map_logits`: BEVSegHead 輸出的 map logits，shape 約為 `[B, 6, 200, 200]`
- `Q_occ`: PQD semantic queries / scene-aware queries，shape 依 ProtoOcc PQD 設定

做法：

1. 使用 predicted `map_logits.sigmoid()`，不使用 GT map mask：

```text
P_map = sigmoid(map_logits)  # [B, 6, H, W]
```

2. 使用 predicted map confidence 對 `map_feature` 做 masked average pooling，得到 6 個 layout tokens：

```text
T_map = MaskedAvgPool(map_feature, P_map)  # [B, 6, D]
```

3. 建立 `6 x K_bg` 的 semantic mask，其中 `K_bg` 只包含 OCC background/layout semantic queries，例如：

```text
OCC background group = [
  driveable_surface, sidewalk, other_flat,
  terrain, manmade, vegetation
]
```

第一版 relation mask 建議：

| Map class | Allowed OCC background queries |
| --- | --- |
| `drivable_area` | `driveable_surface`, `other_flat` |
| `ped_crossing` | `driveable_surface`, `sidewalk` |
| `walkway` | `sidewalk`, `terrain` |
| `stop_line` | `driveable_surface` |
| `carpark_area` | `driveable_surface`, `other_flat` |
| `divider` | `driveable_surface` |

4. 在這個 semantic mask 內學 relation weight，mask 外權重固定為 0：

```text
R_m2o = SemanticMaskedLearnableRelation(mask=M_map_occ)  # [6, K_bg]
Delta_Q_bg = R_m2o(T_map)                                # [B, K_bg, D_occ]
Delta_Q_fg = 0                                           # foreground / dynamic queries 不被 map layout 改
```

5. 用 zero-init residual gate 加到 background OCC query，而不是加到 comprehensive voxel feature：

```text
Q_occ_bg' = Q_occ_bg + alpha_m2o * Gate_m2o(Q_occ_bg, T_map) * Delta_Q_bg
Q_occ_fg' = Q_occ_fg
```

設計原則：

- `alpha_m2o` 初始為 0，確保一開始等價 baseline。
- 第一版 `T_map` 預設 detach，只阻斷 OCC loss 回頭改 map side-path；但 map 主路徑仍正常由 map loss 更新。
- 不使用 GT map mask，不做 GT-guided support mining。
- 不讓 layout prior 直接改 dynamic foreground OCC queries，避免 map layout 把車、人等 movable object prototypes 拉壞。
- Relation 是 **masked learnable**：比純手工 mapping 更有學習能力，也比完全自由 relation 更不容易學到錯誤跨類別捷徑。

### Module B: Occ-to-Map Geometry Adapter

目的：讓 OCC 提供 3D geometry cue 給 Map，但不把 OCC prototype 當成 map prototype 使用，也不在第一版直接改 map feature。

Occ-to-Map 的風險比 Map-to-Occ 高，因為 OCC logits 在 early stage 很 noisy，而且 Map 已經 45.79，提升空間有限。因此它應該是 optional second-stage。

第一版建議走 **logits residual**：

```text
map_feature -> BEVSegHead -> base_map_logits
occ_logits -> HeightPool -> G_occ.detach() -> zero-init logit_adapter -> Delta_map_logits
base_map_logits + beta_o2m * Delta_map_logits -> final_map_logits
```

也可以做較安全的 **gate conditioning** 版本：

```text
map_feature -> BEVSegHead gate / predictor
occ_logits -> HeightPool -> G_occ.detach() -> condition_adapter -> gate_bias
gate + gate_bias -> final map logits
```

只有在上述兩種都穩定後，才做高風險 feature residual ablation：

```text
map_feature + alpha_o2m * Gate_o2m(map_feature, G_occ) * Delta_F_map -> BEVSegHead
```

輸入：

- `occ_logits` 或 `prototype_occ_pred`: coarse/final occupancy prediction
- `map_feature`: map neck feature `[B, 128, 200, 200]`
- `base_map_logits`: BEVSegHead 原始輸出 `[B, 6, 200, 200]`

OCC geometry summary：

```text
G_occ = HeightPool(occ_logits.detach())  # [B, C_geo, H, W]
```

建議包含：

- occupied confidence
- free-space confidence
- ground/layout class group confidence
- objectness / dynamic object group confidence
- height mean / height max / vertical occupancy thickness

logits residual：

```text
Delta_map_logits = LogitAdapter(G_occ)   # [B, 6, H, W]
final_map_logits = base_map_logits + beta_o2m * Delta_map_logits
```

warmup：

```text
epoch 0-6:  beta_o2m = 0, disable Occ-to-Map
epoch 7-10: beta_o2m linear warmup
epoch 11+:  normal training
```

如果是從已訓練好的 `map_loss_weight=4` checkpoint fine-tune，可以縮短成 1-2 epoch warmup；從 scratch 訓練則不要提早開 Occ-to-Map。

stop criterion：

- epoch 11 要和同 epoch baseline 比。如果 baseline epoch 11 約為 Map 43.82，Occ-to-Map 低於 43.5 就停，不繼續調。
- epoch 24 若 Map < 45.6，放棄 Occ-to-Map，不要一直做 detach / gate tuning。
- 如果 Occ-to-Map 只帶來 Map ±0.3，而 OCC 不變，應把它當 optional ablation，不硬放進主方法。

設計原則：

- 第一版 `G_occ` 預設 detach，但這只 detach side-path source，不是 `detach_map_feature=True` 那種硬隔離。
- Map 主路徑仍然正常反傳到 map neck 和 shared BEV feature。
- Adapter 必須 zero-init，讓初始輸出等價 baseline。
- 不使用 OCC prototype 去取代 map prototype；CNN map head 仍是主輸出。
- 不在第一版同時讓 MSFP 和 Occ-to-Map 修改同一份 map feature。

### Module C: Task-Protected Gating

目的：避免互助模組變成新的 feature pollution。

設計：

- `alpha_m2o`、`beta_o2m`、`alpha_o2m` 為可學習 scalar，初始化 0；其中 `alpha_o2m` 只用於高風險 feature residual ablation。
- Gate 使用 sigmoid，輸入是 target feature + source prior 的簡單 concat/projection。
- Source prior 預設走 stop-gradient side path，之後做 no-detach ablation。
- 所有 adapter 都是 residual；第一版 Occ-to-Map 不改原本 map feature，只改 logits 或 gate。

這裡要特別和失敗的 detach 實驗區分：

```text
錯誤方向：detach_map_feature=True
multi_scale_bev.detach() -> map neck -> map head
```

這會切斷 map branch 的主要學習路徑，已經造成 Map mIoU 只有 16.49。

```text
建議方向：side-path source detach
occ_prior.detach() -> geometry adapter -> logits residual / gate conditioning
map_feature -> map head -> map loss -> 正常更新 map 主路徑
```

也就是只保護 cross-task source，不切斷任務自己的主學習通道。

## 預計改動哪些檔案

### 新增檔案

| 檔案 | 用途 |
| --- | --- |
| `projects/mmdet3d_plugin/models/task_modules/layout_geometry_guidance.py` | 實作 `MapToOccLayoutAdapter`、`SemanticMaskedRelation`、`OccToMapLogitAdapter`、gating utilities |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_m2o_masked_relation.py` | Map -> Occ only 主設定，semantic-masked learnable relation |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_o2m_logits.py` | Occ -> Map logits residual optional ablation |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_o2m_gate_condition.py` | Occ -> Map gate conditioning optional ablation |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_o2m_feature_residual.py` | Occ -> Map feature residual 高風險 ablation |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg.py` | 雙向 LGMG 候選 config，只在 Occ-to-Map 通過 stop criterion 後使用 |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_no_detach.py` | no stop-gradient ablation |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_free_relation.py` | 完全自由 learnable relation negative ablation |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_fixed_relation.py` | 固定 semantic relation ablation |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_raw_concat.py` | raw concat / naive fusion negative ablation |
| `TWCC/train_multi_cnn_head_map_neck_lgmg.sh` | TWCC 訓練 launcher |

### 修改檔案

| 檔案 | 預計改動 |
| --- | --- |
| `projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py` | 在 `forward_train` / `simple_test` 接入 query residual 與 logits residual；保留原 baseline path 可關閉 |
| `projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py` | 原則上不改 HFM 主幹；如需更多中間 feature，只新增 optional return，不改既有輸出 |
| `projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py` | 原則上不改主結構；若需要，僅支援回傳 `base_map_logits` / 接收 logits residual |
| `projects/mmdet3d_plugin/models/__init__.py` 或相關 registry import | 註冊新 task module |
| `experiment_results_summary.md` | 實驗完成後新增 LGMG 結果、單向 ablation、失敗案例 |

### 優先不要改

- 不改 `Prototype_Query_Decoder_nuScenes.py` 的核心 PQD 邏輯；第一版只在 detector 端調整 query input 或 query residual。
- 不改 `BEVSegHead` 主結構；map head 繼續使用 CNN head。
- 不把 LGMG 寫進 `Dual_Branch_Encoder` 的 HFM 裡。
- 不讓 Occ-to-Map 第一版直接改 `map_feature`；feature residual 只作高風險 ablation。

## 實驗計劃

### Stage 0: Baseline sanity

確認目前 baseline：

- `CNN head + 128ch map neck + map_loss_weight=4`: Occ 39.72 / Map 45.79
- `overlay_dynamic` epoch 24 若已完成，也加入比較；epoch 5/11 只能當 early signal。

### Stage 1: Map-to-Occ only, semantic-masked learnable relation

目的：確認 map layout prior 是否能提升 OCC，且不傷 map。

設定：

```text
relation_mode = "semantic_masked_learnable"
target_occ_group = "background_only"
map_prior_detach = True
alpha_m2o init = 0
```

預期：

- Occ 提升，尤其 layout-related classes。
- Map 幾乎不變，因為 map 只是 source side。
- 如果 Map 變差，代表 side path 或 detach/gate 設計有問題。

接受條件：

- Occ > 39.72，或至少 layout-related classes 有明確 class-level gain。
- Map >= 45.79 附近，不應因為 OCC loss 透過 side path 回頭影響 map branch。

### Stage 2: Relation 設計 ablation

目的：證明不是任意 relation 都可以，semantic mask 是必要保護。

| Ablation | 預期解讀 |
| --- | --- |
| fixed semantic relation | 若有效，表示語意 mapping 本身有幫助 |
| semantic-masked learnable relation | 主設定，兼顧可學習與語意約束 |
| free learnable relation | 若不穩或退步，證明完全自由跨類別 relation 會學到捷徑 |
| foreground relation enabled | negative ablation；若 OCC dynamic classes 退步，證明 layout 不應改 foreground queries |
| no detach map prior | 檢查 OCC loss 回頭改 map branch 是否造成污染 |

### Stage 3: Optional Occ-to-Map logits residual

目的：確認 3D geometry prior 是否能補 Map。

設定：

```text
occ_logits -> HeightPool -> G_occ.detach() -> zero-init LogitAdapter
final_map_logits = base_map_logits + beta_o2m * Delta_map_logits

epoch 0-6:  beta_o2m = 0
epoch 7-10: beta_o2m linear warmup
epoch 11+:  normal
```

預期：

- Map 不低於 45.79；最低不可低於 45.6。
- 優先觀察 `drivable_area / carpark_area / divider`，但不要預設一定能改善 `stop_line`。
- OCC 幾乎不變，因為 OCC 只是 source side。

stop criterion：

- epoch 11 若 Map 低於同 epoch baseline 0.3 以上，例如低於約 43.5，就停止 Occ-to-Map。
- epoch 24 若 Map < 45.6，放棄 Occ-to-Map，不繼續調 detach/gate。
- 如果只有 Map ±0.3 波動，Occ-to-Map 不應硬放進主方法。

### Stage 4: Bidirectional LGMG, only if Occ-to-Map passes

目的：確認兩個 side path 合起來是否能共同提升。

接受條件：

- Occ > 39.72。
- Map >= 45.79，理想上 > 46。
- 不出現 Map 提升但 OCC 大掉，或 OCC 提升但 Map 掉回 45.6 以下。
- 若 Occ-to-Map 沒通過 Stage 3，最終方法可以只保留 Map-to-Occ，不硬包成 bidirectional。

### Stage 5: 必要 ablations

| Ablation | 目的 |
| --- | --- |
| no detach side path | 檢查 stop-gradient 是否真的避免 source branch 被污染 |
| no gate / alpha fixed 1 | 檢查 gating 是否必要 |
| raw concat fusion | 證明直接 fusion 比受控 side path 差 |
| fixed relation vs semantic-masked learnable relation | 檢查 relation 是否需要學習 |
| free relation vs semantic-masked learnable relation | 檢查語意遮罩是否必要 |
| foreground relation enabled | 證明 dynamic foreground prototypes 不應被 map layout 改 |
| Occ->Map logits residual vs gate conditioning | 找較安全的 O2M 注入方式 |
| Occ->Map feature residual | 高風險 ablation；檢查直接改 map feature 是否和 MSFP/BEVSegHead 打架 |
| no warmup Occ-to-Map | negative ablation；檢查 early noisy OCC 是否傷 map |

### Stage 6: Paper table

建議 table：

| Method | Occ mIoU | Map mIoU | 說明 |
| --- | ---: | ---: | --- |
| ProtoOcc original | 39.56 | - | occ-only external reference |
| Naive Occ+Map MTL | 32.54 | 16.13 | naive map head |
| CNN map neck MTL | 39.82 | 39.94 | task-specific map feature |
| CNN map neck + map priority | 39.72 | 45.79 | strong baseline |
| + overlay dynamic | TBD | TBD | map protection loss |
| + Map-to-Occ masked relation | TBD | TBD | layout prior to OCC background queries |
| + Occ-to-Map logits residual | TBD | TBD | optional geometry prior |
| + LGMG full | TBD | TBD | only if O2M passes stop criterion |
| Map-only upper bound | - | 48.34 | diagnostic upper bound |

## 相關 work

### ProtoOcc

ProtoOcc 的 DBE/HFM 證明 voxel local detail 和 BEV long-range context 對 occupancy 是互補的。PQD 用 Scene-Adaptive / Scene-Agnostic prototypes 做高效 occupancy query decoding。這些是我們的 OCC backbone，不應被大幅改寫。

本計劃和 ProtoOcc 的差異：

- ProtoOcc 是 OCC-only。
- LGMG 是 Occ+Map MTL。
- LGMG 不改 DBE/HFM 主幹，而是在 task-specific branch 之後加入 layout/geometry side guidance。

### MAESTRO

MAESTRO 指出 naive shared-feature MTL 有 task conflict，提出 CPG / TSFG / SPA 來產生 task-specific features 並用 task-oriented prototypes 強化 OCC。它提供兩個重要啟發：

- 不應把同一份 shared feature 直接交給所有 task。
- Map output 可以作為 OCC 的 static background semantic prior。

本計劃和 MAESTRO 的差異：

- 不直接採用 Det/Map/Occ 三任務 foreground/background grouping。
- 不把 MAESTRO-2T 當作本文主體，而是當 strong comparator / diagnostic。
- Map-to-Occ relation 是 semantic-masked learnable，不是 rule-only mapping，也不是完全自由 relation。
- LGMG 是 query/logit-level side adapter，不建立 MAESTRO-style heavy TSFG feature generator。
- Occ-to-Map 是 optional second-stage；若 map 已接近 upper bound 且沒有穩定提升，就不硬放入主方法。

### BEVFusion / BEVerse

這些方法支持 shared backbone + task-specific branch/head 的設計，說明 map segmentation 需要足夠的 BEV feature capacity。這也是我們採用 CNN head + 128ch map neck 作為 strong baseline 的原因。

但它們沒有直接處理 camera-only occupancy + BEV map segmentation 的 mutual guidance，因此 LGMG 的貢獻應該放在「layout and geometry priors for Occ+Map MTL」。

### HintOcc / dynamic weighting

HintOcc 類方法支持 batch-wise class presence / imbalance 的觀點，但主要是 loss reweighting。對我們而言，overlay dynamic balancing 是 map protection component，不是最終 architecture contribution。LGMG 則處理 task interaction。

## 之前的失敗案例與教訓

### 1. PGBR 不適合作為主線

PGBR / prototype refinement 類方法沒有穩定帶來 map 提升，且引入額外複雜度。現階段不應再以 prototype refinement 作為 map 主路徑。

教訓：

- 不要再假設 prototype decoder 一定比 CNN map head 強。
- Map branch 目前應固定在 CNN head + 128ch map neck。

### 2. GT-soft 造成 train-test mismatch

GT-soft epoch 24 結果為 Occ 39.52 / Map 32.40，Map 大幅退步，尤其 `drivable_area` 明顯崩壞。原因很可能是 training 使用 GT-guided support，但 inference 沒有 GT，只能回到 prediction/EMA prototype，造成校準偏移。

教訓：

- 不要再用 GT-dependent support 作為 mainline。
- 新模組 inference 必須 GT-free。
- 若使用 predicted masks/tokens，必須從一開始就按照 inference 可用訊號訓練。

### 3. ProtoMapHead 比不過 CNN head

ProtoMapHead + map neck / PQD-align / 256ch 版本都沒有超過 CNN head + 128ch map neck。多數情況 coarse output 還比 final prototype output 更好。

教訓：

- Map 自己的 prototype refinement 不足以解決問題。
- 不應用 OCC prototype 直接「加強 map prototype」。
- Map 應保留 CNN segmentation path，cross-task 只提供 geometry prior。

### 4. `detach_map_feature=True` 是錯的硬隔離

`detach_map_feature=True` epoch 5 結果為 Occ 38.33 / Map 16.49。這代表把 map branch 的主要學習路徑從 shared BEV feature 切掉，會讓 Map 幾乎學不起來。

教訓：

- 不要在主 map path 使用 `detach_map_feature=True`。
- Stop-gradient 只能用在 cross-task side path 的 source prior。
- Map 主路徑必須正常更新 map neck / shared BEV representation。



## 預期風險

- **Map 已接近 upper bound，提升空間有限**：因此 LGMG 不能只看 Map，必須看 OCC 是否受益。
- **Cross-task guidance 可能變成新的污染源**：需要 zero-init、gate、side-path detach 和 raw concat ablation。
- **OCC prediction early stage 不穩**：Occ-to-Map 必須有明確 warmup；從 scratch 訓練 epoch 0-6 完全 disable，epoch 7-10 線性開啟。
- **Occ-to-Map 收益可能很小**：Map 已經 45.79，距離 map-only 48.34 只剩 2.55。若 epoch 24 Map < 45.6，直接放棄 Occ-to-Map。
- **和 MSFP 相容性不能太樂觀**：MSFP 已經在 map branch 做 active-region residual。LGMG 第一版應先只開 Map-to-Occ，或讓 Occ-to-Map 只做 logits residual / gate conditioning；不要讓兩個模組同時改同一份 `map_feature`。
- **計算量增加**：adapter 必須輕量，避免 MAESTRO-2T 那種資源壓力。
- **語意對應不完全**：Map 六類和 OCC semantic queries 不是一對一。第一版用 semantic-masked learnable relation，並用 fixed / free / foreground-enabled relation ablation 證明設計必要性。

## 預期效果

如果 LGMG 成功，論文主張可以寫成：

> We propose a task-protected layout-geometry mutual guidance framework for camera-only occupancy-map multi-task perception. Instead of directly sharing or fusing raw task features, the proposed method exchanges compact layout and geometry priors through gated residual adapters, enabling occupancy and map segmentation to benefit from each other while avoiding feature interference.

理想結果：

- Occ mIoU 從 39.72 提升到 40.x。
- Map mIoU 維持 45.79 以上，最好推到 46.x。
- 單向 ablation 顯示 Map-to-Occ masked relation 主要幫 OCC，Occ-to-Map logits residual 若有效則主要幫 Map。
- no-detach / no-gate / raw concat ablation 證明受控 side path 的必要性。

如果只提升 OCC 而 Map 持平，仍然有論文價值，因為目前 Map 已經接近 map-only upper bound；此時主張應改成「preserve strong map performance while recovering / improving occupancy through layout guidance」。這種情況下最終方法可以是 `overlay-aware map balancing + Map-to-Occ masked relation`，不一定要包含 Occ-to-Map。

如果雙向沒有提升，保留最有效的單向版本，不硬包成 reciprocal module。從論文角度，誠實地把 Occ-to-Map 放成 optional / negative ablation，會比硬做雙向但分數不穩更有說服力。
