# Layout-Geometry Mutual Guidance 計劃

日期：2026-05-06

更新：2026-05-07。Stage 1 / Stage 2 已暫停；Stage 3 改為獨立的 **Occ-to-Map only map recovery** 實驗，不再和 Stage 1 Map-to-Occ code 綁在一起。

## 動機

目前 Occ+Map 多任務線已經從「map 明顯被壓制」推進到「map 接近 map-only upper bound」：

| Setting | Checkpoint | Occ mIoU | Map mIoU | 備註 |
| --- | --- | ---: | ---: | --- |
| CNN head + 128ch map neck, `map_loss_weight=1` | epoch 24 EMA | 39.82 | 39.94 | map-specific neck 的基本 MTL baseline |
| CNN head + 128ch map neck, `map_loss_weight=4` | epoch 24 EMA | 39.72 | 45.79 | 目前 strong MTL baseline |
| CNN head + 128ch map neck, `overlay_dynamic` | epoch 24 EMA | 39.52 | 46.00 | map protection loss 已完成；Map 略高於 weight4，但 OCC 掉 0.20 |
| CNN head + 128ch map neck, map-only | epoch 24 EMA | - | 48.34 | map-only upper bound |

`map_loss_weight=4` 已經把 MTL map gap 從 `48.34 - 39.94 = 8.40` 縮到 `48.34 - 45.79 = 2.55`，而 OCC 只從 39.82 降到 39.72。這代表 map branch 的主要退化已經不是單純 map head capacity，而是多任務訓練中 map priority 與 task interaction 的問題。

下一階段不能再只寫「把 map 拉起來」。因為 map 已經接近單任務結果，我們需要一個真正的多任務架構，讓 OCC 和 Map 在不互相污染主幹特徵的前提下交換有用資訊，目標是共同提升：

- Map 提供 road layout / static scene structure，幫助 OCC 的地面、道路、sidewalk、manmade、terrain 等類別。
- OCC 提供 3D geometry / height / free-space / objectness cue，幫助 Map 的可行駛區域、邊界、遮擋區域與拓撲判斷。
- Cross-task interaction 必須是可控的 side path，不能破壞已經有效的 DBE / HFM / map neck 主路徑。

因此本計劃提出 **Layout-Geometry Mutual Guidance (LGMG)**：以 `CNN head + 128ch map neck + map_loss_weight=4` 為強 baseline，新增兩個輕量、可關閉、zero-residual 的 cross-task guidance adapter：

1. **Map-to-Occ Layout Query Adapter**：把 map 預測轉成 layout prior，透過 semantic-masked learnable relation 只注入 OCC background semantic queries。
2. **Occ-to-Map Geometry Adapter**：把 OCC 預測轉成 BEV geometry prior。2026-05-07 之後先作為獨立 Stage 3 map recovery path；第一版優先做 map logits residual，不直接改 map feature，也不引入 Stage 1 的 PQD query residual。

這不是 teacher distillation，也不是 GT-guided mining。兩個任務仍然 end-to-end 一起訓練；只是 cross-task guidance 走受控 side path，而不是把 raw feature 直接混進 shared DBE/HFM。

目前實驗策略已調整：Stage 1 的 source-only Map-to-Occ relation 在 epoch 5 EMA 沒有幫到 OCC，因此先暫停；Stage 2 relation ablation 也同步暫停。Stage 3 需從不含 Stage 1 / Stage 2 實作的分支獨立驗證 OCC geometry prior 是否能補 Map。

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
- **Stage 3 當前目標**：先補 Map 成績，正式比較 `map_loss_weight=4` 的 Occ 39.72 / Map 45.79；`overlay_dynamic` 的 Map 46.00 只作為第二階段組合參考，不應在第一個 O2M-only 實驗中混入。
- **Class-level**：Map 優先看 `stop_line / divider / ped_crossing / carpark_area`；OCC 優先看 `driveable_surface / sidewalk / terrain / manmade` 和容易受道路拓撲影響的類別。
- **Ablation**：單向 Map->Occ、單向 Occ->Map、雙向、no-gate、no-stop-gradient、raw concat 必須能說明模組不是偶然調參。

## 要增加什麼模組

### Module A: Map-to-Occ Layout Query Adapter

目的：讓 Map 提供 static layout prior 給 OCC，但不改 DBE/HFM 主幹 feature，也不把 map feature raw concat 到 OCC voxel feature。

第一版建議用 **semantic-masked learnable relation**。也就是 relation 不是純手工固定，也不是完全自由學；它只能在合理的 map-OCC background 對應內學權重。

核心流程用橫向寫法：

```text
map_feature -> BEVSegHead -> map_logits -> sigmoid confidence
          -> masked pooling on map_feature -> 6 layout tokens
          -> Linear(128 -> D_occ)
          -> semantic-masked learnable relation -> background OCC query offsets
          -> optional query_residual for background semantic queries -> PQD
```

輸入：

- `map_feature`: map neck 輸出的 BEV feature，shape 約為 `[B, 128, 200, 200]`
- `map_logits`: BEVSegHead 輸出的 map logits，shape 約為 `[B, 6, 200, 200]`
- `K_bg`: OCC 18 個 semantic queries 中，第一版允許接受 Map-to-Occ residual 的 background/layout subset 大小；主設定不是所有 background 類都接收 layout prior
- `query_residual`: 傳給 PQD 的 optional residual，shape 對齊 PQD 的 semantic query 維度；第一版只對 18 個 occupancy semantic queries 中的 background indices 非零

目前 code hook 要注意：

- 現有 `ProtoOccCnnSegHead.forward_train()` 是先跑 `cnn3d_decoder -> PQD loss`，之後才跑 `BEVSegHead` 算 map logits。因此 Map-to-Occ 不能只在 detector 端「事後」改 query；第一版需要把 map head 前移到 PQD 前，先得到 `map_logits / T_map`，再把 `query_residual` 傳進 PQD。
- PQD 需要新增一個 optional argument，例如 `query_residual=None`。Baseline config 不給這個 argument 時，PQD 行為必須完全不變。
- 不改 PQD 的 mask prediction / loss / RPL 主邏輯，只在 query synthesis 完成後、`query_self_attn` 前，把 residual 加到真正 semantic queries 上。

做法：

1. 使用 predicted `map_logits.sigmoid()`，不使用 GT map mask：

```text
P_map = sigmoid(map_logits)  # [B, 6, H, W]
```

2. 使用 predicted map confidence 對 `map_feature` 做 masked average pooling，得到 6 個 layout tokens。若某一個 map class 在該 frame 幾乎不存在，該 class token 直接設為 0，避免低 confidence mask 退化成 noisy global average：

```text
T_map_raw[c] = MaskedAvgPool(map_feature, P_map[c])
if sum(P_map[c]) < eps:
    T_map_raw[c] = 0

T_map_raw = [B, 6, 128]
```

3. 先做 channel projection，再做 semantic relation。`Linear(128 -> D_occ)` 只負責 channel 對齊；`semantic-masked learnable relation` 只負責 map token 到 OCC query token 的類別 mixing，mask 不作用在 channel projection 上：

```text
T_map = Linear_128_to_Docc(T_map_raw)  # [B, 6, D_occ]
```

4. 建立 `6 x K_bg` 的 semantic mask，其中 `K_bg=4`，只包含第一版真的允許被 map layout prior 修改的 OCC background/layout semantic queries：

```text
OCC layout target group = [
  driveable_surface, sidewalk, terrain, manmade
]
```

第一版 relation mask 建議已用 full train split GT 統計校正。統計來源是 `work_dirs/occ_map_overlap_z0_3_train_non_free.md`，Z=0~3 weighted 的 `Map -> OCC` 顯示：`drivable_area / ped_crossing / stop_line / divider` 主要落在 `driveable_surface`，`walkway` 主要落在 `sidewalk`，`carpark_area` 主要是 `driveable_surface`，但對 `sidewalk` 也有 6.10% 的弱關聯，因此第一版允許 `carpark_area -> sidewalk`。

| Map class | Allowed OCC background queries |
| --- | --- |
| `drivable_area` | `driveable_surface` |
| `ped_crossing` | `driveable_surface`, `sidewalk` |
| `walkway` | `sidewalk`, `terrain`, `manmade` |
| `stop_line` | `driveable_surface` |
| `carpark_area` | `driveable_surface`, `sidewalk` |
| `divider` | `driveable_surface` |

不把 `other_flat` 和 `vegetation` 放進第一版主 mask。雖然 `other_flat -> drivable_area` 在反向 coverage 裡有一定比例，但 Map -> OCC 的 `drivable_area / carpark_area` 對 `other_flat` 只有弱比例；`vegetation` 也沒有穩定對應的 map class。第一版應保持 relation mask 窄一點，這兩類的 `query_residual` 預設固定為 0；`other_flat` 可以放到 weak-relation ablation，`vegetation` 只在後續有明確統計或 class-level need 時再打開。

另外，full train weighted 裡 `walkway -> driveable_surface` 約 6.99%，但主設定故意不開。原因是 `walkway` 和 `driveable_surface` 是邊界上容易互相污染的類別，讓 walkway token 直接推 driveable query 可能把道路/行人區分界拉糊；這條 relation 應放到 weak-relation ablation，而不是 Stage 1 main setting。

5. 在這個 semantic mask 內學 relation weight，mask 外權重固定為 0：

```text
R_m2o = SemanticMaskedLearnableRelation(mask=M_map_occ)       # [6, K_bg]
Delta_Q_bg = einsum("bmd,mk->bkd", T_map, R_m2o)              # [B, K_bg, D_occ]
Delta_Q_other_background = 0                                  # other_flat / vegetation 等未開啟類別
Delta_Q_fg = 0                                                # foreground / dynamic queries 不被 map layout 改
```

6. 用 zero-residual gate 加到 background OCC query，而不是加到 comprehensive voxel feature：

```text
Q_occ_layout' = Q_occ_layout + alpha_m2o * Gate_source(Delta_Q_bg) * Delta_Q_bg
Q_occ_other_background' = Q_occ_other_background
Q_occ_fg' = Q_occ_fg
```

設計原則：

- `alpha_m2o` 初始為 0，確保一開始等價 baseline；adapter / relation projection 本身正常初始化，不再把 adapter 最後一層也 zero-init，避免 `alpha=0` 且 `Delta_Q=0` 造成 residual branch 初期沒有有效學習訊號。
- 第一版 `T_map` 預設 detach，只阻斷 OCC loss 回頭改 map side-path；但 map 主路徑仍正常由 map loss 更新。
- 不使用 GT map mask，不做 GT-guided support mining。
- `MaskedAvgPool` 對 absent / low-confidence map class 使用 zero-token fallback，並讓 gate 根據 zero token 自然壓低該 class 的 residual。
- Stage 1 實作使用 `source_only` gate，只根據 `Delta_Q_bg` 做 sigmoid gating，不讀取 PQD 內部的 `Q_occ_layout`。這讓 adapter 能維持在 detector-side side path，避免第一版大幅改 PQD query synthesis；`target_aware` gate 留到 Stage 5 ablation。
- 不讓 layout prior 直接改 dynamic foreground OCC queries，避免 map layout 把車、人等 movable object prototypes 拉壞。
- Relation 是 **masked learnable**：比純手工 mapping 更有學習能力，也比完全自由 relation 更不容易學到錯誤跨類別捷徑。
- PQD training 有 RPL padding queries。訓練時 PQD 會先把 `RPL_Groups * num_classes` 個 RPL query 放在前面，再接原本 18 個 semantic queries。`query_residual` 只能加在後段真正 semantic queries 的 background indices；RPL query residual 必須固定為 0，避免把 layout prior 加到 augmentation/noise query 上。

### Module B: Occ-to-Map Geometry Adapter

目的：讓 OCC 提供 3D geometry cue 給 Map，但不把 OCC prototype 當成 map prototype 使用，也不在第一版直接改 map feature。

Occ-to-Map 的風險比 Map-to-Occ 高，因為 OCC logits 在 early stage 很 noisy，而且 Map 已經 45.79，提升空間有限。原本它被放在 optional second-stage；但 Stage 1 / Stage 2 已暫停後，Stage 3 改為獨立的 **O2M-only map recovery** 實驗，用乾淨 branch 檢查 3D geometry prior 是否能補 Map。

Stage 3 的硬邊界：

- 不啟用 Map-to-Occ adapter。
- 不修改 PQD semantic query，也不新增 `query_residual`。
- 不做 bidirectional LGMG。
- 不疊 `overlay_dynamic`；第一個 O2M-only 實驗只和 `map_loss_weight=4` baseline 比。
- OCC 只當 side-path source，預設 `detach()`，避免 map loss 或 adapter loss 反向污染 OCC 主路徑。

第一版建議走 **logits residual**：

```text
map_feature -> BEVSegHead -> base_map_logits
coarse prototype occ pred -> HeightPool -> G_occ.detach() -> logit_adapter -> Delta_map_logits
base_map_logits + beta_o2m * Delta_map_logits -> final_map_logits
```

也可以做較安全的 **gate conditioning** 版本：

```text
map_feature -> BEVSegHead gate / predictor
coarse prototype occ pred -> HeightPool -> G_occ.detach() -> condition_adapter -> gate_bias
gate + gate_bias -> final map logits
```

只有在上述兩種都穩定後，才做高風險 feature residual ablation：

```text
map_feature + alpha_o2m * Gate_o2m(map_feature, G_occ) * Delta_F_map -> BEVSegHead
```

輸入：

- `prototype_occ_pred`: `cnn3d_decoder` 直接輸出的 coarse occupancy logits，shape 約為 `[B, 200, 200, 16, 18]`
- `map_feature`: map neck feature `[B, 128, 200, 200]`
- `base_map_logits`: BEVSegHead 原始輸出 `[B, 6, 200, 200]`

第一版明確使用 **coarse prototype occ pred**，不是 final PQD prediction。理由：

- `prototype_occ_pred` 是現有 training path 已經直接可得的 coarse 3D logits，不需要重構 PQD 回傳 final voxel probability。
- final PQD prediction 在目前 training API 只進 loss，不回傳完整 formatted occ probability；若為了 Occ-to-Map 強行改 PQD 回傳 final prediction，會把第一版 LGMG 的改動面變大。
- Stage 3 先用 coarse geometry prior 驗證「3D geometry cue 是否能補 map」。若 coarse 版本無效，不應急著用更重的 final PQD prediction 搶救，避免把失敗歸因變得不乾淨。
- 若後續真的要測 final PQD source，應獨立成 ablation：`o2m_source = coarse_proto | final_pqd_detached`。

OCC geometry summary：

```text
G_occ = HeightPool(prototype_occ_pred.detach())  # [B, C_geo, H, W]
```

建議包含：

- occupied confidence
- free-space confidence
- ground/layout class group confidence
- objectness / dynamic object group confidence
- height mean / height max / vertical occupancy thickness

HeightPool 第一版固定使用 nuScenes 18 類 index，不在不同實驗中任意改 grouping：

| Geometry channel | OCC class indices | 說明 |
| --- | --- | --- |
| `free` | `[17]` | free voxel probability |
| `occupied` | `1 - P_free` | 非 free confidence，包含 unknown / static / dynamic |
| `ground_layout` | `[11, 12, 13, 14]` | `driveable_surface / other_flat / sidewalk / terrain` |
| `structure` | `[15, 16]` | `manmade / vegetation` |
| `dynamic` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` | `barrier` 到 `truck` 的 movable / object group |
| `height_mean` | from `occupied` over z | `sum_z P_occ(z) * z_norm / (sum_z P_occ(z) + eps)` |
| `height_max` | from `occupied` over z | occupied confidence 超過 threshold 的最高 z；若沒有 occupied voxel 則為 0 |
| `occupied_count` | from `occupied` over z | `sum_z P_occ(z)`，表示 vertical thickness / occupancy mass |

`others` class `[0]` 不單獨做 group；它已包含在 `occupied` 裡。若後續發現 unknown / ambiguous voxel 對 map 有幫助，再獨立做 ablation。

logits residual：

```text
Delta_map_logits = LogitAdapter(G_occ)   # [B, 6, H, W]
final_map_logits = base_map_logits + beta_o2m * Delta_map_logits
```

warmup：

```text
beta_o2m(epoch) =
  0                         if epoch <= 6
  (epoch - 6) / 4 * beta_o2m_max
                            if 7 <= epoch <= 10   # 0.25, 0.50, 0.75, 1.00 when beta_o2m_max=1.0
  beta_o2m_max              if epoch >= 11
```

`beta_o2m_max` 是 config knob，第一版用 `1.0`，保留 `0.5` 作為 residual 過強時的 ablation。

本研究線不採 fine-tune，全部從 scratch 訓練。因此 Stage 3 從 scratch 訓練時必須依照上面 warmup：epoch 0-6 residual scale 為 0，epoch 7-10 線性開啟，不能在 epoch 0 就讓 noisy OCC logits 影響 map。實作上不應把 warmup 寫成手動改 config；建議新增 hook 或 detector method，例如 `OccToMapWarmupHook -> model.set_o2m_epoch(runner.epoch)`，讓 `beta_o2m` 在訓練中自動更新。

DDP 注意事項：

- 若 `occ_to_map_adapter` 已在 model 裡，training path 應該每個 epoch 都 forward adapter，包含 `beta_o2m=0` 的 epoch 0-6；此時 adapter 參數在 autograd graph 內，只是 gradient 為 0，避免 DDP 判成 unused。
- 第一版不要設定 `find_unused_parameters=True`；此 repo 既有訓練路徑對這個設定不穩，容易引入額外 DDP 問題。
- 第一版也不要用 `if beta_o2m == 0: skip adapter forward`。若真的要跳過 forward，必須另寫 freeze / unfreeze hook 並獨立驗證，但這不進 Stage 3 主設定。
- inference / evaluation 不需要 warmup，應直接使用 `beta_o2m_eval = beta_o2m_max`，避免 epoch 24 eval 仍看到 base map logits。

stop criterion：

- epoch 11 要和同 epoch baseline (weight4 epoch 11 EMA) 比。如果 baseline epoch 11 約為 Map 43.82，Occ-to-Map 低於 43.5 就停，不繼續調。
- epoch 24 若 Map < 45.6，放棄 Occ-to-Map，不要一直做 detach / gate tuning。
- 如果 Occ-to-Map 只帶來 Map ±0.3，而 thin / boundary classes 沒有穩定改善，應把它當 optional ablation，不硬放進主方法。

設計原則：

- 第一版 `G_occ` 預設 detach，但這只 detach side-path source，不是 `detach_map_feature=True` 那種硬隔離。
- Map 主路徑仍然正常反傳到 map neck 和 shared BEV feature。
- `beta_o2m` 初始為 0 或由 schedule 控制為 0，讓初始輸出等價 baseline；logit adapter 本身正常初始化，不和 `beta_o2m=0` 疊加成雙重 zero-init。
- 不使用 OCC prototype 去取代 map prototype；CNN map head 仍是主輸出。
- 目前 code 還沒有 MSFP，只有 map-specific BEV neck + BEVSegHead。未來如果再做 MSFP / active-region feature residual，不能和 Occ-to-Map feature residual 同時修改同一份 `map_feature`；第一版 Occ-to-Map 只做 logits residual / gate conditioning。

### Module C: Task-Protected Gating

目的：避免互助模組變成新的 feature pollution。

設計：

- `alpha_m2o`、`beta_o2m`、`alpha_o2m` 為 residual scalar，初始化 0 或由 schedule 從 0 開始；其中 `alpha_o2m` 只用於高風險 feature residual ablation。Adapter / projection 層正常初始化，避免 residual scalar 和 adapter output 同時為 0。
- Gate 使用 sigmoid。Stage 1 先採 `source_only` gate，輸入只有 source prior `Delta_Q_bg`；`target_aware` gate 需要讀取 PQD 的 target semantic query，會改動更深，因此留作 Stage 5 ablation。
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

Stage 3 目前主線只新增 O2M-only 需要的檔案；Stage 1 Map-to-Occ 相關檔案暫停，不應出現在這個 branch 的第一版實作裡。

| 檔案 | 用途 |
| --- | --- |
| `projects/mmdet3d_plugin/models/task_modules/occ_to_map_guidance.py` | 實作 `OccToMapGeometryAdapter`、HeightPool、logits residual adapter；不包含 Map-to-Occ relation |
| `projects/mmdet3d_plugin/core/hook/occ_to_map_warmup.py` | 依 epoch 控制 `beta_o2m`：0-6 為 0，7-10 linear warmup，11+ 使用 `beta_o2m_max` |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_o2m_logits.py` | Stage 3 O2M-only 主設定，從 `map_loss_weight=4` baseline 開始；包含 `beta_o2m_max=1.0` |
| `TWCC/train_multi_cnn_head_map_neck_lgmg_o2m_logits.sh` | Stage 3 TWCC 訓練 launcher，work_dir 獨立於 Stage 1 |

後續才考慮、目前不進 Stage 3 第一版：

| 檔案 | 用途 |
| --- | --- |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_o2m_gate_condition.py` | O2M gate conditioning ablation，只有 logits residual 穩定後再做 |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_o2m_feature_residual.py` | O2M feature residual 高風險 ablation，不進第一版 |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_m2o_masked_relation.py` | Stage 1 Map-to-Occ 設定，已暫停，不應混入 Stage 3 branch |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg.py` | 雙向 LGMG 候選 config，只有 Stage 1 和 Stage 3 都通過後才考慮 |

### 修改檔案

| 檔案 | 預計改動 |
| --- | --- |
| `projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py` | 新增 optional `occ_to_map_adapter`。在 `cnn3d_decoder` 取得 `prototype_occ_pred` 後，對 map logits 加 O2M residual；`forward_train` 和 `simple_test` 都必須走同一個 final-logits helper；adapter 關閉時 baseline path 完全不變 |
| `projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py` | 不改；LGMG 第一版不從 HFM / DBE 取新的中間 feature |
| `projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py` | 原則上不改主結構；`base_map_logits` 直接使用既有 forward 輸出，logits residual 在 detector 端相加 |
| `projects/mmdet3d_plugin/models/OccHead/Prototype_Query_Decoder_nuScenes.py` | Stage 3 不改；不新增 `query_residual`，避免帶入 Stage 1 邏輯 |
| `projects/mmdet3d_plugin/models/task_modules/__init__.py`、`projects/mmdet3d_plugin/core/hook/__init__.py` | 註冊 O2M adapter 與 warmup hook |
| `experiment_results_summary.md` | 實驗完成後新增 LGMG 結果、單向 ablation、失敗案例 |

### 優先不要改

- Stage 3 不改 `Prototype_Query_Decoder_nuScenes.py` 的核心 PQD 邏輯、loss、mask prediction、RPL mask/noise 生成，也不新增 optional `query_residual` 入口。
- 不改 `BEVSegHead` 主結構；map head 繼續使用 CNN head。
- 不改 `Dual_Branch_Encoder` / HFM；LGMG 第一版只接在 task-specific head side path。
- 不讓 Occ-to-Map 第一版直接改 `map_feature`；feature residual 只作高風險 ablation。
- 不啟用 `map_to_occ_adapter`、semantic relation、PQD `query_residual`、bidirectional LGMG。
- 不把 `overlay_dynamic` 混進第一個 O2M-only 實驗；若 O2M-only 有效，再做 O2M + overlay 的組合檢查。

## 實驗計劃

### Stage 0: Baseline sanity

確認目前 baseline：

- `CNN head + 128ch map neck + map_loss_weight=4`: Occ 39.72 / Map 45.79
- `overlay_dynamic` epoch 24 已完成：Occ 39.52 / Map 46.00。它是 map protection loss 的目前 best map score，但 OCC 比 weight4 低 0.20；Stage 3 O2M-only 主線仍先和 weight4 比 map recovery，再檢查是否能和 `overlay_dynamic` 組合。
- `overlay_dynamic` epoch 5/11 只能當 early signal，不當 paper-level acceptance criterion。
- Stage 0 已完成

### Stage 1: Map-to-Occ only, semantic-masked learnable relation

目的：確認 map layout prior 是否能提升 OCC，且不傷 map。

設定：

```text
relation_mode = "semantic_masked_learnable"
target_occ_group = "background_only"
map_prior_detach = True
gate_mode = "source_only"
alpha_m2o init = 0
adapter projection normal init
RPL query residual = 0
```

預期：

- Occ 提升，尤其 layout-related classes。
- Map 幾乎不變，因為 map 只是 source side。
- 如果 Map 變差，代表 side path 或 detach/gate 設計有問題。

接受條件：

- Occ > 39.72，或至少 layout-related classes 有明確 class-level gain。
- Map >= 45.79 附近，不應因為 OCC loss 透過 side path 回頭影響 map branch。

實作狀態（2026-05-06）：

- 已新增 `MapToOccLayoutAdapter`，位置在 `projects/mmdet3d_plugin/models/task_modules/layout_geometry_guidance.py`。
- Stage 1 adapter 會用 predicted `map_logits.sigmoid()` 對 `map_feature` 做 masked average pooling，產生 `[B, 6, 128]` layout tokens；absent / low-confidence map class 走 zero-token fallback。
- Adapter 先用 `Linear(128 -> D_occ)` 對齊 PQD query channel，再用 semantic-masked learnable relation 產生 `[B, 4, D_occ]` layout target residual。第一版只寫入 OCC query indices `[11, 13, 14, 15]`，也就是 `driveable_surface / sidewalk / terrain / manmade`；`other_flat / vegetation / foreground` residual 固定為 0。
- `alpha_m2o` 以 learnable scalar 實作，初始為 0；relation / channel projection / `source_only` gate 正常初始化，避免 residual branch 完全無學習訊號。初始 forward 等價 baseline。
- `ProtoOccCnnSegHead` 新增 optional `map_to_occ_adapter`。只有 adapter 開啟時才把 `BEVSegHead` 前移到 PQD 前；adapter 關閉時 baseline 路徑維持原本順序。
- `Prototype_Query_Decoder_nuScenes` 新增 optional `query_residual=None` hook，位置在 `for_query_embed(...)` 之後、`query_self_attn` 之前。`query_residual` 固定使用 `[B, num_queries, C]`，PQD 內部轉成 `[num_queries, B, C]`；訓練時會自動在前面補 `RPL_pad_size` 個 zero residual，確保 RPL padding queries 不吃 layout prior。
- 已新增 Stage 1 config：`projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_m2o_masked_relation.py`，繼承 map-neck baseline，設定 `map_loss_weight=4.0` 與 semantic relation mask。
- `MapToOccLayoutAdapter` 已加入 `target_occ_indices` duplicate check，避免 config 重複 index 造成 silent overwrite。
- 已完成 smoke 檢查：`py_compile`、adapter tensor smoke、config/registry smoke、PQD residual/RPL padding smoke、`git diff --check`。本機環境顯示 `No CUDA runtime is found`，所以尚未跑完整 training smoke。

暫停stage 1:

- epoch 5 ema: map 41.27, occ 37.93
- 沒有幫到occ 反而加強map
- 當前 source-only gated relation 學歪了，主要把 capacity 花在 driveable_surface，這不是我們要的效果。
- 反正OCC很強了，所以就是不加強他了

### Stage 2: Relation 設計 ablation

目的：證明不是任意 relation 都可以，semantic mask 是必要保護。

| Ablation | 預期解讀 |
| --- | --- |
| fixed semantic relation | 若有效，表示語意 mapping 本身有幫助 |
| semantic-masked learnable relation | 主設定，兼顧可學習與語意約束 |
| free learnable relation | 若不穩或退步，證明完全自由跨類別 relation 會學到捷徑 |
| foreground relation enabled | negative ablation；若 OCC dynamic classes 退步，證明 layout 不應改 foreground queries |
| no detach map prior | 檢查 OCC loss 回頭改 map branch 是否造成污染 |

暫停stage 2:

- stage 1 就已經暫停了，所以這些 ablation 也不建議繼續做了。

### 2026-05-07 commit / branch handoff

本次 `layout_geometry_mutual_guidance_plan.md` 的修改用途是保留 Stage 1 / Stage 2 暫停判斷，並明確記錄 Stage 3 需要和 Stage 1 code 分開。這個 commit 只應包含 plan 文件，不要加入 `mmdetection3d/` 或其他未追蹤目錄。

建議 commit message：

```text
docs: record LGMG Stage 1 pause and Stage 3 split
```

建議操作流程：

```bash
git status --short
git add layout_geometry_mutual_guidance_plan.md
git commit -m "docs: record LGMG Stage 1 pause and Stage 3 split"
git status --short
```

後續 Stage 3 不應從目前 `feat/LGMG` HEAD 直接開發，因為 `feat/LGMG` 已經包含 Stage 1 Map-to-Occ adapter 與 debug code。若要做沒有 Stage 1 / Stage 2 的 Occ-to-Map only 分支，建議從只有 LGMG plan、尚未加入 Stage 1 實作的 `0bca839` 開新分支：

```bash
git switch -c feat/LGMG-o2m-only 0bca839
```

若已經在 `feat/LGMG-o2m-only`，則不需要再重開分支；後續只要保持 Stage 3 code 不 cherry-pick Stage 1 implementation commits 即可。若需要把這份最新 plan 紀錄也帶到 `feat/LGMG-o2m-only`，可以在新分支建立後 cherry-pick 上面這個 docs commit。這樣 Stage 3 code base 仍然不含 Stage 1 實作，但 plan 會保留最新暫停紀錄。

### Stage 3: Occ-to-Map only map recovery

目的：在不啟用 Stage 1 / Stage 2 的前提下，確認 OCC coarse 3D geometry prior 是否能補 Map。這是獨立的 O2M-only 實驗，不是 Map-to-Occ 成功後的 second-stage。

分支與比較邊界：

- 建議分支：`feat/LGMG-o2m-only`
- Base：不含 Stage 1 Map-to-Occ adapter / PQD `query_residual` 的 code base
- Baseline：`CNN head + 128ch map neck + map_loss_weight=4`，Occ 39.72 / Map 45.79
- 第一版不疊 `overlay_dynamic`；`overlay_dynamic` 的 Map 46.00 只作為後續組合上限參考

設定：

```text
map_feature -> BEVSegHead -> base_map_logits
coarse prototype occ pred -> HeightPool -> G_occ.detach() -> LogitAdapter
final_map_logits = base_map_logits + beta_o2m * Delta_map_logits

beta_o2m(epoch) =
  0                         if epoch <= 6
  (epoch - 6) / 4 * beta_o2m_max
                            if 7 <= epoch <= 10
  beta_o2m_max              if epoch >= 11
```

實作要求：

- `prototype_occ_pred` 使用 `cnn3d_decoder` 的 coarse logits，不用 final PQD prediction。
- `G_occ` 預設 detach，只阻斷 O2M side source，不切斷 map 主路徑。
- `beta_o2m` 由 hook / schedule 控制，不靠手動改 config；hook 使用 `runner.epoch + 1` 的 1-based human epoch，`beta_o2m_max` 是 config knob，第一版用 `1.0`，保留 `0.5` 作為 ablation。
- `LogitAdapter` body 正常初始化，但最後輸出 `Delta_map_logits` 的 1x1 conv 採 zero-init，避免 epoch 7 warmup 開始時突然把 random residual 加到已經很強的 base map logits。
- final map loss 使用 `final_map_logits`；若輸出 debug，可同時記錄 base / residual logits statistics，但 evaluation 以 final logits 為準。
- `simple_test` 也必須 apply O2M residual：`base_map_logits -> final_map_logits -> bev_seg_head.predict(final_map_logits)`。否則 epoch 24 eval 會用 base logits，O2M 看起來像沒效果。
- inference / evaluation 時使用 `beta_o2m_eval = beta_o2m_max`，等同 warmup 完成後的 epoch 11+ 狀態。
- DDP-safe warmup：training 時即使 `beta_o2m=0`，也應 forward adapter 再乘上 0；第一版不要設定 `find_unused_parameters=True`，也不要條件式 skip adapter forward。
- OCC loss path 不吃 map residual；理論上 OCC 應接近 baseline。
- Stage 3 config 需把 `evaluation.start` 提前到可觀察 epoch 11 stop criterion；第一版用 `start=10`，保留 base config 的 map/occ metric 與 pipeline。

預期：

- Map 至少不低於 45.79；有價值的結果應接近或超過 46.00。
- 優先觀察 `stop_line / divider / carpark_area / drivable_area`。不能只靠 `drivable_area` 拉均值，thin / boundary classes 沒改善就不算強結果。
- OCC 幾乎不變，因為 OCC 只是 source side。
- 若 Map 有提升但 OCC 明顯下降，需檢查是否有 unintended gradient path 或 shared map loss side effect。

stop criterion：

- epoch 11 若 Map 低於同 epoch baseline 0.3 以上，例如低於約 43.5，就停止 Occ-to-Map。
- epoch 24 若 Map < 45.6，放棄 Occ-to-Map，不繼續調 detach/gate。
- epoch 24 若 Map 只在 45.79 附近 ±0.3 波動，且 `stop_line / divider / carpark_area` 沒有穩定改善，Occ-to-Map 不應硬放進主方法。
- epoch 24 若 Map >= 45.79 且 thin / boundary classes 有明確 gain，可以保留為 O2M-only ablation；若接近或超過 46.00，才進一步測 O2M + overlay_dynamic 組合。

#### 2026-05-12 Stage 3 result update: failed

本地 smoke run `work_dirs/smoke_lgmg_o2m_4090/result_4.md` 已完成，結論是 **Stage 3 O2M-only logits residual 第一版失敗**，不應進入 Stage 4，也不應再作為主方法候選。

和本地 `weight_4_4090` baseline 比較：

| Checkpoint | Setting | Occ mIoU | Map mIoU | 判讀 |
| --- | --- | ---: | ---: | --- |
| epoch 11 EMA | `weight_4_4090` baseline | 39.04 | 44.81 | 同機器 baseline |
| epoch 11 EMA | Stage 3 O2M-only | 38.50 | 42.17 | Map `-2.64`、Occ `-0.54`；低於 stop line `43.5`，理論上這裡就該停 |
| epoch 24 EMA | `weight_4_4090` baseline | 39.72 | 45.79 | 正式 strong baseline |
| epoch 24 EMA | Stage 3 O2M-only | 38.87 | 44.38 | Map `-1.41`、Occ `-0.85`；未達 `45.6` guardrail，判定失敗 |

class-level 對 `weight_4_4090` epoch 24 的影響：

- `drivable_area`: `78.76` vs `80.03`，`-1.27`
- `ped_crossing`: `39.88` vs `41.94`，`-2.06`
- `walkway`: `48.87` vs `50.93`，`-2.06`
- `stop_line`: `26.53` vs `28.32`，`-1.79`
- `divider`: `31.80` vs `33.49`，`-1.69`
- `carpark_area`: `40.45` vs `40.00`，只有這一類小幅 `+0.45`

這代表退化不是只發生在 thin class，而是整體 map branch 幾乎全面下降；同時 OCC 也被拖低，表示目前的 coarse `HeightPool -> logits residual` 沒有形成有效 geometry prior，比較像是把 noisy side signal 直接灌進 map logits。

因此目前判斷：

- Stage 3 第一版不通過，不進 Stage 4 `O2M + overlay_dynamic`。
- `Occ-to-Map logits residual` 應降級為 negative ablation / failure case，不作為第三個 contribution 主線。
- 若之後要繼續看 O2M，應先做 debug 型分析：`base vs residual logits` magnitude、`beta_o2m` 開啟後的分布、以及 `HeightPool` 各 channel 是否只學到 coarse drivable shortcut。

#### 2026-05-07 Stage 3 implementation record

已完成 O2M-only logits residual 第一版，實作保持 Stage 3 邊界：不啟用 Map-to-Occ、不改 PQD、不做 bidirectional、不疊 `overlay_dynamic`。

新增 / 修改檔案：

| 檔案 | 實作內容 |
| --- | --- |
| `projects/mmdet3d_plugin/models/task_modules/occ_to_map_guidance.py` | 新增 `OccToMapGeometryAdapter`。輸入 `cnn3d_decoder` coarse occ logits `[B, H, W, Z, 18]`，經 `HeightPool` 產生 8-channel geometry prior：`free / occupied / ground_layout / structure / dynamic / height_mean / height_max / occupied_count`，再輸出 `Delta_map_logits` |
| `projects/mmdet3d_plugin/models/task_modules/__init__.py`、`projects/mmdet3d_plugin/models/__init__.py` | 註冊 task module，讓 config 可透過 `type='OccToMapGeometryAdapter'` 建立 adapter |
| `projects/mmdet3d_plugin/core/hook/occ_to_map_warmup.py`、`projects/mmdet3d_plugin/core/hook/__init__.py` | 新增 `OccToMapWarmupHook`。training 使用 1-based epoch：epoch 1-6 beta=0，epoch 7-10 linear warmup，之後 beta=`beta_o2m_max` |
| `projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py` | 新增 optional `occ_to_map_adapter`、`set_o2m_epoch()`、`_forward_map_logits()`。`forward_train` 和 `simple_test` 都使用 final map logits；adapter 關閉時仍是原本 baseline path |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_o2m_logits.py` | Stage 3 主 config。顯式設定 `map_loss_weight=4.0`、`map_loss_balance_mode='none'`、`beta_o2m_max=1.0`、`detach_occ=True`、`zero_init_output=True`，並把 `evaluation.start=10` |
| `TWCC/train_multi_cnn_head_map_neck_lgmg_o2m_logits.sh` | Stage 3 TWCC launcher，work_dir 獨立為 `work_dirs/ProtoOcc_multi_cnn_head_map_neck_lgmg_o2m_logits` |

重要實作細節：

- `beta_o2m` 是 Python-side schedule value，不放進 parameter / buffer，避免 EMA checkpoint 把 beta 平滑掉；eval / inference 時 adapter 直接使用 `beta_o2m_max`。
- `G_occ` 預設 detach，只阻斷 O2M side source，不切斷 map 主路徑；map loss 仍正常反傳到 map neck / shared feature。
- training 時即使 beta=0 仍 forward adapter 再乘上 0，避免 DDP unused parameter 風險；第一版不開 `find_unused_parameters=True`。
- `Delta_map_logits` 最後一層 zero-init，使初始 residual 等價 0；warmup 開始後先從保守 residual 學起，避免 random residual 直接破壞 strong map baseline。

本地檢查：

- `py_compile` 通過：O2M adapter、warmup hook、`ProtoOccCnnSegHead.py`、Stage 3 config。
- CPU dummy forward 通過：`base_map_logits [2, 6, 4, 5] + occ_logits [2, 4, 5, 3, 18] -> final_map_logits [2, 6, 4, 5]`；epoch 7 beta=`0.25`，eval beta=`1.0`。
- beta=0 的 backward smoke 通過：adapter parameters 仍在 autograd graph 內，符合 DDP-safe warmup 設計。
- `mmcv.Config.fromfile()` 確認 Stage 3 config 保留 `metric=['miou', 'map-miou']`，並覆寫 `evaluation.start=10`。

### Stage 4: O2M + overlay_dynamic combination, only if Stage 3 passes

目的：如果 Stage 3 O2M-only 能補 Map，再檢查它能不能和 `overlay_dynamic` 的 map protection loss 組合。這不是第一版主實驗，避免把 architecture-level O2M 和 loss-level V3 混在一起。

接受條件：

- Map 接近或超過 `overlay_dynamic` 的 46.00。
- OCC 不比 `overlay_dynamic` 的 39.52 更差，理想上回到 39.7 附近。
- 若 O2M-only 沒通過 Stage 3，Stage 4 不做。

Bidirectional LGMG 目前暫停。只有當未來 Map-to-Occ 重新設計後能證明幫 OCC，且 O2M 也通過 Stage 3，才重新考慮 full LGMG。

### Stage 5: 必要 ablations

| Ablation | 目的 |
| --- | --- |
| O2M logits residual main | Stage 3 主設定，先確認 coarse geometry prior 能否補 Map |
| no warmup O2M | negative ablation；檢查 early noisy OCC 是否傷 map |
| no detach O2M source | 檢查 stop-gradient 是否避免 OCC source 被 map loss 污染 |
| beta fixed / no residual schedule | 檢查 zero-start residual 是否必要 |
| HeightPool component ablation | 分別移除 occupied / free-space / ground-layout / objectness / height summary，找出真正有效的 geometry cue |
| O2M gate conditioning | logits residual 穩定後再測，找較安全的注入方式 |
| O2M feature residual | 高風險 ablation；目前不進第一版，避免直接改 `map_feature` |
| O2M + overlay_dynamic | 只有 Stage 3 通過後才測，檢查 architecture prior 和 loss balancing 是否可疊加 |
| Map-to-Occ relation ablations | Stage 1 / 2 已暫停，暫不列入 Stage 3 必要 ablation |

### Stage 6: Paper table

建議 table：

| Method | Occ mIoU | Map mIoU | 說明 |
| --- | ---: | ---: | --- |
| ProtoOcc original | 39.56 | - | occ-only external reference |
| Naive Occ+Map MTL | 32.54 | 16.13 | naive map head |
| CNN map neck MTL | 39.82 | 39.94 | task-specific map feature |
| CNN map neck + map priority | 39.72 | 45.79 | strong baseline |
| + overlay dynamic | 39.52 | 46.00 | map protection loss；Map +0.21，但 OCC -0.20 |
| + Map-to-Occ masked relation | 37.93 @ epoch 5 | 41.27 @ epoch 5 | Stage 1 暫停；目前沒有幫 OCC |
| + Occ-to-Map logits residual | 38.87 | 44.38 | Stage 3 失敗；相較 weight4 baseline 為 Occ `-0.85` / Map `-1.41` |
| + Occ-to-Map logits residual + overlay_dynamic | not run | not run | O2M-only 未通過 Stage 3，因此不進 Stage 4 |
| + LGMG full | TBD | TBD | deferred；需要 M2O 和 O2M 都通過 |
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
- Occ-to-Map 在目前計劃中先作為 Stage 3 O2M-only map recovery 實驗；若 map 已接近 upper bound 且沒有穩定提升，就不硬放入主方法。

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

### 4. Stage 3 O2M logits residual 第一版失敗

`smoke_lgmg_o2m_4090` 的 Stage 3 第一版在 `epoch 11` 就已經掉到 Occ `38.50` / Map `42.17`，低於當初設定的 stop line `43.5`；到 `epoch 24` 也只到 Occ `38.87` / Map `44.38`，依然低於 weight4 baseline 的 Occ `39.72` / Map `45.79`。

教訓：

- coarse OCC `HeightPool -> logits residual` 第一版沒有提供穩定有效的 map geometry prior。
- 問題不是只集中在 `stop_line / divider`，而是 `drivable_area / walkway / ped_crossing` 也一起下降，代表這版不是「thin class 沒補到」，而是整體 map logits 被 side residual 擾動。
- Stage 3 應先作為 negative ablation 保留，不應直接往 Stage 4 疊 `overlay_dynamic`。

### 5. `detach_map_feature=True` 是錯的硬隔離

`detach_map_feature=True` epoch 5 結果為 Occ 38.33 / Map 16.49。這代表把 map branch 的主要學習路徑從 shared BEV feature 切掉，會讓 Map 幾乎學不起來。

教訓：

- 不要在主 map path 使用 `detach_map_feature=True`。
- Stop-gradient 只能用在 cross-task side path 的 source prior。
- Map 主路徑必須正常更新 map neck / shared BEV representation。



## 預期風險

- **Map 已接近 upper bound，提升空間有限**：Stage 3 雖然主看 Map，但不能只看 mean mIoU；必須檢查 `stop_line / divider / carpark_area` 等 class-level 是否真的改善。
- **Cross-task guidance 可能變成新的污染源**：需要 zero-residual scalar、gate、side-path detach 和 raw concat ablation；adapter / projection 本身正常初始化，避免 scalar 和 adapter output 同時為 0。
- **OCC prediction early stage 不穩**：Occ-to-Map 必須有明確 warmup；從 scratch 訓練 epoch 0-6 residual scale 為 0，epoch 7-10 線性開啟；training 仍 forward adapter 以避免 DDP unused 風險。
- **Occ-to-Map 收益可能很小**：Map 已經 45.79，距離 map-only 48.34 只剩 2.55。若 epoch 24 Map < 45.6，直接放棄 Occ-to-Map。
- **和未來 MSFP / overlay_dynamic 相容性不能太樂觀**：Stage 3 第一版只做 O2M logits residual，不直接改 `map_feature`，也不疊 overlay loss。未來若加入 MSFP / active-region feature residual，不要讓兩個模組同時改同一份 `map_feature`。
- **計算量增加**：adapter 必須輕量，避免 MAESTRO-2T 那種資源壓力。
- **O2M 可能只學到 drivable shortcut**：若 mean mIoU 上升但 `stop_line / divider / carpark_area` 沒有改善，不能把它寫成有效 geometry guidance。

## 預期效果

如果 LGMG 成功，論文主張可以寫成：

> We propose a task-protected layout-geometry mutual guidance framework for camera-only occupancy-map multi-task perception. Instead of directly sharing or fusing raw task features, the proposed method exchanges compact layout and geometry priors through gated residual adapters, enabling occupancy and map segmentation to benefit from each other while avoiding feature interference.

Stage 3 理想結果：

- Map mIoU 從 45.79 往 46.x 推進，最好接近或超過 `overlay_dynamic` 的 46.00。
- OCC mIoU 維持 39.72 附近；若下降，需要確認 O2M side path 是否意外影響 OCC 主路徑。
- Class-level 不只看 `drivable_area`，而要看 `stop_line / divider / carpark_area` 是否有穩定改善。
- no-warmup / no-detach / beta schedule / HeightPool component ablation 證明受控 O2M side path 的必要性。

如果 Stage 3 只讓 Map 小幅波動，或只改善 `drivable_area`，就把 Occ-to-Map 寫成 optional / negative ablation，不硬包成 LGMG full method。此時主線仍應回到已驗證的 map-specific neck + map priority / overlay-aware balancing。

如果 Stage 3 有效，再測 Stage 4 的 O2M + `overlay_dynamic`。若組合能接近 46.00 且不再掉 OCC，才有機會把 Occ-to-Map geometry guidance 寫成第三個 cross-task interaction contribution。


PYTHONPATH=. bash tools/dist_train.sh projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_lgmg_o2m_logits.py  1 --work-dir work_dirs/ProtoOcc_multi_cnn_head_map_neck_lgmg_o2m_logits_smoke --cfg-options evaluation.start=1
