# CFV Prototype-TSFG Map Fusion 計劃

日期：2026-05-13

## 參數與名詞定義

| 名稱 | 定義 |
| --- | --- |
| `CFV` | `comprehensive_voxel_feature`，ProtoOcc DBE/HFM 之後的 voxel feature，shape 為 `[B, C_v, X, Y, Z]`，目前 `C_v=48`。 |
| `mask_feat` | `cnn3d_decoder` 的低維 prototype feature，shape 為 `[B, X, Y, Z, C_p]`，目前 `C_p=32`。 |
| `prototype_occ_pred` | `cnn3d_decoder` 的 voxel semantic logits，shape 為 `[B, X, Y, Z, 18]`。 |
| `F_map` | 128-channel map-specific neck 輸出的 BEV feature，shape 為 `[B, C_map, H, W]`，目前 `C_map=128`。 |
| `Q_bg` | 從 ProtoOcc PQD 取出的 background Scene-Aware Query embeddings。第一版使用 `driveable_surface / other_flat / sidewalk / terrain / manmade / vegetation` 對應的 query。 |
| `P_bg` | `Q_bg` 經 adapter-local prototype MLP，或 PQD 內部已算好且 detach 後輸出的 `mask_embed` filters，投到 CFV channel space 後的 background semantic filters，shape 為 `[B, N_bg, C_v]`。 |
| `A_k` | 第 `k` 個 background semantic filter 在每個 voxel 上的 activation / attention map，shape 為 `[B, 1, X, Y, Z]`。 |
| `A_bg` | 對所有 background classes 的 `A_k` 做聚合後得到的 background-aware activation map。 |
| `F_proto_cfv` | 經過 CFV-side prototype-TSFG 後 collapse 到 BEV 的 voxel-to-map side feature。 |
| `gamma(t)` | residual branch 的 scheduled scalar warmup，不作為 learnable parameter。第一版從 0 線性 ramp 到 1，讓初始 forward 等價 baseline。 |
| `G` | gated residual ablation 的 spatial/channel gate，限制 voxel prior 只在有幫助的位置進入 map neck；不是第一版主線。 |

## 一句話定位

> 這不是照抄 MAESTRO 的 TSFG，而是針對 ProtoOcc 的特殊瓶頸：ProtoOcc 已經學到強 occupancy prototype / CFV，但 map branch 只吃 HFM 上游的 map-specific BEV neck feature，沒有安全地使用 CFV 裡的 background semantic prior。本計劃用 **CFV-side background prototype-TSFG side branch** 產生 voxel-aware map prior，再以 **scheduled residual fusion** 融合回 128ch map neck。

簡化成一條路徑：

```text
CFV + background OCC prototypes
  -> CFV-side prototype-wise activation
  -> prototype-aware channel/spatial refinement
  -> background-aware height collapse
  -> projection to 128ch
  -> scheduled residual fusion into map neck
  -> BEVSegHead
```

## 現有證據

### 1. Map drop 是 Occ+Map MTL 的真問題

目前 canonical ladder：

| Setting | Occ mIoU | Map mIoU |
| --- | ---: | ---: |
| CNN head + 128ch map neck, `map_loss_weight=1` | 39.82 | 39.94 |
| CNN head + 128ch map neck, `map_loss_weight=4` | 39.72 | 45.79 |
| CNN head + 128ch map neck, overlay dynamic V3-lite | 39.60 | 46.38 |
| CNN head + 128ch map neck, map-only | - | 48.34 |

`map_loss_weight=4` 已經把 gap 從 `48.34 - 39.94 = 8.40` 縮到 `48.34 - 45.79 = 2.55`，但仍低於 map-only。這代表第一層問題是 task priority / negative transfer，下一層問題才是如何把 occupancy semantic knowledge 轉成 map gain。

### 2. Background OCC class 和 map class 有資料對應

`occ_map_overlap_z0_3_train_non_free.md` 的 z=0~3 weighted 統計：

| Map class | 主要 OCC 對應 |
| --- | --- |
| `drivable_area` | `driveable_surface` 71.75% |
| `ped_crossing` | `driveable_surface` 75.61%，`sidewalk` 5.11% |
| `walkway` | `sidewalk` 52.63%，`terrain` 7.53%，`driveable_surface` 6.99%，`manmade` 6.87% |
| `stop_line` | `driveable_surface` 75.39%，`car` 3.16% |
| `carpark_area` | `driveable_surface` 52.57%，`car` 15.36%，`sidewalk` 6.10% |
| `divider` | `driveable_surface` 74.81%，`car` 4.06% |

這支持 background OCC prototypes 不是任意 prior。它們對 map layout 有明確對應，尤其是 `driveable_surface` 和 `sidewalk`。

### 3. Plain VAMI 還不夠成為最強故事

既有 `VoxelAwareMapIngest` 已經完成：

```text
CFV -> avg/max Z-collapse -> projection -> residual fusion -> map neck
```

它是必要 ablation，但故事比較像「把 voxel feature 多塞給 map」。這次主線要再往前一步：先用 background prototypes 在 CFV space 做 semantic filtering，再把 filtered CFV 轉給 map。

## Challenge / Motivation / Contribution

### Challenge

Camera-only Occ+Map MTL 裡，occupancy 和 map 共享 background layout semantics，但 naive sharing 會讓 map 明顯退化。ProtoOcc 的 CFV/PQD/prototype 對 occupancy 很強，卻沒有自然轉成 map gain。

### Motivation

MAESTRO 的 ablation 顯示 map gain 主要來自 prototype-guided task-specific feature generation，而不是單純多任務互助。但直接照搬 MAESTRO 會變成「二任務版 TSFG」。ProtoOcc 的真正特色是它已有強 CFV 和 occupancy prototypes，因此方法應該圍繞「如何把 CFV 裡的 background semantic prior 安全轉給 map branch」。

### Contribution

1. **Diagnosis**：指出 ProtoOcc-style Occ+Map MTL 的 map drop 不是 map head 上限不足，而是 CFV / occupancy prototype semantics 沒有被 map branch 消化。
2. **Method**：提出 CFV-side background prototype-TSFG，從 ProtoOcc 的 background Scene-Aware Queries 和 CFV 產生 voxel-to-map semantic prior。
3. **Protected Fusion**：保留 128ch map-specific neck 主路徑，用 two-layer residual fusion + scheduled `gamma(t)` 融合 `F_proto_cfv`，避免 map branch 被 48ch occupancy bottleneck 取代。

## 方法設計

### 整體路徑

```text
Dual_Branch_Encoder
  -> CFV: comprehensive_voxel_feature [B, 48, X, Y, Z]
  -> F_map: map_bev_feature [B, 128, H, W]

cnn3d_decoder(CFV)
  -> mask_feat [B, X, Y, Z, 32]
  -> prototype_occ_pred [B, X, Y, Z, 18]

Prototype_Query_Decoder
  -> Q_scene: Scene-Aware Queries [B, 18, 48]
  -> Q_bg = Q_scene[background_class_ids] [B, N_bg, 48]

CFVPrototypeTSFGMapFusion
  -> project Q_bg through prototype MLP to P_bg [B, N_bg, 48]
  -> produce prototype-wise activation A_bg over CFV
  -> generate F_proto_cfv by background-aware height collapse
  -> project F_proto_cfv to 128ch
  -> F_map' = F_map + gamma(t) * Delta(F_map, F_proto_cfv)

BEVSegHead(F_map')
```

### Step 1：Extract background Scene-Aware Query embeddings

第一版不重新做 GT-based prototype mining，也不另外從 GT map / GT occ 挖 prototype。直接使用 ProtoOcc PQD 已經算好的 **Scene-Aware Queries**：

```text
Q_scene = learnable_query
        + global_protoEMA_agg(Scene-Agnostic Prototype)
        + local_protoEMA_agg(Scene-Adaptive Prototype)

Q_scene = for_query_embed(Q_scene)
Q_scene = query_self_attn(Q_scene)
```

取出 background OCC classes 對應的 query。這裡要注意實作裡 `Prototype_Query_Decoder_nuScenes.forward()` 內部的 `query_feat` 在進 `forward_head()` 前是 `[Q_total, B, C]`，不是 `[B, Q_total, C]`。因此不要在 detector 端憑空切 tensor；應該在 PQD 內部把 RPL padding 移除、轉成 batch-first，再透過 optional return expose 出來：

```text
background classes = [driveable_surface, other_flat, sidewalk, terrain, manmade, vegetation]

# training with RPL:
# Q_scene_all = [RPL noisy queries, real class queries]
Q_real_qbc = Q_scene_all[RPL_pad_size:, :, :]  # [18, B, 48]
Q_real = Q_real_qbc.transpose(0, 1).contiguous()
                                                   # [B, 18, 48]
assert Q_real.shape[1] == num_classes

# inference / no RPL:
Q_real = Q_scene_all.transpose(0, 1).contiguous()
                                                   # [B, 18, 48]

Q_bg = Q_real[:, background_class_ids, :]      # [B, N_bg, 48]
Q_bg_source = Q_bg.detach()                    # block map loss -> PQD query path
```

設計選擇：

- 這一步只是 **extract existing ProtoOcc semantic representation**，不是新的 prototype mining contribution。
- 使用 Scene-Aware Query 而不是 raw Scene-Adaptive Prototype，因為它已經融合 learnable class query、scene-agnostic EMA prototype、scene-adaptive local prototype，並且是 ProtoOcc 最後真正拿去 decode occupancy mask 的 class representation。
- Training 有 RPL padding 時，只取非 RPL 的 real class queries；RPL queries 不進 map branch。這裡要加 assert，避免 silent bug 把 RPL noisy query 當成 background prototype。
- 第一版 `detach_query=True`，但這裡的 detach 必須包含 **PQD shared projection weights** 的保護。若在 adapter 端直接呼叫 PQD 的 `post_norm/mask_embed`，即使 `Q_bg.detach()`，map loss 仍會更新 PQD 的 `post_norm/mask_embed` 參數，這會污染 occupancy path。
- 因此第一版只允許兩種安全介面：
  1. `Prototype_Query_Decoder_nuScenes.py` 在自己的 `forward()` 內算好 `mask_embed_real_bqc = mask_embed(post_norm(Q_real))`，再以 `mask_embed_real_bqc.detach()` optional return 給 detector；adapter 只消費 detached filter，不回傳梯度到 PQD。
  2. adapter 自己建立 local `query_to_filter_mlp`，輸入 `Q_bg.detach()`，輸出 `P_bg`。這個 local MLP 可被 map loss 訓練，但不共享 PQD 權重。
- 推薦第一版用 `query_to_filter='pqd_mask_embed_detached'` 作主線，因為它最接近 PQD 原本 decode occupancy mask 的 filter 語意；local MLP 作為 ablation 或 fallback。
- 這需要在 `Prototype_Query_Decoder_nuScenes.py` expose batch-first real-class query/filter，例如 `query_embed_real_bqc` 和 `mask_embed_real_bqc`，shape 固定 `[B, 18, 48]`。實作時要保持原 occupancy loss / prediction path 不變。

### Step 2：Prototype-wise Feature

把 background Scene-Aware Query embeddings 投到 CFV channel space，**不是把 CFV 壓到 prototype space**。接著用它們和 CFV 做 dot product，產生 MAESTRO-style `Prototype-wise Feature`：

```text
CFV_source = CFV.detach()              # first version blocks map loss -> DBE/HFM

# preferred: consume PQD-computed detached mask filters
P_bg_all = mask_embed_real_bqc.detach()
                                      # [B, 18, 48]
P_bg = P_bg_all[:, background_class_ids, :]
                                      # [B, N_bg, 48]

# fallback / ablation: adapter-local projection, no shared PQD weights
# P_bg = Q_bg_source + zero_init_local_MLP(Q_bg_source)

Q_cfv = CFV_source                     # [B, 48, X, Y, Z]

P_bg_norm = normalize(P_bg, dim=-1)
Q_cfv_norm = normalize(Q_cfv, dim=1)

A_k = sigmoid(tau * dot(Q_cfv_norm, P_bg_norm[k]))  # [B, N_bg, X, Y, Z]
```

其中：

- `pqd_mask_embed_detached` 是主線，因為 PQD 原本就是用 `mask_embed(query)` 去和 voxel feature 做 dot product。但這個 filter 必須在 PQD 內部算好後 detach 再輸出；adapter 不直接持有或呼叫 PQD 的 `post_norm/mask_embed`，避免 map loss 更新 PQD 權重。
- 若使用 adapter-local MLP，應採 residual/identity-style init：`P_bg = Q_bg + zero_init_local_MLP(Q_bg)`，避免一開始變成 random projection。這條 path 的 MLP 可訓練，但它是 CFV-to-map adapter 的參數，不是 PQD 參數。
- `normalize` 是為了把 dot product 變成 cosine-like similarity，避免 activation 主要被 feature norm / query norm 主導。若不 normalize，訓練可能學到「放大向量 norm」而不是語意對齊。這應保留成 ablation：`normalize=True/False`。
- `A_k` 就是 `Prototype-wise Feature`：第 `k` 個 background query-derived filter 在每個 voxel 上的 spatial activation / attention score。
- `tau` 是 temperature，控制 activation sharpness。normalize 後 cosine 落在 `[-1, 1]`，所以 `tau=1` 會讓 sigmoid 只落在約 `[0.27, 0.73]`，空間選擇性太弱。第一版設 `tau=10`，並在 ablation 測 `8/10/15`。

可選 activation ablation：

```text
class_competition = softmax(tau * cos_k, dim=background_class)
backgroundness = sigmoid(tau_bg * max_k cos_k)
A_k = backgroundness * class_competition_k
```

直接 softmax over `k` 會強迫每個 voxel 都分給某個 background class，缺少「不屬於任何 background prior」的選項，所以不作第一版主線。

第一版先保留所有 background class 的 activation map：

```text
F_wise = A_k                          # [B, N_bg, X, Y, Z]
```

`F_wise` 是 spatial/prototype activation，不是 48-channel dense feature。這點和 MAESTRO 對齊：Prototype-wise Feature 本身是「每個 prototype 在空間上 activate 哪裡」。

如果要做更便宜的 ablation，才把它聚成單一 activation：

```text
A_bg = max_k A_k                      # [B, 1, X, Y, Z]
F_wise_single = A_bg                  # [B, 1, X, Y, Z]
```

`max_k` 的意思是對 background class 維度取最大值：每個 voxel 只要和任一 background semantic filter 對齊，就被保留下來。它不是對 spatial 維度取 max。

後續 ablation 才測 class-wise relation：

```text
A_map_c = sum_k R[c, k] * A_k
```

其中 `R[c,k]` 可以由 overlap 統計初始化，例如 `drivable_area -> driveable_surface`、`walkway -> sidewalk`。

### Step 3：Prototype-aware Feature + Enhanced CFV

平行於 `Prototype-wise Feature`，用 background query-derived filters 產生 MAESTRO-style `Prototype-aware Feature`。這一步是 channel attention：prototype group 告訴 CFV 哪些 channels 對 background/map prior 比較重要。

```text
P_avg = mean(P_bg, dim=1)              # [B, 48]
P_max = max(P_bg, dim=1)               # [B, 48]
C_gate = 1+MLP(cat(P_avg, P_max))     # final layer zero-init
                                      # [B, 48]

F_aware = CFV * C_gate[:, :, None, None, None]
                                      # [B, 48, X, Y, Z]
```

`F_aware = CFV * C_gate` 是 channel-wise scaling，不是空間遮罩。它的意思是：如果 background Scene-Aware Queries 表示這個 scene 需要某些 semantic/geometric channels，就把 CFV 的那些 channels 放大或保留。

`C_gate` 有 collapse 成 scene-invariant global channel scaling 的風險，因為 6 個 background query 可能在不同 scene 變化不大。因此它必須做 ablation：

```text
c_gate_mode='none'       # F_aware = CFV_source
c_gate_mode='prototype'  # P_bg avg/max -> MLP，主線
c_gate_mode='cfv'        # global pooled CFV_source -> MLP
c_gate_mode='hybrid'     # cat(P_bg pooled, CFV pooled) -> MLP
```

如果 `none` 和 `prototype` 差不多，代表 prototype-aware channel gate 沒有提供 scene-conditional 行為，後續可以簡化掉這條 path。

接著把 `Prototype-wise Feature` 和 `Prototype-aware Feature` concat，再用 3D conv 轉回 48-channel enhanced CFV：

```text
F_ref = Conv3d(cat(F_wise, F_aware), N_bg + 48 -> 48)
                                      # [B, 48, X, Y, Z]
```

這才是對應 MAESTRO 的：

```text
Enhanced Feature = Conv(Cat[Prototype-wise Feature, Prototype-aware Feature])
```

可選再加 lightweight Feature Suppression：

```text
S_supp = sigmoid(Conv3d(F_ref))        # [B, 1, X, Y, Z]
F_ref = F_ref * S_supp
```

第一版不要額外加 supervised suppression loss，避免把問題變成新的 loss engineering。先讓 map loss 自己學哪些 voxel prior 對 map 有用。若要更貼近 MAESTRO，可以把 Feature Suppression 作為 V2 ablation。

### Step 4：Background-aware height collapse

不直接 avg/max collapse 原始 CFV，而是 collapse 經過 `Prototype-wise + Prototype-aware` enhancement 後的 `F_ref`：

```text
F_avg = mean_z(F_ref)
F_max = max_z(F_ref)
F_proto_cfv = Conv2d(cat(F_avg, F_max), 96 -> 128)
```

如果要更貼近 map/ground layout，可加 low-Z prior：

```text
z_weight = learnable softmax over Z
F_lowz = sum_z z_weight[z] * F_ref[..., z]
F_proto_cfv = Conv2d(cat(F_avg, F_max, F_lowz), 144 -> 128)
```

第一版建議先用 `avg_max_concat`，避免增加過多自由度。若效果正，再測 learnable low-Z。

這裡不是「兩種 collapse 方法同時競爭」，而是一個 height aggregation strategy 裡使用兩個統計量：

- `mean_z`：保留整個 height column 的平均 layout / ground-context 訊息。
- `max_z`：保留某個高度上最強的 semantic response，對 thin / sparse 結構比較友善。

後續 ablation 才比較 `avg only`、`max only`、`avg+max`、`learnable low-Z`。

### Step 5：Scheduled residual fusion 到 map neck

第一版不要做直接 concat，也先不要上完整 gate。主路徑仍是 128ch map neck，CFV branch 只提供 residual correction：

```text
U = cat(F_map, F_proto_cfv)                              # [B, 256, H, W]
Delta = Conv3x3(256 -> 128) + BN + ReLU
Delta = Conv1x1(128 -> 128)                              # final conv zero-init
F_map' = F_map + gamma(t) * Delta
```

初始化：

- `gamma(t)` 是 schedule，不是 learnable parameter
- `gamma(t) = 0` at epoch 0，之後 2-3 epochs 線性 ramp 到 1
- `Delta` 最後一層 conv zero-init
- `detach_voxel_source=True`
- `gamma(t)` 需要由 dedicated warmup hook 或 detector helper 在每個 epoch/iter 更新；不能只把 `gamma_schedule` 寫在 config 裡但沒有 runtime setter。

這保證 step 0：

```text
F_map' == F_map
```

因此第一個 iteration 完全等價 `CNN head + 128ch map neck + map_loss_weight=4` baseline，降低訓練初期破壞 map neck 的風險。

不要同時使用 `learnable gamma = 0` 和 `Delta` final conv zero-init。若兩者都為 0，`gamma` 會因為 `Delta=0` 沒梯度，`Delta` 也會因為 `gamma=0` 沒梯度，容易形成 dead branch。第一版採用 scheduled `gamma(t)`，所以當 warmup 大於 0 後，zero-init residual branch 仍能收到梯度。

實作上新增 `CFVProtoTSFGWarmupHook`：

```text
epoch 0: adapter.set_gamma(0.0)
epoch 1..3: linearly ramp gamma to 1.0
epoch >= 3: adapter.set_gamma(1.0)
```

hook 要支援 DDP：若 model 被 `MMDistributedDataParallel` 包住，要透過 `runner.model.module` 找到 detector，再呼叫 `set_cfv_proto_tsfg_gamma(value)`。debug log 必須印出當前 `gamma(t)`，避免 branch 因為 hook 沒接上而永遠是 0。

#### Fusion 版本階梯

| 版本 | 公式 | 角色 |
| --- | --- | --- |
| A. Scalar residual | `F_map' = F_map + alpha(t) * F_proto_cfv` | 最簡 ablation。檢查 CFV prior 直接相加是否已有幫助；`alpha(t)` 用 schedule。 |
| B. Two-layer residual fusion | `F_map' = F_map + gamma(t) * Delta(cat(F_map, F_proto_cfv))` | **第一版主線**。保留 residual 保護，同時用 3x3+BN+ReLU+1x1 讓 `F_map` 和 `F_proto_cfv` 有 local/spatial + nonlinear interaction。 |
| C. Gated residual fusion | `F_map' = F_map + gamma(t) * G(cat(...)) * Delta(cat(...))` | 後續 robustness ablation。若 B 有效但不穩，才用 gate 控制位置/channel。 |
| D. Direct fusion | `F_map' = Conv(cat(F_map, F_proto_cfv))` | 風險對照組。容易變成加容量或覆蓋 map neck，不作主線。 |

主線先選 B。A 用來判斷「只要 residual prior 就夠不夠」；C 用來判斷 gate 是否真的需要；D 只用來證明直接 fusion 風險。

## 和 MAESTRO 的差異

| 面向 | MAESTRO | 本計劃 |
| --- | --- | --- |
| 主要問題 | tri-task MTL conflict | ProtoOcc Occ+Map 中 CFV/prototype semantic 未被 map 消化 |
| prototype 來源 | shared voxel feature 產生 foreground/background prototype groups | ProtoOcc PQD 的 background Scene-Aware Queries，經 detached PQD mask filter 或 adapter-local MLP 轉成 CFV-space semantic filters |
| TSFG 作用位置 | 為各 task 生成 task-specific features | 只在 CFV side branch 做 voxel-to-map semantic transfer |
| map branch | TSFG 直接產生 map task feature | 保留 128ch map-specific neck，只加 scheduled residual prior |
| 安全性 | general MTL module | scheduled `gamma(t)` + source detach + zero-init residual branch，保護 occupancy 和 map neck |
| 論文故事 | generic prototype-guided MTL | ProtoOcc-specific background prototype to map transfer |

論文文字不要寫：

```text
We adopt MAESTRO TSFG for map segmentation.
```

應該寫：

```text
Inspired by the observation that prototype-guided feature filtering can mitigate task conflict, we target a ProtoOcc-specific bottleneck: the BEV map branch does not consume the background semantic priors already encoded in the occupancy CFV/prototypes. We therefore introduce a protected CFV-to-map residual adapter guided by background occupancy prototypes.
```

## 預計改動哪些檔案

### 新增

| 檔案 | 用途 |
| --- | --- |
| `projects/mmdet3d_plugin/models/task_modules/cfv_prototype_tsfg_map_fusion.py` | 新 module：background Scene-Aware Query to CFV activation、height collapse、scheduled residual fusion，並保留 scalar/gated/direct fusion ablation 開關。 |
| `projects/mmdet3d_plugin/core/hook/cfv_proto_tsfg_warmup.py` | 新 hook：依 `gamma_schedule` 更新 adapter 的 runtime `gamma`，支援 DDP wrapper。 |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_cfv_proto_tsfg.py` | 主實驗 config，繼承 `ProtoOcc_multi_cnn_head_map_neck.py`，設定 `map_loss_weight=4.0`。 |
| `TWCC/train_multi_cnn_head_map_neck_cfv_proto_tsfg.sh` | TWCC launcher。 |

### 修改

| 檔案 | 預計改動 |
| --- | --- |
| `projects/mmdet3d_plugin/models/OccHead/Prototype_Query_Decoder_nuScenes.py` | 增加 helper 或 optional return，expose 非 RPL、batch-first 的 `query_embed_real_bqc` / `mask_embed_real_bqc.detach()`。原本 occupancy loss、mask prediction、RPL 邏輯不改。 |
| `projects/mmdet3d_plugin/models/task_modules/__init__.py` | export `CFVPrototypeTSFGMapFusion`。 |
| `projects/mmdet3d_plugin/core/hook/__init__.py` | export `CFVProtoTSFGWarmupHook`。 |
| `projects/mmdet3d_plugin/models/task_modules/voxel_aware_map_ingest.py` | `forward(..., **kwargs)` 忽略額外參數，確保舊 VAMI config 仍相容。 |
| `projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py` | 接收 PQD 回傳的 background query/filter，並傳給 adapter；train/eval 都走同一路徑；新增 `set_cfv_proto_tsfg_gamma(value)` 供 hook 更新 gamma。 |

### 不改

- `dual_branch_encoder.py`：保留 128ch map-specific neck 和 DBE/HFM 主結構。
- `cnn3d_decoder.py`：不改既有 18-class CE+Lovász 與 `mask_feat/prototype_occ_pred` 產生方式。
- `bev_seg_head.py`：map loss 和 head 不動。

## 實作 correctness guardrails

1. **PQD optional return 固定 batch-first**
   `Prototype_Query_Decoder_nuScenes.py` 應在 RPL slicing 後輸出：
   ```text
   query_embed_real_bqc: [B, 18, 48]
   mask_embed_real_bqc:  [B, 18, 48]
   ```
   不讓 detector 直接處理 `[Q_total, B, C]` 的內部 layout。

2. **不共享可訓練 PQD projection 給 map loss**
   第一版的 adapter 不直接呼叫 PQD 的 `post_norm/mask_embed`。若使用 `pqd_mask_embed_detached`，PQD 端要先算好 filter 再 `.detach()` optional return；若使用 local MLP，該 MLP 必須是 adapter 自己的參數。

3. **train / eval 對稱**
   `forward_train()` 和 `simple_test()` 都必須透過同一個 helper 取得 query/filter，並呼叫同一個 adapter path。eval 不能只跑 base `map_feature`，否則正式評估會 bypass CFV Prototype-TSFG。

4. **runtime gamma 真的更新**
   `gamma_schedule` 必須由 `CFVProtoTSFGWarmupHook` 或 detector helper 實際呼叫 `set_gamma()`。smoke test 要檢查 `gamma=0` 時輸出等價 baseline，warmup 後 `gamma > 0` 且 residual norm 非零。

## Config 草案

```python
_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

voxel_out_channels = 48
query_channels = 48
map_bev_channels = 128

model = dict(
    map_loss_weight=4.0,
    voxel_aware_map_ingest=dict(
        type='CFVPrototypeTSFGMapFusion',
        voxel_in_channels=voxel_out_channels,
        query_channels=query_channels,
        map_channels=map_bev_channels,
        occ_num_classes=18,
        background_class_ids=(11, 12, 13, 14, 15, 16),
        z_collapse_mode='avg_max_concat',
        detach_cfv=True,
        detach_query=True,
        query_source='scene_aware',
        query_to_filter='pqd_mask_embed_detached',
        prototype_mlp_hidden_channels=128,
        mlp_proto_init='residual_zero',
        normalize_similarity=True,
        activation_mode='sigmoid_cosine',
        temperature=10.0,
        c_gate_mode='prototype',
        fusion_mode='residual_two_layer',
        residual=True,
        zero_init_delta=True,
        gamma_schedule=dict(
            type='linear',
            start_epoch=0,
            end_epoch=3,
            start_value=0.0,
            end_value=1.0,
        ),
    ),
)

custom_hooks = [
    dict(
        type='CFVProtoTSFGWarmupHook',
        start_epoch=0,
        end_epoch=3,
        start_value=0.0,
        end_value=1.0,
    ),
]

evaluation = dict(start=10)
```

`background_class_ids` 須在實作前再和 dataset class order 確認。依目前 result class order，預設假設為：

```text
11 driveable_surface
12 other_flat
13 sidewalk
14 terrain
15 manmade
16 vegetation
```

## 實驗階梯

### V0：Clean strong baseline（已完成）

| Setting | Occ | Map | 角色 |
| --- | ---: | ---: | --- |
| `CNN head + 128ch map neck + map_loss_weight=4` | 39.72 | 45.79 | 正式 strong baseline |
| `overlay_dynamic V3-lite` | 39.60 | 46.38 | 目前 MTL map best，作為 secondary target |
| `map-only` | - | 48.34 | upper bound |

### V1：Plain VAMI ablation

目的：測「只把 CFV collapse 到 map neck」是否已經有效。

設定：既有 `ProtoOcc_multi_cnn_head_map_neck_vami.py`。

判讀：

- 若 V1 沒超過 45.79，代表單純 CFV collapse 不是答案，prototype-TSFG 必須證明 semantic filtering 有額外價值。
- 若 V1 超過 46.38，代表 CFV consumption 本身很強；V2 仍可作為 prototype-guided refinement，看是否補 `stop_line/divider`。

### V2：CFV Prototype-TSFG + scheduled residual（主實驗）

目的：測 background prototype-guided CFV semantic filtering 是否能超過 plain VAMI 和 V3-lite。

成功標準：

| 等級 | 條件 | 判斷 |
| --- | --- | --- |
| Fail | Map < 45.79 | 不如 clean weight4，放棄主線 |
| Weak positive | 45.79 <= Map < 46.38 | 有增益但不夠當主方法，只能當 ablation |
| Useful | Map >= 46.38 且 Occ >= 39.0 | 超過目前 MTL best，可繼續 |
| Strong | Map >= 47.0 且 `stop_line/divider/ped_crossing` 至少 2 類提升 | 有 paper method 潛力 |
| Very strong | Map 接近或超過 48.34 | 可主張縮小或突破 map-only upper-bound gap |

Class-level 優先順序：

1. `stop_line`：weight4 28.32，map-only 33.62，gap +5.30。
2. `divider`：weight4 33.49，map-only 35.43，gap +1.94。
3. `ped_crossing`：weight4 41.94，map-only 44.58，gap +2.64。
4. `carpark_area`：weight4 40.00，map-only 42.87，gap +2.87。

若 mean 提升只來自 `drivable_area`，視為 weak result，不能當主 contribution。

### V3：必要 ablations

| Ablation | 目的 |
| --- | --- |
| `Plain VAMI` | 證明 prototype-TSFG 不只是多一條 CFV residual。 |
| `Map-neck TSFG only` | 對應使用者提出的第 1 種：prototype 直接作用在 `F_map`，測是否只是 capacity/attention。 |
| A. scalar residual | `F_map' = F_map + alpha(t) * F_proto_cfv`，測最簡 residual prior 是否已足夠。 |
| B. two-layer residual fusion | 主線版本，測 `F_map` 和 `F_proto_cfv` 的 nonlinear residual interaction。 |
| C. gated residual fusion | 若 B 有效但不穩，測 gate 是否能提升 robustness。 |
| D. direct fusion | `Conv(cat(F_map, F_proto_cfv))` 風險對照，檢查直接覆蓋 map neck 是否退化。 |
| `activation_mode` variants | `sigmoid_cosine` vs `backgroundness * softmax_over_k`，確認 prototype-wise activation 是否有足夠空間選擇性。 |
| `temperature` variants | `tau=8/10/15`，避免 `tau=1` 造成 activation 近似常數。 |
| `query_to_filter` variants | `pqd_mask_embed_detached` vs adapter-local residual-zero MLP vs adapter-local random MLP，確認 query-to-CFV filter 初始化是否重要。 |
| `c_gate_mode` variants | `none/prototype/cfv/hybrid`，確認 prototype-aware channel gate 是否真的提供 scene-conditional 訊息。 |
| `gamma_schedule` variants | 2/3/6 epoch ramp，確認 residual branch warmup 長度。 |
| non-zero residual / no warmup | 證明 zero-init + warmup 對保護 baseline 重要。 |
| detach variants | 第一版 `detach_cfv=True, detach_query=True`；後續個別測 `detach_cfv=False`、`detach_query=False`、或細分 `detach_cfv_for_wise / detach_cfv_for_aware`。 |
| `query_source` variants | `scene_aware` / raw `scene_adaptive` / `ema_only`，確認 Scene-Aware Query 是否比單一 prototype source 穩。 |
| `background_class_ids` variants | 測是否加入 `car` 能補 `carpark_area/stop_line`，但避免動態物件污染 layout。 |
| `learnable_low_z` | 若 V2 有效，再測 learned height collapse 是否比 avg/max 更好。 |

## Stop / Go 條件

### Early stop

- epoch 11 EMA Map < 43.3：停。代表 side branch 早期已明顯破壞 map。
- epoch 11 Occ < 38.8：停。代表 despite detach/fusion，整體訓練已傷到 occupancy。

### Full run 判斷

- epoch 24 EMA Map < 45.79：放棄此主線。
- 45.79 <= Map < 46.38：寫成 negative/weak ablation，不再加複雜模組。
- Map >= 46.38 且 Occ >= 39.0：進 ablation。
- Map >= 47.0 且 thin classes 有改善：可作為主方法候選。

## Debug 指標

實作時建議加入可選 `debug=True`，每 N iter log：

```text
background query norm per class
query_embed_real_bqc / mask_embed_real_bqc shape
mask_embed_real_bqc.requires_grad should be False when using pqd_mask_embed_detached
P_bg norm per class after query_to_filter
min/max/mean of A_k per background class
activation entropy or max-class ratio over background classes
mean/std of A_bg
abs mean of Delta
gamma(t) value
||gamma(t) * Delta|| / ||F_map||
mean/std of gate G（只有 C gated residual ablation 需要）
```

如果 schedule 已經 ramp 到 1 但 residual norm 仍接近 0，表示 branch 沒被使用。
如果 residual norm 很大但 map 掉，表示 voxel prior 在污染 map neck。
如果 `driveable_surface` 的 `A_k` 長期壓過其它 background query，表示 branch 可能學成 driveable-only prior，要測 class-balanced relation 或 map-class-specific activation。
如果 `A_k` 長期落在 0.4-0.6 附近，表示 temperature 太小或 query/filter 沒對齊，prototype-wise activation 幾乎沒有空間選擇性。

## 預期風險

1. **學成 driveable-only prior**
   `driveable_surface` 會主導大多數 map class，可能只提升 `drivable_area`。解法是 class-balanced prototype pooling 或 per-map-class relation ablation。

2. **變成 MAESTRO clone**
   論文寫法不能說採用 TSFG。要強調這是 ProtoOcc CFV/prototype 未被 map branch 使用的問題，TSFG 只是 side-branch filtering mechanism。

3. **Scene-Aware Query 初期不穩**
   Scene-Aware Query 仍受 early `prototype_occ_pred` / local prototype quality 影響。zero-init residual 可保護初期；若 sparse background query 太弱，可測 EMA-only 或 raw Scene-Adaptive Prototype ablation。

4. **map loss 回流拉壞 OCC/PQD**
   第一版必須 `detach_cfv=True`、`detach_query=True`，而且不能讓 adapter 直接呼叫 PQD 的可訓練 `post_norm/mask_embed`。若使用 `pqd_mask_embed_detached`，optional return 的 `mask_embed_real_bqc` 必須已 detach。若 no-detach ablation 讓 Occ 掉超過 0.3，就不採用。

5. **Prototype-wise activation 太平**
   normalize 後若 `temperature` 太小，`A_k` 會接近常數，`F_wise` 失去意義。第一版用 `tau=10`，並 log `A_k` distribution。

6. **只增加 capacity 而非語意轉移**
   必須做 Plain VAMI 和 Map-neck TSFG only ablation。如果它們和 V2 差不多，論文不能主張 background prototype transfer 是關鍵。

## Paper story 寫法

主標題方向：

```text
Prototype-guided Voxel-to-Map Transfer for Camera-only Occupancy and BEV Map Joint Learning
```

摘要中的核心句：

```text
Although ProtoOcc learns strong scene-aware occupancy queries from local and global prototypes, the BEV map branch in a joint Occ+Map setting does not directly consume these occupancy semantics or the occupancy-supervised CFV. We introduce a background-query-guided voxel-to-map adapter that filters CFV in the voxel space and injects the resulting semantic prior into the map-specific BEV neck through a protected scheduled residual path.
```

和 MAESTRO 的 positioning：

```text
MAESTRO shows that prototype-guided task-specific feature generation can mitigate MTL conflict. However, directly adopting its TSFG does not address the ProtoOcc-specific bottleneck: the map branch is separated before the occupancy-supervised CFV is formed, while ProtoOcc's strongest semantic representation is its scene-aware occupancy query. Our method therefore performs query-guided filtering on CFV and transfers only a residual semantic prior to the protected map-specific neck.
```

不要在 paper story 裡過度承諾「background prototypes 會直接解決 `stop_line/divider`」。目前 overlap 統計顯示很多 thin / overlay map classes 仍高度落在 `driveable_surface` 上，所以合理說法是：background prototypes 提供可檢驗的 layout prior，是否能補 thin classes 要靠 class-level ablation 證明。若只提升 `drivable_area`，這條方法只能寫成 weak ablation。

## 第一版實作順序

1. 更新 `Prototype_Query_Decoder_nuScenes.py`，先 expose batch-first `query_embed_real_bqc` / detached `mask_embed_real_bqc`，確認 RPL slicing 後 shape 是 `[B, 18, 48]`。
2. 新增 `CFVPrototypeTSFGMapFusion` module，先支援 GT-free background query/filter activation、avg+max collapse、two-layer scheduled residual fusion。
3. 新增 `CFVProtoTSFGWarmupHook`，讓 `gamma(t)` 在 runtime 真正從 0 ramp 到 1，並支援 DDP wrapper。
4. 更新 `ProtoOccCnnSegHead` helper，讓 adapter 可收到 background query/filter；`forward_train()` 和 `simple_test()` 必須走同一路徑。
5. 保持既有 `VoxelAwareMapIngest` config 可跑，作為 Plain VAMI ablation。
6. 新增 `ProtoOcc_multi_cnn_head_map_neck_cfv_proto_tsfg.py`。
7. 做 CPU/config build 檢查，確認 baseline config 不受影響。
8. 先跑 1-GPU smoke：檢查 debug log 中 `gamma(t)=0` 時 residual output 等價 baseline、warmup 後 residual norm 非零、background query/filter shape 正常、`mask_embed_real_bqc.requires_grad=False`。
9. 正式 TWCC 跑 epoch 11 EMA；通過 stop criterion 才跑滿 epoch 24。
