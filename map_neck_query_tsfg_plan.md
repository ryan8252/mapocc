# Map-neck Query-TSFG 計劃

日期：2026-05-17

## 一句話定位

> 用 ProtoOcc PQD 的 background Scene-Aware Queries 當作 MAESTRO-style prototype group，直接引導 128ch map-specific BEV feature；不再把 48ch CFV collapse 成 map residual。

這版主線不是舊的 `CFVPrototypeTSFGMapFusion`。舊版走：

```text
background PQD filter -> CFV activation/gating -> Z collapse -> 128ch projection -> map residual
```

這次改成：

```text
background post-norm Scene-Aware Query
  -> detach
  -> MLP 48 -> 128
  -> prototype-wise response on F_map
  -> prototype-aware channel modulation on F_map
  -> zero-init residual delta
  -> BEVSegHead
```

核心差異是：**prototype 直接作用在 128ch map branch**，而不是先作用在 occupancy CFV。

## Baseline 與問題

目前正式 comparison point 仍是乾淨的 CNN map head + 128ch map neck：

| Setting | Occ mIoU | Map mIoU | 用途 |
| --- | ---: | ---: | --- |
| CNN head + 128ch map neck, `map_loss_weight=1` | 39.82 | 39.94 | naive MTL reference |
| CNN head + 128ch map neck, `map_loss_weight=4` | 39.72 | 45.79 | clean strong baseline |
| CNN head + 128ch map neck, V3-lite | 39.60 | 46.38 | current MTL map best, loss-balancing add-on |
| CNN head + 128ch map neck, map-only | - | 48.34 | map upper bound |

`map_loss_weight=4` 已經證明大部分 map drop 來自 task-level map suppression，但仍比 map-only 低 2.55。V3-lite 把 Map 推到 46.38，但主要增益在 `carpark_area`，`stop_line / divider` 仍沒有被根本解決。

舊 `CFVPrototypeTSFGMapFusion` 的問題不是 prototype 概念本身，而是它把 map improvement 綁到 48ch occupancy CFV residual；結果沒有穩定超過 weight4 baseline，後期還出現退化。因此下一版應把 query prototype 留在 map feature space 使用。

## 設計原則

1. **query detach 交給 adapter 控制**
   PQD 只負責 expose `query_norm_real_bqc = post_norm(query_real_qbc)`，不要在 PQD 內部 `no_grad()` / `detach()` 這個欄位。第一版由 `MapNeckQueryTSFG(detach_query=True)` 在 adapter 內 detach，避免 map loss 反打 PQD / occupancy decoder；之後 no-detach ablation 只需要改 config。

2. **作用在 128ch map branch**
   直接吃 `map_bev_feature` 或 map neck 輸出的 `F_map: [B, 128, H, W]`。不使用 `comprehensive_voxel_feature`，不做 Z collapse。

3. **fixed gamma = 1**
   不使用 `gamma_schedule`。舊 CFV 版的 schedule 讓 residual 到後期被強迫拉到 full strength，對高風險 CFV prior 不友善。這版讓 residual branch 自己靠 map loss 學出有用 delta。

4. **zero-init residual**
   `F_out = F_map + Delta`，其中 `Delta` 最後一層 conv zero-init，所以 step 0 完全等價 baseline。不要同時設定 `gamma=0`，否則 `gamma=0` + `Delta=0` 會讓 branch 沒有有效梯度。

5. **保留 map neck 主路徑**
   新 module 是 side adapter，不取代 `map_bev_feature -> BEVSegHead`。若 residual 無用，模型可以維持接近 baseline。

## Tensor Path

### 1. PQD 端 expose post-norm real query

目前 PQD 已有：

```python
query_real_qbc = query_feat[RPL_pad_size:]           # [18, B, 48]
query_embed_real_bqc = query_real_qbc.transpose(0, 1)
mask_embed_real_bqc = self.mask_embed(
    self.post_norm(query_real_qbc)).transpose(0, 1)
```

這次需要的 prototype source 是：

```python
query_norm_real_bqc = self.post_norm(query_real_qbc).transpose(0, 1)
```

建議新增 optional return 欄位：

```python
query_norm_real_bqc: [B, 18, 48]
```

不要讓 detector 端重建 `post_norm(query_real_qbc)`，因為 RPL slicing、layout、normalization 都屬於 PQD 內部語意。

### 2. Background class selection

第一版只取 static/background OCC classes：

| OCC id | name | 理由 |
| ---: | --- | --- |
| 11 | `driveable_surface` | 對 `drivable_area / stop_line / divider` 最直接 |
| 12 | `other_flat` | flat background 補充 |
| 13 | `sidewalk` | 對 `walkway / ped_crossing` 重要 |
| 14 | `terrain` | walkway/road 邊界補充 |
| 15 | `manmade` | walkway / static structure prior |
| 16 | `vegetation` | map background context |

不放 `free`，也不放 dynamic foreground。

### 3. Prototype projection

```python
p_bg_raw = query_norm_real_bqc[:, background_ids, :]
p_bg_48 = p_bg_raw.detach() if detach_query else p_bg_raw
p_bg_128 = prototype_proj(p_bg_48)
```

Shape：

```text
p_bg_48:  [B, 6, 48]
p_bg_128: [B, 6, 128]
F_map:    [B, 128, H, W]
```

`prototype_proj` 是 adapter-local MLP，不共享 PQD 的 `mask_embed`。理由是這次要對齊 map feature space，而不是 occupancy mask-feature space。

## Map-neck Query-TSFG Module

暫定 module 名稱：

```text
MapNeckQueryTSFG
```

可放在：

```text
projects/mmdet3d_plugin/models/task_modules/map_neck_query_tsfg.py
```

### Prototype-wise branch

對應 MAESTRO 的 prototype-wise feature：

```python
if similarity_mode == 'cosine':
    p_sim = F.normalize(p_bg_128, dim=-1)
    f_sim = F.normalize(F_map, dim=1)
elif similarity_mode == 'dot':
    p_sim = p_bg_128
    f_sim = F_map
else:
    raise ValueError(f'Unsupported similarity_mode: {similarity_mode}')

A = torch.einsum('bnc,bchw->bnhw', p_sim, f_sim)
```

Shape：

```text
A: [B, 6, H, W]
```

第一版不使用 temperature scaling，避免 `A` 的強度多一個難歸因超參數。主線先用 `similarity_mode='cosine'` 作為保守版本；因為 MAESTRO 原版 prototype-wise feature 是 dot-product，必須把 `similarity_mode='dot'` 列為必要 ablation。dot-product 更貼近 MAESTRO，但會吃 `p_bg_128` / `F_map` 的 feature norm，因此 debug log 要記錄 `A.mean/std/min/max`。

### Prototype-aware branch

對應 MAESTRO 的 prototype-aware channel modulation：

```python
p_avg = p_bg_128.mean(dim=1)
p_max = p_bg_128.max(dim=1).values
gate = 1.0 + gate_mlp(torch.cat([p_avg, p_max], dim=-1))
F_aware = F_map * gate[:, :, None, None]
```

Shape：

```text
gate:    [B, 128]
F_aware: [B, 128, H, W]
```

`gate_mlp` 最後一層建議 zero-init，讓初始 `gate=1.0`。

### Enhanced feature / residual delta

```python
enhanced_input = torch.cat([F_map, F_aware, A], dim=1)
Delta = fusion(enhanced_input)
F_out = F_map + Delta
```

Shape：

```text
enhanced_input: [B, 128 + 128 + 6, H, W]
Delta:          [B, 128, H, W]
F_out:          [B, 128, H, W]
```

`fusion` 建議：

```text
Conv2d(262 -> 128, 3x3, padding=1, bias=False)
BN
ReLU
Conv2d(128 -> 128, 1x1, bias=True)
```

最後一層 `Conv2d(128 -> 128)` zero-init。這樣 step 0：

```text
Delta = 0
F_out = F_map
```

### Debug monitoring

`MapNeckQueryTSFG` 的 debug log 要走 `logging.getLogger('mmdet')`，寫進 TWCC `work_dirs/*.log`，不要只用 stdout。當 `debug=True` 且 step 命中 `debug_interval` 時至少記錄：

```text
A_mean / A_std / A_min / A_max
gate_minus_1_abs_mean
delta_abs_mean
delta_abs_max
fmap_abs_mean
delta_to_fmap_ratio = delta_abs_mean / (fmap_abs_mean + eps)
```

判讀：

```text
training 前幾百 iter: delta_to_fmap_ratio 接近 0 是正常的 zero-init 現象
訓練中逐步爬到 0.05 ~ 0.2: branch 有實際介入但未壓過 map neck
epoch 11 仍 < 0.01: branch 近似 no-op，需先檢查梯度 / learning rate / fusion 設計
很快 > 0.5: residual 可能太強，優先測 fixed gamma=0.5 或縮小 fusion
```

這個 monitoring 是 training-time 必備診斷，不改變主方法；正式訓練可先 `debug=False`，但 smoke / epoch-11 diagnosis 應開啟短 interval 版本確認 branch 不是裝飾。

## Gamma 決策

第一版：

```text
gamma = 1.0 fixed
F_out = F_map + Delta
```

不做：

```text
gamma schedule
learnable gamma init 0
gamma=0 + zero-init Delta
```

理由：

- zero-init residual 已經提供 baseline-equivalence，不需要 schedule。
- `gamma=0` 加上 zero-init delta 會讓 branch 初始沒有有效梯度。
- learnable gamma 與最後一層 residual conv 的 scale 高度重複，第一版會讓 attribution 變差。
- 如果 branch 有害，zero-init residual 可以靠訓練維持小 delta；不用 schedule 強迫它變大。

後續 ablation 可測：

| Variant | 說明 |
| --- | --- |
| fixed `gamma=0.5` | 檢查 residual 強度是否過大 |
| bounded learnable gamma, init 0.5 | 只作為第二階段，不當第一版 |
| no zero-init + fixed gamma | 風險對照，預期較容易破壞 baseline |

## Detector Integration

優先接在 `ProtoOccCnnSegHead`，沿用既有 `voxel_aware_map_ingest` slot 或新增更準確的 optional slot 都可。為了少改 code 和保留 config 相容，第一版可以沿用現有 slot 名稱，但 module 本身不吃 `voxel_feature`。

建議 interface：

```python
forward(
    map_feature,
    voxel_feature=None,
    query_norm_real_bqc=None,
    query_embed_real_bqc=None,
    mask_embed_real_bqc=None,
    **kwargs)
```

`requires_pqd_query_info = True`，讓 detector 在需要時要求 PQD 回傳 query info。舊 VAMI / baseline path 不受影響。

## Config 草案

新增：

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_query_tsfg.py
```

繼承：

```python
_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']
```

設定：

```python
model = dict(
    map_loss_weight=4.0,
    voxel_aware_map_ingest=dict(
        type='MapNeckQueryTSFG',
        query_channels=48,
        map_channels=128,
        occ_num_classes=18,
        background_class_ids=(11, 12, 13, 14, 15, 16),
        detach_query=True,
        prototype_hidden_channels=128,
        similarity_mode='cosine',
        use_prototype_wise=True,
        use_prototype_aware=True,
        residual=True,
        fixed_gamma=1.0,
        zero_init_delta=True,
        zero_init_gate=True,
        debug=False,
        debug_interval=100,
    )
)
```

不要加 `CFVProtoTSFGWarmupHook`。`custom_hooks` 只保留原本需要的 `MEGVIIEMAHook`。

## 實作檔案清單

### 新增

| 檔案 | 用途 |
| --- | --- |
| `projects/mmdet3d_plugin/models/task_modules/map_neck_query_tsfg.py` | `MapNeckQueryTSFG` module。 |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_query_tsfg.py` | 主實驗 config。 |
| `TWCC/train_multi_cnn_head_map_neck_query_tsfg.sh` | TWCC launcher。 |

### 修改

| 檔案 | 改動 |
| --- | --- |
| `projects/mmdet3d_plugin/models/OccHead/Prototype_Query_Decoder_nuScenes.py` | 在 `return_query_info` 中新增 `query_norm_real_bqc`，保留既有 `query_embed_real_bqc` / `mask_embed_real_bqc`。 |
| `projects/mmdet3d_plugin/models/task_modules/__init__.py` | export `MapNeckQueryTSFG`。 |
| `projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py` | 若沿用 `_apply_voxel_aware_map_ingest()`，只需確保 `query_info` 會 pass through。若目前已可 pass `**query_info`，理論上不需大改。 |

不需要修改：

- `Dual_Branch_Encoder`
- `BEVSegHead`
- `cnn3d_decoder`
- `ProtoOccMultitask`，除非要把 prototype-map-head branch 也支援這個 adapter

### 2026-05-17 實作紀錄

新增：

| 檔案 | 內容 |
| --- | --- |
| `projects/mmdet3d_plugin/models/task_modules/map_neck_query_tsfg.py` | 新增 `MapNeckQueryTSFG`，支援 `query_norm_real_bqc` / `query_embed_real_bqc` / `mask_embed_real_bqc` source、`similarity_mode='cosine'|'dot'`、prototype-wise response、prototype-aware channel modulation、fixed-gamma zero-init residual、`mmdet` logger debug。 |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_query_tsfg.py` | 主實驗 config，繼承 128ch map neck baseline，設定 `map_loss_weight=4.0`、啟用 `MapNeckQueryTSFG`、保留 `MEGVIIEMAHook`、`evaluation.start=10`。 |
| `TWCC/train_multi_cnn_head_map_neck_query_tsfg.sh` | TWCC launcher，輸出到 `work_dirs/ProtoOcc_multi_cnn_head_map_neck_query_tsfg`。 |

修改：

| 檔案 | 內容 |
| --- | --- |
| `projects/mmdet3d_plugin/models/OccHead/Prototype_Query_Decoder_nuScenes.py` | `return_query_info` 新增 `query_norm_real_bqc`；PQD 內不 detach，detach 由 adapter 的 `detach_query` 控制。 |
| `projects/mmdet3d_plugin/models/task_modules/__init__.py` | export `MapNeckQueryTSFG`。 |
| `map_neck_query_tsfg_plan.md` | 補上實作紀錄與驗證結果。 |

已完成驗證：

```text
python -m py_compile map_neck_query_tsfg.py Prototype_Query_Decoder_nuScenes.py
bash -n TWCC/train_multi_cnn_head_map_neck_query_tsfg.sh
conda run -n mapocc Config.fromfile(...)  # map_loss_weight=4.0, type=MapNeckQueryTSFG, similarity_mode=cosine, EMA hook, evaluation.start=10
conda run -n mapocc tensor smoke          # F_out shape [2,128,8,8], zero-init max diff=0, query grad None, map grad exists, final delta conv grad exists
conda run -n mapocc dot-path smoke        # similarity_mode='dot' forward shape [2,128,8,8]
conda run -n mapocc build_detector(...)   # CPU-only monkeypatch torch.Tensor.cuda, registry/config build OK
```

梯度 smoke 細節：因為 `zero_init_delta=True` 與 `zero_init_gate=True`，第一次 backward 只有 final delta conv 會立刻有有效梯度；第二次 backward 可看到 `prototype_proj`、fusion 前段、gate final Linear 開始有梯度；gate MLP 前層到第三次 backward 才開始有梯度。這符合 zero-init residual/gate 的預期。

## Validation / Smoke Tests

1. **shape smoke**

確認：

```text
query_norm_real_bqc: [B, 18, 48]
p_bg_128:           [B, 6, 128]
F_map:              [B, 128, H, W]
A:                  [B, 6, H, W]
F_out:              [B, 128, H, W]
```

2. **baseline-equivalence smoke**

在 zero-init delta 下：

```text
max_abs(F_out - F_map) == 0
```

這比舊 CFV 版的 `gamma=0` smoke 更直接，因為 training config 實際使用的就是 `fixed_gamma=1`。

3. **gradient smoke**

因為 `Delta` 最後一層 conv zero-init，第一個 backward 不應要求所有前段 module 立刻都有有效梯度。正確 smoke 分兩段：

第一個 backward，用 toy loss `F_out.mean().backward()` 確認：

```text
query_norm_real_bqc.grad is None
map_feature.grad exists
fusion final Conv2d weights have grads
PQD weights do not receive map-loss grads
```

跑一次 `optimizer.step()` 後，再做第二個 backward，確認：

```text
prototype_proj has grads
gate_mlp final Linear has grads if use_prototype_aware=True
fusion first Conv2d / BN path has grads
query_norm_real_bqc.grad is still None when detach_query=True
```

若 `zero_init_gate=True`，gate MLP 前層可能要到第二次 `optimizer.step()` 後的第三個 backward 才開始有梯度，因為 gate 的最後 Linear 也是 zero-init。這是正常現象；重點是最後一層先收到梯度並逐步打開 residual / gate path。這比 `gamma=0 + zero-init Delta` 安全，因為 `fixed_gamma=1` 不會額外切斷 branch 梯度。

4. **prototype discriminative sanity check**

在 main run 送 TWCC 前，先用已訓練 PQD checkpoint 跑一小段 val / train sample，收集 `query_norm_real_bqc[:, background_ids, :]`，檢查 background prototype 是否真的有 class-discriminative 訊號：

```text
bg_query_cosine_matrix: [6, 6]
offdiag_mean
offdiag_max
per_pair_mean_similarity
```

不要只用單一 sample 判死刑，因為 `query_norm_real_bqc` 是 scene-aware。建議統計多個 scene：

```text
如果跨 scene 的 offdiag_mean / offdiag_max 長期接近 0.9+
=> 6 個 background prototypes 幾乎 collapsed
=> cosine response maps 可能高度相似，prototype-wise branch 資訊量不足
```

若 prototype similarity 太高，先不要直接調大 fusion。優先檢查：

```text
1. A response map correlation 是否也 collapsed
2. similarity_mode='dot' 是否比 cosine 有更多差異
3. mask_embed_real_bqc source 是否比 post-norm query source 更有 class separation
4. background_class_ids 是否需要縮小或調整
```

5. **config smoke**

確認：

```text
build_detector(cfg.model) 成功
det.voxel_aware_map_ingest.requires_pqd_query_info == True
custom_hooks 沒有 CFVProtoTSFGWarmupHook
map_loss_weight == 4.0
```

## Experiment Ladder

### Main run

| Experiment | Goal |
| --- | --- |
| `MapNeckQueryTSFG` full | prototype-wise + prototype-aware，fixed gamma=1，zero-init residual |

比較點：

| Setting | Occ | Map | 判斷 |
| --- | ---: | ---: | --- |
| weight4 baseline | 39.72 | 45.79 | 必須先超過 |
| V3-lite | 39.60 | 46.38 | 若超過才有主方法潛力 |
| map-only | - | 48.34 | upper bound |

### Ablations

| Ablation | 目的 |
| --- | --- |
| prototype-wise only | 只使用 `A`，檢查 spatial response 是否主因 |
| prototype-aware only | 只使用 channel gate，檢查 scene-level channel modulation 是否主因 |
| `similarity_mode='dot'` | MAESTRO-faithful dot-product response，對照第一版 cosine response |
| `mask_embed_real_bqc` source | 與 post-norm query source 對照：occupancy mask filter vs query prototype |
| add `car` to background ids | 測 `carpark_area`，但若 static classes 掉太多不採用 |
| fixed `gamma=0.5` | 檢查 residual 過強問題 |
| no detach query | 只在主線有效後測，觀察 Occ 是否掉超過 0.3 |
| stack on V3-lite | 第二階段組合，不用於第一個 attribution run |

## Success / Stop Criteria

| 等級 | 條件 | 判斷 |
| --- | --- | --- |
| Fail | Map < 45.79 | 不如 clean weight4，停止主線 |
| Weak positive | 45.79 <= Map < 46.38 | 只能當 ablation，不當主 contribution |
| Useful | Map >= 46.38 且 Occ >= 39.0 | 超過現有 MTL map best，可繼續 |
| Strong | Map >= 47.0 且 `stop_line/divider/ped_crossing` 至少兩類提升 | 有 paper method 潛力 |
| Very strong | 接近 48.34 | 顯著縮小 map-only gap |

Class-level 優先順序：

1. `stop_line`：weight4 28.32，map-only 33.62。
2. `ped_crossing`：weight4 41.94，map-only 44.58。
3. `divider`：weight4 33.49，map-only 35.43。
4. `carpark_area`：weight4 40.00，map-only 42.87。

如果 mean gain 只來自 `drivable_area` 或只來自 `carpark_area`，不能說是完整 TSFG 解決方案。

## Paper Story

暫定方法名：

```text
Background Query-Guided Map Feature Generation
```

可寫成：

> Instead of collapsing occupancy voxel features into the map branch, we reuse ProtoOcc's background scene-aware queries as semantic prototypes and apply MAESTRO-style prototype-wise and prototype-aware feature generation directly on the protected 128-channel map representation.

中文論點：

> 舊 CFV 路線證明「把 occupancy feature 塞給 map」不夠穩；新的 query-TSFG 路線把 ProtoOcc 已學到的 background Scene-Aware Queries 當成 semantic prototype，但只在 map-specific 128ch branch 上做 feature generation，避免 map branch 被 occupancy bottleneck 取代。

## 下一步

1. 在 PQD optional return 補 `query_norm_real_bqc`。
2. 新增 `MapNeckQueryTSFG`，先只支援 detached query、fixed gamma=1、zero-init residual。
3. 新增 config / TWCC launcher。
4. 跑 shape + gradient + baseline-equivalence smoke。
5. 跑 prototype discriminative sanity check，確認 background query 沒有明顯 collapsed。
6. 開 debug 短跑確認 `A`、`gate`、`Delta/F_map` monitoring 會寫入 `work_dirs/*.log`。
7. 先訓練到 epoch 11，與 weight4 epoch 11 Map 43.82 比；若明顯低於 43.82，先停下看 debug log，不直接跑滿 24。
8. 若 epoch 11 正常，再跑 epoch 24，正式和 weight4 45.79 / V3-lite 46.38 比。
