# ProtoMapHead 對齊 ProtoOcc PQD 的修正紀錄

日期: 2026-04-30
範圍: `projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py`

---

## 1. 動機

實驗結果 [experiment_results_summary.md](experiment_results_summary.md) 顯示:

| 設定 | Map mIoU |
|---|---:|
| ProtoMapHead + 256ch map neck EMA, **coarse output** | 39.44 |
| ProtoMapHead + 256ch map neck EMA, **final output** | 39.09 |

Coarse output(只有 CNN 預測,還沒做 prototype refinement)比 final output 高 0.35。
**Prototype refinement 目前是負貢獻**,但 ProtoOcc 在 occ 上 prototype refinement 是正向的,沒理由搬到 map 上會壞掉,所以懷疑是「對齊不完整」的 bug。

對照 ProtoOcc 的 `Prototype_Query_Decoder_nuScenes` (PQD) 後,確認 ProtoMapHead 有四個跟 PQD **設計意圖不一致**的點。本次修 P0 + P1。

---

## 2. 改了什麼

### P0 #1 — Prototype 維度 bottleneck(主要修正)

**問題**: ProtoMapHead 把 `mask_feat` (`hidden_channels` 維,在 config 是 128) 同時用在:
1. AdaPG 的 prototype 池化
2. AgnoPG 的 EMA bank
3. Scene-aware query 的 local/global agg
4. 最終 dot product

**PQD 的設計** (`projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py:160`):
- `cnn3d_decoder` `out_dim=32` → `mask_feat` 是 32 維
- `prototype_query_decoder` `feat_channels=48`, `out_channels=48` → query / dot product 是 48 維
- EMA bank `nn.Embedding(num_classes, 32)` 在 32 維低維空間
- aggs `Linear(32, 48)` 投影到 query 空間
- 最終 dot product 用的是 **`voxel_feats` (48 維)**,不是 `mask_feat` (32 維)

**為什麼這個 bottleneck 重要**:
- 高維特徵在 dot product 路徑上承擔 mask prediction loss 的 gradient
- 如果 prototype 也定義在這個高維空間,prototype refinement gradient 跟 mask loss gradient **共用同一個特徵空間**,造成 prototype 跟著 mask loss 跑而不是反映語意中心
- 線狀類別 (`stop_line`, `divider`, `ped_crossing`) 像素少,mask loss gradient 在這些像素特別 noisy,直接打回 prototype 就會讓 prototype 跟著 noise 跑 → final output 比 coarse output 還差

**修法**:
- 加入 `proto_bottleneck = nn.Conv2d(hidden_channels, proto_dim, 1)`
- AdaPG / AgnoPG / scene-aware query 全部改用 `proto_feat` (低維,預設 `proto_dim=32`)
- aggs 從 `Linear(hidden, hidden)` 改成 `Linear(proto_dim, hidden)`
- EMA bank 從 `Embedding(K, hidden)` 改成 `Embedding(K, proto_dim)`
- `mask_feat` (hidden 維) 仍保留給最終 dot product 用,不變

**對應 PQD 行為**: 現在 `proto_dim=32` 預設值跟 PQD 的 `cnn3d_decoder out_dim=32` 完全一致。

---

### P0 #2 — AdaPG 加入「不確定 pixel 過濾」

**問題**: PQD 的 AdaPG 在挑 prototype 來源 voxel 時,過濾掉 top-1 跟 top-2 信心差距小的不確定 voxel:
```python
top2_values, _ = torch.topk(mask_, 2, dim=1)
difference = top2_values[:, 0] - top2_values[:, 1]
scores = 1.0 - difference          # difference 大 → score 小 → 確定
vis_mask = scores >= 0.9            # vis_mask 是「不確定」mask
mask_cls *= ~vis_mask               # 排除不確定 voxel
```

只有「模型確定屬於某類 + 沒有競爭類別接近」的 voxel 才會進 prototype 池。

**ProtoMapHead 之前的版本**:
```python
conf_mask = soft_masks[:, k] > 0.5    # 只要 sigmoid > 0.5 全部進池
```

`stop_line` 邊界、`ped_crossing` 模糊區那些「sigmoid 0.51,模型其實不太確定」的 pixel 全部進池,稀釋 prototype 品質。

**修法**:
- 新增 `prototype_mining_thresh` 參數 (預設 0.7)
- AdaPG 改成 `soft_masks[:, k] > prototype_mining_thresh`
- 由於 map 是 per-class sigmoid (類別可重疊),不能直接套 PQD 的 top-2 trick;這裡是 **近似移植** PQD 的設計意圖,不是嚴格等效。差異在於:PQD 的 top-2 margin 排除「兩 class 同時高分」的 ambiguous voxel,sigmoid 絕對值 threshold 排不掉這種情況(對 map 而言 `drivable_area` ∩ `lane` 多類同時高分本來就合法,所以這層保護在 sigmoid setting 下也用不上)。0.7 = 0.5 + 0.2 margin 等於要求模型該類別至少有 0.2 的把握離決策邊界。

**對應 PQD 行為**: 「只用模型最確定的 pixel」這個 intent 被搬過來,但 PQD 的 multi-class disambiguation 那層保護在 sigmoid 設定下用不上。對 sparse 線狀類別仍有保護作用,因為邊界區占比高,過濾掉後 prototype 不會被邊界 noise 拉走。

---

### P1 #3 — FFN(評估後決定不做)

原本想加 FFN,因為 `nn.MultiheadAttention` 沒帶 FFN block,query 之間做 self-attention 後沒有 channel-wise 的非線性混合。

**但仔細看 PQD config** (`ProtoOcc_proto_map_head.py:194`):
```python
operation_order=['self_attn', 'norm',]
```

PQD 的 transformer 雖然 config 上有 `ffn_cfgs`,但 `operation_order` **沒包含 'ffn'**,所以 FFN 是被宣告但實際沒用。PQD 的 transformer 真實結構就是 `self_attn → norm`,沒 FFN。

**結論**: 為了忠於 PQD,**不加 FFN**。已經紀錄在程式碼註解裡。

---

### P1 #4 — `post_norm`

**問題**: PQD 在 `forward_head` 一進來就先做 LayerNorm:
```python
def forward_head(self, decoder_out, mask_feature):
    decoder_out = self.post_norm(decoder_out)   # ← 在用之前先 norm
    decoder_out = decoder_out.transpose(0, 1)
    cls_pred = self.cls_embed(decoder_out)
    mask_embed = self.mask_embed(decoder_out)
```

ProtoMapHead 之前沒有這層。Self-attn 出來的 q 直接餵 `mask_embed`,scale 不穩。

**修法**:
- 加 `self.post_norm = nn.LayerNorm(hidden_channels)`
- `_forward_head` 在 self-attn 之後、`mask_embed` 之前加 `q = self.post_norm(q)`

**附帶**: `nn.MultiheadAttention` 也加上 `dropout=0.1`,跟 PQD 的 `MultiheadAttention(... dropout=0.1)` 一致。

---

## 3. 程式碼差異總覽

| 位置 | 修改 |
|---|---|
| `__init__` 參數 | 新增 `proto_dim=32`、`prototype_mining_thresh=0.7` |
| `__init__` 屬性 | 保存 `self.proto_dim`、`self.prototype_mining_thresh` |
| `__init__` 模組 | 新增 `self.proto_bottleneck = Conv2d(hidden, proto_dim, 1)` |
| `__init__` EMA bank | `Embedding(K, hidden)` → `Embedding(K, proto_dim)` |
| `__init__` aggs | `Linear(hidden, hidden)` → `Linear(proto_dim, hidden)` (兩個 agg 的第一層) |
| `__init__` self-attn | 加 `dropout=0.1` |
| `__init__` post_norm | 新增 `self.post_norm = LayerNorm(hidden)` |
| `_cnn_proto_generator` | 多回傳 `proto_feat = proto_bottleneck(mask_feat)` |
| `_adaPG` | 改吃 `proto_feat`,池化在 proto_dim;閾值改用 `prototype_mining_thresh` |
| `_gt_guided_adaPG` | 改吃 `proto_feat`,池化在 proto_dim |
| `_scene_aware_queries` | `query_feat` 零張量 shape 改成 `(K, hidden_channels)` (因為 input 維度從 hidden 變成 proto_dim) |
| `_forward_head` | self-attn 後加 `q = self.post_norm(q)` |
| `forward` | 解三個值 `(mask_feat, proto_feat, coarse_pred)`,把 `proto_feat` 傳進 AdaPG |

---

## 4. 受影響的設定 / 相容性

### 4.1 設定檔不需要改

`ProtoOcc_proto_map_head_map_neck.py`、`ProtoOcc_proto_map_head_map_neck_256.py` 等檔案**不用動**。新加的兩個參數都有 default:
- `proto_dim=32`
- `prototype_mining_thresh=0.7`

如果想 ablation 不同 `proto_dim` (例如 64),直接在 config 的 `proto_map_head=dict(...)` 裡加一行 `proto_dim=64,` 即可。

### 4.2 舊 checkpoint **不能直接 load**

下列權重 shape 改變(以 hidden=128 的 `*_map_neck.py` 為例;不同 config hidden 不同,推廣規則 = 表中 128 都換成該 config 的 hidden):
- `*_map_neck.py` hidden=128 → 表中數字
- `*_map_neck_256.py` hidden=256 → (6,256)/(256,256)
- canonical `ProtoOcc_proto_map_head.py` hidden=96 → (6,96)/(96,96)

| 權重 | 舊 shape (hidden=128) | 新 shape |
|---|---|---|
| `prototype_EMA_feat.weight` | (6, 128) | (6, 32) |
| `global_protoEMA_agg.0.weight` | (128, 128) | (128, 32) |
| `global_protoEMA_agg.0.bias` | (128,) | (128,) |
| `local_protoEMA_agg.0.weight` | (128, 128) | (128, 32) |
| `local_protoEMA_agg.0.bias` | (128,) | (128,) |

新增權重(舊 ckpt 沒有,要從頭 train):

- `proto_bottleneck.weight`、`proto_bottleneck.bias`
- `post_norm.weight`、`post_norm.bias`
- `self_attn` 的 dropout 不影響權重結構

**結論**: 舊 work_dirs 的 checkpoint 都需要重新訓練。這預期之內,因為原本 prototype refinement 就是負貢獻,目標就是訓出新版的可用權重。

### 4.3 PGBR / GT-soft 路徑

PGBR (`PrototypeGroundedBEVRefiner`) 跟 `_gt_guided_adaPG` 都還在,但這次修改:
- PGBR 完全沒動,介面不變
- `_gt_guided_adaPG` 同步改成池化 in proto_dim,跟 `_adaPG` 一致

使用者目前先不打開 PGBR / GT-soft (見 [experiment_results_summary.md](experiment_results_summary.md) 結論 9, 10)。

---

## 5. 預期效益與下一步驗證

### 5.1 預期效益

| 修正 | 對 final map mIoU 的預期影響 |
|---|---|
| P0 #1 (dim bottleneck) | +0.5 ~ +1.0,讓 final 至少不輸 coarse |
| P0 #2 (信心過濾) | +0.5 ~ +1.0,線狀類別 ped_crossing/stop_line/divider 預期 +1~+2 |
| P1 #4 (post_norm + dropout) | +0.1 ~ +0.3 (穩定性) |
| **合計** | **39.02 → 40.5+,有機會贏過 CNN head 39.94** |

### 5.2 下一步訓練

跑一次 `ProtoOcc_proto_map_head_map_neck.py` 完整 24 epoch,跟舊版同 config 比:

| 指標 | 舊 ProtoMapHead EMA | 新 ProtoMapHead EMA (預期) | 說明 |
|---|---:|---:|---|
| Occ mIoU | 39.71 | ≥ 39.5 | 不能傷到 occ |
| Map mIoU final | 39.02 | **≥ 40.5** | 主要驗證點 |
| Map mIoU coarse | 39.44 | 持平 | coarse 跟之前差不多就好 |
| `final - coarse` | −0.35 | **≥ +0.3** | refinement 從負貢獻變正貢獻才算修對 |

若 `final - coarse ≥ 0` 達成 → P0 #1 修對 (主嫌排除)。
若 ped_crossing / stop_line / divider 三類各 +1 以上 → P0 #2 修對 (副嫌排除)。

### 5.3 後續 ablation

修對之後再做這些 ablation 撐 paper:
- `proto_dim` ∈ {16, 32, 64, 128} → 確認 32 是甜蜜點
- `prototype_mining_thresh` ∈ {0.5, 0.6, 0.7, 0.8} → 看線狀類別曲線
- 開回 PGBR、GT-soft 看是否還壞 (現在 prototype 空間乾淨後,可能不再壞)

---

## 6. 對「ProtoOcc 框架下的 MTL」故事的意義

修這 4 個地方不只是 work-around,而是**忠實將 ProtoOcc 的 prototype 機制泛化到 BEV map 任務**。論文裡可以寫成:

> "We faithfully extend ProtoOcc's prototype-based decoder to BEV map segmentation, including (1) the **dim-bottlenecked prototype space** that decouples prototype gradients from the dot-product feature space, (2) **uncertainty-aware prototype mining** that excludes ambiguous pixels from the prototype pool, and (3) **transformer-based query refinement with post-LayerNorm**. To adapt to map's per-class sigmoid setting (vs occupancy's mutually-exclusive softmax), we replace PQD's top-2 margin filter with a per-class confidence threshold that approximates the same design intent (high-confidence pixels only)."

這就是論文「貢獻 C3」(忠實移植 ProtoOcc 到 map) 的具體段落,**作為 prototype-level cross-task MTL bridge 的合格起點**。

---

## 7. Codex review 後續補強(2026-05-01)

把 P0/P1 alignment 完成後拿給 codex review,他指出四點。逐一處理結果如下。

### 7.1 P0 #5 — Random EMA 污染(真實 bug,已修)

**問題**: `_adaPG` 對 sparse 類在沒 pixel 過 0.7 threshold 時回傳 `valid=False` zero proto;`_agno_PG_update_and_get` 對應地不更新該 class 的 EMA(正確),**但仍然 return 整張 `prototype_EMA_feat.weight`**,包含尚未初始化的 random embedding row。下游 `_scene_aware_queries` 不分 class 直接 `global_protoEMA_agg(ema_query)`,把 random EMA 灌進該 class 的 query。

舊 0.5 threshold 時 sparse 類幾乎都會湊到 valid pixel,bug 沒觸發;改 0.7 後早期訓練 sparse 類(`ped_crossing` / `stop_line` / `divider`)很可能整個 batch 都過不了門檻,bug 變常態。PQD 在 occ 上沒踩到是因為 softmax 互斥 + 類別像素量大,sparse 類的 EMA 早期就會 init 好。

**修法**: `_scene_aware_queries(for_query, ema_query, valid_mask)` 多吃一個 `valid_mask`:
- EMA 路徑用 `~self.proto_first_flag` 做 per-class mask,EMA 還沒被任何 iter 初始化的 row 不貢獻
- Local 路徑用 `valid_mask` 做 per-class mask,本 iter 沒 confident pixel 的 class 不貢獻 (`local_protoEMA_agg(0)` 也只是 bias term,放進去沒意義)
- 兩條路都被擋掉時,該 class 自動 fallback 到 `learnable_query`,不會被 noise 污染

PQD 沒做這層 mask,因為它的 setting 不會踩到。這是 map-specific safeguard,程式碼註解有寫。

### 7.2 P1 #6 — `conf_thresh` 死參數(已移除)

舊 P0 #2 換成 `prototype_mining_thresh` 之後,`conf_thresh` 變成沒人讀的 attribute,但 `__init__` 還在收、config 還在傳。已從 `proto_map_head.py` 跟 `ProtoOcc_proto_map_head.py` 都移除,避免之後 ablation 看到兩個門檻參數誤會。

### 7.3 文件 — Shape 表只對 hidden=128 正確(已補註)

第 4.2 節原本只列 `*_map_neck.py` 那個 config 的 shape (hidden=128)。已加上各 config 的 hidden 對照:`map_neck` 128 / `map_neck_256` 256 / canonical 96,推廣規則寫清楚。

### 7.4 文件 — 0.7 threshold「等效 PQD top-2」說法太強(已軟化)

PQD top-2 margin 是 softmax 上的 multi-class disambiguation,sigmoid 絕對值 threshold 沒辦法做這件事(map 多類同時高分本來合法)。第 2 節 P0 #2 描述跟第 6 節論文段落都已改成「approximates the same design intent」/「近似移植」,不再說「等效」。

### 7.5 順帶 — PQD 變數命名對齊

P0 修完後,ProtoMapHead 內 `mask_feat` (hidden 96, dot product) 跟 `proto_feat` (proto_dim 32, prototype pool) 跟 PQD 的 `voxel_feats` (48, dot product) / `mask_feat` (32, prototype pool) 邏輯角色一致但命名對不上。為了之後讀 paper / source 對照不混亂,程式碼內 local 變數 rename:

| 角色 | PQD 命名 | ProtoMapHead 舊名 | ProtoMapHead 新名 |
|---|---|---|---|
| 高維 dot-product 特徵 | `voxel_feats` (input) / `mask_feature` (`forward_head` arg) | `mask_feat` | `mask_feature` |
| 低維 prototype-pool 特徵 | `mask_feat` (input) | `proto_feat` | `mask_feat` |

這只動 local variable / function 參數名稱,**沒動 module attribute**(`self.proto_bottleneck`、`self.prototype_EMA_feat`、`self.global_protoEMA_agg` 等都不變),所以**不影響 checkpoint 載入**。

### 7.6 訓練前驗證建議

不要直接跑 24 epoch。建議:
1. Smoke test 1~2 epoch,iter 印 log 看每 class 過 `prototype_mining_thresh=0.7` 的 pixel 數,跟 `proto_first_flag` 何時 flip 成 False
2. 如果 sparse 類前幾百 iter 大量 invalid → 可考慮 threshold ramp-up (0.5 → 0.7) 或前 N iter 走 `gt_hard` 預熱 EMA
3. 修完上面幾項後再跑完整 24 epoch 對照 baseline

#### Smoke test 實作方式

`ProtoMapHead` 內已加 debug 開關,預設關閉。跑 smoke test 時用 CLI 覆寫打開:

```bash
cd Ryan/ProtoOcc

PYTHONPATH=. python tools/train.py projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck.py \
  --work-dir work_dirs/proto_map_head_smoke_debug \
  --cfg-options \
    runner.max_epochs=1 \
    log_config.interval=20 \
    model.proto_map_head.prototype_debug=True \
    model.proto_map_head.prototype_debug_interval=20
```

看 256ch map neck 就把 config 換成:

```bash
projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_256.py
```

debug log 會寫進 `work_dirs/.../*.log`。`*.log.json` 只記 scalar metrics,不會有這種文字訊息。log 會長這樣:

```text
[ProtoMapHead debug] iter=20 mode=pred_threshold thresh=0.70 support_pixels=drivable_area=..., ped_crossing=..., walkway=..., stop_line=..., carpark_area=..., divider=... valid=drivable_area=1, ped_crossing=0, ... proto_first_flag_false=drivable_area,walkway flipped_this_iter=walkway
```

判讀重點:
- `support_pixels`:該 iter/batch 每個 class 的「prototype 池貢獻 pixel 數」。語意隨 `mode` 切換:
  - `pred_threshold`(預設):coarse sigmoid 超過 `prototype_mining_thresh` 的 pixel 數
  - `gt_hard` / `gt_soft`(GT 預熱模式):GT-positive pixel 數(`gt_mask.sum()`)
  log header 已印 `mode=...`,對照解讀即可
- `valid`:該 class 這次有沒有 local prototype 可用
- `flipped_this_iter`:該 iter 哪些 class 的 EMA row 第一次被初始化,也就是 `proto_first_flag` 從 True 變 False
- `proto_first_flag_false`:目前已經初始化過 EMA 的 class

如果 `ped_crossing` / `stop_line` / `divider` 幾百 iter 都是 `support_pixels=0`、`valid=0`、也沒有出現在 `proto_first_flag_false`,代表 0.7 對 early training 太硬,再考慮 threshold ramp-up 或 `gt_hard` 預熱。
