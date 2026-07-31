# Prototype-Grounded BEV Refinement

> 2026-04-23 update：PGBR 目前只保留為 ablation 用 optional submodule。canonical `ProtoMapHead` 不傳 `pgbr_cfg`，因此不會建立 PGBR。

## 這次新增了什麼

在 [ProtoMapHead](/home/robot/Desktop/Ryan/ProtoOcc/projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py) 裡新增了一個 **Prototype-Grounded BEV Refinement（PGBR）** stage，位置放在：

```text
bev_feature
  -> CNN proto generator
  -> coarse_pred + mask_feat
  -> AdaPG / AgnoPG / scene-aware queries
  -> Prototype-Grounded BEV Refinement
  -> refined_mask_feat
  -> self-attn + mask_embed x refined_mask_feat
  -> final_masks
```

它不是另外開一個新 detector，也不是把 `ProtoMapHead` 換掉，而是把 prototype 的作用從「只在 query 形成與 final decoding 出現」往前延伸到 **BEV feature grounding**。

目前 PGBR 已被降級成 optional submodule。canonical config 不傳任何 PGBR 相關欄位；只有需要 ablation 時才提供：

```python
pgbr_cfg=dict(
    temperature=1.0,
    detach_query=True)
```

---

## 為什麼要做這個模組

目前的 `ProtoMapHead` 已經做到一件重要的事：  
**讓 prototype 直接參與 final map decoding。**

但還有一個缺口：

- `final_masks` 的確是 prototype-guided decoder 輸出的
- 可是在 decoder 之前，`mask_feat` 本身仍然只是 shared BEV feature 經過淺層 CNN 轉出來的 feature
- 它還沒有被 prototype 顯式地做過 map-oriented grounding

這會帶來一個很實際的問題：

- shared / multitask 的 `bev_feature` 同時服務 occ 與 map
- 即使 decoder 很強，輸入給 decoder 的 feature 仍可能含有很多對 map 無關或模糊的空間訊號

所以 PGBR 的目的不是取代 `ProtoMapHead`，而是先回答這個問題：

> 在 final prototype decoding 之前，我們能不能先用 scene-aware prototypes 去重新標定哪些 BEV 區域更值得被強化、哪些區域應該被壓抑？

---

## 模組怎麼做

### 1. 以 scene-aware query 作為 grounding source

PGBR 直接重用 `ProtoMapHead` 已經算好的 `query_feat`：

```text
query_feat = learnable_query
           + global_protoEMA_agg(ema_query)
           + local_protoEMA_agg(for_query)
```

這代表 refinement 用的 prototype 不是額外再算一套，而是直接使用目前 decoder 真正要拿去解碼的那組 **class-wise scene-aware queries**。

PGBR ablation config 預設使用：

- `detach_query=True`

也就是先用 `query_feat.detach()` 做 refinement，避免 refinement branch 在 early stage 反向把 query 本身拉壞。這是穩定性優先的選擇。

### 2. 算 class-wise grounding logits

對 `mask_feat` 與 `query_feat` 做 channel-wise normalization 後，計算每個 class prototype 對每個 BEV cell 的相似度：

```text
ground_logits[b, k, h, w]
  = < normalize(query_k), normalize(mask_feat[b, :, h, w]) > / temperature
```

得到的 `ground_logits` 有兩種用途：

- `softmax(ground_logits, dim=class)`：決定一個位置應該混入哪些 class prototype
- `sigmoid(ground_logits)`：保留這個位置與 prototype 的絕對相似強度

這裡刻意沒有只做單一 normalize。原因是：

- 單純的 class-normalized weighting 會讓每個位置都被硬分配到某種 prototype mixture
- 即使那個位置其實對所有 class 都不太像，也仍然會被增強

因此正式版採用：

- `softmax` 做 **class mixture**
- `sigmoid` 做 **absolute confidence gating**

### 3. Enhancement branch

先把 `query_feat` 投影成可回寫到 feature map 的 `proto_value`，再依照 class-wise grounding weights 做加權合成：

```text
enhance = sum_k ground_weight[:, k] * proto_value[k]
enhance = enhance * max(sigmoid_ground_map)
```

這個設計的含義是：

- 哪個 class prototype 該主導某個位置，由 `softmax` 決定
- 這個位置到底值不值得被 prototype 強化，由 `sigmoid` 最大值決定

換句話說，PGBR 不只是「把 prototype 平均灌回 feature」，而是有意識地區分：

- **哪一類 prototype 該出手**
- **這個位置值不值得被出手**

### 4. Suppression branch

如果某個位置對所有 prototype 都不太像，那它更可能是：

- 與 map 任務關聯較弱的區域
- 或是目前不夠確定、容易污染 decoder 的區域

因此 suppression branch 會用：

```text
suppress_gate = 1 - max(sigmoid_ground_map)
```

去強化「prototype 不支持」的位置的抑制效果，再配合一個學習式的 `suppress_proj`：

```text
suppress = suppress_proj(mask_feat) * suppress_gate
```

這樣 suppression 不是硬切掉 feature，而是讓網路自己學會在低-grounding 區域要壓什麼訊號。

### 5. Residual fuse

最後把三路特徵拼起來：

- 原始 `mask_feat`
- `enhance`
- `suppress`

再用一個小型 CNN fuse 成 residual update：

```text
delta = refine_fuse([mask_feat, enhance, suppress])
refined_mask_feat = mask_feat + delta
```

這裡採 residual 形式，是因為：

- 如果 refinement 真有幫助，它可以在原 feature 上做有方向的修正
- 如果 refinement 在某些位置還不成熟，殘差寫法比較不容易破壞原本已經可用的 `mask_feat`

---

## 這個模組在挑戰什麼

我會把 PGBR 挑戰的 prior assumption 寫成下面兩句。

### 挑戰 1：shared BEV feature 可以直接丟給 map decoder

很多 multi-task 架構會默認：

> 只要 shared encoder 足夠強，map head 直接吃 shared / BEV feature 就夠了。

PGBR 挑戰的是這個假設。  
我們的立場是：

- shared BEV feature 不一定天然對 map 夠友善
- 尤其在 ProtoOcc 這種 backbone/feature 主體原本偏向 occ 的設定下，更可能需要 map-oriented grounding

所以我們主張：

> 在 final map decoding 之前，應該先做 prototype-guided spatial grounding。

### 挑戰 2：prototype 只要出現在單一階段就夠了

前人的 prototype 使用方式通常只覆蓋局部階段：

- **ProtoOcc**：prototype decoding 很強，但主要集中在 occupancy decoder
- **MAESTRO**：prototype 會進 feature shaping，但 map final mask 仍由 BEVFusion-style dense CNN head 產生

PGBR 加上 `ProtoMapHead` 的組合，挑戰的是另一個更強的觀點：

> prototype 不應只停留在 feature prior，或只停留在 final decoder；它應該貫穿整個 map branch，既參與 feature grounding，也參與 final decoding。

這也是為什麼我更建議把整體方法寫成：

- **prototype-centric map branch**

而不是把 refinement 和 decoder 視為兩個互不相干的小技巧。

---

## 和 MAESTRO、ProtoOcc 的關係

### 相對 ProtoOcc

ProtoOcc 的啟發在於：

- prototype query formation
- scene-adaptive prototype
- EMA global memory
- final prototype-based decoding

但 ProtoOcc 的 prototype decoder 主要是為 **occupancy** 設計。  
PGBR 的作用，是把 prototype 的影響更早地帶到 **2D BEV map branch** 裡。

### 相對 MAESTRO

MAESTRO 告訴我們一件重要的事：

- prototype-guided feature shaping 對 multi-task perception 很重要

但 MAESTRO 的 map 分支最終仍是：

- task-specific BEV feature
- downstream BEVFusion-style CNN map head

所以我們不是在複製 MAESTRO 的 map decoder，而是在延續它對 task grounding 的啟發，並往前走一步：

- MAESTRO：prototype 主要做 feature shaping
- 我們：prototype 同時做 **feature grounding + final decoding**

---

## 目前這版的設計取捨

### 1. 先不加 auxiliary loss

第一版沒有額外加 `ground_map` 的 supervision。原因很簡單：

- 先確認 refinement 本身能不能靠原本的 map losses 帶動
- 避免一開始就多一個 loss，讓 ablation 難拆解

目前仍然只保留：

- `loss_map_coarse_bce`
- `loss_map_coarse_dice`
- `loss_map_focal`
- `loss_map_dice`

### 2. 用 `softmax + sigmoid`，不是單一 normalize

這是正式版 forward 比較重要的設計點。

- `softmax`：負責 class-wise mixture
- `sigmoid`：保留絕對相似度，避免低相關位置被硬增強

### 3. 先 detach query

PGBR ablation config 預設 `detach_query=True`，是為了讓：

- `query_feat` 先穩定服務 decoder
- refinement branch 先學會「怎麼利用 query」
- 而不是一開始就和 query formation 互相拉扯

如果後面訓練穩定，可以再 ablate：

- `detach_query=False`

---

## 本次實作影響到的檔案

- [proto_map_head.py](/home/robot/Desktop/Ryan/ProtoOcc/projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py)
- [ProtoOcc_proto_map_head.py](/home/robot/Desktop/Ryan/ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py)

`ProtoOccMultitask` 的 public API 沒有改：

- train 仍然回傳 `coarse_pred, final_masks`
- test 仍然使用 `predict(final_masks)` 輸出 map probability

因此 PGBR 是一個 **內聚在 `ProtoMapHead` 內部的 optional refinement submodule**，不需要額外改 detector 的訓練與推論接口。canonical `ProtoMapHead` 不提供 `pgbr_cfg`，所以不會建立這個 submodule。
