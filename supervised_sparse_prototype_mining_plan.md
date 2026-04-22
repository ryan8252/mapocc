# Supervised Sparse Prototype Mining 改動計劃

日期：2026-04-22

## 一句話定位

這份計劃是下一階段的 C2 貢獻：

> Sparse semantic classes often fail to provide reliable prototype support under prediction-driven prototype mining. We propose GT-guided soft prototype mining to stabilize class-wise prototypes for both occupancy and BEV map semantics during training.

簡單說：

```text
PGBR / TSFG 解決「prototype 怎麼用」。
Supervised Sparse Prototype Mining 解決「prototype 怎麼可靠地來」。
```

這不是要取代 PGBR，也不是要把 GT 帶進 inference。GT mask 只在 training 階段幫助 prototype extraction，避免小物件或細線類別在 early training 時抽不到 prototype 或污染 EMA bank。

## Related Work Positioning

這個方法必須先承認它的 lineage：training-time GT mask / label mask 來做 class-wise feature aggregation 不是全新的概念。

相關方向包含：

| 方向 | 相似處 | 本計劃的差異 |
| --- | --- | --- |
| OCR / OCNet 類 context aggregation | 使用 supervised class region representation 聚合 pixel feature | OCR 主要用於同一張影像內的 context representation；本計劃放在 scene-adaptive prototype + EMA bank pipeline，目標是穩定跨 iteration 的 prototype memory |
| Few-shot segmentation prototype | 用 support mask 抽 class prototype | few-shot 的 support set 是外部 support image；本計劃使用 same-scene training GT 作為 prototype mining support，inference 不提供 GT |
| Supervised / pixel contrastive learning | 用 GT 定義 positive set，提升 class-wise feature discriminability | contrastive learning 通常作為 auxiliary representation loss；本計劃直接替換 AdaPG / CPG-style prototype mining 的 support 來源 |
| MAESTRO CPG | 都與 class-wise prototype generation 有關 | MAESTRO 著重 prototype 如何做 task-specific feature generation；本計劃只改 prototype 來源可靠性，不把自己包裝成新的 TSFG |
| ProtoOcc AdaPG | 都產生 scene-adaptive class prototypes 並更新 EMA bank | ProtoOcc 使用 prediction-driven masks；本計劃使用 training-time GT gate + prediction confidence reweighting 解決 sparse class cold-start |

因此 C2 的 novelty 不應寫成：

```text
we use GT masks to pool prototypes
```

而應寫成：

```text
we address sparse-class cold-start in scene-adaptive prototype and EMA-bank pipelines by replacing hard prediction-driven support with supervised soft support during training.
```

## 放在整體三個貢獻中的位置

目前建議把論文故事整理成三個層次：

| 貢獻 | 挑戰的破綻 | 對應方法 |
| --- | --- | --- |
| C1. Task-specific map feature generation | ProtoOcc 原本的 48-channel occupancy-oriented BEV bottleneck 不適合 map segmentation，naive hard sharing 會造成 map collapse | 從 shared multi-scale BEV feature 分出 map-specific BEV neck |
| C2. Supervised sparse prototype mining | MAESTRO CPG / ProtoOcc AdaPG 依賴 prediction-driven prototype mining，對 sparse / thin / small-area semantics 不穩 | training 時用 GT-guided soft mask 保證 prototype support，prediction confidence 只當 soft weighting |
| C3. Unified cross-task semantic prototype bank | 現有方法缺少 occ / map 之間顯式對齊的 shared semantic prototype space | 用 shared meta-prototypes 或 soft correspondence matrix 建立 occ-map prototype interaction |

本文件只處理 C2。

## 要挑戰什麼

### 1. MAESTRO 的 CPG

MAESTRO 的 Class-wise Prototype Generator (CPG) 會從 shared voxel feature 產生 foreground/background prototype groups，後續在 TSFG 中用來做 task-specific feature enhancement / suppression。

它的強項是：

```text
prototype-guided task-specific feature generation
```

但它沒有明確處理：

```text
sparse map classes 的 prototype support 是否穩定
```

尤其對 BEV map segmentation 的小面積與線狀類別：

```text
ped_crossing
stop_line
divider
carpark_area
```

這些類別在單一 scene 中可能面積很小，甚至不存在。如果 prototype generation 過度依賴模型早期預測或 feature grouping，容易被 `drivable_area`、`walkway` 這類 dominant region 淹掉。

因此我們不是說 MAESTRO 的 TSFG 錯，而是指出：

```text
MAESTRO focuses on how prototypes guide task-specific features,
but does not explicitly address whether sparse semantic classes can reliably form prototypes.
```

### 2. ProtoOcc 的 AdaPG

ProtoOcc 的 Scene-Adaptive Prototype Generator (AdaPG) 目前使用 prediction-driven mask 來抽 prototype。

在 occupancy decoder 中，目前程式位於：

```text
projects/mmdet3d_plugin/models/OccHead/Prototype_Query_Decoder_nuScenes.py
```

核心邏輯是：

```python
mask_target = occ_pred.clone().detach().permute(0, 4, 1, 2, 3)
mask_ = F.softmax(mask_target, dim=1)
top2_values, top2_indices = torch.topk(mask_, 2, dim=1)
difference = top2_values[:, 0] - top2_values[:, 1]
scores = 1.0 - difference
vis_mask = scores >= 0.9
mask = mask_.argmax(dim=1)

for i in range(self.num_classes):
    mask_cls = mask == i
    mask_cls *= ~vis_mask
    query_ = mask_feat[mask_cls, :].mean(dim=0)
```

也就是：

```text
先讓 shallow CNN / coarse occ classifier 猜每個 voxel 是什麼類別，
再排除 top-2 margin 太小的 ambiguous voxels，
再用猜出來的 class mask 去平均 feature，得到 class prototype。
```

所以不能說 ProtoOcc 完全沒有處理 low-confidence voxels。更精準的弱點是：

```text
margin filter 可以排除模糊 voxel，
但無法保證 sparse class 在 early epoch 有足夠 positive support。
```

這對大類通常可行，但對小物件仍有風險：

```text
bicycle
motorcycle
pedestrian
traffic_cone
```

如果 early epoch 的 coarse classifier 沒把這些類別猜出來：

```text
該 class 的 predicted mask 可能是空的
=> scene-adaptive prototype = zero
=> EMA bank 不更新
=> decoder query 品質變差
```

在 map branch 中，目前 `ProtoMapHead._adaPG()` 也是 prediction-driven：

```text
projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py
```

核心邏輯是：

```python
soft_masks = torch.sigmoid(coarse_pred)
conf_mask = soft_masks[:, k] > self.conf_thresh
proto = mask_feat_perm[conf_mask].mean(0)
```

也就是：

```text
sigmoid(coarse_pred) > 0.5 才能參與 prototype pooling
```

但目前 map-neck no-PGBR 的 early evaluation 中，多數 map class 的 best evaluation threshold 落在 `0.35`。這是實驗觀察，不是 code 設定；它表示 early / mid training 的 map confidence 仍偏低。這會導致：

```text
GT 裡有 stop_line
但 coarse_pred(stop_line) 沒有任何 pixel > 0.5
=> stop_line prototype = zero
=> EMA bank 不更新
```

## 目標

### 方法目標

1. 在 training 階段，用 GT mask 保證 sparse class 能抽到 prototype。
2. 保留 shallow CNN / coarse predictor 的角色，讓 prediction confidence 只負責 soft weighting，而不是 hard selection。
3. 當某個 scene 不存在某類別時，不用 zero prototype 覆蓋 EMA bank，而是跳過更新並沿用 scene-agnostic prototype。
4. 同一套概念同時支援 3D occupancy prototype 和 2D BEV map prototype。
5. 為下一階段 unified occ-map semantic prototype bank 打基礎。

### 實驗目標

Occupancy 希望改善小物件：

```text
bicycle
motorcycle
pedestrian
traffic_cone
construction_vehicle
```

Map 希望改善小面積與線狀類別：

```text
ped_crossing
stop_line
divider
carpark_area
```

整體目標不是只追 mIoU，而是要證明：

```text
GT-guided soft mining 能提升 sparse class 的 prototype reliability。
```

## 核心概念

Prototype 是「從某一類的 feature 平均出來的代表向量」。

例如 map 中的 pedestrian crossing prototype：

```text
prototype_ped_crossing = 所有 ped_crossing BEV cell 的 feature 平均
```

或 occupancy 中的 bicycle prototype：

```text
prototype_bicycle = 所有 bicycle voxel 的 feature 平均
```

目前 prediction-driven mining 的問題是：

```text
模型 early epoch 猜不出小類別
=> 小類別沒有 positive support
=> 抽不到 prototype
```

Supervised Sparse Prototype Mining 的核心是：

```text
GT mask 決定哪些位置可以參與 prototype pooling。
prediction confidence 決定這些 GT-positive 位置的權重大小。
```

通用公式：

```python
weight = gt_mask * (alpha + (1 - alpha) * pred_prob.detach())
```

白話解釋：

```text
GT 決定誰可以參加。
pred_prob 決定誰講話比較大聲。
alpha 決定 GT 保底權重有多強。
```

建議第一版使用：

```python
alpha = 0.5
```

如果某個位置是該類 GT positive：

```text
gt_mask = 1
```

若模型 confidence 高：

```text
pred_prob = 0.8
alpha = 0.5
weight = 1 * (0.5 + 0.5 * 0.8) = 0.9
```

若模型 confidence 低：

```text
pred_prob = 0.2
alpha = 0.5
weight = 1 * (0.5 + 0.5 * 0.2) = 0.6
```

兩者都會參與 prototype pooling，但高 confidence 位置權重更大。

如果某個位置不是該類 GT positive：

```text
gt_mask = 0
weight = 0
```

它完全不參與該類 prototype。

## 為什麼不用純 GT hard mining

純 GT hard mining 是：

```python
weight = gt_mask
```

這可以當 ablation，但不建議直接作為最終主方法，因為它會讓 prototype extraction 完全不依賴 shallow CNN / coarse predictor，造成 train-test mismatch 變大。

推薦的 GT-soft mining：

```python
weight = gt_mask * (alpha + (1 - alpha) * pred_prob.detach())
```

好處是：

1. GT 提供保底，避免小類別 support 消失。
2. prediction confidence 仍然影響 prototype quality。
3. `detach()` 避免模型透過操控 pooling weight 走捷徑。
4. shallow CNN 仍然透過 coarse segmentation loss 學習，不會被拿掉。

### alpha sweep

`alpha=0.5` 只是第一版 default，不應寫成固定常數。建議做 sweep：

| alpha | 意義 |
| ---: | --- |
| 0.0 | GT gate 內完全由 prediction confidence 決定權重 |
| 0.3 | prediction 權重較強 |
| 0.5 | GT 保底與 prediction reweighting 折中 |
| 0.7 | GT 保底較強 |
| 1.0 | GT-hard mining，所有 GT positive 權重相同 |

這個 ablation 可以直接回答：

```text
sparse classes 需要多少 GT 保底？
prediction confidence 在 prototype mining 裡到底有沒有幫助？
```

若要再進一步，可以測 curriculum：

```python
weight = gt_mask * (alpha_t + (1 - alpha_t) * pred_prob.detach())
```

其中 `alpha_t` 在 early epoch 較高，late epoch 較低，例如：

```text
epoch 1-6:   alpha_t = 1.0
epoch 7-12:  alpha_t = 0.7
epoch 13+:   alpha_t = 0.5
```

這個 schedule 的直覺是：

```text
early epoch prediction 還不可靠，多靠 GT；
late epoch prediction 較可靠，讓 prediction confidence 多一點影響力。
```

curriculum 建議放在第二階段 ablation，不要第一版就打開。

## Training 和 Inference 的差別

### Training

training 時有 GT：

```text
occupancy: voxel_semantics / gt_occ
map: gt_masks_bev
```

所以可以用 GT mask 幫助 prototype mining。

流程：

```text
feature
  + coarse prediction
  + GT mask
  -> GT-guided soft prototype mining
  -> local scene prototype
  -> update EMA bank if class exists
  -> scene-aware query / refinement / decoder
```

### Pooling scope

現有 ProtoOcc occupancy AdaPG 和目前的 `ProtoMapHead._adaPG()` 都是 local-batch pooling：

```text
同一張 GPU 上的 B 個 samples 被一起 pooling 成一組 class prototypes。
```

這個行為嚴格來說是：

```text
local-batch-adaptive prototype
```

而不是每個 scene 都有完全獨立 prototype。第一版建議保留這個行為，原因是：

1. 和 released ProtoOcc / 目前 ProtoMapHead 行為一致。
2. 改動最小，方便比較。
3. DDP 下每張 GPU 的 local batch 不做 cross-GPU prototype sync，這也和目前程式一致。

但論文敘事若要強調 `scene-adaptive`，更乾淨的版本是 per-sample pooling：

```text
每個 sample 各自抽 prototype，再各自更新 / 聚合 EMA。
```

因此建議新增 config：

```python
prototype_pooling_scope='batch_local'  # first version, matches current code
prototype_pooling_scope='sample'       # stricter scene-adaptive ablation
```

第一版實作使用 `batch_local`；若 reviewer 或實驗需要，再測 `sample`。

### Inference

inference 時沒有 GT，所以不能用 GT mask。

此時有兩種 prototype 來源：

1. **EMA bank**：training 階段累積下來的 scene-agnostic prototype。
2. **prediction-based local prototype**：用 shallow CNN / coarse predictor 的 prediction threshold 或 top-k 來抽 local prototype。

推薦第一版 inference：

```text
以 EMA bank 作為主要穩定來源，
prediction-based local prototype 作為 scene-specific 補充。
```

也就是：

```text
training: 用 GT 保證 prototype 抽得準
inference: 用 EMA bank 和模型自己的 prediction
```

這不算作弊，因為 GT 只存在於 training-time prototype support，不進入測試階段。

train-test mismatch 的關鍵緩解點是：

```text
inference 主要依賴 training 累積出的 EMA bank，
不是完全依賴 test-time prediction 重新 mine 出乾淨 prototype。
```

eval mode 下 EMA bank 必須 frozen：

```text
model.eval() 時不更新 prototype_EMA_feat
proto_first_flag / EMA state 不應被 test samples 改寫
```

## 當 scene 中真的沒有某個類別怎麼辦

如果某個 scene 沒有 pedestrian：

```text
gt_mask_pedestrian.sum() == 0
```

此時不應該：

```text
用 zero prototype 覆蓋 pedestrian EMA
```

正確做法：

```text
local prototype = None / zero placeholder
EMA 不更新
query 主要使用 learnable query + EMA global prototype
```

也就是：

```python
eps = 1e-6
if support < eps:
    proto = zero_placeholder
    valid = False
else:
    proto = weighted_average(...)
    valid = True
```

EMA update：

```python
if valid:
    update_ema(class_id, proto)
else:
    skip_update(class_id)
```

這正是 AgnoPG / Scene-Agnostic Prototype Generator 的用途：

```text
scene 有該類：用 local prototype 更新全局記憶
scene 沒該類：沿用過去累積的全局 prototype
```

## Map 版本設計

### 目前問題

目前 `ProtoMapHead._adaPG()` 使用：

```python
soft_masks = torch.sigmoid(coarse_pred)
conf_mask = soft_masks[:, k] > self.conf_thresh
```

這對 map 的 sparse classes 不穩。

### 改動方向

新增一個 GT-guided soft mining 分支：

```python
def _gt_guided_adaPG(self, coarse_pred, mask_feat, gt_masks_bev):
    prob = torch.sigmoid(coarse_pred).detach()
    feat = mask_feat.permute(0, 2, 3, 1)
    gt = gt_masks_bev.float()
    eps = 1e-6
    alpha = self.prototype_mining_alpha

    for k in range(self.num_classes):
        weight = gt[:, k] * (alpha + (1 - alpha) * prob[:, k])
        support = weight.sum()

        if support < eps:
            proto = zero
            valid = False
        else:
            proto = (feat * weight[..., None]).sum(dim=(0, 1, 2)) / support
            valid = True
```

`ProtoMapHead.forward()` 改為支援 optional GT：

```python
def forward(self, bev_feat, gt_masks_bev=None):
    mask_feat, coarse_pred = self._cnn_proto_generator(bev_feat)

    if self.training and gt_masks_bev is not None and self.prototype_mining_mode == 'gt_soft':
        for_query, valid_mask = self._gt_guided_adaPG(coarse_pred, mask_feat, gt_masks_bev)
    else:
        for_query, valid_mask = self._adaPG(coarse_pred, mask_feat)

    ema_query = self._agno_PG_update_and_get(for_query, valid_mask)
```

Detector 呼叫處：

```python
coarse_pred, final_masks = self.proto_map_head(
    map_feature,
    gt_masks_bev=gt_masks_bev)
```

### Map sparse class support expansion 選項

對 `stop_line` / `divider` / `ped_crossing` 這類細線，可以測一個 optional support expansion：

```text
只在 training prototype pooling 使用擴張後的 support mask。
segmentation loss 仍然使用原始 GT mask。
```

但 binary dilation 會改變 prototype 語意，例如 `divider` 可能混入周邊 road surface feature，造成：

```text
prototype support 變多
prototype purity 變差
```

因此若要做 support expansion，建議優先測 distance-transform soft weighting，而不是 binary dilation：

```text
離 GT 中心線越近，權重越高；
離中心越遠，權重越低。
```

這個應作為第二階段 ablation，不建議第一版就開。

Map GT 目前通常沒有 ignore index；若未來加入 void / ignore map label，必須在 prototype pooling 時排除，和 segmentation loss 對齊。

## Occupancy 版本設計

### 目前問題

目前 `Prototype_Query_Decoder_nuScenes.forward()` 使用 coarse `occ_pred` 做 argmax，並用 top-2 margin filter 排除 ambiguous voxels 後抽 prototype：

```python
mask_ = F.softmax(mask_target, dim=1)
top2_values, top2_indices = torch.topk(mask_, 2, dim=1)
difference = top2_values[:, 0] - top2_values[:, 1]
scores = 1.0 - difference
vis_mask = scores >= 0.9
mask = mask_.argmax(dim=1)
mask_cls = mask == i
mask_cls *= ~vis_mask
query_ = mask_feat[mask_cls, :].mean(dim=0)
```

這對小物件類別可能不穩：

```text
bicycle
motorcycle
pedestrian
traffic_cone
```

### 改動方向

讓 occupancy decoder 在 training 時接收 `gt_occ`：

目前 `forward_train()` 已經有 `gt_occ`，但 `self(...)` 沒有傳入：

```python
all_cls_scores, all_mask_preds, RPL_args = self(
    voxel_feats, img_metas, mask_feat, occ_pred)
```

建議改成：

```python
all_cls_scores, all_mask_preds, RPL_args = self(
    voxel_feats, img_metas, mask_feat, occ_pred, gt_occ=gt_occ)
```

然後 `forward()` 支援：

```python
def forward(self, voxel_feats, img_metas, mask_feat, occ_pred=None, gt_occ=None):
```

GT-soft mining pseudo-code：

```python
prob = F.softmax(occ_pred.detach(), dim=-1)  # B, X, Y, Z, C
feat = mask_feat                             # B, X, Y, Z, D
gt = gt_occ                                  # B, X, Y, Z
eps = 1e-6
alpha = self.prototype_mining_alpha

for c in range(self.num_classes):
    gt_mask = (gt == c)
    valid_gt = gt_mask & (gt != 255)

    weight = valid_gt.float() * (alpha + (1 - alpha) * prob[..., c])
    support = weight.sum()

    if support < eps:
        proto = zero
        valid = False
    else:
        proto = (feat * weight[..., None]).sum(dim=(0, 1, 2, 3)) / support
        valid = True
```

注意：

```text
ignore_index = 255 的 voxel 不參與 prototype pooling。
```

是否要套 `mask_camera`：

```text
第一版建議和原本 occ loss 設定保持一致。
如果原本 supervision 使用 mask_camera，prototype pooling 也可以只取 visible voxels。
如果原本 class prototype 想利用完整 gt_occ，則先只排除 255。
```

較安全的 default 是和 loss 對齊，避免從未被 loss supervise 的 hidden voxels 產生 noisy prototype：

```python
use_camera_mask_for_proto=True
```

但這會讓 sparse classes 的 support 更少，因此需要 ablation：

```python
use_camera_mask_for_proto=True   # align with visible-voxel loss
use_camera_mask_for_proto=False  # more support, but possible feature noise
```

## EMA bank update 需要改什麼

目前 map 和 occ 都用 `for_query.sum(1) != 0` 判斷某類是否有 prototype。

GT-soft mining 後，建議改成顯式回傳：

```python
for_query:  (K, C)
valid_mask: (K,)
```

EMA update 應改為：

```python
cur_assign_flag = valid_mask
```

而不是：

```python
cur_assign_flag = for_query.sum(1) != 0
```

原因：

```text
某個合法 prototype 的 feature sum 理論上也可能接近 0，
用 valid_mask 更清楚。
```

### Occ decoder 的 buffer / indexing 清理

在修改 occupancy decoder 時，建議順手修兩個容易踩雷的點。

第一，`Prototype_Query_Decoder_nuScenes` 目前用 plain tensor 建立 `proto_first_flag`：

```python
self.proto_first_flag = torch.ones(self.num_classes).cuda().bool()
```

建議改成 buffer：

```python
self.register_buffer(
    'proto_first_flag',
    torch.ones(self.num_classes, dtype=torch.bool))
```

原因：

```text
buffer 會跟 checkpoint 存取
model.to(device) 會自動搬移
DDP / eval 行為比較穩
```

第二，目前 EMA update 使用：

```python
cur_query_assign_flag = for_query[-RPL_pad_size:].sum(1) != 0
```

`for_query` 本身是 `(num_classes, D)`，而 `RPL_pad_size = RPL_Groups * num_classes`，所以當 `RPL_pad_size >= num_classes` 時，`for_query[-RPL_pad_size:]` 等於整個 `for_query`。這在目前程式邏輯上可運作，但可讀性很差。

重構時建議改成：

```python
cur_query_assign_flag = valid_mask
```

或至少：

```python
cur_query_assign_flag = for_query_valid_mask
```

避免未來改 RPL 時把 indexing 寫壞。

## 與 PGBR 的關係

PGBR 是：

```text
用 scene-aware prototype 去做 BEV feature grounding / enhancement / suppression
```

Supervised Sparse Prototype Mining 是：

```text
在 PGBR 使用 prototype 之前，先讓 sparse class 的 prototype 抽得更可靠
```

所以兩者關係是：

```text
GT-soft mining -> better prototype
PGBR -> better use of prototype
```

ablation 應該要拆開：

| 實驗 | 是否 GT-soft mining | 是否 PGBR |
| --- | --- | --- |
| baseline ProtoMapHead no PGBR | 否 | 否 |
| GT-soft mining only | 是 | 否 |
| PGBR only | 否 | 是 |
| GT-soft mining + PGBR | 是 | 是 |

這樣才能證明 C2 不是 PGBR 的另一種說法。

## 與 CNN map head 的關係

第一版 C2 可以先只改 ProtoMapHead / occupancy decoder，因為它們目前真的有 prototype query。

但如果要讓 C2 對 CNN head 也有價值，可以做第二階段 head-agnostic module：

```text
map_feature
  -> supervised sparse prototype mining
  -> prototype-guided feature conditioner
  -> BEVSegHead 或 ProtoMapHead
```

這時 CNN head 不需要 prototype decoder，它只吃被 prototype-conditioned 的 `enhanced_map_feature`。

不過這會更接近 MAESTRO TSFG / PGBR，因此建議第二階段才做。第一階段先專注：

```text
prototype mining reliability
```

不要同時改：

```text
feature conditioner
decoder head
prototype mining
```

否則 ablation 會變得不乾淨。

## 建議實作順序

### Step 1：Map-only GT-soft mining

修改：

```text
projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py
projects/mmdet3d_plugin/models/detectors/ProtoOccMultitask.py
```

內容：

1. `ProtoMapHead.forward()` 新增 `gt_masks_bev=None`。
2. 新增 `_gt_guided_adaPG()`。
3. `_adaPG()` 和 `_gt_guided_adaPG()` 都回傳 `(for_query, valid_mask)`。
4. `_agno_PG_update_and_get()` 改接 `valid_mask`。
5. config 新增：

```python
prototype_mining_mode='pred_threshold'  # baseline
prototype_mining_mode='gt_hard'
prototype_mining_mode='gt_soft'
prototype_mining_alpha=0.5
prototype_pooling_scope='batch_local'
```

新增 config：

```text
projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_no_pgbr_gt_soft.py
projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_gt_soft.py
```

目的：

```text
先證明 map sparse classes 能不能受益。
```

### Step 2：Occupancy GT-soft mining

修改：

```text
projects/mmdet3d_plugin/models/OccHead/Prototype_Query_Decoder_nuScenes.py
projects/mmdet3d_plugin/models/detectors/ProtoOccMultitask.py
projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py
```

內容：

1. `forward_train()` 呼叫 `self(..., gt_occ=gt_occ)`。
2. `forward()` 新增 `gt_occ=None`。
3. 新增 `_gt_guided_adaPG()` 或 `_build_scene_adaptive_prototypes()` helper。
4. RPL 暫時維持原本 prediction-mask 版本，避免第一版牽動太多。
5. config 新增：

```python
prototype_mining_mode='pred_argmax'
prototype_mining_mode='gt_hard'
prototype_mining_mode='gt_soft'
prototype_mining_alpha=0.5
prototype_pooling_scope='batch_local'
use_camera_mask_for_proto=True
```

若同一份 config 同時有 map prototype 和 occ prototype，可以用明確 prefix：

```python
map_prototype_mining_mode='gt_soft'
occ_prototype_mining_mode='gt_soft'
```

但 module 內部參數名稱建議統一成 `prototype_mining_mode`，避免未來抽 shared helper 時做 key 翻譯。

目的：

```text
證明 occupancy 小物件 prototype 也能受益。
```

### Step 3：整理 shared helper

如果 Step 1 / Step 2 有效，再抽出共用 helper：

```text
projects/mmdet3d_plugin/models/utils/prototype_mining.py
```

可能 API：

```python
weighted_class_prototypes_2d(feat, gt_mask, pred_prob, mode)
weighted_class_prototypes_3d(feat, gt_label, pred_prob, mode)
```

先不要一開始就抽 abstraction，避免過早抽象讓 debug 困難。

## 建議實驗表

### Map 實驗

| 實驗 | Config | 重點 |
| --- | --- | --- |
| ProtoMapHead map neck no PGBR baseline | `ProtoOcc_proto_map_head_map_neck_no_pgbr.py` | prediction threshold mining |
| GT-hard map mining | 新增 | 只用 GT mask mean pooling |
| GT-soft map mining | 新增 | GT mask + prediction confidence weighting |
| GT-soft + PGBR | 新增 | 測 reliable prototype 是否讓 PGBR 更有效 |
| alpha sweep | 新增 | `alpha in {0, 0.3, 0.5, 0.7, 1.0}` |
| pooling scope ablation | 新增 | `batch_local` vs `sample` |

主要報：

```text
map/mean/iou@max
ped_crossing
stop_line
divider
carpark_area
```

### Occupancy 實驗

| 實驗 | 重點 |
| --- | --- |
| Original ProtoOcc AdaPG | prediction argmax mining |
| GT-hard occ mining | GT class mask mean pooling |
| GT-soft occ mining | GT class mask + prediction confidence weighting |
| alpha sweep | `alpha in {0, 0.3, 0.5, 0.7, 1.0}` |
| camera mask ablation | `use_camera_mask_for_proto=True/False` |

主要報：

```text
Occ mIoU
bicycle
motorcycle
pedestrian
traffic_cone
construction_vehicle
```

### Cross-task 多任務實驗

| 實驗 | 重點 |
| --- | --- |
| map GT-soft only | 只穩定 map prototype |
| occ GT-soft only | 只穩定 occ prototype |
| map + occ GT-soft | 同時穩定兩邊 prototype |

觀察：

```text
map prototype 穩定是否傷害 occ
occ prototype 穩定是否幫助 map
兩者同時使用是否互補
```

## 評估時要額外記錄的診斷量

為了證明 C2 是 prototype reliability，而不是只看最後 IoU，建議 log：

1. 每個 class 每個 batch 是否有 valid local prototype。
2. 每個 class 的 prototype support size。
3. EMA bank 每個 class 的 update 次數。
4. `for_query` norm / EMA prototype norm。
5. map sparse classes 在 early epoch 的 valid rate。
6. occ small object classes 在 early epoch 的 valid rate。
7. 同一 class prototype across epochs 的 cosine similarity。
8. 不同 class prototype 之間的 cosine distance / similarity matrix。

例如：

```text
map_proto_valid_rate/stop_line
map_proto_support/ped_crossing
occ_proto_valid_rate/bicycle
occ_proto_support/pedestrian
map_proto_consistency/stop_line
occ_proto_consistency/bicycle
map_proto_inter_class_min_distance
occ_proto_inter_class_min_distance
```

如果 baseline 中 `stop_line` 常常 valid rate 很低，而 GT-soft mining 明顯提高 valid rate，就能直接支撐 C2 的論點。

更強的論證是：

```text
GT-soft 不只讓 support 變多，
也讓同類 prototype 跨 epoch 更穩定，
並讓不同類 prototype 更可分。
```

## 風險與注意事項

### 1. Train-test mismatch

training 用 GT，inference 沒 GT，會有 mismatch。

緩解方式：

```text
不用純 GT hard 作為主方法。
使用 GT-soft：GT 保底，prediction confidence 仍然參與。
inference 使用 EMA bank 作為穩定來源。
```

### 2. Shallow CNN 可能變弱

如果 prototype 完全不依賴 prediction，shallow CNN 的 prototype-related role 會變弱。

緩解方式：

```text
coarse_pred 繼續用 BCE/Dice 或 CE loss 監督。
prototype weight 使用 pred_prob.detach()，但 pred_prob 仍影響 pooling weight。
```

### 3. GT mask 太稀疏

`stop_line` / `divider` 可能 support 很少。

可選解：

```text
training prototype pooling 使用 distance-transform soft support。
segmentation loss 仍使用原始 GT mask。
```

不要第一時間使用 binary dilation 當主方法，因為它可能讓線狀類別 prototype 混入周邊背景語意。若要測 dilation，必須同時報 prototype purity / inter-class separability。

### 4. RPL 互動

ProtoOcc 的 RPL 會對 predicted mask 加 noise，再生成 noisy prototypes。

第一版建議：

```text
主 for_query 使用 GT-soft mining。
RPL_for_query 先維持原本 prediction/noisy mask 邏輯。
```

等主效果確認後，再討論 RPL 是否也要改成 GT-guided noisy mask。

### 5. 類別 absent 不應更新 EMA

如果某個 scene 沒有某類別，必須 skip EMA update。

不要讓 zero prototype 寫入 EMA bank。

### 6. alpha 是 hyperparameter

`alpha=0.5` 沒有理論上唯一正確性。它只是折中 baseline。

必須用 ablation 支撐：

```text
alpha in {0, 0.3, 0.5, 0.7, 1.0}
```

若 `alpha=1.0` 最好，表示 GT-hard support 已足夠，prediction confidence 反而有 noise。

若 `alpha=0.0` 最好，表示 prediction confidence 對 prototype purity 很重要。

若中間值最好，才支撐 GT support + prediction confidence 的雙重設計。

### 7. Batch-level prototype vs scene-level prototype

目前 first version 保持 local-batch pooling。這和現有 code 一致，但如果要在論文中嚴格使用 `scene-adaptive`，最好補一個 per-sample pooling ablation。

不要在沒有 ablation 的情況下過度宣稱：

```text
each scene has fully independent prototypes
```

比較穩的說法是：

```text
batch-local scene-adaptive prototypes following ProtoOcc implementation
```

## 預期結果

理想情況：

```text
Map:
  stop_line / ped_crossing / divider 上升
  map mean 上升

Occupancy:
  bicycle / motorcycle / pedestrian / traffic_cone 上升
  Occ mIoU 不下降或小幅上升
```

若只提升 sparse classes，但 mean 提升有限，仍可作為 C2 論點：

```text
GT-soft mining improves prototype support for sparse semantics.
```

若 GT-hard 比 GT-soft 更好：

```text
表示 prediction confidence 在 early stage 可能仍然有 noise。
可以考慮 curriculum 或降低 pred_prob 權重。
```

若 GT-soft 無效：

需要檢查：

```text
1. GT mask shape / class order 是否正確
2. support 是否真的增加
3. EMA update 是否有被 valid_mask 控制
4. prototype 是否被後續 query_embed 壓掉
5. sparse class loss 是否太弱
```

## 可以交給其他 AI 的最小任務描述

請在 ProtoOcc 中實作 `GT-guided soft prototype mining`：

1. 修改 `ProtoMapHead`，讓 `forward(bev_feat, gt_masks_bev=None)` 支援 training-time GT-soft prototype mining。
2. `_adaPG()` 和新的 `_gt_guided_adaPG()` 都回傳 `(for_query, valid_mask)`。
3. EMA update 改用 `valid_mask`，避免 absent class 用 zero prototype 更新 bank。
4. 在 `ProtoOccMultitask.forward_train()` 呼叫 `self.proto_map_head(map_feature, gt_masks_bev=gt_masks_bev)`。
5. 新增 no-PGBR 和 PGBR 的 GT-soft config。
6. 新增 `prototype_mining_alpha`，支援 alpha sweep。
7. 第一版 pooling scope 先保持 `batch_local`，並在 config 中保留未來 `sample` ablation 的入口。
8. eval mode 下不得更新 EMA bank。
9. 第一版先不要改 CNN head，也先不要抽 shared module。
10. 通過 `py_compile` 和 config loading smoke test。

第二階段再把同樣概念搬到 occupancy decoder：

1. `Prototype_Query_Decoder_nuScenes.forward_train()` 將 `gt_occ` 傳入 `forward()`。
2. training 時用 `gt_occ` + `softmax(occ_pred)` 做 GT-soft class prototype mining。
3. 修正 `proto_first_flag`，用 `register_buffer` 儲存。
4. EMA update 改用顯式 `valid_mask`，不要沿用脆弱的 `for_query[-RPL_pad_size:]` indexing。
5. 補 `use_camera_mask_for_proto=True/False` ablation。
6. 先保留 RPL 原本邏輯不動。

## 最終論文敘事

C2 可以寫成：

```text
Prediction-driven prototype mining is fragile for sparse semantic categories.
In ProtoOcc AdaPG, margin filtering removes ambiguous voxels, but it still cannot guarantee positive support for sparse classes when early coarse predictions miss them.
Similarly, MAESTRO-style prototype generation does not explicitly guarantee sparse map class support.
We introduce supervised sparse prototype mining, which uses GT masks during training to guarantee positive support and uses prediction confidence as alpha-controlled soft weighting.
This stabilizes both scene-adaptive prototypes and EMA prototype banks without using GT at inference time.
```

中文版本：

```text
前人的 prototype generator 主要關心如何從 feature / prediction 產生 prototype，
但對小物件、線狀 map 元素這類 sparse semantics，prediction-driven mask 在訓練早期容易抽不到可靠 support。
即使 ProtoOcc AdaPG 有 top-2 margin filter，它仍無法保證 sparse class 一定有 positive support。
我們提出 GT-guided soft prototype mining，在訓練期用 GT mask 保證正樣本 support，
再用模型 prediction confidence 做 soft weighting，讓 prototype 既有監督保底，也保留模型自適應性。
```
