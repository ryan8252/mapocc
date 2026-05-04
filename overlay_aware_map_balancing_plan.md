# MTL-Aware Overlay Map Balancing 計劃

## Summary

這份計劃的核心論點需要從「只加強 `ped_crossing / stop_line / divider`」更新成兩層問題：

1. **Task-level map suppression**：在 Occ+Map 多任務訓練中，OCC 的 dense 3D supervision 會主導 shared BEV feature 和整體 loss priority，導致 map branch 學得比 map-only 明顯差。
2. **Map-internal overlay/thin imbalance**：在 map task 內部，`ped_crossing / stop_line / divider` 是疊在 road surface 上的 overlay/thin classes，正樣本面積小、形狀細長，仍然是最容易被壓制的類別。

因此我們要挑戰的不是 MAESTRO 的三任務設定本身，而是現有 Occ+Map 或 Occ+Det+Map 架構通常沒有明確處理 HD map 在多任務訓練中的 task-level priority 和 overlay/thin class imbalance。目標是在 end-to-end、不蒸餾、不分階段訓練的條件下，把 MTL map performance 往 map-only upper bound 靠近，同時讓 OCC 不明顯低於 original ProtoOcc baseline。

這個故事可以借鑑 HintOcc 的 batch-wise dynamic weighting：HintOcc 指出 dataset-wise/static class weighting 沒有考慮每個 batch 中 class 是否真的出現，會低估 present under-represented classes。我的設定更進一步：HD map 是 multi-label BEV segmentation，且 map task 同時受到 OCC dense supervision 的 task-level suppression。因此方法不應只是套用 occupancy CE class weighting，而是設計 **MTL-aware, batch-wise, overlay-positive map balancing**。

## Parameter Glossary

先把後面會用到的參數定義清楚：

- `map_loss_weight`：整個 map loss 的 global weight，用來控制 map task 在 Occ+Map MTL 裡的整體 priority。例如 `map_loss_weight=4` 代表所有 map loss 先整體放大 4 倍。
- `map_class_weights`：V1 static weighting 使用的 per-class multiplier，class order 是 `[drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider]`。
- `overlay_class_indices`：overlay/thin classes 的 channel index，對應 `{ped_crossing, stop_line, divider}` = `[1, 3, 5]`。
- `batch_pos_ratio_c`：目前 mini-batch 裡 class `c` 的正樣本比例，定義為 `sum(gt[:, c, :, :]) / (B * H * W + eps)`。這只看單一 channel，不把六個 map channels 混在一起。
- `ref_pos_ratio_c`：class `c` 在 nuScenes train split 裡的固定參考正樣本比例。第一版建議訓練前 pre-compute 一次後寫進 config，不另外新增 loss-ratio EMA。
- `dynamic_overlay_gamma`：dynamic weight 的平滑係數。`gamma < 1` 會讓權重變化比較溫和；first trial 用 `0.5`。
- `dynamic_overlay_min_weight`：dynamic overlay weight 的下限。first trial 用 `1.0`，意思是最低維持原本 loss，不因為某個 batch overlay 較多就降低它。
- `dynamic_overlay_max_weight`：dynamic overlay weight 的上限。first trial 用 `5.0`，避免 `stop_line` 這類極稀少 class 造成 loss 爆掉。
- `overlay_dynamic_weight_c`：V3 對 class `c` 算出的 effective multiplier。它只改變 overlay GT-positive pixels 的 BCE element weight，以及 present overlay class 的 Dice class weight，不額外加一個重複 loss。
- `map_bce_loss_weight` / `map_dice_loss_weight`：現有 config 中 `BEVSegHead` 的 map BCE 權重是 `5.0`，Dice 權重是 `1.0`。V1+ 手動 loss path 必須保留這個比例，不能把 BCE 和 Dice 直接等權相加。
- `positive_mask_c`：class `c` 的 GT-positive binary mask。它只用來建立 supervised BCE element weight，不是 GT support / GT prototype mining，inference 時完全不需要 GT。

## Current Evidence

`epoch_24_ema_weight_4` 已完成，因此 `map_loss_weight=4` 現在可以視為正式 strong baseline，而不是只看 early/mid-training snapshot：

| Setting | Checkpoint | Occ mIoU | Map mIoU | Drivable | Ped. Cross. | Walkway | Stop Line | Carpark | Divider |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MTL CNN head + 128ch map neck, `map_loss_weight=1` | epoch 11 EMA | 39.46 | 36.95 | 74.94 | 26.30 | 42.99 | 17.26 | 35.02 | 25.19 |
| MTL CNN head + 128ch map neck, `map_loss_weight=4` | epoch 5 EMA | 38.17 | 39.55 | 76.10 | 32.50 | 44.77 | 19.35 | 36.95 | 27.64 |
| MTL CNN head + 128ch map neck, `map_loss_weight=4` | epoch 11 EMA | 39.26 | 43.82 | 78.59 | 37.82 | 48.71 | 24.54 | 42.25 | 31.03 |
| MTL CNN head + 128ch map neck, `map_loss_weight=1` | epoch 24 EMA | 39.82 | 39.94 | 76.36 | 30.52 | 45.15 | 20.73 | 39.25 | 27.63 |
| MTL CNN head + 128ch map neck, `map_loss_weight=4` | epoch 24 EMA | 39.72 | 45.79 | 80.03 | 41.94 | 50.93 | 28.32 | 40.00 | 33.49 |
| Map-only CNN head + 128ch map neck | epoch 24 EMA | - | 48.34 | 80.84 | 44.58 | 52.72 | 33.62 | 42.87 | 35.43 |
| MAESTRO R50 | reported | 38.60 | 51.30 | 80.30 | 45.90 | 55.40 | 36.10 | 48.30 | 41.80 |

這個結果改變了原本的判斷：

- `map_loss_weight=4` 不只是 early-learning boost。到 epoch 24，Map mIoU 從 weight1 的 39.94 提升到 45.79，OCC 只從 39.82 下降到 39.72，代價約 0.10。
- MTL map gap 從 weight1 的 `48.34 - 39.94 = 8.40` 縮到 weight4 的 `48.34 - 45.79 = 2.55`。這強力支持「map 在多任務中主要先被 loss priority / task-level suppression 壓制」的論點。
- 提升不只出現在 `ped_crossing / stop_line / divider`，`drivable_area / walkway / carpark_area` 也明顯提升。因此不能把貢獻寫成單純 class-wise rare-class weighting。
- 相對於 map-only，剩餘 gap 最大的是 `stop_line`：28.32 vs 33.62，差 5.30。`ped_crossing` 和 `divider` 已經接近 map-only，但仍有改善空間。
- 下一步的 overlay-aware balancing 不應該只是打敗 weight1，而是要在 V0 `map_loss_weight=4` 這個 strong baseline 上，進一步補 `stop_line / divider / ped_crossing`，同時維持 OCC。

## Motivation

Map-only CNN head + 128ch map neck 可達 48.34 mIoU，代表 map GT、BEV map head、rasterization pipeline 本身不是主要瓶頸。真正的退化出現在 Occ+Map MTL：`map_loss_weight=1` epoch 24 只有 39.94 mIoU，和 map-only 差 8.40 points。

`map_loss_weight=4` epoch 24 達到 45.79 mIoU，且 OCC 仍有 39.72，說明 map branch 不是沒有能力學，而是在 MTL 中 loss priority 不足。這是一個比「nuScenes dataset 有問題」更精準、也更適合寫論文的說法：nuScenes HD map 的 overlay/thin classes 本來就難，MAESTRO supplemental Table 2 也顯示 `ped_crossing / stop_line / divider` 低於大面積類別；但我們的額外觀察是，OCC 多任務訓練會進一步放大這個困難。

HintOcc 的 batch-wise dynamic weighting 提供一個可以借用的論述方向：under-represented classes 的問題不只來自 dataset-level frequency，也來自 mini-batch / scene-level class presence。對 occupancy semantic CE 來說，batch-wise weighting 可以忽略 absent classes 對 present classes 權重的影響；對 HD map segmentation 來說，這個問題更明顯，因為 `ped_crossing / stop_line / divider` 是 multi-label overlay/thin masks，positive pixels 很少，且一個 batch 中可能完全 absent。

因此本計劃不是直接複製 HintOcc 的 single-label CE dynamic class weighting，而是把它改成 Occ+Map MTL 的 map-specific 版本：先用 global map balancing 修正 task-level suppression，再用 batch-wise overlay-positive balancing 只加強當前 batch 中真的出現的 overlay positives，同時保留 absent classes 的 negative BCE 以避免 false positives。

## Challenge

- **Task-level imbalance**：OCC loss 來自 dense 3D occupancy volume，監督訊號更密集，容易主導 shared feature learning；map loss 在預設權重下被壓制。
- **Overlay/thin class imbalance**：`ped_crossing / stop_line / divider` 面積小、正樣本少，且通常依附在 road surface 上，不是獨立的大面積 semantic region。
- **Multi-label loss domination**：BCE/Dice 同時看六個 map channels，大面積類別與大量 easy negatives 會稀釋 overlay/thin positive pixels。
- **Static weighting 的限制**：dataset-wise/static class weights 不知道某個 overlay class 在當前 batch 是否真的出現，容易把 absent sparse classes 和 present sparse positives 混在一起處理。
- **Global weighting 的限制**：`map_loss_weight=4` 很有效，而且 OCC 幾乎沒有受損；但它同時加強所有 map classes。剩餘問題主要集中在 `stop_line` 等 overlay/thin classes，因此仍需要更細的 task-class balancing。

## Goal

Primary goal:

- 建立一個 end-to-end Occ+Map MTL 的 map balancing 方法，先解決 task-level map suppression，再處理 overlay/thin class imbalance。

Metric goals:

- Strong baseline：`map_loss_weight=4` epoch 24 是新的必要 baseline，結果為 Occ 39.72 / Map 45.79。
- Map target：下一步 overlay-aware loss 應該至少維持 45.79 附近，並優先提升 `stop_line`，其次是 `divider / ped_crossing`。
- OCC target：OCC mIoU 至少維持在 39.0 以上；理想情況不要比 V0 39.72 低超過 0.3，並盡量接近 `map_loss_weight=1` epoch 24 的 39.82。
- Paper wording：**MTL-aware batch-wise HD map balancing for occupancy-map joint learning**，不要只寫成 class-wise weighting。

## Method

Map class taxonomy:

- overlay/thin classes: `{ped_crossing, stop_line, divider}`，主要是 road surface 上的 painted markings / thin structures。
- area classes: `{drivable_area, walkway, carpark_area}`，主要是大面積 surface regions。`walkway` 雖然也受 MTL 影響，但不應被寫成 painted overlay class。

### V0-V3 Difference Summary

| Version | 做什麼 | 主要解決的問題 | 和前一版的差別 | 風險 / 判讀方式 |
| --- | --- | --- | --- | --- |
| V0 | 只把 global `map_loss_weight` 從 1 提到 4，所有 map classes 等比例加強。 | Task-level map suppression。 | 不做 class-specific balancing，只確認 map task 在 MTL 裡是否被 OCC 壓制。 | 現在結果是 Occ 39.72 / Map 45.79，已經是 strong baseline；後續不能只跟 weight1 比。 |
| V1a | 在 V0 上加 static `map_class_weights=[1,1.5,1,2,1,1.5]`。 | Overlay/thin classes 在 map loss 內仍偏弱。 | 保留 V0 的 global map priority，再額外提高 `ped_crossing / stop_line / divider`。 | 如果 area classes 掉很多，代表只是把錯誤從 overlay 轉移到 area。 |
| V1b | 降低 global `map_loss_weight=2`，但 overlay class weights 更高，effective weights 是 `[2,4,2,6,2,4]`。 | 測 OCC-map trade-off。 | 不是主線 improvement，而是檢查如果 V0/V1a 太偏 map，OCC 能不能回升。 | Map mIoU 預期可能低於 V0；只有 OCC 明顯回升時才有報告價值。 |
| V2 | 不再整個 channel 加權，只 static 加強 overlay GT-positive pixels 和 class-wise Dice。 | Sparse positive pixels 被大面積類別和 easy negatives 稀釋。 | 從 class-level static weighting 改成 positive-only weighting，避免把 overlay negative pixels 也一起放大。 | 要檢查 false positives 和 fixed-threshold IoU，不能只看 `iou@max`。 |
| V3 | 根據每個 batch 的 `batch_pos_ratio_c / ref_pos_ratio_c` 動態調整 overlay positive weight。 | Present-but-rare overlay classes 在某些 batch 中被低估。 | 從固定 overlay weight 改成 batch-wise dynamic weight，類似 HintOcc 但改成 multi-label map positive balancing。 | moving parts 最多；需要 debug log 確認權重只在 class present 且稀疏時啟動。 |

### V0: Strong Global Map Baseline

先正式建立 `map_loss_weight=4` baseline。

- Config 建議獨立成 `ProtoOcc_multi_cnn_head_map_neck_weight4.py`，避免反覆手改主 config。
- Epoch 24 結果為 Occ 39.72 / Map 45.79，距離 map-only 48.34 只剩 2.55 points。
- 這不是最終貢獻，但它是所有新方法必須比較的 strong baseline。
- 這代表 task-level balancing 已經解掉大部分 MTL gap；後續方法的價值要看能不能在這個基準上補 overlay/thin classes。

### V1a: Overlay-on-Top Static Weighting

在 V0 的 task-level map priority 之上，再疊加 overlay/thin class emphasis：

- class order: `[drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider]`
- first trial: `map_loss_weight=4.0`, class weights `[1.0, 1.5, 1.0, 2.0, 1.0, 1.5]`
- effective weights: `[4, 6, 4, 8, 4, 6]`

這是主要 static weighting baseline。它不降低 area classes 的 map priority，因此不會重新引入 task-level map suppression。主要問題是：在保住 `drivable_area / walkway / carpark_area` 的前提下，是否能額外提升 `ped_crossing / stop_line / divider`。

### V1b: Pareto Trade-Off Static Weighting

如果 V0 或 V1a 的 epoch 24 OCC 明顯下降，再測一組偏向 OCC recovery 的 trade-off：

- class order: `[drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider]`
- trial: `map_loss_weight=2.0`, class weights `[1.0, 2.0, 1.0, 3.0, 1.0, 2.0]`
- effective weights: `[2, 4, 2, 6, 2, 4]`

這組不是主線方法，而是 Pareto diagnostic。它預期可能犧牲部分 mean Map mIoU，尤其 area classes，來換 OCC recovery。如果它的 Map mIoU 明顯低於 V0 weight4，就只能作為「OCC-map trade-off」報告，不應包裝成主要 improvement。

### V2: Static Overlay-Positive Reweighting

只加強 overlay/thin classes 的 positive pixels 和 class-wise Dice：

- strengthen: `ped_crossing / stop_line / divider`
- keep negative BCE normal or only mildly weighted
- avoid model 為了衝 rare class 而產生大量 false positives

這比直接對整個 channel 加權更合理，因為 HD map overlay classes 的主要問題是 positive support 太少，不是所有 negative pixels 都需要加強。

### V3: Batch-Wise Dynamic Overlay-Positive Balancing

借鑑 HintOcc 的 batch-wise dynamic weighting，但改成 map-specific multi-label 版本：

- 根據當前 batch 每個 overlay class 的 positive pixel ratio 計算權重。
- 只對當前 batch 中真的出現的 overlay positives 加 boost。
- 若某 class 在 batch 中完全 absent，不加強 positive / Dice 權重，但保留 normal negative BCE 和 normal classwise Dice，避免 false positives 沒有懲罰。
- 權重需要 clamp，避免 `stop_line` 因為極少而造成 training unstable。
- 動態權重只作用在 `ped_crossing / stop_line / divider`，大面積 classes 仍由 base BCE/Dice 和 global `map_loss_weight` 控制。
- 這裡不使用 additive double-counting。Overlay positive pixels 的 effective weight 直接由 `overlay_dynamic_weight_c` 決定。

Dynamic weighting 的 ratio 定義要固定為 per-channel positive ratio，而不是把六個 map channels 混在一起：

```text
valid_pixels = B * H * W
batch_pos_ratio_c = sum(gt[:, c, :, :]) / (valid_pixels + eps)

if sum(gt[:, c, :, :]) > 0:
    overlay_dynamic_weight_c = clamp(
        (ref_pos_ratio_c / (batch_pos_ratio_c + eps)) ** gamma,
        min_weight,
        max_weight,
    )
else:
    overlay_dynamic_weight_c = 1.0
```

First trial:

- `ref_pos_ratio_c`: pre-compute from the nuScenes train split for each overlay channel and store in config.
- `eps = 1e-6`
- `gamma = 0.5` for stability
- `min_weight = 1.0`
- `max_weight = 5.0`

`min_weight = 1.0` makes the dynamic weight one-sided: present-but-rare overlay classes receive extra weight, while present-but-abundant overlay classes stay at normal weight 1.0. We do not down-weight overlay-rich batches because the problem is sparse positives being suppressed, not overlay-rich batches being over-dominant.

Loss composition without additive double-counting:

```text
BCE_per_pixel = BCEWithLogits(seg_logits, gt, reduction='none')
bce_element_weight = ones_like(gt)

for each overlay class c:
    if gt[:, c].sum() > 0:
        bce_element_weight[:, c] =
            1 + (overlay_dynamic_weight_c - 1) * gt[:, c]

loss_map_bce =
    map_bce_loss_weight * mean(BCE_per_pixel * bce_element_weight)

Dice_per_class = naive_dice_loss(sigmoid(seg_logits), gt, reduction='none')
dice_class_weight = ones(B, C)

for each overlay class c:
    if gt[:, c].sum() > 0:
        dice_class_weight[:, c] = overlay_dynamic_weight_c

loss_map_dice =
    map_dice_loss_weight * mean(Dice_per_class * dice_class_weight)
```

因此 area positive pixels 的 effective weight 仍是 baseline 的 `map_loss_weight`，overlay positive pixels 的 effective weight 近似是 `map_loss_weight * overlay_dynamic_weight_c`。但 reduction 仍然沿用原本 BCE over `(B,C,H,W)` 的 mean，不把 positive / negative 分開平均後再相加，避免 `stop_line` 這類極稀少 class 被額外放大到不可控。

這裡跟 HintOcc 的差異是：HintOcc 是 single-label occupancy CE，可以把 absent class 的 CE term 權重設為 0；這裡是 multi-label map BCE/Dice，因此 absent overlay class 只是不加 positive/Dice boost，不能把整個 channel 的 negative loss 關掉。

Global `map_loss_weight` 不在 `BEVSegHead.loss()` 內再乘一次。現有 detector 的 `_scale_map_losses()` 已經負責把 `loss_map_bce` / `loss_map_dice` 乘上 `map_loss_weight`；如果 head 裡又乘一次，`map_loss_weight=4` 會變成實質 16 倍，和 V0 baseline 不可比。

## Expected Code Changes

- `projects/mmdet3d_plugin/models/dense_heads/bev_seg_head.py`
  - Add optional class-wise BCE/Dice weighting.
  - Add optional overlay-positive weighting.
  - Add optional batch-wise dynamic overlay weights.
  - Add debug logging for per-class positive pixels and per-class loss contribution.
  - Keep default behavior unchanged when balancing options are not enabled.

- `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py`
  - Keep default behavior unchanged.
  - Add optional config fields only when balancing is enabled:
    - `map_class_weights`
    - `overlay_class_indices`
    - `overlay_pos_weight`
    - `dynamic_overlay_ref_pos_ratio`
    - `dynamic_overlay_gamma`
    - `dynamic_overlay_min_weight`
    - `dynamic_overlay_max_weight`
    - `map_loss_balance_mode`

- New config variants:
  - `ProtoOcc_multi_cnn_head_map_neck_weight4.py`
  - `ProtoOcc_multi_cnn_head_map_neck_overlay_static_top.py`
  - `ProtoOcc_multi_cnn_head_map_neck_overlay_static_pareto.py`
  - `ProtoOcc_multi_cnn_head_map_neck_overlay_dynamic.py`

- `experiment_results_summary.md`
  - Add `map_loss_weight=4` epoch 5, epoch 11, and epoch 24 rows.
  - Add overlay-static and overlay-dynamic rows after experiments finish.

- Implementation detail:
  - Do not rely on the built-in `CrossEntropyLoss.class_weight` for map class weighting.
    In this repo, sigmoid BCE passes `class_weight` into `pos_weight`, which only reweights positive terms and may not match `(B, C, H, W)` map logits safely.
  - For V1+ balancing modes, bypass the `build_loss` BCE/Dice path and compute weighted BCE/Dice manually inside `BEVSegHead.loss()`. Keep the existing `build_loss` path only for default baseline behavior.
  - Implement map balancing directly inside `BEVSegHead.loss()`:
    - compute BCE with `reduction='none'`
    - reshape class weights as `(1, C, 1, 1)`
    - keep the existing BCE/Dice loss ratio: BCE `loss_weight=5.0`, Dice `loss_weight=1.0`
    - keep global `map_loss_weight` in detector-level `_scale_map_losses()` only; do not multiply it inside `BEVSegHead.loss()`
    - for V1 full-channel static weighting, multiply BCE elements by `(1, C, 1, 1)` class weights before the same global mean reduction
    - compute Dice as per-class `(B, C)` losses, then apply matching class weights before mean reduction
    - for V2/V3 overlay-positive mode, build an element-wise BCE weight map `1 + (overlay_weight - 1) * gt_positive_mask` for overlay channels only
    - do not split positive BCE and negative BCE into two separately averaged terms; that changes the loss scale and can over-amplify rare classes
    - keep normal negative BCE for absent overlay classes
  - Pre-compute `dynamic_overlay_ref_pos_ratio` from the nuScenes train split after the training pipeline has produced final `gt_masks_bev` tensors, using the same map resolution and channel order as training. Do not use raw polygon area ratios.
  - Do not add a new loss-ratio EMA in the first version. This is independent from model-weight EMA checkpoints, which should still be used for evaluation.
  - Log per-class positive pixels, dynamic weights, and per-class BCE/Dice contribution, so the effect is inspectable instead of only relying on final mIoU.


## Experiment Plan

1. Treat `map_loss_weight=4` epoch 24 as established V0.
   - V0 result: Occ 39.72 / Map 45.79.
   - Compare every later method against V0, `map_loss_weight=1` epoch 24, and map-only upper bound.
   - Update the paired baseline table: original ProtoOcc / occ-only, map-only, Occ+Map weight1, Occ+Map weight4.

2. Run V1a overlay-on-top static weighting.
   - Start with `map_loss_weight=4.0`, class weights `[1,1.5,1,2,1,1.5]`.
   - Expected behavior: map close to or better than weight4, `stop_line/divider/ped_crossing` better, area classes not obviously lower.
   - Purpose: test whether overlay emphasis can be added without undoing task-level map balancing.
   - Before full training, run a loss-parity smoke: balancing mode disabled must match the original `build_loss` path, and V1 with all class weights set to `1` must match V0 within numerical tolerance.

3. Run V1b Pareto trade-off only if needed.
   - Use `map_loss_weight=2.0`, class weights `[1,2,1,3,1,2]`.
   - Expected behavior: mean map may drop compared with weight4, but OCC may recover.
   - Purpose: report the OCC-map trade-off if V0/V1a over-prioritize map.
   - Since V0 OCC is already 39.72, this is not urgent unless V1a/V2/V3 causes a larger OCC drop.

4. Run V2 static overlay-positive weighting.
   - Compare against global weight4, not only against weight1.
   - Main question: can it improve thin overlay classes without increasing false positives?
   - Record both `iou@max` and fixed-threshold scores such as `iou@0.50`, because positive-only boost may change probability calibration.

5. Run V3 dynamic overlay balancing only after static weighting has a clear signal.
   - Dynamic weighting has more moving parts, so it should not be the first implementation target.
   - Compare against V1a/V1b and V2, not only against `map_loss_weight=1`.
   - Main question: can batch-wise present-class weighting improve `stop_line/divider/ped_crossing` without sacrificing dominant map classes or OCC?

## Acceptance Criteria

Use `map_loss_weight=4` epoch 24, Occ 39.72 / Map 45.79, as the formal benchmark. Epoch 5 / epoch 11 results are only early-training signals and should not be used as paper-level acceptance criteria.

- Overlay-aware loss must either beat global weight4 on at least two overlay classes, or keep similar map mIoU while recovering OCC.
- Since global weight4 already reaches map-only minus 2.55 points with acceptable OCC, the paper contribution should emphasize task-level map balancing first, and overlay/thin balancing as the second-stage refinement.
- For V1a/V2/V3 mainline results, mean Map mIoU should stay close to 45.79; large drops in `drivable_area / walkway / carpark_area` mean the method is only shifting errors.
- A useful overlay-aware result should especially improve `stop_line` beyond V0 28.32, ideally above 30, while not reducing mean Map mIoU by more than about 0.5-1.0 point.
- For V1b Pareto results, a mean Map mIoU drop is acceptable only if OCC recovery is clear; report it separately as trade-off, not as the main improvement.
- OCC should stay at least above 39.0; ideal results should remain close to V0 39.72 and should not fall below V0 by more than about 0.3.
- Dynamic overlay balancing should not only improve `iou@max`; it should also avoid obvious calibration regression at fixed thresholds such as `iou@0.50`.
- Debug logs should show that dynamic weights activate mainly when overlay classes are present and stay within the clamp range.
- Loss implementation must preserve V0 comparability: no second `map_loss_weight` multiplication, no BCE/Dice ratio change, and no positive/negative BCE split-average unless it is reported as a separate ablation.

## Notes

- Do not claim "nuScenes dataset has a problem." The safer and stronger claim is: nuScenes HD map segmentation has layered overlay/thin annotations that are naturally imbalanced, and Occ+Map MTL amplifies that imbalance.
- `map_loss_weight=4` is now a very strong diagnostic baseline. Do not call V1/V2/V3 an improvement unless they beat or match this baseline under the acceptance criteria above.
- Because PGBR and GT-soft prototype mining performed poorly, the next method should stay on the CNN head + 128ch map neck baseline first.
- Do not use GT road-support constrained loss as the next step. GT-soft already showed train-test mismatch / calibration risk, so the mainline should avoid GT-dependent training hints that are unavailable at inference.
- Do not copy HintOcc's single-label CE formula directly. The correct adaptation here is multi-label overlay-positive balancing with base negative BCE preserved.
- If the method works on CNN head + 128ch map neck, test transfer to ProtoMapHead later with no-PGBR / no-GT-soft first, preferably on coarse map output before involving final prototype refinement.
- Do not use epoch 11 weight4 as the formal benchmark. It can guide early decisions, but all claims should be checked against weight4 epoch 24: Occ 39.72 / Map 45.79.
- If writing the MTL suppression story for a paper, include a paired baseline table with original ProtoOcc / occ-only, map-only, Occ+Map weight1, Occ+Map weight4, and ours.
