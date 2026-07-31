# Path B：2-task 公平戰策略 + Encoder 實驗計劃

日期：2026-05-31

Branch：`path_b_map_experiments`（off `MapResidualAdapter`，尚未 commit）

> 本文件同時是**策略決策紀錄**與**實作筆記**。它記錄一個關鍵判斷：在 2-task(Occ+Map)
> 框架下，map 的天花板基本是 map-only 的 48.34，硬追 MAESTRO 的 51.30 並不是
> dual-branch encoder 能解的問題。因此主線改走 Path B：用 occ→map 正向遷移衝破自己的
> 48.34，並主打「occ 領先 + map 具競爭力」的乾淨 2-task 故事。本文件附帶兩個已完成、已
> 驗證的 encoder 實驗。

---

## 1. 動機與關鍵判斷

### 1.1 起點問題

ProtoOcc + BEVFusion-style CNN seg head 的 map 成績一直追不上 MAESTRO，懷疑是
dual-branch encoder 沒做出夠好的 occ-specific / map-specific feature。為此做了多輪
encoder / head 改造（adapter、map_hfm、TGDE、CFV-TSFG、VAMI…），但 map 始終卡在
45–47 區間。

### 1.2 所有相關實驗的並排（R50，single-frame，nuScenes）

| 設定 | Occ mIoU | Map mIoU | 備註 |
| --- | ---: | ---: | --- |
| CNN map neck, `map_loss_weight=1` | 39.82 | 39.94 | naive MTL |
| CNN map neck, `map_loss_weight=4` | 39.72 | 45.79 | **strong baseline** |
| overlay dynamic V3 | 39.52 | 46.00 | loss-level balancing |
| map_hfm（注入 voxel 幾何 residual） | 39.55 | 46.61 | +0.8 |
| MapResidualAdapter（2D zero-init refine） | 39.64 | 46.70 | +0.9 |
| CFV proto-TSFG（抄 MAESTRO TSFG） | 39.54 | **45.27** | **掉分 ❌** |
| VAMI（voxel-aware map ingest） | 39.56 | 46.04 | +0.2 |
| TGDE HTG+MTE cat_z | 38.32 | 37.69 | 只到 epoch5，~e6 放棄；weighted 從未訓練 |
| **map-only 上限** | – | **48.34** | 現行 map pipeline 天花板 |
| BEVFusion R50 (STL) | – | 47.10 | 外部單任務參考 |
| **MAESTRO R50 (3-task)** | 38.60 | **51.30** | 外部多任務參考 |

兩個事實：

1. 每一個 2-task 變體（含抄得最像 MAESTRO 的 CFV-TSFG）都落在 45.3–46.7，**沒有任何一個
   逼近 map-only 48.34，更別說 51.30**。
2. **你的 map-only 48.34 已經贏 BEVFusion R50 STL 的 47.10；你的 occ 39.6 也贏
   MAESTRO 的 occ 38.6。**

### 1.3 MAESTRO 為什麼是 51.30（讀 paper Figure 1 + abstract）

MAESTRO 是 **3 個 task（Detection + Map + Occupancy）一起訓練**：

```text
Det:  STL 42.5  Baseline-MTL 38.2  MAESTRO 43.2  (+5.0 NDS)
Map:  STL 48.3  Baseline-MTL 43.5  MAESTRO 51.3  (+7.8 mIoU)
Occ:  STL 37.4  Baseline-MTL 36.0  MAESTRO 38.6  (+2.6 mIoU)
```

機制（CPG / TSFG / SPA）：

```text
CPG: 類別分成 foreground / background 兩組 prototype
  foreground prototype -> detection
  background prototype -> map
  兩組都 -> occupancy
TSFG: 用這些 prototype 做 task-specific enhance / suppress
SPA:  用 detection head + map head 的輸出去強化 occ 的 prototype
```

**關鍵推論**：MAESTRO 的 Map 51.30 > 它自己的單任務 STL 48.3，高出 3.0。
「多任務 map 超過單任務 map」只可能來自 **positive cross-task transfer**，不可能來自
encoder decoupling。而這個 transfer 來自第三個 task（detection）的 foreground prototype
+ SPA cross-head 訊號。

這也解釋了**為什麼 CFV-TSFG 抄了反而掉分**：TSFG 不是獨立模組，它吃的是 CPG 從
detection 分出來的 fg/bg prototype + SPA cross-head 訊號。把 TSFG 單獨搬到
2-task(Occ+Map) 等於搬空殼。

### 1.4 結論（決定 Path B）

- MAESTRO 對你的 Map 差距（48.34 → 51.30，+2.96）**不是 dual-branch encoder 能補的**，
  本質是「3-task + cross-task prototype transfer」。
- **在 2-task 下，~48.34 基本就是天花板。**
- 差距集中在 thin/structured 類別（你 vs MAESTRO）：`stop_line −7.5`、`divider −7.6`、
  `carpark −5.4`、`walkway −3.9`、`ped_crossing −3`；**drivable 已打平（~80.3）**。

### 1.5 方向選擇

提供三條路，已選 **Path B**：

| Path | 內容 | 取捨 |
| --- | --- | --- |
| A | 2-task 打公平戰 + 主打 occ 領先 ✅ **選這條** | 接受 ~48.34 天花板；用 occ→map 正向遷移衝破自己的 48.34；主打 occ 39.6 > 38.6、架構更簡單。乾淨、不算抄、可達成度高 |
| B | 加第三個 task(detection) 硬追 51.3 | 可能逼近 51.3，但工程量大、且很難說「沒抄 MAESTRO」 |
| C | 只收 encoder residual 紅利、不追 | 收 map_hfm+adapter 到 ~47.x 當定稿，時間轉去寫論文 |

> 命名沿用對話：選定的這條在討論中稱為「Path B」。

---

## 2. Path B 的論文故事骨架

```text
We target a clean 2-task (occupancy + BEV map) framework rather than MAESTRO's
3-task design. Our occupancy already surpasses MAESTRO R50 (39.6 vs 38.6) and
the original ProtoOcc, while our map segmentation is competitive with the
single-task upper bound. Instead of detection-driven cross-task prototypes, we
use occupancy geometry as the positive-transfer source to push multi-task map
above its single-task ceiling, with task-specific feature capacity added as
zero-init residuals on top of the co-trained shared BEV trunk.
```

兩個明確賣點：

1. **Occ 領先**：39.6 > MAESTRO 38.6，> 原始 ProtoOcc。
2. **Map 競爭力 + 正向遷移**：用 occ→map（幾何）把 2-task map 推過自己的 map-only 48.34。

---

## 3. 一個已確立的經驗規律（決定實作風格）

把四個 encoder 實驗連起來：

```text
adapter（在 shared map feature 上加 residual）           -> +0.9
map_hfm（在 shared map feature 上加 voxel 幾何 residual） -> +0.8
TGDE MTE（用 map-private backbone 取代 shared feature）   -> epoch5 就落後
detach_map_feature=True（切斷 shared trunk）             -> 16.49 崩盤
```

> **規律：在 occ co-trained 的 shared map feature 上「加」task-specific residual 會贏；
> 用 from-scratch 的 map-private branch「取代」shared feature 會輸。** occ co-trained 的
> shared BEV 對 map 是強 pretrained 資產，不只是干擾。

因此本階段所有 encoder 改動都採「**zero-init residual、可開關、可 ablate、不破壞既有
forward**」原則。

---

## 4. 已完成的實作

### 4.1 Task #1：合併 map_hfm + MapResidualAdapter

兩個各自 +0.9 的模組，機制正交（map_hfm 注入新幾何資訊；adapter 做 2D refine），先前
在不同 branch、從未一起跑過。本次把它們串成單一 map path。

Tensor path：

```text
multi_scale_bev
  -> map_hfm   (注入 voxel-branch 幾何: mean+max over Z, zero-init residual)
  -> adapter   (PerScaleMapResidualAdapter, 2D zero-init refine)
  -> map_bev_encoder_neck
  -> map_bev_feature [B,128,H,W]
```

兩者皆 zero-init，step-0 等同 `map_loss_weight=4` baseline。

### 4.2 Task #2：MTE-weighted map-only 診斷

把 `MapOnly_BEV_Encoder` 的「固定 cat-Z flatten + 共享 backbone」換成 map-private 的
`MapTopologyEncoder`（**learned softmax-Z pooling**），backbone 深度/寬度/stride 與
baseline 完全一致，唯一差別 = learned-Z vs cat-Z。

```text
LSS voxel X [B,80,16,H,W]
  -> MapTopologyEncoder
       weighted_pool:  z_weight = softmax_Z(Conv3d(X)); X_bev = sum_Z(X * z_weight)
       -> 1x1 Conv2d(80 -> 160)
       -> CustomBEVBackbone(num_channels=[160,320,640], num_layer=[2,2,2], stride=[2,2,2])
  -> map_bev_encoder_neck (Custom_FPN_LSS)
  -> map_bev_feature [B,128,H,W]
```

---

## 5. 改動的檔案

### 新增檔案

| 檔案 | 用途 |
| --- | --- |
| `projects/mmdet3d_plugin/models/backbones/task_modules/map_topology_encoder.py` | `MapTopologyEncoder`（weighted_pool / cat_z 兩種 Z projection），從 TGDE branch 移植 |
| `projects/mmdet3d_plugin/models/backbones/task_modules/__init__.py` | 匯出 `MapTopologyEncoder` |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter.py` | Task #1 合併實驗 config |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_map_only_mte_weighted.py` | Task #2 map-only 診斷 config |

### 修改檔案

| 檔案 | 改動 |
| --- | --- |
| `projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py` | 新增 `MapHFMFusionLayer`、`Dual_Branch_Encoder` 加 `use_map_hfm`/`map_hfm_lower_source`/`map_hfm_with_cp` 接線（map path 串 hfm→adapter）、新增 `MapOnly_MTE_Encoder` |
| `projects/mmdet3d_plugin/models/backbones/__init__.py` | 匯入 task_modules 觸發 `MapTopologyEncoder` 註冊；`__all__` 補 `MapOnly_MTE_Encoder`/`MapTopologyEncoder` |

### 刻意不改

- 不改 voxel branch / HFM occ 主幹、cnn3d_decoder、PQD、BEVSegHead。
- `use_map_hfm` 預設 False，既有 baseline / adapter-only / map_hfm 行為完全不變（已回歸驗證）。

---

## 6. 驗證紀錄

| 檢查 | 結果 |
| --- | --- |
| `py_compile`（encoder / MTE / __init__） | 通過 |
| 載入 `..._hfm_adapter.py` config | 通過（`use_map_hfm=True`、adapter=PerScaleMapResidualAdapter、`map_loss_weight=4`） |
| Task #1 encoder forward smoke（`[1,80,16,16,16]`） | 通過，3 輸出 `(1,48,16,16,16)`/`(1,48,16,16)`/`(1,128,16,16)`；hfm 24/24/192 與 adapter 160/320/640 channel 對齊 |
| 載入 `..._map_only_mte_weighted.py` config | 通過（`MapOnly_MTE_Encoder` + `weighted_pool`） |
| Task #2 encoder forward smoke（`[1,80,16,32,32]`） | 通過，輸出 `(1,128,32,32)` |
| 既有 adapter-only config 回歸 | 通過（`use_map_hfm` absent → default False，行為不變） |

> 註：完整 detector build 在純 CPU 上仍會卡在既有 PQD 的 `.cuda()`（與本次改動無關），故只在
> encoder 層級做 forward smoke。

---

## 7. 執行方式

```bash
# Task #1：合併 residual（目標：逼近 map-only 48.34）
bash tools/dist_train.sh \
  projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter.py <num_gpus>

# Task #2：MTE-weighted map-only 診斷（目標：判斷天花板能否被墊高）
bash tools/dist_train.sh \
  projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_map_only_mte_weighted.py <num_gpus>

# 評測（沿用既有慣例）
bash tools/dist_test.sh <config> <work_dir>/epoch_24_ema.pth 1 --eval miou map-miou
```

---

## 8. 實驗計劃與判讀

### Stage 1：Task #1 合併

| 設定 | 預期 |
| --- | --- |
| baseline weight4 | Occ 39.72 / Map 45.79 |
| map_hfm | 39.55 / 46.61 |
| MapResidualAdapter | 39.64 / 46.70 |
| **hfm + adapter（本實驗）** | Map 目標 > 46.7，逼近 48.34；Occ 不低於 39.5 |

判讀：

- 兩者都在「同一張 map BEV feature」上加 residual，可能 **sub-additive**（不會剛好
  46.7 + 46.6）。能穩定 > 46.7 並往 48.34 靠就算成功。
- 若 Map 反而低於單獨任一個，代表兩個 residual 在搶同一份預算 → 之後改成串接不同層級
  （hfm 在前段、adapter 在後段）或只保留較強的一個。

### Stage 2：Task #2 map-only 診斷（Path B 的關鍵分叉）

| 結果 | 解讀 | 下一步 |
| --- | --- | --- |
| map-only Map **> 48.34** | learned-Z map-private backbone 能墊高天花板 | 把 map-private 分支接成 2-task 的 occ→map 並聯 |
| map-only Map **≤ 48.34** | 單一 weighted-Z 丟掉太多高度細節（cat-Z 保留全 16 層，weighted 只留一個加權和） | 放棄 map-private replace，occ→map 改走 prior/條件化，不做 feature replace |

建議補一個 **cat_z parity 對照**（`z_projection='cat_z'`、`z_channels=16`）驗證 MTE wrapper
本身能重現 ~48.34，避免把「wrapper 寫壞」誤判成「learned-Z 沒用」。

---

## 9. Path B 主菜（這兩個只是收紅利 / 診斷）

合併 residual 的上限是 48.34。**Path B 要贏，核心賭注是 occ→map 正向遷移，把多任務 map
推過自己的 48.34**（與 MAESTRO 用 detection 墊高 map 同邏輯，但來源換成 occ 幾何，故不算抄）。
載體是既有 `feat/LGMG` branch（目前 O2M 仍停在 ~46）。

接續方向由 Stage 2 結果決定：

- 若 MTE-weighted 破 48.34 → map-private 分支有料，接成 occ→map 並聯（zero-init、source detach）。
- 若沒破 → occ→map 用「幾何 prior / layout 條件化」而非 feature replace，把 LGMG 的 O2M 做對。

---

## 10. 風險

1. **合併 sub-additive**：兩個 residual 搶同一份預算，加總效益小於各自。需 Stage 1 確認。
2. **MTE weighted-Z 丟資訊**：weighted_pool 只留一個加權高度組合，可能輸給 cat-Z。需 cat_z parity 對照。
3. **天花板誤判**：把 2-task 的天花板 48.34 當成「encoder 沒做好」會持續白做 encoder 變體；Path B 已明確把主戰場移到 occ→map transfer。
4. **occ 不能掉**：Path B 的賣點是 occ 領先，任何 map-side 改動讓 Occ 明顯低於 39.5 都要回退。

---

## 11. 之前的失敗教訓（支持本計劃的設計選擇）

| 失敗案例 | 結果 | 教訓 → 對本計劃的影響 |
| --- | --- | --- |
| `detach_map_feature=True` | Occ 38.33 / Map 16.49 | 不可切斷 shared trunk → 所有改動保留 co-training |
| CFV proto-TSFG（抄 MAESTRO） | Map 45.27（掉分） | TSFG 在 2-task 是空殼 → 不抄 TSFG，改走 occ→map transfer |
| TGDE MTE replace（cat_z） | epoch5 Map 37.69，放棄 | replace shared feature 起步太慢 → encoder 用 residual-add，不 replace |
| GT-soft | Map 32.40 | train-test mismatch → 不用 GT 直接改 feature |
| 只調 `map_loss_weight` | 45.79，離 48.34 仍差 2.55 | loss-level 不足 → 需 feature-level / cross-task |

---

## 12. 相關文件

- `task_specific_geometry_decomposition_encoder_plan.md` / `tgde_module_plans/`：TGDE 完整計畫（HTG+MTE 已實作、03–07 僅 plan；`MapTopologyEncoder` 由此移植）。
- `layout_geometry_mutual_guidance_plan.md`：LGMG，Path B 主菜的載體。
- `overlay_aware_map_balancing_plan.md`：map-side loss-level 貢獻。
- `map_specific_feature_protection_plan.md`：head-level MSFP（與本文件的 encoder-level 改動不同層）。
- `map_only_upper_bound_experiment.md`：48.34 天花板的由來。
