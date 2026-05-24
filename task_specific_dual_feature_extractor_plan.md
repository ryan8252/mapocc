# Task-Specific Dual Feature Extractor 計劃

日期：2026-05-24

## 一句話定位

> 從 LSS voxel feature 直接產生 OCC-specific voxel feature 與 Map-specific BEV feature；OCC decoder 固定使用 ProtoOcc `cnn3d_decoder + PQD`，Map decoder 固定使用 BEVFusion-style `BEVSegHead`，方法貢獻放在 encoder-side task-specific feature extraction。

這份計劃的主線不是「ProtoOcc 加一個 map head」，也不是「照抄 MAESTRO CPG+TSFG」。更準確的定位是：

```text
Multi-task perception methods can improve either the encoder or the decoders.
We keep strong task decoders unchanged and improve the encoder by generating
task-specific voxel/BEV features from LSS voxel features.
```

因此：

- OCC head：沿用 ProtoOcc 的 `cnn3d_decoder + Prototype_Query_Decoder_nuScenes`。
- Map head：沿用 BEVFusion-style `BEVSegHead`。
- `map_bev_encoder_neck`：視為 map-specific encoder neck，不是 decoder。
- 新方法：改造 `Dual_Branch_Encoder` 的 Dual Feature Extractor / feature fusion，使它輸出 task-specific features。

暫定方法名稱：

```text
Geometry-Aware Task-Specific Dual Feature Extractor (GA-TSDFE)
```

## Baseline 與比較邊界

正式比較應以目前最強、乾淨的 map-neck multitask baseline 為主：

| Setting | Occ mIoU | Map mIoU | 用途 |
| --- | ---: | ---: | --- |
| CNN head + 128ch map neck, `map_loss_weight=1` | 39.82 | 39.94 | naive MTL reference |
| CNN head + 128ch map neck, `map_loss_weight=4` | 39.72 | 45.79 | strong clean baseline |
| CNN head + 128ch map neck, map-only | - | 48.34 | map upper bound diagnostic |

第一版方法必須固定 decoder：

```text
same ProtoOcc PQD:      F_occ_new vs original CVF
same BEVSegHead:        F_map_new vs 128ch map-neck baseline
same map_loss_weight=4: fair comparison against strong MTL baseline
```

如果改了 PQD 或 BEVSegHead，方法故事會從 encoder enhancement 變成 decoder/head 改良，不利於和 MAESTRO-style 敘事對齊。

## 現有 ProtoOcc Dual Branch Encoder 的問題

目前 `Dual_Branch_Encoder` 的核心路徑：

```text
LSS voxel feature X [B, C, Z, H, W]
  |
  +-- Small-kernel voxel branch
  |     -> vox_res / vox1 / vox2 / vox3
  |
  +-- BEV branch
        -> torch.cat(X.unbind(dim=2), dim=1)
        -> down_sample_for_3d_pooling
        -> CustomBEVBackbone
        -> multi_scale_bev
```

之後 `multi_scale_bev` 被三個地方使用：

```text
1. HFM / BEV-to-voxel fusion:
   multi_scale_bev -> bev_ch1/2/3 -> fixed addition into voxel branch

2. OCC residual lift:
   multi_scale_bev -> bev_encoder_neck -> voxelize_module -> CVF

3. Map branch:
   multi_scale_bev -> map_bev_encoder_neck -> map_bev_feature
```

主要弱點：

1. **Z-collapse 是 fixed concat**
   `torch.cat(X.unbind(dim=2), dim=1)` 把所有 height bins 當 channel 攤平。Map 和 OCC 需要的高度訊息不同，但現在沒有 task-specific height selection。

2. **BEV feature 是 shared**
   同一份 `multi_scale_bev` 同時給 OCC fusion 和 map neck。Map loss 與 OCC loss 都會拉同一個 BEV trunk，容易形成 shared-feature conflict。

3. **BEV-to-voxel fusion 是 fixed addition**
   現在是 `vox + bev_ch(multi_scale_bev)` 的固定融合，沒有判斷哪些 BEV context 對 OCC 有用、哪些會干擾 OCC。

4. **map neck 只是在 shared BEV 後面補救**
   `map_bev_encoder_neck` 已經能保護 map feature，但它吃到的 source 仍是 shared `multi_scale_bev`。所以它是有效 baseline，但還不是完整的 task-specific encoder。

## 方法總覽

目標是把現有 DBE：

```text
X -> Dual_Branch_Encoder -> CVF + map_bev_feature
```

改成：

```text
X -> GA-TSDFE
     -> F_occ: occ-specific comprehensive voxel feature
     -> F_map: map-specific BEV feature
```

整體 tensor path：

```text
LSS voxel feature X [B, 80, 16, 200, 200]
  |
  +-- Local voxel branch
  |     -> V_local
  |
  +-- Large-kernel BEV branch
        -> shared multi-scale BEV
        -> task-specific BEV adapters
             -> multi_scale_bev_occ
             -> multi_scale_bev_map

multi_scale_bev_occ + V_local
  -> task-specific HFM / lift
  -> F_occ [B, 48, 200, 200, 16]
  -> cnn3d_decoder + PQD
  -> occupancy prediction

multi_scale_bev_map
  -> map_bev_encoder_neck
  -> F_map [B, 128, 200, 200]
  -> BEVSegHead
  -> BEV map prediction
```

核心原則：

- 新模組只輸出 feature，不輸出 mask/logit/query prediction。
- 不修改 PQD 的 loss、RPL、query matching、mask decoding。
- 不修改 BEVSegHead 的 BCE/Dice loss。
- 所有新增 residual path 初始等價 baseline，降低第一輪實驗風險。

## 子模組 A：Task-Aware Height Reweighting

### Motivation

Map 和 OCC 從 LSS voxel feature 需要不同 height cue：

| Task | 需要的 height cue |
| --- | --- |
| Map | ground plane, road topology, lane/sidewalk boundary, low-height static layout |
| OCC | object volume, vertical structure, free space, visibility, occlusion |

原本 `torch.cat(X.unbind(dim=2), dim=1)` 沒有 task bias。它讓同一個 height representation 同時服務 Map 和 OCC。

### 設計

第一版不直接替換掉原本 z-concat，而是做可關閉的 height reweighting：

```text
X: [B, C, Z, H, W]

G_occ = HeightGate_occ(X)  # [B, 1, Z, H, W]
G_map = HeightGate_map(X)  # [B, 1, Z, H, W]

X_occ = X * (1 + G_occ)
X_map = X * (1 + G_map)

pooled_occ = cat_z(X_occ) -> Conv1x1(C*Z -> 160)
pooled_map = cat_z(X_map) -> Conv1x1(C*Z -> 160)
```

這樣保留原始 DBE 的 height-channel representation，但讓兩個 task 可以強調不同高度。

### V1 實作選項

為了降低成本，V1 可先不跑兩套完整 BEV backbone：

```text
pooled_shared = original cat_z(X) -> shared CustomBEVBackbone -> multi_scale_bev
height_context_occ/map = HeightGate(X)
multi_scale_bev_occ/map = TaskAdapter(multi_scale_bev, height_context_occ/map)
```

V2 再評估是否改成：

```text
pooled_occ -> shared-weight CustomBEVBackbone -> multi_scale_bev_occ
pooled_map -> shared-weight CustomBEVBackbone -> multi_scale_bev_map
```

V1 優先，因為可以保持大部分 DBE 行為不變。

## 子模組 B：Task-Specific Multi-Scale BEV Adapters

### Motivation

現在 `multi_scale_bev` 是一份 shared feature。Map branch 雖然有 `map_bev_encoder_neck`，但 source 還是 shared。OCC branch 也直接用同一份 `multi_scale_bev` 做 HFM 與 `voxelize_module`。

### 設計

在 `CustomBEVBackbone` 後插入 task adapters：

```text
multi_scale_bev = [F0, F1, F2]

multi_scale_bev_occ = []
multi_scale_bev_map = []

for each scale s:
    Delta_occ_s = Adapter_occ_s(F_s, height_context_occ)
    Delta_map_s = Adapter_map_s(F_s, height_context_map)

    F_occ_s = F_s + Delta_occ_s
    F_map_s = F_s + Delta_map_s
```

Adapter 建議使用輕量 residual block：

```text
Conv1x1(C -> C/r)
BN/ReLU
Depthwise large-kernel Conv(k=7 or 11)
BN/ReLU
Conv1x1(C/r -> C)
zero-init last conv
```

Step 0：

```text
Delta_occ_s = 0
Delta_map_s = 0
F_occ_s = F_s
F_map_s = F_s
```

這保證初始化時等價 baseline。

### 為什麼不是 decoder

這裡不產生 map logits、不產生 occ masks、不產生 class query。它只產生更適合 task heads 使用的 feature：

```text
encoder output: task-specific features
decoder output: final predictions
```

因此它仍然是 encoder-side enhancement。

## 子模組 C：Geometry Tokens / Registers

### Motivation

MAESTRO 的 CPG+TSFG 用 prototypes 做 task-aware feature enhancement。這個概念可以借，但不應照抄它的 foreground/background grouping，因為 OCC+Map 的關係不是三任務 detection/map/occ 的簡化版。

我們可以把 prototype 概念改成 encoder-level geometry tokens：

```text
T_ground
T_boundary
T_volume
T_free_space
T_occlusion
```

這些 token 不是 final class prototypes，也不是 PQD queries。它們是 encoder memory，用來產生 feature gates。

### V1 設計

V1 不做重 transformer。先用全域/區域 pooling 產生 geometry context：

```text
T_geo = MLP(GlobalPool(multi_scale_bev) + GlobalPool(vox_branch_feature))

gate_occ = MLP_occ(T_geo)
gate_map = MLP_map(T_geo)
```

然後作用在 task adapters：

```text
Delta_occ_s = Adapter_occ_s(F_s) * gate_occ_s
Delta_map_s = Adapter_map_s(F_s) * gate_map_s
```

### V2 設計

如果 V1 有效，再做 token-to-feature cross attention：

```text
geometry registers T_geo
  -> cross-attend to BEV feature
  -> task-specific spatial/channel gates
```

V2 才考慮 masked attention 或 register tokens，避免第一版過重。

## 子模組 D：Task-Specific OCC Fusion / Lift

### Motivation

如果只改善 map path，OCC 不會超過 ProtoOcc 原本 CVF。要讓 OCC 變強，必須讓 PQD 吃到新的 `F_occ`，而不是原本 `comprehensive_voxel_feature`。

### 設計

把 OCC path 中使用 `multi_scale_bev` 的地方全部改成 `multi_scale_bev_occ`：

```text
vox3 = vox3 + vox2 + bev_ch1(multi_scale_bev_occ[2])
vox_2 = vox_2 + vox1 + bev_ch2(multi_scale_bev_occ[1])
vox_raw = vox_res + vox_raw + bev_ch3(multi_scale_bev_occ[0])

bev_occ = bev_encoder_neck(multi_scale_bev_occ)
vox_occ = voxelize_module(bev_occ)
F_occ = vox_raw + vox_occ
```

同時 map path 使用 `multi_scale_bev_map`：

```text
F_map = map_bev_encoder_neck(multi_scale_bev_map)
```

這樣輸出邊界很清楚：

```text
F_occ: encoder feature for ProtoOcc PQD
F_map: encoder feature for BEVSegHead
```

## 與 MAESTRO / ProtoOcc 的差異

### vs ProtoOcc

ProtoOcc 的 DBE 是：

```text
small-kernel voxel branch + large-kernel BEV branch + fixed HFM
```

GA-TSDFE 是：

```text
small-kernel voxel branch + large-kernel BEV branch
  + task-aware height reweighting
  + task-specific multi-scale BEV adapters
  + task-specific OCC lift and map neck input
```

差異不是 head，而是 encoder 內的 feature extraction / fusion 從 shared fixed 變成 task-specific adaptive。

### vs MAESTRO

MAESTRO 的核心是：

```text
CPG + TSFG: prototype-guided task-specific feature enhancement
```

GA-TSDFE 的核心是：

```text
voxel-geometry-guided task-specific dual feature extraction
```

差異：

- 不使用 MAESTRO 的 foreground/background grouping 當主設計。
- 不把 map/OCC 關係簡化成 hard prototype groups。
- 從 LSS voxel geometry 直接學 task-specific split。
- 保留 ProtoOcc PQD 和 BEVFusion-style map head，凸顯 encoder contribution。

可以寫成 paper wording：

```text
Unlike MAESTRO, which enhances task features through semantic prototype groups,
our method performs task-specific feature extraction directly from LSS voxel
features, producing an occupancy-specific voxel representation and a
map-specific BEV representation before fixed task decoders.
```

## 實作計劃

### Stage 0：Baseline 對齊

目標：確認改動前比較基準一致。

- 使用 `ProtoOcc_multi_cnn_head_map_neck.py`。
- 設定 `map_loss_weight=4.0`。
- 固定 OCC decoder 與 BEVSegHead。
- 記錄 strong baseline：
  - Occ mIoU 39.72
  - Map mIoU 45.79

### Stage 1：Task Adapters after `multi_scale_bev`

最小可驗證版本。

新增 optional module：

```text
projects/mmdet3d_plugin/models/backbones/task_specific_dual_feature_extractor.py
```

或直接先在 `dual_branch_encoder.py` 內掛：

```python
task_specific_dual_feature_extractor=dict(
    type='TaskSpecificDualFeatureExtractor',
    in_channels=[160, 320, 640],
    use_height_context=False,
    zero_init=True)
```

修改 DBE forward：

```text
multi_scale_bev = bev_encoder_backbone(pooled_x)
multi_scale_bev_occ, multi_scale_bev_map = task_adapter(multi_scale_bev, x)

OCC fusion uses multi_scale_bev_occ
map_bev_encoder_neck uses multi_scale_bev_map
```

第一版先不改 `torch.cat(x.unbind(dim=2), 1)`。

### Stage 2：Task-Aware Height Reweighting

在 Stage 1 有正向跡象後加入。

新增：

```text
HeightTaskGate(X) -> height_context_occ / height_context_map
```

先作為 adapter conditioning，不直接跑兩套 BEV backbone：

```text
TaskAdapter(F_s, height_context_task)
```

如果 Stage 2 有效，再考慮 V2 的 task-specific z-collapse。

### Stage 3：Geometry Tokens / Registers

在 Stage 1/2 都有跡象後加入。

第一版使用 pooling-based geometry tokens：

```text
T_geo = MLP(GlobalPool(voxel branch + BEV branch))
```

用於產生 task-specific gates。

不要一開始就做 cross-attention transformer，否則如果失敗很難歸因。

### Stage 4：Full Task-Specific Height Split

最後再做較重版本：

```text
X_occ = X * (1 + G_occ)
X_map = X * (1 + G_map)

pooled_occ -> BEV backbone/adapter -> multi_scale_bev_occ
pooled_map -> BEV backbone/adapter -> multi_scale_bev_map
```

這階段要小心顯存與訓練時間。若 Stage 1/2 已經能超過 baseline，不一定需要 Stage 4。

## Ablation 設計

| ID | Setting | 目的 |
| --- | --- | --- |
| A0 | weight4 map-neck baseline | strong baseline |
| A1 | shared `multi_scale_bev` + map neck only | 現有方法 |
| A2 | task adapters, OCC path only | 測 `F_occ` 是否能超過 CVF |
| A3 | task adapters, Map path only | 測 `F_map` 是否能超過 map neck source |
| A4 | task adapters, OCC + Map | 主方法 |
| A5 | A4 + height context | 測 height-aware split 是否有效 |
| A6 | A5 + geometry tokens | 測 geometry gate 是否有效 |
| A7 | no zero-init residual | 驗證 baseline-equivalent initialization 是否必要 |
| A8 | detach map branch source | 測 shared gradient conflict |
| A9 | no task split, larger channels only | 排除只是 capacity 增加 |

關鍵 comparison：

```text
A2 vs A0: same PQD, only F_occ changed
A3 vs A0: same BEVSegHead, only F_map source changed
A4 vs A0: full task-specific encoder
A9 vs A4: prove task-specific split matters, not just parameters
```

## Debug 與監控指標

需要加 debug log，避免只看最終 mIoU 才知道失敗。

每隔固定 iteration 記錄：

```text
delta_occ_s_norm / F_s_norm
delta_map_s_norm / F_s_norm
gate_occ_mean/std/min/max
gate_map_mean/std/min/max
cosine(F_occ_s, F_map_s)
height_gate_occ_z_distribution
height_gate_map_z_distribution
map loss gradient norm on shared BEV trunk
occ loss gradient norm on shared BEV trunk
```

健康跡象：

- step 0 或初期 `delta_norm` 接近 0，但後續逐漸非零。
- `F_occ_s` 和 `F_map_s` 不應完全 collapse 成同一個 feature。
- `height_gate_map` 應偏向低高度 / ground-related bins，但不能全壓成單一 Z。
- `height_gate_occ` 應保留更多 vertical diversity。
- Map mIoU 上升時 OCC 不應快速掉超過 0.2。

## 成功標準

正式 epoch-24 或等價完整訓練比較：

| 指標 | 最低接受 | 理想 |
| --- | ---: | ---: |
| Occ mIoU | >= 39.72 | 40.0+ |
| Map mIoU | >= 45.79 | 46.5+ |
| Map thin classes | stop_line/divider/ped_crossing 不退 | 至少兩類上升 |
| decoder 改動 | 0 | 0 |

若結果是：

- Map 上升但 OCC 掉到 < 39.5：不能當主方法，只能當 map enhancement。
- OCC 上升但 Map 低於 45.79：沒有解決 Occ+Map MTL，不能當主線。
- 兩者都小幅上升但參數暴增：需要 A9 排除 capacity effect。
- 只有 1/4 train 有效、full train 無效：不能 claim method，需回到 gradient/debug 分析。

## 風險與對策

### 風險 1：兩個 task adapters 只是增加參數

對策：必做 A9 `larger shared channels only`，證明 task split 比單純加寬有效。

### 風險 2：Map branch 改善但 OCC 被污染

對策：

- OCC path 使用 `multi_scale_bev_occ`，Map path 使用 `multi_scale_bev_map`。
- residual zero-init。
- 保留 `detach_map_feature` ablation。

### 風險 3：Height gate collapse

對策：

- 記錄 height distribution。
- 可加 entropy regularization 作 ablation，但不要放進第一版主線。

### 風險 4：方法太像 MAESTRO

對策：paper wording 強調：

```text
MAESTRO: semantic prototype-guided task feature enhancement.
Ours: LSS voxel geometry-guided task-specific dual feature extraction.
```

不要使用 CPG / TSFG 命名，不使用 MAESTRO hard grouping 當主模組。

### 風險 5：方法太像 ProtoOcc DBE 小改

對策：主圖和 ablation 要清楚顯示：

```text
ProtoOcc DBE: shared BEV feature + fixed fusion
Ours: task-specific BEV features + task-specific OCC/map encoder outputs
```

## 建議主圖畫法

```text
Images
  -> Image Encoder + Depth + LSS
  -> LSS Voxel Feature
  -> Geometry-Aware Task-Specific Dual Feature Extractor
       |                                |
       |                                |
       v                                v
  Occ-specific Voxel Feature       Map-specific BEV Feature
       |                                |
       v                                v
  ProtoOcc CNN3D + PQD             map_bev_encoder_neck + BEVSegHead
       |                                |
       v                                v
  Occupancy Prediction             BEV Map Prediction
```

圖上不要展開 PQD 內部 RPL / prototype mining，也不要展開 BEVSegHead 的 BCE/Dice。它們是 fixed task decoders。

## 預期論文敘事

Challenge：

```text
Existing Occ+Map multitask models either share a common BEV/voxel encoder or
improve task decoders. Such shared representation is insufficient because
occupancy and map segmentation require different height and geometry cues.
```

Motivation：

```text
MAESTRO shows that task-aware feature enhancement is important, while ProtoOcc
shows that dual voxel/BEV feature extraction is effective for occupancy.
However, neither directly generates task-specific voxel/BEV representations
from LSS voxel features for camera-only OCC + BEV map segmentation.
```

Contribution：

```text
We propose a geometry-aware task-specific dual feature extractor that splits
LSS voxel features into occupancy-specific voxel features and map-specific
BEV features before fixed task decoders. This strengthens the encoder while
preserving ProtoOcc PQD and BEVFusion-style map heads.
```

## 下一步

第一個可實作版本建議只做：

```text
Stage 1: Task adapters after multi_scale_bev
```

理由：

- 不改 LSS / depth / PQD / BEVSegHead。
- 不改原本 z-concat。
- 初始等價 baseline。
- 能直接驗證 task-specific encoder split 是否有用。

若 Stage 1 連 1/4 train 都沒有改善，再做 height split 風險太高；應先回來看 gradient conflict / delta norm / task feature cosine。
