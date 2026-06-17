# Map-TSFG Supervised Suppression — 實作規格(給 Codex 執行)

Repo: `mapocc`,branch: `aaaaaaaaaaaaaaa`(以下所有路徑以 repo root 為準)

## 0. 背景與目標

目前最強的 multi-task 設定(`...active_enhance_gate_bevfusion_aligned_mapw8`,map mean 0.4574 / occ 39.61)相對 map-only(0.4909)有 ~3.4 點的任務衝突損失。MAESTRO 的 TSFG ablation 顯示「prototype 引導的 enhancement + **被 GT 監督的乘法 suppression**」在 map 上值 +3.8 mIoU。

本 repo 之前試過的三個相關模組都缺了關鍵成分,**這次要補的就是這些**:

| 已試過的模組 | 缺什麼 |
|---|---|
| `MapNeckQueryTSFG` | 無監督 suppression、zero-init **加法**殘差(只能加資訊不能移除干擾)、prototype detach |
| `CFVPrototypeTSFGMapFusion` | 同上(detach + 加法殘差 + 無監督) |
| `bev_seg_head` 的 `active_enhance_gate` | gate 下限 1.0(只能增強不能抑制),且在 head **尾端**(decoded feature 之後) |

新模組 `MapTSFGSuppression` 的三個關鍵差異:
1. **乘法 gate**:`F_out = F_tilde * sigmoid(score)`,score 可以把區域壓到 0。
2. **監督式 suppression**:score map 用 focal loss 對齊「6 類 map GT 的 union」。
3. **預設不 detach**:讓 map 的 suppression 梯度回流塑形共享特徵與 PQD query(detach 留作 ablation 開關)。

插入點:`ProtoOccCnnSegHead` 既有的 `voxel_aware_map_ingest` hook(map neck 輸出的 128-ch BEV feature 之後、`bev_seg_head` 之前)。這個 hook 已經有 PQD query 的 plumbing(`requires_pqd_query_info` → `query_norm_real_bqc` 等),只需要補「傳入 `gt_masks_bev` + 回收 aux loss」。

**不要動** dual_branch_encoder、bev_seg_head、PQD 的任何現有行為。所有改動必須對未配置新模組的舊 config 完全向後相容。

---

## 1. 變更清單

| 動作 | 檔案 |
|---|---|
| 新增 | `projects/mmdet3d_plugin/models/task_modules/map_tsfg_suppression.py` |
| 修改 | `projects/mmdet3d_plugin/models/task_modules/__init__.py`(註冊) |
| 修改 | `projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py`(hook 傳 GT + 回收 loss,兩個 call site) |
| 新增 | `projects/configs/ProtoOcc/ProtoOcc_..._bevfusion_aligned_mapw8_tsfg_supp.py`(實驗 A) |
| 新增 | `projects/configs/ProtoOcc/ProtoOcc_..._bevfusion_aligned_mapw8_tsfg_full.py`(實驗 B) |
| 新增 | `projects/configs/ProtoOcc/ProtoOcc_..._bevfusion_aligned_mapw8_tsfg_full_detachq.py`(實驗 C) |

(config 完整檔名見第 4 節,沿用 repo 命名慣例。)

---

## 2. 新模組:`map_tsfg_suppression.py`

完整檔案內容如下。介面刻意模仿 `MapNeckQueryTSFG`(同一個 hook、同樣吃 `**query_info`),但 forward 回傳 `(feature, losses_dict)`,並以 `returns_aux_losses = True` 讓 detector 辨識。

```python
"""Map-TSFG supervised suppression.

MAESTRO-style task-specific feature generation for the map branch:
prototype-wise + prototype-aware enhancement, followed by a GT-supervised
multiplicative suppression gate. Unlike the earlier MapNeckQueryTSFG /
CFVPrototypeTSFGMapFusion adapters, this module
  (1) gates multiplicatively (can remove occ interference, not just add),
  (2) supervises the suppression score with focal loss against the union of
      map GT masks, and
  (3) keeps gradients flowing into the PQD queries / shared features by
      default (detach_prototypes is an ablation switch only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class MapTSFGSuppression(BaseModule):

    # Detector-side protocol flags.
    returns_aux_losses = True

    def __init__(self,
                 mode='full',                       # 'full' | 'suppress_only'
                 map_channels=128,
                 query_channels=48,
                 occ_num_classes=18,
                 background_class_ids=(11, 12, 13, 14, 15, 16),
                 prototype_source='query_norm_real_bqc',
                 detach_prototypes=False,
                 hidden_channels=128,
                 enhance_residual=True,             # full mode: F_tilde = x + zero-init conv(...)
                 score_hidden_channels=64,
                 score_init_bias=2.0,               # sigmoid(2.0) ≈ 0.88 at start
                 supp_loss_weight=1.0,
                 focal_gamma=2.0,
                 focal_alpha=-1.0,
                 gt_dilation=0,                     # px; >0 dilates union GT (protects thin classes)
                 loss_name='loss_map_supp',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert mode in ('full', 'suppress_only')
        if prototype_source not in ('query_norm_real_bqc',
                                    'query_embed_real_bqc',
                                    'mask_embed_real_bqc'):
            raise ValueError(f'bad prototype_source: {prototype_source!r}')

        self.mode = mode
        self.map_channels = map_channels
        self.query_channels = query_channels
        self.occ_num_classes = occ_num_classes
        self.background_class_ids = list(background_class_ids)
        self.prototype_source = prototype_source
        self.detach_prototypes = bool(detach_prototypes)
        self.enhance_residual = bool(enhance_residual)
        self.supp_loss_weight = float(supp_loss_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.gt_dilation = int(gt_dilation)
        self.loss_name = loss_name

        # Only request PQD query plumbing when we actually use prototypes.
        self.requires_pqd_query_info = (mode == 'full')

        num_bg = len(self.background_class_ids)
        if mode == 'full':
            self.prototype_proj = nn.Sequential(
                nn.Linear(query_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, map_channels),
            )
            self.prototype_gate = nn.Sequential(
                nn.Linear(2 * map_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, map_channels),
                nn.Sigmoid(),
            )
            self.enhance_conv = nn.Sequential(
                nn.Conv2d(map_channels + num_bg, map_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(map_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(map_channels, map_channels, kernel_size=1,
                          bias=True),
            )
            if self.enhance_residual:
                nn.init.zeros_(self.enhance_conv[-1].weight)
                nn.init.zeros_(self.enhance_conv[-1].bias)

        self.score_predictor = nn.Sequential(
            nn.Conv2d(map_channels, score_hidden_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(score_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(score_hidden_channels, 1, kernel_size=1, bias=True),
        )
        # Smooth start: spatially-uniform gate at sigmoid(score_init_bias).
        nn.init.zeros_(self.score_predictor[-1].weight)
        nn.init.constant_(self.score_predictor[-1].bias, score_init_bias)

    # ------------------------------------------------------------------
    # prototype helpers (full mode only)
    # ------------------------------------------------------------------
    def _get_bg_prototypes(self, query_info):
        q = query_info.get(self.prototype_source)
        if q is None:
            raise ValueError(
                f'{self.prototype_source} missing; the detector must pass '
                'PQD query_info (requires_pqd_query_info).')
        if q.dim() != 3 or q.shape[1] < self.occ_num_classes:
            raise ValueError(
                f'{self.prototype_source} must be [B, >= {self.occ_num_classes}, C], '
                f'got {tuple(q.shape)}')
        q = q[:, self.background_class_ids, :]
        if self.detach_prototypes:
            q = q.detach()
        return self.prototype_proj(q)          # (B, N_bg, map_channels)

    def _enhance(self, x, p):
        # Prototype-wise: raw dot product (MAESTRO Fig.4), no cosine norm.
        f_wise = torch.einsum('bnc,bchw->bnhw', p, x)
        # Prototype-aware: channel scaling from pooled prototypes.
        gamma = self.prototype_gate(
            torch.cat([p.mean(dim=1), p.max(dim=1).values], dim=-1))
        f_aware = x * gamma[:, :, None, None]
        delta = self.enhance_conv(torch.cat([f_wise, f_aware], dim=1))
        f_tilde = x + delta if self.enhance_residual else delta
        return f_tilde, f_aware

    # ------------------------------------------------------------------
    # suppression loss
    # ------------------------------------------------------------------
    def _make_supp_target(self, gt_masks_bev, like_logits):
        # gt_masks_bev: (B, num_map_classes, H, W) binary -> union (B,1,H,W)
        target = gt_masks_bev.float().amax(dim=1, keepdim=True)
        if self.gt_dilation > 0:
            k = 2 * self.gt_dilation + 1
            target = F.max_pool2d(target, kernel_size=k, stride=1,
                                  padding=self.gt_dilation)
        if target.shape[-2:] != like_logits.shape[-2:]:
            target = F.interpolate(target, size=like_logits.shape[-2:],
                                   mode='nearest')
        return target

    def _focal_loss(self, logits, target):
        logits = logits.float()
        target = target.float()
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target,
                                                reduction='none')
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        loss = ce * ((1.0 - p_t) ** self.focal_gamma)
        if self.focal_alpha >= 0:
            alpha_t = (self.focal_alpha * target
                       + (1.0 - self.focal_alpha) * (1.0 - target))
            loss = alpha_t * loss
        return loss.mean()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self,
                map_feature,
                voxel_feature=None,        # unused; kept for hook signature
                gt_masks_bev=None,
                **query_info):
        if map_feature.dim() != 4 or map_feature.shape[1] != self.map_channels:
            raise ValueError(
                f'expected map_feature [B,{self.map_channels},H,W], '
                f'got {tuple(map_feature.shape)}')

        if self.mode == 'full':
            f_tilde, f_aware = self._enhance(
                map_feature, self._get_bg_prototypes(query_info))
            score_logits = self.score_predictor(f_aware)
        else:
            f_tilde = map_feature
            score_logits = self.score_predictor(map_feature)

        out = f_tilde * torch.sigmoid(score_logits)

        losses = {}
        if self.training and gt_masks_bev is not None:
            target = self._make_supp_target(gt_masks_bev, score_logits)
            losses[self.loss_name] = (
                self.supp_loss_weight * self._focal_loss(score_logits, target))
        return out, losses
```

`__init__.py` 加上:

```python
from .map_tsfg_suppression import MapTSFGSuppression
```
並加入 `__all__`。

---

## 3. Detector patch:`ProtoOccCnnSegHead.py`

### 3.1 改寫 `_apply_voxel_aware_map_ingest`

現行版本回傳單一 feature。改成統一回傳 `(map_feature, aux_losses)`,以 `returns_aux_losses` 屬性區分新舊模組,**舊模組行為不變**:

```python
def _apply_voxel_aware_map_ingest(self,
                                  map_feature,
                                  voxel_feature,
                                  query_info=None,
                                  gt_masks_bev=None):
    if self.voxel_aware_map_ingest is None:
        return map_feature, {}
    query_info = query_info or {}
    module = self.voxel_aware_map_ingest
    returns_aux = getattr(module, 'returns_aux_losses', False)

    def _run(feat):
        if returns_aux:
            return module(
                map_feature=feat,
                voxel_feature=voxel_feature,
                gt_masks_bev=gt_masks_bev,
                **query_info)
        return module(
            map_feature=feat,
            voxel_feature=voxel_feature,
            **query_info), {}

    if isinstance(map_feature, dict):
        updated = dict(map_feature)
        new_feat, aux = _run(self._get_map_feature_tensor(map_feature))
        updated['bev_feature'] = new_feat
        return updated, aux
    return _run(map_feature)
```

### 3.2 兩個 call site

`forward_train`(注意:呼叫點在 `gt_masks_bev = self._normalize_map_targets(...)` **之後**,直接傳 normalize 過的 tensor):

```python
map_feature, ingest_losses = self._apply_voxel_aware_map_ingest(
    map_feature, occ_voxel_feature, query_info,
    gt_masks_bev=gt_masks_bev)
...
map_losses = self.bev_seg_head.loss(bev_seg_logits, gt_masks_bev)
losses.update(self._scale_map_losses(map_losses))
losses.update(ingest_losses)   # 注意:supp loss 不乘 map_loss_weight,
                               # 權重由模組自己的 supp_loss_weight 控制
```

`simple_test`:

```python
map_feature, _ = self._apply_voxel_aware_map_ingest(
    map_feature, occ_voxel_feature, query_info)
```

**重要**:suppression gate 是推論路徑的一部分(不是只在訓練時開),`simple_test` 必須同樣經過模組。

`ProtoOccMultitask.py` 不需要改(本實驗的 config 鏈都走 `ProtoOccCnnSegHead`)。

---

## 4. 三個實驗 config

全部繼承目前最強的 aligned 設定。基底:
`projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8.py`

### A. `..._bevfusion_aligned_mapw8_tsfg_supp.py` — 只加監督式 suppression(最乾淨的新成分隔離)

```python
_base_ = [
    './ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8.py'
]

# Supervised multiplicative suppression gate at the map-neck output
# (the ingredient missing from all previous adapter attempts).
model = dict(
    voxel_aware_map_ingest=dict(
        type='MapTSFGSuppression',
        mode='suppress_only',
        map_channels=128,
        score_init_bias=2.0,
        supp_loss_weight=1.0,
        focal_gamma=2.0,
        gt_dilation=0))
```

### B. `..._bevfusion_aligned_mapw8_tsfg_full.py` — 完整 TSFG(enhancement + suppression),prototype 不 detach

```python
_base_ = [
    './ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8.py'
]

model = dict(
    voxel_aware_map_ingest=dict(
        type='MapTSFGSuppression',
        mode='full',
        map_channels=128,
        query_channels=48,
        occ_num_classes=18,
        background_class_ids=(11, 12, 13, 14, 15, 16),
        prototype_source='query_norm_real_bqc',
        detach_prototypes=False,        # 關鍵差異:梯度回流
        enhance_residual=True,          # enhancement 走 zero-init 殘差,保護主路徑
        score_init_bias=2.0,
        supp_loss_weight=1.0,
        focal_gamma=2.0,
        gt_dilation=0))
```

### C. `..._bevfusion_aligned_mapw8_tsfg_full_detachq.py` — 同 B,但 `detach_prototypes=True`

```python
_base_ = ['./<config B 檔名>.py']

# Ablation: does gradient flow into PQD queries matter, or is the
# supervised suppression alone responsible for the gain?
model = dict(
    voxel_aware_map_ingest=dict(detach_prototypes=True))
```

---

## 5. 驗證流程

### 5.1 先做的 sanity check(任何訓練之前)

1. `python -c "from projects.mmdet3d_plugin.models.task_modules import MapTSFGSuppression"` import 成功。
2. 用 config B build model(`mmcv Config.fromfile` + `build_detector`),跑一個 dummy `forward_dummy` 或 50 iter 的 1quarter smoke,確認:
   - losses 裡出現 `loss_map_supp` 且為有限值;
   - 沒有 NaN/Inf;
   - 舊 config(不含 `voxel_aware_map_ingest`)照常可訓練(向後相容)。
3. 確認 `query_norm_real_bqc` 的 shape 是 `(B, 18, 48)`(`return_query_info` 路徑在 `Prototype_Query_Decoder_nuScenes.py` 已存在,模組宣告 `requires_pqd_query_info` 後 detector 會自動開啟)。
4. 在 log 加一行(或用既有 debug 慣例)輸出 suppression score 統計:`S.mean()` 初始應 ≈0.88,訓練中應分化(map GT union 區域→1、其餘→0)。若整張圖收斂到全 1,代表 supp loss 權重太低或 focal 失衡。

### 5.2 訓練

沿用之前 full run(`nano4_h200`)完全相同的啟動方式與排程,只換 config。**不要用 1quarter smoke 來做指標決策**——既有紀錄顯示 smoke 下 ped_crossing/stop_line IoU 全為 0,thin 類在 smoke 中不可見,a0–a5 全部擠在 0.220±0.004 就是雜訊;smoke 只拿來確認不會 crash。

### 5.3 比較基準與成功標準

| Run | 對照 | 期望 |
|---|---|---|
| A(supp only) | mapw8 baseline:map 0.4574 / occ 39.61 | map ≥ +1.0;occ 不低於 39.3 |
| B(full TSFG) | 同上 | map +2~3(MAESTRO ablation 全套值 +3.8);occ 不低於 39.3 |
| C(detach) | 對照 B | 若 C ≈ B,表示增益主要來自監督 suppression 本身;若 B > C,證明梯度回流有貢獻 |

評估與報表沿用既有 `result.md` 流程(map_mean / thin avg / ped / stop / divider / OCC mIoU),特別關注 thin avg 與 stop_line(目前最弱的類)。

### 5.4 失敗時的調整順序

1. occ 明顯掉(<39.0):先試 `detach_prototypes=True`(config C);再試 `supp_loss_weight=0.5`。
2. map 沒動、`S.mean()` 貼著 1:`supp_loss_weight` 加到 2.0,或 `score_init_bias` 降到 0.0。
3. thin 類變差:`gt_dilation=1`(union GT 膨脹 1px,避免 gate 在細線邊界誤殺)。

---

## 6. 給 Codex 的注意事項

- `map_feature` 在這條 config 鏈通常是純 tensor,但 detector 有 dict 分支(`_get_map_feature_tensor`),patch 必須兩種都處理(第 3.1 節的寫法已涵蓋)。
- 不要把 `loss_map_supp` 乘進 `_scale_map_losses`(map_loss_weight=8 是 seg loss 的權重,supp loss 權重由模組自管)。
- `background_class_ids=(11..16)` 與 repo 既有模組(`CFVPrototypeTSFGMapFusion`、`MapNeckQueryTSFG`)一致,對應 Occ3D 的 drivable_surface/other_flat/sidewalk/terrain/manmade/vegetation;不要改。
- 既有的 `active_enhance_gate`(在 bev_seg_head 尾端)**保持開啟不動**,以隔離本次改動的增量;兩個 gate 位置不同不衝突。
- 改 `_apply_voxel_aware_map_ingest` 簽名後,全 repo grep 該函式名,確認沒有其他呼叫點(目前只有 `forward_train` 與 `simple_test` 兩處)。
