# Map-HFM Clean Weight4 Ablation 實作計畫

## Map-HFM 是做什麼的
Map-HFM 是放在 `Dual_Branch_Encoder` 裡的 map-specific feature path。它不是改 occupancy HFM，也不是改 PQD；它的目標是把 DBE voxel branch 中間特徵的 3D 幾何訊息，轉成 BEV map 可用的多尺度特徵，再餵給現有 `map_bev_encoder_neck`。

做法是：
- 從 `vox_res / vox1 / vox3` 取 voxel 中間特徵。
- 沿 Z 軸做 `mean_z` 和 `max_z`，concat 成 BEV 表徵。
- 和對應尺度的 `multi_scale_bev` 融合。
- 輸出 channel 與 spatial level 都和原本 `multi_scale_bev` 相同的 3-level feature list。
- 讓現有 128ch map neck 可以原封不動重用。

核心原則：Map-HFM 只改善 map branch 的輸入特徵，不改 loss、不改 map head、不改 dataset、不改 evaluation、不動 occupancy/PQD 主路徑。

## 修改檔案
本輪是計畫模式，尚未修改任何檔案。實作時預計修改/新增：

- 修改：`projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py`
- 新增：`projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_weight4.py`
- 新增：`projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_weight4_map_hfm.py`

不新增第一版 V3-lite composition config。`ProtoOcc_multi_cnn_head_map_neck_map_hfm_v3_lite.py` 不作為第一個 ablation；之後只有 A1 有希望時才考慮。

## 實作重點
在 `Dual_Branch_Encoder` 加參數：

```python
use_map_hfm=False
map_hfm_lower_source='vox3'
map_hfm_with_cp=None
```

新增內部 module `MapHFMFusionLayer`：

- 輸入：`bev_i` 和對應 voxel feature。
- voxel layout 使用 early DBE layout：`[B, C, Z, H, W]`。
- Z collapse：
  ```python
  v_bev = torch.cat([v.mean(dim=2), v.max(dim=2).values], dim=1)
  ```
- resize `v_bev` 到對應 BEV level spatial size。
- concat 後 ConvBNReLU 融合。
- 必須 residual 且 identity-safe：
  ```python
  out = bev_i + delta
  ```
- final projection zero-init，確保剛初始化時 `out == bev_i`，不破壞 baseline 起點。
- 不 hard-code channel，從既有 config channel 推導。

標準 shape flow：

- `multi_scale_bev[0]`: `[B,160,100,100]`
  - `vox_res`: `[B,24,16,200,200]`
  - collapse: `[B,48,200,200]`
  - resize: `[B,48,100,100]`
  - output: `[B,160,100,100]`

- `multi_scale_bev[1]`: `[B,320,50,50]`
  - `vox1`: `[B,24,8,100,100]`
  - collapse: `[B,48,100,100]`
  - resize: `[B,48,50,50]`
  - output: `[B,320,50,50]`

- `multi_scale_bev[2]`: `[B,640,25,25]`
  - default lower source `vox3`: `[B,192,4,50,50]`
  - collapse: `[B,384,50,50]`
  - resize: `[B,384,25,25]`
  - output: `[B,640,25,25]`

之後：
```python
map_source = map_hfm_features if use_map_hfm else multi_scale_bev
map_bev = self.map_bev_encoder_neck(map_source)
map_bev_feature = map_bev[0]
```

回傳格式維持：
```python
comprehensive_voxel_feature, bev_feature, map_bev_feature
```

## Detach 與 Checkpoint 規則
`detach_map_feature=True` 時：

- detach raw sources：
  - `multi_scale_bev`
  - Map-HFM 使用的 voxel intermediates
- 不 detach Map-HFM outputs，否則 Map-HFM 參數不會收到 gradient。

Checkpoint helper：

```python
use_checkpoint = (
    self.training and
    self.map_hfm_with_cp and
    any(t.requires_grad for t in inputs if torch.is_tensor(t))
)
```

只有 `use_checkpoint=True` 才用 `cp.checkpoint`，否則正常 forward。這避免 detached inputs 下 Map-HFM 參數 gradient 被吃掉。

## Config 策略
新增 clean weight4 baseline：

```python
# projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_weight4.py
_base_ = ['./ProtoOcc_multi_cnn_head_map_neck.py']

model = dict(
    map_loss_weight=4.0)
```

新增第一個 Map-HFM ablation：

```python
# projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_weight4_map_hfm.py
_base_ = ['./ProtoOcc_multi_cnn_head_map_neck_weight4.py']

model = dict(
    dual_branch_encoder=dict(
        use_map_hfm=True,
        map_hfm_lower_source='vox3',
        detach_map_feature=False))
```

不得設定或 override：

```python
dynamic_overlay_gamma
dynamic_overlay_max_weight
map_loss_balance_mode
overlay_class_indices
dynamic_overlay_ref_pos_ratio
```

## 測試
最小測試：

- config parse：
  - `ProtoOcc_multi_cnn_head_map_neck.py`
  - `ProtoOcc_multi_cnn_head_map_neck_weight4.py`
  - `ProtoOcc_multi_cnn_head_map_neck_weight4_map_hfm.py`
- synthetic encoder smoke：
  - `use_map_hfm=False` shape 不變。
  - `use_map_hfm=True` shape 不變。
- identity-safe check：
  - 初始化時 Map-HFM output 應等於 raw `multi_scale_bev`。
- detach check：
  - `detach_map_feature=True` 時 raw source 無 gradient，Map-HFM 參數仍有 gradient。
- one-batch train smoke：
  - loss keys 不新增。
  - map feature size 對齊 `gt_masks_bev`。
  - PQD / occupancy output 正常。

## Ablation 表
A0 clean baseline：

- Config: `ProtoOcc_multi_cnn_head_map_neck_weight4.py`
- `use_map_hfm=False`
- `map_loss_weight=4.0`
- Reference: Occ 39.72 / Map 45.79

A1 first Map-HFM ablation：

- Config: `ProtoOcc_multi_cnn_head_map_neck_weight4_map_hfm.py`
- `use_map_hfm=True`
- `map_hfm_lower_source='vox3'`
- `detach_map_feature=False`
- `map_loss_weight=4.0`

A2 later only if A1 is promising：

- Overlay dynamic V3-lite + Map-HFM
- 這是 composition，不是第一個 clean architecture ablation。
