# ProtoOcc Map-Specific Feature Generator 改動計劃

日期：2026-04-21

## 實作狀態

已完成第一版 map-specific BEV neck 實作。

修改檔案：

| 檔案 | 狀態 | 內容 |
| --- | --- | --- |
| `projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py` | 已修改 | 新增 `map_bev_encoder_neck`、`return_map_feature`、`detach_map_feature`，可從 `multi_scale_bev` 分出 map-specific feature |
| `projects/mmdet3d_plugin/models/detectors/ProtoOccMultitask.py` | 已修改 | 支援 encoder 回傳 3 個值，map branch 優先使用 `map_bev_feature`，加入解析度檢查與 `map_loss_weight` |
| `projects/mmdet3d_plugin/models/necks/lss_fpn.py` | 已修改 | `Custom_FPN_LSS` 的 checkpoint 改成只在 training 時啟用，避免 inference 時多餘 checkpoint |
| `projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py` | 已修改 | `use_bev_refinement=False` 時不再建立 PGBR modules，避免 no-PGBR ablation 在 DDP 中產生 unused parameters |
| `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck.py` | 已新增 | 128-channel map-specific neck 主實驗 |
| `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_detach.py` | 已新增 | 128-channel map-specific neck + detach ablation |
| `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_256.py` | 已新增 | 256-channel map-specific neck 高容量版本 |
| `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_no_pgbr.py` | 已新增 | 128-channel map-specific neck + no PGBR ablation |
| `projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py` | 已修改 | CNN head detector 支援 `map_bev_feature`、解析度檢查與 `map_loss_weight` |
| `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py` | 已新增 | 128-channel map-specific neck + CNN `BEVSegHead` 對照組 |

已完成檢查：

| 檢查 | 結果 |
| --- | --- |
| `python -m py_compile` | 通過 |
| `git diff --check` | 通過 |
| `conda run -n mapocc` 讀取 128-channel config | 通過，`return_map_feature=True`，map neck/head channel 為 128 |
| `conda run -n mapocc` 讀取 detach config | 通過，`detach_map_feature=True`，map neck channel 為 128 |
| `conda run -n mapocc` 讀取 256-channel config | 通過，map neck/head channel 為 256，`catconv_in_channels2=672` |
| `conda run -n mapocc` 讀取 no-PGBR config | 通過，map neck/head channel 為 128，`use_bev_refinement=False` |
| `conda run -n mapocc` 讀取 CNN-head config | 通過，model type 為 `ProtoOccCnnSegHead`，map neck/head channel 為 128 |
| no-PGBR head 參數檢查 | 通過，`proto_value_proj`、`suppress_proj`、`refine_fuse` 不再出現在 no-PGBR head 參數中 |
| PGBR head 參數檢查 | 通過，PGBR 版本仍保留 11 個 refinement 相關參數 |
| 小尺寸 encoder smoke test | 通過，輸出 `(occ_feature, bev_feature, map_bev_feature)`，shape 為 `(1,48,32,32,16)`、`(1,48,32,32)`、`(1,128,32,32)` |
| 小尺寸 CNN-head smoke test | 通過，`map_bev_feature` shape 為 `(1,128,32,32)`，CNN map logits shape 為 `(1,6,32,32)` |

第一個建議執行的實驗：

```bash
bash tools/dist_train.sh projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck.py 1
```

如果出現 OOM，優先保留 128-channel config，並確認 `with_cp=True` 已啟用；如果仍然 OOM，再考慮降低 batch 或先測 detach 版本。

## 目標

目前 ProtoOcc 多任務版本的 map segmentation 表現偏低。最佳的 `ProtoMapHead no PGBR` 目前約為：

| 方法 | Occ mIoU | Map mIoU |
| --- | ---: | ---: |
| ProtoMapHead no PGBR, epoch 24 | 37.61 | 32.95 |
| BEVFusion R50 | - | 47.10 |
| MAESTRO R50 | 38.60 | 51.30 |

主要差距在 map task，尤其是小面積與線狀類別：

| 類別 | ProtoMapHead epoch 24 | BEVFusion R50 | MAESTRO R50 |
| --- | ---: | ---: | ---: |
| ped_crossing | 24.52 | 42.80 | 45.90 |
| stop_line | 15.22 | 31.30 | 36.10 |
| divider | 21.50 | 37.80 | 41.80 |
| carpark_area | 25.88 | 43.10 | 48.30 |

因此這次改動的核心目標是：

1. 讓 map branch 不再只吃 48-channel 的 occupancy-oriented BEV feature。
2. 從 ProtoOcc 已有的 multi-scale BEV feature 分出 map-specific branch。
3. 保留 occupancy 主幹，避免 map loss 破壞原本 ProtoOcc 的 occupancy 表現。
4. 讓架構更接近 BEVFusion / BEVerse / MAESTRO 的 multi-task feature 設計。

## 目前問題

現在 `ProtoOccMultitask` 的 map branch 使用的是 `Dual_Branch_Encoder` 回傳的 `bev_feature`：

```text
voxel_feat
  -> Dual_Branch_Encoder
      -> multi_scale_bev
      -> bev_encoder_neck
      -> bev_feature, 48 channels
      -> voxelize_module
      -> comprehensive_voxel_feature

bev_feature, 48 channels
  -> ProtoMapHead
```

問題是這個 `bev_feature` 不是專門為 map segmentation 設計的。它會被 `voxelize_module` 投回 3D voxel feature，服務 occupancy decoder，所以它比較像 occupancy-oriented BEV bottleneck。

目前設定：

```python
numC_Trans = 80
voxel_out_channels = 48

bev_encoder_backbone:
    output channels roughly [160, 320, 640]

bev_encoder_neck:
    out_channels = voxel_out_channels = 48

proto_map_head:
    in_channels = 48
    hidden_channels = 96
```

這代表 map head 只拿到已經被壓縮到 48 channels 的 BEV feature。相比之下：

| 模型 | map 前 BEV feature channel |
| --- | ---: |
| ProtoOcc 目前 | 48 |
| BEVFusion camera-only | 256 |
| BEVFusion camera+lidar fusion | 512 |
| BEVerse map branch | 256 |

所以目前的 map branch capacity 明顯偏低。

## 靈感來源

### 1. BEVFusion

BEVFusion 的 map segmentation head 不是直接吃很薄的 BEV feature，而是吃 decoder neck 後的高維 BEV feature。

camera-only 設定：

```text
LSSTransform output: 80 channels
GeneralizedResNet BEV decoder: [160, 320, 640]
LSSFPN neck: 256 channels
BEVSegmentationHead input: 256 channels
```

camera+lidar fusion 設定：

```text
ConvFuser output: 256 channels
SECOND BEV backbone: [128, 256]
SECONDFPN neck: [256, 256]
concat output: 512 channels
BEVSegmentationHead input: 512 channels
```

重點不是 head 很複雜，而是 map head 前面的 BEV feature 有足夠 channel 與多尺度融合。

### 2. BEVerse

BEVerse 更直接支持這次設計。它的 shared BEV input 是 64 channels，但 `MultiTaskHead` 內部會為不同 task 建立 task-specific `BevEncode`。

簡化後是：

```text
shared BEV feature, 64 channels
  -> map-specific BevEncode
  -> 256-channel map feature
  -> MapHead

shared BEV feature, 64 channels
  -> detection-specific BevEncode
  -> 256-channel detection feature
  -> CenterHead

shared BEV feature, 64 channels
  -> motion-specific BevEncode
  -> 256-channel motion feature
  -> MotionHead
```

這說明多任務不一定要每一層都 hard sharing。更合理的方式是：

```text
shared backbone + task-specific feature generator + task head
```

### 3. MAESTRO

MAESTRO 的核心概念之一是 task-specific feature generation。它不是單純把同一份 BEV feature 丟給所有 task，而是透過 task-specific feature transformation、adaptive feature enhancement、feature suppression 等機制產生更適合各 task 的 feature。

這給我們的啟發是：

```text
ProtoOcc 的 shared BEV backbone 可以保留
但 map branch 應該有自己的 feature generator
```

## 核心改動設計

新的設計如下：

```text
voxel_feat
  -> Dual_Branch_Encoder
      -> pooled_x
      -> bev_encoder_backbone
      -> multi_scale_bev
          ├─ occ branch:
          │    -> original bev_encoder_neck
          │    -> 48-channel bev_feature
          │    -> voxelize_module
          │    -> comprehensive_voxel_feature
          │    -> occupancy decoder
          │
          └─ map branch:
               -> new map_bev_encoder_neck
               -> 128/256-channel map_bev_feature
               -> ProtoMapHead / PGBR
               -> map segmentation
```

也就是說，共享部分到 `multi_scale_bev` 為止；之後 occupancy 和 map 各自有 task-specific neck。

這不違背多任務設計，因為 shared representation 仍然存在：

```text
image backbone
view transformer
voxel feature
pooled BEV feature
BEV backbone / multi_scale_bev
```

只是最後一段 feature refinement 不再強迫 map 和 occupancy 共用。

## 為了改善什麼

### 改善 1：提升 map feature capacity

目前 map head input 是 48 channels。新設計會讓 map branch 使用 128 或 256 channels。

第一版建議優先使用：

```python
map_bev_channels = 128
```

原因是 ProtoOcc 的 occupancy decoder 本身已經很吃顯存，直接把 map branch 拉到 256 channels 有 OOM 風險。若 128 channels 已經能讓 Map mIoU 從 32.95 明顯提升，例如提升到 38 左右，就能證明方向正確。

第二階段再測：

```python
map_bev_channels = 256
```

原因是 BEVFusion / BEVerse 的 map branch 都是 256 channels 起跳。256 channels 是較接近外部方法的目標設定，但不應該作為第一個冒險實驗。

### 改善 2：保留細節類別

`divider`、`stop_line`、`ped_crossing` 都需要高解析度與多尺度融合。直接使用 48-channel bottleneck 可能會讓細線結構在 map head 前就已經被壓掉。

新設計從 `multi_scale_bev` 分支，能重新融合：

```text
high-resolution feature: 160 channels
mid-resolution feature: 320 channels
low-resolution semantic feature: 640 channels
```

這會比直接使用最後 48-channel `bev_feature` 更適合 map segmentation。

### 改善 3：降低 map 對 occ 的負面干擾

原本 map loss 會直接更新同一條 48-channel BEV neck，而這條 neck 也會影響 occupancy。

新設計中：

```text
occ branch 使用原本 bev_encoder_neck
map branch 使用 map_bev_encoder_neck
```

因此 map loss 主要更新 map-specific neck 和 map head，對 occupancy branch 的直接干擾會變小。

如果訓練初期 occupancy 仍然掉分，可以加一個開關：

```python
detach_map_feature=True
```

讓 map branch 暫時不反傳到 shared `multi_scale_bev`，先確認 map head 本身能不能學好。

## 實作風險與防呆

### 1. 顯存風險

新增 map-specific neck 後，map branch 會多吃一份 multi-scale BEV fusion 的顯存。若直接使用 256 channels，顯存壓力會比目前 48-channel map branch 高很多。

建議順序：

```text
先跑 128-channel map neck
再跑 256-channel map neck
必要時開啟 checkpoint / AMP / detach_map_feature
```

`Custom_FPN_LSS` 本身已支援 `with_cp`，如果 256 channels OOM，可以先把 map neck 的 `with_cp=True` 加進 config。

### 2. Loss 權重需要重新平衡

map branch capacity 變大後，map loss 的梯度強度和原本 48-channel branch 會不同。如果實驗出現：

```text
Map mIoU 上升
Occ mIoU 明顯下降
```

不能立刻否定 map-specific neck。這可能只是 map loss 太強，需要調整 loss weight。

建議在後續 config 中預留：

```python
map_loss_weight = 1.0
```

並做：

| 實驗 | map_loss_weight | 目的 |
| --- | ---: | --- |
| default | 1.0 | baseline |
| weaker map | 0.5 | 保護 occupancy |
| stronger map | 2.0 | 若 map 學不動再測 |

### 3. 解析度必須對齊

`ProtoMapHead` 原本吃的是 `bev_feature`，其 spatial size 應該與 `gt_masks_bev` 對齊。新的 `map_bev_feature` 必須保持同樣解析度，否則 loss 或 evaluation 會出問題。

實作時需要在 train 階段加防呆：

```python
if map_input_feature.shape[-2:] != gt_masks_bev.shape[-2:]:
    raise ValueError(
        f'map feature size {map_input_feature.shape[-2:]} '
        f'does not match gt_masks_bev size {gt_masks_bev.shape[-2:]}'
    )
```

如果解析度不同，應優先調整 `map_bev_encoder_neck` 的 upsample 設定，而不是在 loss 前硬 interpolate，避免把 thin classes 模糊掉。

### 4. `Custom_FPN_LSS` 只是第一版

第一階段複用 `Custom_FPN_LSS` 是為了快速驗證 feature bottleneck。但它原本是服務 ProtoOcc 的 BEV-to-voxel 路線，不一定是最適合 2D map segmentation 的 neck。

如果 map neck 版本有提升但仍卡在 thin classes，可以再新增專門的 `Map_FPN_LSS`：

```text
multi_scale_bev
  -> stronger high-resolution skip connection
  -> 2D detail-preserving fusion
  -> optional deformable convolution
  -> map_bev_feature
```

但這應該放在第二階段，不要一開始就增加太多變因。

## 預計新增或修改的檔案

### 1. 修改 `dual_branch_encoder.py`

檔案：

```text
projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py
```

改動：

1. 新增 constructor 參數：

```python
map_bev_encoder_neck=None
return_map_feature=False
detach_map_feature=False
```

2. 在 `__init__` 中建立 map-specific neck：

```python
self.return_map_feature = return_map_feature
self.detach_map_feature = detach_map_feature
self.map_bev_encoder_neck = (
    builder.build_neck(map_bev_encoder_neck)
    if map_bev_encoder_neck is not None else None
)
```

3. 在 forward 中，從 `multi_scale_bev` 分出 map feature：

```python
map_bev_feature = None
if self.map_bev_encoder_neck is not None:
    map_source = multi_scale_bev
    if self.detach_map_feature:
        map_source = [feat.detach() for feat in multi_scale_bev]
    map_bev = self.map_bev_encoder_neck(map_source)
    map_bev_feature = map_bev[0] if isinstance(map_bev, (list, tuple)) else map_bev
```

4. 回傳格式需要保留舊行為：

```python
if self.return_map_feature:
    assert self.return_bev_feature
    return comprehensive_voxel_feature, bev_feature, map_bev_feature
elif self.return_bev_feature:
    return comprehensive_voxel_feature, bev_feature
return comprehensive_voxel_feature
```

其中 `return_map_feature=True` 應該隱含 `return_bev_feature=True`，因為 map branch 和舊 map fallback 都需要 `bev_feature`。建議在 `__init__` 加防呆：

```python
if return_map_feature and not return_bev_feature:
    raise ValueError('return_map_feature=True requires return_bev_feature=True.')
```

### 2. 修改 `ProtoOccMultitask.py`

檔案：

```text
projects/mmdet3d_plugin/models/detectors/ProtoOccMultitask.py
```

改動：

1. 更新 `_split_encoder_output`，支援三個輸出：

```python
def _split_encoder_output(self, encoder_output):
    if isinstance(encoder_output, tuple):
        if len(encoder_output) == 3:
            return encoder_output
        if len(encoder_output) == 2:
            comprehensive_voxel_feature, bev_feature = encoder_output
            return comprehensive_voxel_feature, bev_feature, None
    return encoder_output, None, None
```

2. `forward_train` 和 `simple_test` 兩個 call site 都必須從兩個值解包改成三個值：

```python
comprehensive_voxel_feature, bev_feature, map_bev_feature = \
    self._split_encoder_output(encoder_output)
```

如果只改 `_split_encoder_output`，但忘了改這兩個地方，會直接出現：

```text
ValueError: too many values to unpack
```

3. forward train 中 map branch 優先使用 `map_bev_feature`：

```python
map_input_feature = map_bev_feature if map_bev_feature is not None else bev_feature
coarse_pred, final_masks = self.proto_map_head(map_input_feature)
```

4. simple test 同樣使用 `map_bev_feature`。

5. train 階段加入解析度檢查：

```python
if map_input_feature.shape[-2:] != gt_masks_bev.shape[-2:]:
    raise ValueError(...)
```

這樣舊模型不會壞，新模型可以切到 map-specific feature。

### 3. 新增或擴充 map neck

可選方案 A：複用現有 `Custom_FPN_LSS`

檔案：

```text
projects/mmdet3d_plugin/models/necks/lss_fpn.py
```

目前 `Custom_FPN_LSS` 已經能吃三層 multi-scale BEV feature，因此可以先直接複用，不一定要新增 class。

建議 128-channel 第一版 config：

```python
map_bev_channels = 128

map_bev_encoder_neck=dict(
    type='Custom_FPN_LSS',
    catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
    catconv_in_channels2=numC_Trans * 2 + map_bev_channels * 2,
    out_channels=map_bev_channels,
    input_feature_index=(0, 1, 2),
)
```

但要注意：`catconv_in_channels2` 取決於 `Custom_FPN_LSS` 第二次 concat 的 channel 數。若 `out_channels=256`，第二層 concat 是：

```text
multi_scale_bev[0]: 160 channels
x2_up: out_channels * 2 = 512 channels
total: 672 channels
```

所以 `catconv_in_channels2` 應該是：

```python
numC_Trans * 2 + map_bev_channels * 2
```

如果 `map_bev_channels=256`，就是 `160 + 512 = 672`。

可選方案 B：新增更清楚的 `Map_FPN_LSS`

檔案：

```text
projects/mmdet3d_plugin/models/necks/lss_fpn.py
```

新增一個專門給 map branch 的 neck，避免和 occupancy neck 的命名與設定混在一起：

```python
class Map_FPN_LSS(nn.Module):
    ...
```

第一階段建議先複用 `Custom_FPN_LSS`，等確認有效後再抽成 `Map_FPN_LSS`。

### 4. 修改 config

新增 config：

```text
projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck.py
```

內容基於：

```text
projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py
```

新增設定：

```python
map_bev_channels = 128

model = dict(
    dual_branch_encoder=dict(
        return_bev_feature=True,
        return_map_feature=True,
        detach_map_feature=False,
        map_bev_encoder_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
            catconv_in_channels2=numC_Trans * 2 + map_bev_channels * 2,
            out_channels=map_bev_channels,
            input_feature_index=(0, 1, 2),
        ),
    ),
    proto_map_head=dict(
        in_channels=map_bev_channels,
        hidden_channels=map_bev_channels,
    ),
)
```

後續再新增 256-channel 版本：

```text
projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_256.py
```

```python
_base_ = ['./ProtoOcc_proto_map_head_map_neck.py']

map_bev_channels = 256

model = dict(
    dual_branch_encoder=dict(
        map_bev_encoder_neck=dict(
            catconv_in_channels2=numC_Trans * 2 + map_bev_channels * 2,
            out_channels=map_bev_channels,
        ),
    ),
    proto_map_head=dict(
        in_channels=map_bev_channels,
        hidden_channels=map_bev_channels,
    ),
)
```

`hidden_channels=map_bev_channels` 是第一階段的保守設定。若顯存允許，可以再測 `hidden_channels=map_bev_channels * 2`，但這會明顯增加 `ProtoMapHead` 的 attention / prototype / mask embedding 成本。

也可以再新增一個 detach 版本做保護 occupancy 的 ablation：

```text
projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_detach.py
```

```python
_base_ = ['./ProtoOcc_proto_map_head_map_neck.py']

model = dict(
    dual_branch_encoder=dict(
        detach_map_feature=True,
    ),
)
```

### 5. 可能修改 `proto_map_head.py`

檔案：

```text
projects/mmdet3d_plugin/models/dense_heads/proto_map_head.py
```

第一階段不一定要改。只要把 `in_channels` 和 `hidden_channels` 改成 256，它理論上可以直接接新的 map feature。

後續可改項目：

1. 訓練階段加入 GT-guided AdaPG。
2. 加入 soft/top-k prototype pooling，避免 `sigmoid > 0.5` 對稀疏類別太嚴格。
3. 加入 coarse/final mask fusion。
4. 將 PGBR 放在 256-channel map feature 上，而不是 48-channel bottleneck 上。

## 實作順序

### Step 1：只加 map-specific neck，不改 ProtoMapHead 邏輯

目的：驗證 map feature bottleneck 是否是主因。

改動：

```text
dual_branch_encoder.py
ProtoOccMultitask.py
新增 ProtoOcc_proto_map_head_map_neck.py
```

預期：

```text
Map mIoU 應該明顯高於 32.95
Occ mIoU 不應該比目前 37.61 掉太多
```

### Step 2：做 channel-only 對照組

目的：區分「map 進步是因為 channel 變多」還是「因為從 multi-scale BEV 分出 task-specific neck」。

建議加入兩個對照：

| 對照組 | 說明 | 目的 |
| --- | --- | --- |
| 48->128/256 map adapter | 原本 48ch `bev_feature` 後面接 1x1/3x3 conv 升維，再進 ProtoMapHead | 測單純升維是否足夠 |
| widened shared neck + occ projection | 原本 shared `bev_encoder_neck` 輸出 128/256，map 使用寬 feature；occ 透過 1x1 conv 投回 48 再 voxelize | 測不分支、只加寬 shared neck 是否足夠 |

注意：第二個對照組不是單純改一個數字就能跑，因為 `voxelize_module` 和 occupancy decoder 目前假設 `voxel_out_channels=48`。若直接把 shared `bev_encoder_neck.out_channels` 改成 256，會造成 channel mismatch。因此需要加一個 `occ_bev_proj: 256 -> 48`。

### Step 3：做 detach ablation

目的：檢查 map loss 是否污染 shared BEV backbone。

比較：

| 實驗 | detach_map_feature | 預期觀察 |
| --- | --- | --- |
| map neck joint | False | map 較強，但可能影響 occ |
| map neck detach | True | occ 較穩，但 map 可能略低 |

如果 detach 版本 map 仍然大幅提升，代表主要問題是 map head input feature 太弱，而不是 joint training。

### Step 4：再加 PGBR / GT-guided AdaPG

目的：在更強的 map feature 上測 prototype refinement。

這一步才是比較公平的 PGBR 實驗，因為 PGBR 不應該只在 48-channel bottleneck 上補救。

## 預期實驗表

| 實驗 | Occ mIoU | Map mIoU | 目的 |
| --- | ---: | ---: | --- |
| current ProtoMapHead no PGBR | 37.61 | 32.95 | baseline |
| CNN head + map neck 128ch | TBD | TBD | 檢查 map-specific feature 對 naive CNN head 是否也有效 |
| 48->128/256 map adapter | TBD | TBD | 單純升維對照 |
| widened shared neck + occ projection | TBD | TBD | channel vs decoupling 對照 |
| map neck 128ch | TBD | TBD | 檢查 channel capacity |
| map neck 256ch | TBD | TBD | 對齊 BEVFusion / BEVerse |
| map neck 256ch detach | TBD | TBD | 檢查 map loss 干擾 |
| map neck 256ch + PGBR | TBD | TBD | 測 prototype-guided refinement |
| map neck 256ch + GT-guided AdaPG | TBD | TBD | 改善稀疏 map class prototype |

## 成功標準

短期成功：

```text
Map mIoU 從 32.95 提升到 38+
Occ mIoU 維持在 37 附近
```

中期成功：

```text
Map mIoU 接近或超過 42
Occ mIoU 回到 38+
```

長期目標：

```text
Map mIoU 接近 BEVFusion R50 47.10
Occ mIoU 接近原始 ProtoOcc 39.56 或 MAESTRO R50 38.60
```

## 一句話總結

這次改動不是放棄 shared feature，而是把 ProtoOcc 從「hard sharing 到 48-channel occupancy bottleneck」改成「shared BEV backbone + map-specific feature generator」。這個方向同時受到 BEVFusion 的高維 BEV decoder、BEVerse 的 task-specific `BevEncode`、以及 MAESTRO 的 task-specific feature generation 啟發。
