# Map-only Upper-bound Experiment

日期：2026-04-27

## 為什麼要做

目前 best 多任務結果是 `CNN head + 128ch map neck EMA`：

| 方法 | Occ mIoU | Map mIoU |
| --- | ---: | ---: |
| CNN head + 128ch map neck EMA | 39.82 | 39.94 |

這個結果的 occupancy 已經高於原始 ProtoOcc R50 與 MAESTRO R50，但 map segmentation 仍卡在 40 左右，距離 BEVFusion R50 的 47.10 與 MAESTRO R50 的 51.30 還有明顯差距。

主要差距集中在 sparse / thin map classes：

| Class | 目前 best | BEVFusion R50 | 差距 |
| --- | ---: | ---: | ---: |
| ped_crossing | 30.52 | 42.80 | -12.28 |
| stop_line | 20.73 | 31.30 | -10.57 |
| divider | 27.63 | 37.80 | -10.17 |

因此先做 map-only upper-bound，不是要把最後方法改成 map-only，而是要判斷目前卡住的原因：

1. 如果 map-only 可以到 45 或更高，代表 map head / map GT 本身有潛力，主要問題是 multi-task conflict 或 loss balance。
2. 如果 map-only 仍卡在 40 左右，代表單純改 `map_loss_weight` 不太可能大幅改善，需要優先檢查 map GT 對齊、feature resolution、head capacity 或 thin-class loss。

## 做了什麼

### 1. 新增乾淨的 map-only BEV encoder

修改檔案：

```text
projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py
projects/mmdet3d_plugin/models/backbones/__init__.py
```

新增 `MapOnly_BEV_Encoder`，只保留 map segmentation 需要的 BEV path：

```text
voxel feature
  -> collapse Z into BEV channels
  -> down_sample_for_3d_pooling
  -> CustomBEVBackbone
  -> Custom_FPN_LSS map neck
  -> map BEV feature
```

它不建立 voxel branch、hierarchical 3D fusion、voxelize module、occupancy decoder 相關 feature path。

### 2. 新增真正的 map-only detector

新增檔案：

```text
projects/mmdet3d_plugin/models/detectors/ProtoOccMapOnly.py
```

訓練時只建並使用：

- image backbone / image neck
- `CM_DepthNet`
- `LSSViewTransformer_depthGT`
- `MapOnly_BEV_Encoder`
- `BEVSegHead`

不建：

- `cnn3d_decoder`
- `prototype_query_decoder`
- `ProtoMapHead`
- occupancy loss / prototype loss

第一版仍保留 depth auxiliary supervision，因為 depth 是 view-transform 的幾何監督，不是最終 occupancy 任務。

### 3. 新增完整 map-only config

新增檔案：

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_map_only.py
```

這個 config 不再繼承多任務 config，而是完整宣告 map-only 需要的模型與 data pipeline。

模型核心：

```python
model = dict(
    type='ProtoOccMapOnly',
    map_bev_encoder=dict(type='MapOnly_BEV_Encoder', ...),
    bev_seg_head=dict(type='BEVSegHead', ...),
    train_depth=True)
```

data pipeline 只收 map/depth 需要的 key：

```python
keys=[
    'img_inputs',
    'sa_gt_depth',
    'sa_gt_semantic',
    'gt_masks_bev',
]
```

因為 occupancy modules 不會被建立，所以不需要 `find_unused_parameters=True`。

## 怎麼跑

單卡：

```bash
bash tools/dist_train.sh projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_map_only.py 1
```

如果只要測 checkpoint：

```bash
bash tools/dist_test.sh \
  projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_map_only.py \
  work_dirs/ProtoOcc_multi_cnn_head_map_neck_map_only/epoch_24.pth \
  1 \
  --eval map-miou
```

## 怎麼判讀

| Map-only 結果 | 判讀 | 下一步 |
| ---: | --- | --- |
| `>= 45` | map branch/GT 有潛力，multi-task 設定壓住 map | 做 `map_loss_weight` sweep、detach、staged training |
| `42-45` | loss balance 有幫助，但 thin classes 仍是瓶頸 | 加 focal / class-wise positive weighting |
| `39-41` | 不是單純 multi-task conflict | 優先查 GT 對齊、feature resolution、map head/loss |
| `< 39` | map-only 都不如 multi-task best | 檢查 config 是否真的載入 best map neck、evaluation 是否正確 |

## 注意事項

這個實驗只用來量 map upper bound。即使 map-only 分數很高，也不能直接拿來當最終 multi-task 結果；它的作用是決定後續應該優先處理 multi-task conflict 還是 map branch 本身。
