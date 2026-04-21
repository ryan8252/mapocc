# ProtoOcc 多任務實驗結果整理

日期：2026-04-21

## 實驗摘要

目前整理三組已測結果：

| 實驗 | Config | Checkpoint | Occ mIoU | Map mIoU |
| --- | --- | --- | ---: | ---: |
| Naive MTL CNN map head | `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head.py` | `work_dirs/ProtoOcc_multi_cnn_head_4090/epoch_15.pth` | 32.54 | 16.13 |
| ProtoMapHead no PGBR | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_no_pgbr.py` | `work_dirs/ProtoOcc_proto_map_head_TWCC/epoch17.pth` | 36.77 | 32.03 |
| ProtoMapHead no PGBR | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_no_pgbr.py` | `work_dirs/ProtoOcc_proto_map_head_TWCC/epoch_24.pth` | 37.61 | 32.95 |

外部參考數字：

| 方法 | Backbone | Occ mIoU | Map mIoU | 備註 |
| --- | --- | ---: | ---: | --- |
| 原始 ProtoOcc | R50 | 39.56 | - | 使用者提供的原始 occupancy 表現 |
| BEVFusion | R50 | - | 47.10 | MAESTRO supplemental Table 2 |
| MAESTRO | R50 | 38.60 | 51.30 | MAESTRO supplemental Table 2 / Table 4 |
| BEVFusion | Swin-T | - | 56.60 | MAESTRO supplemental Table 3 |
| MAESTRO | Swin-T | - | 57.50 | MAESTRO supplemental Table 3 |

## Map Segmentation 分類結果

| 實驗 | Drivable | Ped. Cross. | Walkway | Stop Line | Carpark | Divider | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive MTL CNN head, epoch 15 | 54.34 | 7.19 | 20.49 | 5.26 | 0.25 | 9.23 | 16.13 |
| ProtoMapHead no PGBR, epoch 17 | 71.36 | 22.67 | 37.51 | 14.98 | 24.52 | 21.11 | 32.03 |
| ProtoMapHead no PGBR, epoch 24 | 72.14 | 24.52 | 38.44 | 15.22 | 25.88 | 21.50 | 32.95 |
| BEVFusion R50 | 78.00 | 42.80 | 49.70 | 31.30 | 43.10 | 37.80 | 47.10 |
| MAESTRO R50 | 80.30 | 45.90 | 55.40 | 36.10 | 48.30 | 41.80 | 51.30 |

## Occupancy 分類結果

| Class | Naive MTL CNN epoch 15 | ProtoMapHead no PGBR epoch 17 | ProtoMapHead no PGBR epoch 24 |
| --- | ---: | ---: | ---: |
| others | 6.94 | 9.91 | 10.65 |
| barrier | 40.92 | 43.87 | 43.87 |
| bicycle | 22.10 | 21.36 | 21.52 |
| bus | 39.19 | 42.05 | 43.31 |
| car | 45.96 | 49.63 | 50.54 |
| construction_vehicle | 20.82 | 20.65 | 20.87 |
| motorcycle | 22.49 | 22.58 | 26.05 |
| pedestrian | 23.40 | 26.59 | 25.29 |
| traffic_cone | 22.58 | 25.84 | 26.66 |
| trailer | 29.35 | 28.32 | 31.72 |
| truck | 34.25 | 34.81 | 35.60 |
| driveable_surface | 75.42 | 80.10 | 81.01 |
| other_flat | 26.28 | 41.83 | 42.78 |
| sidewalk | 40.90 | 50.24 | 51.17 |
| terrain | 42.80 | 53.41 | 53.05 |
| manmade | 33.44 | 40.27 | 40.87 |
| vegetation | 26.34 | 33.63 | 34.37 |
| mIoU | 32.54 | 36.77 | 37.61 |

## 主要比較

### ProtoMapHead no PGBR epoch 24 vs epoch 17

| 指標 | epoch 17 | epoch 24 | 差異 |
| --- | ---: | ---: | ---: |
| Occ mIoU | 36.77 | 37.61 | +0.84 |
| Map mIoU | 32.03 | 32.95 | +0.92 |

Map 各類提升：

| Class | epoch 17 | epoch 24 | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 71.36 | 72.14 | +0.78 |
| ped_crossing | 22.67 | 24.52 | +1.85 |
| walkway | 37.51 | 38.44 | +0.93 |
| stop_line | 14.98 | 15.22 | +0.24 |
| carpark_area | 24.52 | 25.88 | +1.36 |
| divider | 21.11 | 21.50 | +0.39 |

### ProtoMapHead no PGBR epoch 24 vs Naive MTL CNN epoch 15

| 指標 | Naive MTL CNN | ProtoMapHead no PGBR | 差異 |
| --- | ---: | ---: | ---: |
| Occ mIoU | 32.54 | 37.61 | +5.07 |
| Map mIoU | 16.13 | 32.95 | +16.82 |

Map 各類提升：

| Class | Naive MTL CNN | ProtoMapHead no PGBR | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 54.34 | 72.14 | +17.80 |
| ped_crossing | 7.19 | 24.52 | +17.33 |
| walkway | 20.49 | 38.44 | +17.95 |
| stop_line | 5.26 | 15.22 | +9.96 |
| carpark_area | 0.25 | 25.88 | +25.63 |
| divider | 9.23 | 21.50 | +12.27 |

### ProtoMapHead no PGBR epoch 24 vs 外部方法

| 對比方法 | Occ 差距 | Map 差距 |
| --- | ---: | ---: |
| 原始 ProtoOcc | -1.95 | - |
| MAESTRO R50 | -0.99 | -18.35 |
| BEVFusion R50 | - | -14.15 |

## 目前結論

1. Naive MTL CNN map head 表現很差：Occ mIoU 只有 32.54，Map mIoU 只有 16.13。這表示直接在目前 ProtoOcc BEV feature 上加淺層 CNN map head，會造成嚴重多任務退化。

2. ProtoMapHead 明顯優於 Naive MTL CNN head：epoch 24 達到 Occ 37.61 / Map 32.95，相比 Naive MTL CNN 分別提升 +5.07 / +16.82。這可以支撐「prototype map head 有效緩解 naive multitask 退化」。

3. 目前最大問題仍是 map 分數離 BEVFusion/MAESTRO 太遠。ProtoMapHead epoch 24 的 Map mIoU 是 32.95，距離 BEVFusion R50 的 47.10 還差 14.15，距離 MAESTRO R50 的 51.30 還差 18.35。

4. Map 的主要瓶頸是小區域和線狀類別：`stop_line` 只有 15.22，`divider` 只有 21.50，`ped_crossing` 只有 24.52。這些類別遠低於 BEVFusion/MAESTRO。

5. PGBR 可能能改善 map，但不太可能單獨補上 14 到 18 個 mIoU 的差距。下一步更應該優先檢查 map-specific feature branch、GT-guided prototype、coarse/final mask 融合，而不是只加 refinement。

## 建議下一步實驗

1. 測 `ProtoMapHead` 的 coarse-only、final-only、coarse+final：
   - 如果 coarse 比 final 好，代表 prototype dot-product head 正在破壞 map 預測。
   - 如果 coarse+final 好，後續可以直接做融合。

2. 將 AdaPG 訓練階段改成 GT-guided prototype：
   - 現在 `_adaPG` 用 `sigmoid(coarse_pred) > 0.5` 抽 prototype。
   - 對 `stop_line`、`divider`、`ped_crossing` 這種稀疏類別，早期很容易抽不到 prototype。

3. 建立 map-specific BEV branch：
   - 目前 map head 使用 `Dual_Branch_Encoder` 輸出的 48-channel `bev_feature`。
   - 這個 feature 同時被 voxelize 回 occupancy branch，較偏向 occ 任務，不一定適合 map segmentation。

4. 保護 occupancy 主任務：
   - 可以先測 `bev_feature.detach()` 給 map head，避免 map loss 反向污染 occ backbone。
   - 目標先變成保持 Occ 接近 39.56，再逐步提升 Map。

