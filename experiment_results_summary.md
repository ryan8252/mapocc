# ProtoOcc 多任務實驗結果整理

日期：2026-05-04

## 實驗摘要

目前整理十三組主要訓練結果，另補 output ablation：

| 實驗 | Config | Checkpoint | Occ mIoU | Map mIoU |
| --- | --- | --- | ---: | ---: |
| Naive MTL CNN map head | `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head.py` | `work_dirs/ProtoOcc_multi_cnn_head_4090/epoch_15.pth` | 32.54 | 16.13 |
| CNN map head + 128ch map neck (EMA) | `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_TWCC/epoch_24_ema.pth` | 39.82 | 39.94 |
| CNN map head + 128ch map neck, `map_loss_weight=4` (EMA) | `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_TWCC/epoch_5_ema_weight_4.pth` | 38.17 | 39.55 |
| CNN map head + 128ch map neck, `map_loss_weight=4` (EMA) | `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_TWCC/epoch_11_ema_weight_4.pth` | 39.26 | 43.82 |
| **CNN map head + 128ch map neck, `map_loss_weight=4` (EMA)** | `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_TWCC/epoch_24_ema_weight_4.pth` | **39.72** | **45.79** |
| **CNN map head + 128ch map neck, map-only (EMA)** | `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_map_only.py` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_map_only/epoch_24_ema.pth` | - | **48.34** |
| ProtoMapHead canonical (no PGBR) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py` | `work_dirs/ProtoOcc_proto_map_head_TWCC/epoch17.pth` | 36.77 | 32.03 |
| ProtoMapHead canonical (no PGBR) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py` | `work_dirs/ProtoOcc_proto_map_head_TWCC/epoch_24.pth` | 37.61 | 32.95 |
| ProtoMapHead + 128ch map neck (no PGBR) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck.py` | `work_dirs/ProtoOcc_proto_map_head_map_neck_no_pgbr_TWCC/epoch_24.pth` | 37.04 | 37.14 |
| ProtoMapHead + 128ch map neck (no PGBR, EMA) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck.py` | `work_dirs/ProtoOcc_proto_map_head_map_neck_no_pgbr_TWCC/epoch_24_ema.pth` | 39.71 | 39.02 |
| ProtoMapHead + 128ch map neck PQD-align (EMA) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck.py` | `work_dirs/ProtoOcc_proto_map_head_map_neck_pqd_align/epoch_24_ema.pth` | 39.78 | 38.75 |
| ProtoMapHead + 128ch map neck GT-soft (EMA) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_gt_soft.py` | `work_dirs/ProtoOcc_proto_map_head_map_neck_gt_soft/epoch_24_ema.pth` | 39.52 | 32.40 |
| ProtoMapHead + 256ch map neck (no PGBR, EMA) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck_256.py` | `work_dirs/ProtoOcc_proto_map_head_map_neck_256/epoch_24_ema.pth` | 39.76 | 39.09 |

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
| CNN head + 128ch map neck, epoch 24 EMA | 76.36 | 30.52 | 45.15 | 20.73 | 39.25 | 27.63 | 39.94 |
| CNN head + 128ch map neck, `map_loss_weight=4`, epoch 5 EMA | 76.10 | 32.50 | 44.77 | 19.35 | 36.95 | 27.64 | 39.55 |
| CNN head + 128ch map neck, `map_loss_weight=4`, epoch 11 EMA | 78.59 | 37.82 | 48.71 | 24.54 | 42.25 | 31.03 | 43.82 |
| **CNN head + 128ch map neck, `map_loss_weight=4`, epoch 24 EMA** | **80.03** | **41.94** | **50.93** | **28.32** | **40.00** | **33.49** | **45.79** |
| ProtoMapHead no PGBR, epoch 17 | 71.36 | 22.67 | 37.51 | 14.98 | 24.52 | 21.11 | 32.03 |
| ProtoMapHead no PGBR, epoch 24 | 72.14 | 24.52 | 38.44 | 15.22 | 25.88 | 21.50 | 32.95 |
| ProtoMapHead + 128ch map neck no PGBR, epoch 24 | 73.97 | 28.29 | 41.91 | 18.96 | 33.71 | 26.00 | 37.14 |
| ProtoMapHead + 128ch map neck no PGBR, epoch 24 EMA | 75.43 | 30.50 | 44.55 | 20.54 | 35.47 | 27.62 | 39.02 |
| ProtoMapHead + 128ch map neck PQD-align, epoch 24 EMA (final) | 75.82 | 31.19 | 44.17 | 20.55 | 33.12 | 27.63 | 38.75 |
| ProtoMapHead + 128ch map neck PQD-align, epoch 24 EMA (coarse) | 75.89 | 31.45 | 44.48 | 20.47 | 34.10 | 27.69 | 39.01 |
| ProtoMapHead + 128ch map neck GT-soft, epoch 24 EMA | 47.60 | 29.50 | 39.57 | 17.35 | 36.43 | 23.96 | 32.40 |
| ProtoMapHead + 256ch map neck no PGBR, epoch 24 EMA (final) | 74.52 | 32.10 | 45.03 | 21.30 | 33.51 | 28.05 | 39.09 |
| ProtoMapHead + 256ch map neck no PGBR, epoch 24 EMA (coarse) | 76.15 | 32.47 | 45.30 | 20.92 | 33.67 | 28.15 | 39.44 |
| **CNN head + 128ch map neck, map-only, epoch 24 EMA** | **80.84** | **44.58** | **52.72** | **33.62** | **42.87** | **35.43** | **48.34** |
| BEVFusion R50 | 78.00 | 42.80 | 49.70 | 31.30 | 43.10 | 37.80 | 47.10 |
| MAESTRO R50 | 80.30 | 45.90 | 55.40 | 36.10 | 48.30 | 41.80 | 51.30 |

## Occupancy 分類結果

| Class | Naive MTL CNN epoch 15 | CNN head + 128ch neck epoch 24 EMA | CNN head + 128ch neck weight4 epoch 24 EMA | ProtoMapHead no PGBR epoch 17 | ProtoMapHead no PGBR epoch 24 | 128ch map neck epoch 24 | 128ch map neck epoch 24 EMA | 128ch map neck PQD-align epoch 24 EMA | 128ch map neck GT-soft epoch 24 EMA | 256ch map neck epoch 24 EMA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| others | 6.94 | 12.17 | 11.82 | 9.91 | 10.65 | 11.61 | 12.05 | 12.34 | 12.31 | 12.05 |
| barrier | 40.92 | 47.95 | 47.22 | 43.87 | 43.87 | 44.59 | 47.73 | 47.90 | 47.62 | 48.24 |
| bicycle | 22.10 | 26.34 | 24.03 | 21.36 | 21.52 | 23.51 | 25.81 | 24.98 | 25.12 | 24.88 |
| bus | 39.19 | 44.81 | 44.77 | 42.05 | 43.31 | 36.46 | 44.52 | 44.58 | 44.52 | 44.20 |
| car | 45.96 | 51.91 | 51.94 | 49.63 | 50.54 | 49.01 | 51.95 | 52.06 | 51.94 | 51.84 |
| construction_vehicle | 20.82 | 23.47 | 25.86 | 20.65 | 20.87 | 20.39 | 22.86 | 24.50 | 22.98 | 23.48 |
| motorcycle | 22.49 | 26.54 | 26.68 | 22.58 | 26.05 | 25.26 | 26.89 | 26.35 | 26.43 | 26.40 |
| pedestrian | 23.40 | 27.64 | 27.72 | 26.59 | 25.29 | 25.79 | 27.10 | 27.84 | 27.84 | 27.85 |
| traffic_cone | 22.58 | 28.26 | 27.52 | 25.84 | 26.66 | 26.04 | 28.26 | 27.79 | 27.49 | 28.09 |
| trailer | 29.35 | 33.31 | 33.06 | 28.32 | 31.72 | 29.20 | 31.50 | 32.23 | 32.09 | 32.98 |
| truck | 34.25 | 37.16 | 37.61 | 34.81 | 35.60 | 33.34 | 37.12 | 36.63 | 37.03 | 37.20 |
| driveable_surface | 75.42 | 81.92 | 81.97 | 80.10 | 81.01 | 81.30 | 82.28 | 82.15 | 81.82 | 82.18 |
| other_flat | 26.28 | 46.13 | 45.89 | 41.83 | 42.78 | 43.57 | 46.59 | 46.56 | 45.90 | 46.45 |
| sidewalk | 40.90 | 53.28 | 53.49 | 50.24 | 51.17 | 51.80 | 53.66 | 53.34 | 52.78 | 53.46 |
| terrain | 42.80 | 56.46 | 56.36 | 53.41 | 53.05 | 54.58 | 56.44 | 56.93 | 56.47 | 56.63 |
| manmade | 33.44 | 42.89 | 42.66 | 40.27 | 40.87 | 38.03 | 43.51 | 43.03 | 42.67 | 42.97 |
| vegetation | 26.34 | 36.67 | 36.64 | 33.63 | 34.37 | 35.14 | 36.86 | 37.05 | 36.78 | 37.06 |
| mIoU | 32.54 | 39.82 | 39.72 | 36.77 | 37.61 | 37.04 | 39.71 | 39.78 | 39.52 | 39.76 |

## 主要比較

### ProtoMapHead + 256ch map neck EMA vs 128ch map neck EMA

| 指標 | 128ch map neck EMA | 256ch map neck EMA | 差異 |
| --- | ---: | ---: | ---: |
| Occ mIoU | 39.71 | 39.76 | +0.05 |
| Map mIoU | 39.02 | 39.09 | +0.07 |

Map 各類差異：

| Class | 128ch map neck EMA | 256ch map neck EMA | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 75.43 | 74.52 | -0.91 |
| ped_crossing | 30.50 | 32.10 | +1.60 |
| walkway | 44.55 | 45.03 | +0.48 |
| stop_line | 20.54 | 21.30 | +0.76 |
| carpark_area | 35.47 | 33.51 | -1.96 |
| divider | 27.62 | 28.05 | +0.43 |

### ProtoMapHead PQD-align EMA vs 舊 128ch map neck EMA

`pqd_align` 是把 ProtoMapHead 的 prototype 空間對齊 ProtoOcc PQD 後重新訓練的 128ch map neck 版本。測試使用 final output。

| 指標 | 舊 128ch map neck EMA | PQD-align EMA | 差異 |
| --- | ---: | ---: | ---: |
| Occ mIoU | 39.71 | 39.78 | +0.07 |
| Map mIoU | 39.02 | 38.75 | -0.27 |

Map 各類差異：

| Class | 舊 128ch map neck EMA | PQD-align EMA | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 75.43 | 75.82 | +0.39 |
| ped_crossing | 30.50 | 31.19 | +0.69 |
| walkway | 44.55 | 44.17 | -0.38 |
| stop_line | 20.54 | 20.55 | +0.01 |
| carpark_area | 35.47 | 33.12 | -2.35 |
| divider | 27.62 | 27.63 | +0.01 |

PQD-align 沒有傷到 occupancy，Occ mIoU 小幅高於舊 128ch EMA，也高於原始 ProtoOcc R50 的 39.56。但 Map 沒有突破舊版，主要退步集中在 `carpark_area` -2.35；其他類別大多持平或小幅上升。這表示 PQD 對齊修掉了一部分 prototype 空間問題，但沒有解決 map branch 的主要瓶頸。

### ProtoMapHead PQD-align EMA coarse vs final output

這是同一個 `ProtoOcc_proto_map_head_map_neck.py` pqd_align epoch 24 EMA checkpoint 的 output ablation。`final` 是 prototype refinement 後輸出；`coarse` 是 map coarse branch 直接輸出。

| 指標 | final output | coarse output | 差異 |
| --- | ---: | ---: | ---: |
| Map mIoU | 38.75 | 39.01 | +0.26 |

Map 各類差異：

| Class | final output | coarse output | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 75.82 | 75.89 | +0.06 |
| ped_crossing | 31.19 | 31.45 | +0.26 |
| walkway | 44.17 | 44.48 | +0.30 |
| stop_line | 20.55 | 20.47 | -0.07 |
| carpark_area | 33.12 | 34.10 | +0.98 |
| divider | 27.63 | 27.69 | +0.05 |

PQD-align epoch 24 仍然是 coarse output 略高於 final output（+0.26 mIoU）。Final 只在 `stop_line` 小幅較好（+0.07），但 `carpark_area` 低於 coarse 0.98。這和舊 256ch ablation 的方向一致：prototype refinement 在後期沒有穩定帶來正貢獻，尤其會壓低部分大面積/區塊類別。

### ProtoMapHead + 256ch map neck EMA coarse vs final output

這是同一個 `ProtoOcc_proto_map_head_map_neck_256.py` epoch 24 EMA checkpoint 的 output ablation。`final` 是 prototype refinement 後輸出；`coarse` 是 map coarse branch 直接輸出。

| 指標 | final output | coarse output | 差異 |
| --- | ---: | ---: | ---: |
| Map mIoU | 39.09 | 39.44 | +0.36 |

Map 各類差異：

| Class | final output | coarse output | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 74.52 | 76.15 | +1.63 |
| ped_crossing | 32.10 | 32.47 | +0.37 |
| walkway | 45.03 | 45.30 | +0.28 |
| stop_line | 21.30 | 20.92 | -0.39 |
| carpark_area | 33.51 | 33.67 | +0.16 |
| divider | 28.05 | 28.15 | +0.10 |

Coarse output 整體比 final output 高 0.36 mIoU，主要來自 `drivable_area` +1.63。Final output 只有在 `stop_line` 明顯較好（+0.39），其餘類別都沒有超過 coarse。這表示 prototype refinement 目前沒有穩定提升 map segmentation，反而會犧牲大面積類別的校準。

### CNN head + 128ch map neck EMA vs ProtoMapHead + 128ch map neck EMA

| 指標 | ProtoMapHead + 128ch neck EMA | CNN head + 128ch neck EMA | 差異 |
| --- | ---: | ---: | ---: |
| Occ mIoU | 39.71 | 39.82 | +0.11 |
| Map mIoU | 39.02 | 39.94 | +0.92 |

Map 各類差異：

| Class | ProtoMapHead + 128ch neck EMA | CNN head + 128ch neck EMA | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 75.43 | 76.36 | +0.93 |
| ped_crossing | 30.50 | 30.52 | +0.02 |
| walkway | 44.55 | 45.15 | +0.60 |
| stop_line | 20.54 | 20.73 | +0.19 |
| carpark_area | 35.47 | 39.25 | +3.78 |
| divider | 27.62 | 27.63 | +0.01 |

這組 ablation 顯示 map-specific neck 才是主要增益來源。當 CNN head 也接上 128ch map neck 後，在 `map_loss_weight=1` 設定下已經不輸 ProtoMapHead，達到 Map mIoU 39.94，主要多在 `carpark_area` +3.78。

### CNN head + 128ch map neck weight4 vs weight1

這是同一個 CNN head + 128ch map neck 架構，只改 global map task loss priority。`map_loss_weight=4` 使用 epoch 24 EMA。

| 指標 | weight1 epoch 24 EMA | weight4 epoch 24 EMA | 差異 |
| --- | ---: | ---: | ---: |
| Occ mIoU | 39.82 | 39.72 | -0.10 |
| Map mIoU | 39.94 | 45.79 | +5.85 |

Map 各類差異：

| Class | weight1 epoch 24 EMA | weight4 epoch 24 EMA | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 76.36 | 80.03 | +3.67 |
| ped_crossing | 30.52 | 41.94 | +11.42 |
| walkway | 45.15 | 50.93 | +5.78 |
| stop_line | 20.73 | 28.32 | +7.59 |
| carpark_area | 39.25 | 40.00 | +0.75 |
| divider | 27.63 | 33.49 | +5.86 |

`map_loss_weight=4` 不是單純 early-learning boost；到 epoch 24 仍把 Map mIoU 從 39.94 拉到 45.79，OCC 只下降 0.10。這強力支持 MTL map drop 的第一層瓶頸是 task-level map suppression，而不是 CNN map head 或 rasterization pipeline 本身能力不足。後續 overlay-aware loss 必須和這個 weight4 baseline 比，而不是只和 weight1 比。

### ProtoMapHead + 128ch map neck epoch 24 / EMA vs no-neck epoch 24

| 指標 | no-neck epoch 24 | map neck epoch 24 | map neck epoch 24 EMA | map neck 差異 | EMA 差異 | 最終差異 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Occ mIoU | 37.61 | 37.04 | 39.71 | -0.57 | +2.67 | +2.10 |
| Map mIoU | 32.95 | 37.14 | 39.02 | +4.19 | +1.88 | +6.07 |

Map neck 各類提升，使用 non-EMA epoch 24 對比 no-neck epoch 24：

| Class | no-neck epoch 24 | map neck epoch 24 | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 72.14 | 73.97 | +1.83 |
| ped_crossing | 24.52 | 28.29 | +3.77 |
| walkway | 38.44 | 41.91 | +3.47 |
| stop_line | 15.22 | 18.96 | +3.74 |
| carpark_area | 25.88 | 33.71 | +7.83 |
| divider | 21.50 | 26.00 | +4.50 |

EMA 對 map neck epoch 24 的提升：

| Class | map neck epoch 24 | map neck epoch 24 EMA | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 73.97 | 75.43 | +1.46 |
| ped_crossing | 28.29 | 30.50 | +2.21 |
| walkway | 41.91 | 44.55 | +2.64 |
| stop_line | 18.96 | 20.54 | +1.58 |
| carpark_area | 33.71 | 35.47 | +1.76 |
| divider | 26.00 | 27.62 | +1.62 |

### ProtoMapHead + 128ch map neck GT-soft EMA vs no-GT-soft EMA

| 指標 | 128ch map neck EMA | 128ch map neck GT-soft EMA | 差異 |
| --- | ---: | ---: | ---: |
| Occ mIoU | 39.71 | 39.52 | -0.19 |
| Map mIoU | 39.02 | 32.40 | -6.62 |

Map 各類差異：

| Class | 128ch map neck EMA | GT-soft EMA | 差異 |
| --- | ---: | ---: | ---: |
| drivable_area | 75.43 | 47.60 | -27.83 |
| ped_crossing | 30.50 | 29.50 | -1.00 |
| walkway | 44.55 | 39.57 | -4.98 |
| stop_line | 20.54 | 17.35 | -3.19 |
| carpark_area | 35.47 | 36.43 | +0.96 |
| divider | 27.62 | 23.96 | -3.66 |

GT-soft 這組 occupancy 幾乎守住，但 map final output 明顯退步。主要問題是 `drivable_area` 從 75.43 掉到 47.60，直接把 Map mIoU 拉低 6.62。另一方面，`carpark_area` 反而小幅提升，表示不是整個 map branch 失效，而是 final mask 的類別校準或 inference-time prototype fallback 出現偏移。

這次所有 map 類別的 `iou@max` 都落在最低 threshold 0.35，代表輸出 probability 偏保守。GT-soft training 使用 GT-guided support，但 inference 沒有 GT，只能回到 prediction/EMA prototype，這裡可能存在 train-test mismatch。

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

### CNN head + 128ch map neck map-only EMA vs MTL EMA（MTL 代價量化）

相同架構（CNN head + 128ch map neck EMA），只差是否同時訓練 occupancy：

| 指標 | MTL weight1 | MTL weight4 | map-only | weight1 代價 | weight4 剩餘代價 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Occ mIoU | 39.82 | 39.72 | - | - | - |
| Map mIoU | 39.94 | 45.79 | 48.34 | **-8.40** | **-2.55** |

Map 各類差異（map-only minus MTL）：

| Class | MTL weight1 | MTL weight4 | map-only EMA | weight1 gap | weight4 gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| drivable_area | 76.36 | 80.03 | 80.84 | +4.48 | +0.81 |
| ped_crossing | 30.52 | 41.94 | 44.58 | **+14.06** | +2.64 |
| walkway | 45.15 | 50.93 | 52.72 | +7.57 | +1.79 |
| stop_line | 20.73 | 28.32 | 33.62 | **+12.89** | **+5.30** |
| carpark_area | 39.25 | 40.00 | 42.87 | +3.62 | +2.87 |
| divider | 27.63 | 33.49 | 35.43 | +7.80 | +1.94 |

這是目前最直接的 MTL map 損耗量化。移除 occupancy 任務後，map-only 達到 **48.34**，超越 BEVFusion R50（47.10）+1.24，距離 MAESTRO R50（51.30）僅差 2.96。`map_loss_weight=4` 把 MTL gap 從 -8.40 縮到 -2.55，代表大部分退化來自 task-level map priority 不足；剩餘 gap 最大的是 `stop_line`（+5.30），才是下一階段 overlay/thin balancing 應優先處理的部分。

### 目前 best vs 外部方法

| 對比方法 | Occ 差距 | Map 差距 |
| --- | ---: | ---: |
| 原始 ProtoOcc | +0.16 using CNN head + 128ch neck weight4 EMA | - |
| MAESTRO R50 | +1.12 using CNN head + 128ch neck weight4 EMA | -5.51 using CNN head + 128ch neck weight4 EMA |
| BEVFusion R50 | - | -1.31 using CNN head + 128ch neck weight4 EMA |
| BEVFusion R50 | - | **+1.24 using map-only** |
| MAESTRO R50 | - | **-2.96 using map-only** |

## 目前結論

1. Naive MTL CNN map head 表現很差：Occ mIoU 只有 32.54，Map mIoU 只有 16.13。這表示直接在目前 ProtoOcc shared BEV feature 上加淺層 CNN map head，會造成嚴重多任務退化。

2. ProtoMapHead 明顯優於沒有 map neck 的 Naive MTL CNN head：canonical no PGBR epoch 24 達到 Occ 37.61 / Map 32.95，相比 Naive MTL CNN 分別提升 +5.07 / +16.82。但這不能證明 prototype head 本身優於 CNN head，因為 CNN head + 128ch map neck 在 weight1 下已達 Occ 39.82 / Map 39.94，在 weight4 下更達 Occ 39.72 / Map 45.79。

3. 128ch map-specific BEV neck 是目前最有效的架構改動。non-EMA epoch 24 相比 no-neck epoch 24，Map mIoU 從 32.95 提升到 37.14（+4.19），六個 map 類別全部上升；Occ mIoU 小降 0.57。

4. EMA 對 map neck 實驗非常重要。map neck epoch 24 EMA 達到 Occ 39.71 / Map 39.02，相比 non-EMA 分別提升 +2.67 / +1.88；相比 no-neck epoch 24 則提升 +2.10 / +6.07。

5. MTL 本身確實對 map 造成明顯損耗，但 `map_loss_weight=4` 大幅縮小這個 gap。相同架構 CNN head + 128ch map neck EMA，map-only 是 **48.34**；weight1 MTL 只有 39.94，gap 為 -8.40；weight4 MTL 達到 45.79，gap 縮到 -2.55。這表示第一層主要問題是 task-level map suppression，而不是 map head 能力上限。

6. Map-only 48.34 已超越 BEVFusion R50（47.10）+1.24，距離 MAESTRO R50（51.30）僅差 2.96。weight4 MTL 45.79 仍低於 BEVFusion R50 1.31、低於 MAESTRO R50 5.51，但已比 weight1 的 39.94 明顯接近外部 map baseline。

7. 目前 MTL best 是 CNN head + 128ch map neck `map_loss_weight=4` epoch 24 EMA：Occ 39.72 / Map 45.79。Occ 仍高於原始 ProtoOcc R50 39.56 和 MAESTRO R50 38.60，Map 則把與 BEVFusion R50 的差距縮到 1.31。

8. weight1 MTL 的線狀類別瓶頸非常明顯：`stop_line` 20.73、`divider` 27.63、`ped_crossing` 30.52。weight4 後分別提升到 28.32 / 33.49 / 41.94，已接近 map-only 的 33.62 / 35.43 / 44.58。剩餘最明顯 gap 是 `stop_line`，因此下一步 overlay-aware balancing 應優先處理它，而不是只追求整體 Map mIoU。

9. PQD-align epoch 24 EMA 達到 Occ 39.78 / Map 38.75。Occ 小幅高於舊 128ch EMA 的 39.71，但 Map 低 0.27，主要由 `carpark_area` -2.35 造成。這表示 PQD 對齊沒有造成 occupancy 退化，也修掉一部分 prototype 空間問題，但仍沒有讓 ProtoMapHead 超越 CNN head + 128ch map neck；和 weight4 CNN baseline 45.79 相比，差距更明顯。

10. PGBR ablation 目前沒有改善 map，反而讓 Occ / Map 都下降。GT-soft epoch 24 EMA 也沒有改善 final map output：Occ 幾乎持平（39.52 vs 39.71），但 Map 從 39.02 掉到 32.40，主要是 `drivable_area` 大幅下降。下一步不應把目前這版 GT-soft 直接當主線，而要先診斷 train-test mismatch 和 probability calibration。

11. Coarse/final output ablation 顯示兩個 epoch 24 EMA checkpoint 都是 coarse output 高於 final output：256ch 舊版為 39.44 vs 39.09（+0.36），128ch PQD-align 為 39.01 vs 38.75（+0.26）。Prototype refinement 目前只穩定保住或小幅改善 `stop_line`，但會壓低 `drivable_area` 或 `carpark_area` 等區塊類別，因此後續不應直接假設 final mask 是最佳輸出，應測 coarse-only、coarse+final fusion 或 class-wise selection。

12. 目前最強證據支持的是 map-specific feature branch + task-level map loss balancing：map-only 48.34 > CNN head + 128ch neck weight4 MTL 45.79 >> CNN head + 128ch neck weight1 MTL 39.94 > ProtoMapHead + 256ch neck EMA coarse 39.44 > ProtoMapHead + 128ch PQD-align coarse 39.01。Prototype refinement 本身的貢獻目前仍不足，不應作為下一步主線。

## 建議下一步實驗

1. 以 weight4 作為正式 strong baseline：
   - 128ch branch 已證實比直接使用 48-channel shared `bev_feature` 更適合 map segmentation。
   - CNN head + 128ch map neck + `map_loss_weight=4` 已是目前 MTL best，後續方法必須和 Occ 39.72 / Map 45.79 比，而不是只和 weight1 39.94 比。

2. 實作 overlay-aware map balancing：
   - 目標不是再證明 map task 需要更大 loss，而是在 weight4 strong baseline 上補 `stop_line / divider / ped_crossing`。
   - 成功標準應是 Map mIoU 維持 45.79 附近，`stop_line` 優先超過 30，且 Occ 不低於 39.0。

3. 保護 occupancy 主任務：
   - weight4 已可讓 Occ 達到 39.72，後續目標是保持 Occ 不低於原始 ProtoOcc 39.56，最好不要比 weight4 低超過 0.3。


## 目前試過但沒有用的方法

1. 補測 `ProtoOcc_proto_map_head_map_neck_256.py` 的 `epoch_24.pth`：
   - ema 一定比較好 不用去測non-EMA
   - 目前 256ch EMA 只比 128ch EMA 微幅提升，需要 non-EMA 結果確認 256ch 本身的效果，避免把 EMA 差異誤判成 channel 差異。


2. 繼續測 `ProtoMapHead` 的 coarse+final 融合：
   - 結論 通常coarse output 比較好 ProtoMapHead屁用沒有
   - 256ch 舊版 coarse output 39.44 高於 final output 39.09；128ch PQD-align coarse output 39.01 高於 final output 38.75。
   - Final output 主要只在 `stop_line` 較好；後續可測 class-wise fusion，例如大面積類別用 coarse，線狀類別嘗試 final 或 weighted ensemble。
   - GT-soft checkpoint 也應補測 coarse-only，確認 32.40 是 final prototype refinement 的問題，還是 coarse map prediction 本身已退化。

3. 診斷 GT-soft prototype mining：
   - 結論 根據smoke test 一開始訓練的時候 的確有比較多的prototype可以抓到 但是在做推論的時候沒有了GT的支持 反而都抓不到prototype 推論結果很差
   - 目前 GT-soft epoch 24 EMA 結果是 Occ 39.52 / Map 32.40，final map output 明顯低於 no-GT-soft EMA 的 39.02。
   - 所有 map 類別最佳 IoU 都在 threshold 0.35，應檢查 probability histogram、coarse/final logit scale，以及 inference 時 prediction/EMA prototype fallback 是否和 training-time GT support 不一致。
   
