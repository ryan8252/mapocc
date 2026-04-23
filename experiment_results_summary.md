# ProtoOcc 多任務實驗結果整理

日期：2026-04-24

## 實驗摘要

目前整理六組已測結果：

| 實驗 | Config | Checkpoint | Occ mIoU | Map mIoU |
| --- | --- | --- | ---: | ---: |
| Naive MTL CNN map head | `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head.py` | `work_dirs/ProtoOcc_multi_cnn_head_4090/epoch_15.pth` | 32.54 | 16.13 |
| ProtoMapHead canonical (no PGBR) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py` | `work_dirs/ProtoOcc_proto_map_head_TWCC/epoch17.pth` | 36.77 | 32.03 |
| ProtoMapHead canonical (no PGBR) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head.py` | `work_dirs/ProtoOcc_proto_map_head_TWCC/epoch_24.pth` | 37.61 | 32.95 |
| ProtoMapHead + 128ch map neck (no PGBR) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck.py` | `work_dirs/ProtoOcc_proto_map_head_map_neck_no_pgbr_TWCC/epoch_24.pth` | 37.04 | 37.14 |
| ProtoMapHead + 128ch map neck (no PGBR, EMA) | `projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck.py` | `work_dirs/ProtoOcc_proto_map_head_map_neck_no_pgbr_TWCC/epoch_24_ema.pth` | 39.71 | 39.02 |
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
| ProtoMapHead no PGBR, epoch 17 | 71.36 | 22.67 | 37.51 | 14.98 | 24.52 | 21.11 | 32.03 |
| ProtoMapHead no PGBR, epoch 24 | 72.14 | 24.52 | 38.44 | 15.22 | 25.88 | 21.50 | 32.95 |
| ProtoMapHead + 128ch map neck no PGBR, epoch 24 | 73.97 | 28.29 | 41.91 | 18.96 | 33.71 | 26.00 | 37.14 |
| ProtoMapHead + 128ch map neck no PGBR, epoch 24 EMA | 75.43 | 30.50 | 44.55 | 20.54 | 35.47 | 27.62 | 39.02 |
| ProtoMapHead + 256ch map neck no PGBR, epoch 24 EMA | 74.52 | 32.10 | 45.03 | 21.30 | 33.51 | 28.05 | 39.09 |
| BEVFusion R50 | 78.00 | 42.80 | 49.70 | 31.30 | 43.10 | 37.80 | 47.10 |
| MAESTRO R50 | 80.30 | 45.90 | 55.40 | 36.10 | 48.30 | 41.80 | 51.30 |

## Occupancy 分類結果

| Class | Naive MTL CNN epoch 15 | ProtoMapHead no PGBR epoch 17 | ProtoMapHead no PGBR epoch 24 | 128ch map neck epoch 24 | 128ch map neck epoch 24 EMA | 256ch map neck epoch 24 EMA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| others | 6.94 | 9.91 | 10.65 | 11.61 | 12.05 | 12.05 |
| barrier | 40.92 | 43.87 | 43.87 | 44.59 | 47.73 | 48.24 |
| bicycle | 22.10 | 21.36 | 21.52 | 23.51 | 25.81 | 24.88 |
| bus | 39.19 | 42.05 | 43.31 | 36.46 | 44.52 | 44.20 |
| car | 45.96 | 49.63 | 50.54 | 49.01 | 51.95 | 51.84 |
| construction_vehicle | 20.82 | 20.65 | 20.87 | 20.39 | 22.86 | 23.48 |
| motorcycle | 22.49 | 22.58 | 26.05 | 25.26 | 26.89 | 26.40 |
| pedestrian | 23.40 | 26.59 | 25.29 | 25.79 | 27.10 | 27.85 |
| traffic_cone | 22.58 | 25.84 | 26.66 | 26.04 | 28.26 | 28.09 |
| trailer | 29.35 | 28.32 | 31.72 | 29.20 | 31.50 | 32.98 |
| truck | 34.25 | 34.81 | 35.60 | 33.34 | 37.12 | 37.20 |
| driveable_surface | 75.42 | 80.10 | 81.01 | 81.30 | 82.28 | 82.18 |
| other_flat | 26.28 | 41.83 | 42.78 | 43.57 | 46.59 | 46.45 |
| sidewalk | 40.90 | 50.24 | 51.17 | 51.80 | 53.66 | 53.46 |
| terrain | 42.80 | 53.41 | 53.05 | 54.58 | 56.44 | 56.63 |
| manmade | 33.44 | 40.27 | 40.87 | 38.03 | 43.51 | 42.97 |
| vegetation | 26.34 | 33.63 | 34.37 | 35.14 | 36.86 | 37.06 |
| mIoU | 32.54 | 36.77 | 37.61 | 37.04 | 39.71 | 39.76 |

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

### ProtoMapHead + 256ch map neck epoch 24 EMA vs 外部方法

| 對比方法 | Occ 差距 | Map 差距 |
| --- | ---: | ---: |
| 原始 ProtoOcc | +0.20 | - |
| MAESTRO R50 | +1.16 | -12.21 |
| BEVFusion R50 | - | -8.01 |

## 目前結論

1. Naive MTL CNN map head 表現很差：Occ mIoU 只有 32.54，Map mIoU 只有 16.13。這表示直接在目前 ProtoOcc BEV feature 上加淺層 CNN map head，會造成嚴重多任務退化。

2. ProtoMapHead 明顯優於 Naive MTL CNN head：canonical no PGBR epoch 24 達到 Occ 37.61 / Map 32.95，相比 Naive MTL CNN 分別提升 +5.07 / +16.82。這可以支撐「prototype map head 有效緩解 naive multitask 退化」。

3. 128ch map-specific BEV neck 是目前最有效的架構改動。non-EMA epoch 24 相比 no-neck epoch 24，Map mIoU 從 32.95 提升到 37.14（+4.19），六個 map 類別全部上升；Occ mIoU 小降 0.57。

4. EMA 對 map neck 實驗非常重要。map neck epoch 24 EMA 達到 Occ 39.71 / Map 39.02，相比 non-EMA 分別提升 +2.67 / +1.88；相比 no-neck epoch 24 則提升 +2.10 / +6.07。

5. 256ch map neck EMA 是目前數字上的 best，達到 Occ 39.76 / Map 39.09。不過相對 128ch map neck EMA 只提升 +0.05 / +0.07，屬於微幅提升，表示單純加寬 map branch 已接近 plateau。

6. 256ch 對小區域和線狀類別有幫助：`ped_crossing` +1.60，`stop_line` +0.76，`divider` +0.43；但 `drivable_area` -0.91、`carpark_area` -1.96 抵消了大部分提升。

7. 目前 best 的 Occ mIoU 39.76 已經高於使用者提供的原始 ProtoOcc R50 39.56，也高於 MAESTRO R50 的 38.60；但 Map mIoU 39.09 仍距離 BEVFusion R50 47.10 差 8.01，距離 MAESTRO R50 51.30 差 12.21。

8. Map 的主要瓶頸仍是小區域和線狀類別：`stop_line` 21.30，`divider` 28.05，`ped_crossing` 32.10。雖然已比 no-neck epoch 24 明顯上升，但仍遠低於 BEVFusion/MAESTRO。

9. PGBR ablation 目前沒有改善 map，反而讓 Occ / Map 都下降。下一步更應該優先沿著 map-specific BEV neck 與 GT-guided prototype 前進，而不是繼續把 PGBR 當主線。

10. `ProtoMapHead` 的輸出 ablation 顯示，目前 inference 以 `coarse` 最好、`final` 次之、`coarse_final` 最差。以 `ProtoOcc_proto_map_head_map_neck_256.py` 的 `epoch_24_ema.pth` 為例，Occ mIoU 在三者下都維持 39.76 不變，但 Map mIoU 依序為 `39.44 > 39.09 > 38.89`。這表示目前 prototype dot-product head 還沒有穩定優於 coarse branch，而直接做 `coarse_pred + final_masks` 的融合也沒有帶來增益，反而造成些微退化。

## 建議下一步實驗

1. 補測 `ProtoOcc_proto_map_head_map_neck_256.py` 的 `epoch_24.pth`：
   - 目前 256ch EMA 只比 128ch EMA 微幅提升，需要 non-EMA 結果確認 256ch 本身的效果，避免把 EMA 差異誤判成 channel 差異。

2. 測 `ProtoMapHead` 的 coarse-only、final-only、coarse+final：
   - 如果 coarse 比 final 好，代表 prototype dot-product head 正在破壞 map 預測。
   - 如果 coarse+final 好，後續可以直接做融合。

3. 將 AdaPG 訓練階段改成 GT-guided prototype：
   - 現在 `_adaPG` 用 `sigmoid(coarse_pred) > 0.5` 抽 prototype。
   - 對 `stop_line`、`divider`、`ped_crossing` 這種稀疏類別，早期很容易抽不到 prototype。

4. 繼續改進 map-specific BEV branch：
   - 128ch branch 已證實比直接使用 48-channel shared `bev_feature` 更適合 map segmentation。
   - 256ch 只有微幅提升，後續若要改 branch，應優先測 detach map feature 或更強的 map feature fusion，而不是只繼續加 channel。

5. 保護 occupancy 主任務：
   - 可以先測 `bev_feature.detach()` 給 map head，避免 map loss 反向污染 occ backbone。
   - 目前 256ch map neck + EMA 已可讓 Occ 達到 39.76，後續目標是保持 Occ 不低於原始 ProtoOcc 39.56，再逐步提升 Map。
