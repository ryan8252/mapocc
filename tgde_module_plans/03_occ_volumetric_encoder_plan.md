# Occ Volumetric Encoder 模組計劃

日期：2026-05-25

## 一句話定位

> 為 ProtoOcc PQD 產生 occupancy-specific volumetric feature，避免 OCC feature 被 map-specific BEV branch 干擾。

模組名稱：

```text
Occ Volumetric Encoder (OVE)
```

## 動機

ProtoOcc PQD 需要的不是 generic BEV feature，而是 occupancy voxel feature。現有 detector / PQD 邊界使用 legacy layout：

```text
comprehensive_voxel_feature [B, 48, H, W, Z]
```

但 OVE 內部統一使用更自然的 3D conv layout：

```text
F_occ_zyx [B, 48, Z, H, W]
```

只在接回現有 ProtoOcc decoder 前做 layout adapter：

```text
F_occ_legacy = F_occ_zyx.permute(0, 1, 3, 4, 2).contiguous()
```

原本 DBE 會從 voxel branch 和 BEV branch fusion 出 CVF，但 BEV branch 是 shared 的。如果 map loss 拉動 shared BEV trunk，OCC 的 CVF 也會被影響。

OVE 的目標是讓 OCC path 有自己的 volumetric encoder：

```text
X_occ -> F_occ_zyx -> layout adapter -> cnn3d_decoder + PQD
```

## 挑戰

1. PQD 對 input feature shape 很敏感，不能亂改輸出格式。
2. 完整替換 DBE 後 OCC 可能掉點。
3. 若 OVE 太像 ProtoOcc DBE，方法差異不夠。
4. 若 OVE 太新太大，第一版很難 debug。

## 目標

輸入：

```text
X_occ [B, 80, Z, H, W]
```

輸出：

```text
F_occ_zyx [B, 48, Z, H, W]
```

固定 downstream：

```text
F_occ_zyx -> cnn3d_decoder
F_occ_legacy = F_occ_zyx.permute(0, 1, 3, 4, 2).contiguous()
F_occ_legacy -> PQD
```

## 要做什麼

### V1：ProtoOcc-compatible volumetric branch

第一版保留 ProtoOcc voxel branch 的安全形狀：

```text
X_occ
  -> 3D conv local branch
  -> downsample / upsample voxel pyramid
  -> F_occ_local [B,48,Z,H,W]
```

再加一個 task-specific BEV context：

```text
X_occ
  -> occ-specific z projection
  -> occ BEV context
  -> voxel lift
  -> F_occ_context [B,48,Z,H,W]
```

融合：

```text
F_occ_zyx = F_occ_local + F_occ_context
```

### V2：Geometry-conditioned voxel fusion

如果 V1 穩定，再把固定 addition 改成 gate：

```text
alpha_occ = sigmoid(Conv([F_occ_local, F_occ_context]))
F_occ_zyx = F_occ_local + alpha_occ * F_occ_context
```

這是 OVE 自己內部的 confidence fusion，不是 M2O。

### OVE 不是 DBE 的具體差異

OVE 可以借 ProtoOcc voxel branch 的安全形狀，但它不是原本 DBE。具體差異：

| 項目 | ProtoOcc DBE | OVE |
| --- | --- | --- |
| 任務目標 | 產生 shared CVF，並同時供 OCC / map 上游使用 | 只產生 OCC-specific voxel feature |
| 輸入 | 原始 `X` | HTG 後的 `X_occ` |
| BEV branch | shared `multi_scale_bev`，同時餵 HFM / voxelize / map neck | occ-specific context，只餵 OCC path |
| Map output | 可回傳 `map_bev_feature` | 不輸出 map feature |
| Fusion | fixed HFM addition | fixed addition 或 gated OCC-only fusion |
| Layout | legacy output `[B,C,H,W,Z]` | internal `[B,C,Z,H,W]`，只在 PQD boundary 轉 legacy |

## 預計改動哪些檔案

新增：

```text
projects/mmdet3d_plugin/models/backbones/task_modules/occ_volumetric_encoder.py
```

config：

```python
occ_volumetric_encoder=dict(
    type='OccVolumetricEncoder',
    in_channels=80,
    out_channels=48,
    z_size=[4, 8, 16],
    with_cp=True,
    use_context_gate=False,
)
```

不改：

```text
projects/mmdet3d_plugin/models/dense_heads/cnn3d_decoder.py
projects/mmdet3d_plugin/models/dense_heads/Prototype_Query_Decoder_nuScenes.py
```

## 實驗設計

| ID | Setting | 目的 |
| --- | --- | --- |
| O0 | weight4 map-neck baseline | 對照 |
| O2 | HTG + OVE | 測 X_occ 是否有幫助 |
| O3 | OVE with fixed addition | 對照 |
| O4 | OVE with context gate | 測 gated voxel fusion |
| O5 | same params shared OVE | 排除只是 capacity |

## Debug 指標

```text
F_occ_norm
F_occ_local_norm
F_occ_context_norm
alpha_occ_internal_mean/std
cnn3d_decoder loss
PQD loss_cls / loss_mask / loss_dice
free_ratio / non_free_recall
```

特別注意：

- 如果 free ratio 異常高，代表 OCC feature collapse。
- 如果 `F_occ_context` norm 遠大於 `F_occ_local`，可能破壞 volume。
- 如果 PQD loss 明顯惡化，先退回 fixed addition。

## 預期效果

最低成功標準：

```text
Occ mIoU >= 39.72
Map mIoU 不應因 OVE 單獨改動而明顯下降
```

強成功標準：

```text
Occ mIoU > 40.0
Map mIoU >= 45.79
```

如果 OVE 有效，代表 replacement DBE 不只是保護 map，也能讓 PQD 吃到更乾淨的 OCC-specific CVF。
