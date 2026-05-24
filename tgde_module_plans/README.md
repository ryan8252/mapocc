# TGDE Module Plans

日期：2026-05-25

## 用法

這個資料夾把原本過大的 `Task-Specific Geometry Decomposition Encoder` 拆成多個獨立模組計劃。每一份 plan 都應該可以單獨討論、單獨實作、單獨 ablation，避免一開始把所有想法混在一起。

共同實驗邊界：

```text
Input:  LSS voxel feature X [B, 80, 16, 200, 200]
Occ:    fixed ProtoOcc cnn3d_decoder + PQD
Map:    fixed BEVFusion-style BEVSegHead
Baseline: CNN head + 128ch map neck, map_loss_weight=4
```

共同 baseline：

| Setting | Occ mIoU | Map mIoU | 用途 |
| --- | ---: | ---: | --- |
| Original ProtoOcc R50 OCC-only | 39.56 | - | OCC-only safety reference |
| CNN head + 128ch map neck, `map_loss_weight=1` | 39.82 | 39.94 | naive MTL reference |
| CNN head + 128ch map neck, `map_loss_weight=4` | 39.72 | 45.79 | strong clean baseline |
| CNN head + 128ch map neck, map-only | - | 48.34 | map upper bound diagnostic |

注意：

- `map_loss_weight=4` 是 TGDE 模組的正式 strong baseline。
- Original ProtoOcc R50 是 OCC-only reference，不是 map branch 對照；新模組若讓 OCC 低於 39.56，要視為明顯風險。
- 單一 seed 的小幅差異不要過度解讀。經驗上 `Occ/Map <= 0.2` 的變化先當成可能噪聲；`0.2-0.5` 需要同 protocol 或同 seed 再確認；`> 0.5` 且 class-level 改善一致，才比較適合進下一輪 ablation。

## 拆分後的模組

| Plan | 模組 | 核心問題 |
| --- | --- | --- |
| `01_height_task_gate_plan.md` | Height Task Gate | LSS voxel 的 height bins 應該對 map / occ 分開選 |
| `02_map_topology_encoder_plan.md` | Map Topology Encoder | map 不應該吃 shared BEV，而要吃 map-specific topology BEV |
| `03_occ_volumetric_encoder_plan.md` | Occ Volumetric Encoder | OCC 應該保留 volume，而不是只依賴 shared BEV fusion |
| `04_geometry_register_bank_plan.md` | Geometry Register Bank | 用 geometry memory 取代 MAESTRO-style semantic grouping |
| `05_occ_to_map_geometry_prior_plan.md` | O2M Geometry Prior | 讓 OCC volume 幫 map，但只傳 geometry prior |
| `06_map_to_occ_topology_lift_plan.md` | M2O Topology Lift | 讓 map topology 幫 OCC，但不直接灌 raw map feature |
| `07_confidence_gated_fusion_controller_plan.md` | Confidence-Gated Fusion Controller | 控制 O2M / M2O 何時融合，避免 negative transfer |

## 模組組合與介面契約

這張表是 reproduce ablation 的核心。每個組合都必須遵守固定 input/output shape，不然不同實驗之間無法比較。

| 組合 | 使用模組 | Input contract | Output contract | Decoder contract | 必要對照 |
| --- | --- | --- | --- | --- | --- |
| B0 | Baseline | `X [B,80,Z,H,W]` | DBE legacy `CVF [B,48,H,W,Z]`, `F_map [B,128,H,W]` | fixed PQD + fixed BEVSegHead | weight4 baseline |
| C1 | HTG only sanity | `X [B,80,Z,H,W]` | `X_occ/X_map [B,80,Z,H,W]` | no metric claim unless consumed by OVE/MTE | gate gradient sanity |
| C2 | HTG + MTE | `X_map [B,80,Z,H,W]` | `multi_scale_map`, then neck output `F_map [B,128,H,W]` | fixed BEVSegHead | same-params shared map branch |
| C3 | HTG + OVE | `X_occ [B,80,Z,H,W]` | internal `F_occ_zyx [B,48,Z,H,W]`; legacy `F_occ [B,48,H,W,Z]` only at PQD boundary | fixed cnn3d_decoder + PQD | original DBE OCC path |
| C4 | HTG + MTE + OVE | `X_occ/X_map [B,80,Z,H,W]` | `F_map [B,128,H,W]`, `F_occ_zyx [B,48,Z,H,W]` | fixed PQD + fixed BEVSegHead | weight4 + same-params no-split |
| C5 | C4 + GRB | `F_map`, `F_occ_zyx`, `X` | modulated `F_map_reg`, `F_occ_reg_zyx` | fixed decoders | C4 without GRB |
| C6 | C4/C5 + O2M | `F_occ_zyx`, `F_map` | `O2M_prior [B,128,H,W]`, `F_map_out [B,128,H,W]` | fixed BEVSegHead | direct raw O2M / detached O2M |
| C7 | C4/C5 + M2O | `F_map`, `F_occ_zyx` | `M2O_prior [B,48,Z,H,W]`, `F_occ_out_zyx [B,48,Z,H,W]` | fixed PQD after layout adapter | direct raw M2O / detached M2O |
| C8 | C4/C5 + O2M + M2O + CGFC | `F_map`, `F_occ_zyx`, priors | gated `F_map_out`, `F_occ_out_zyx` | fixed PQD + fixed BEVSegHead | fixed-lambda SOGDet-style fusion |

Layout rule:

```text
Internal OVE / M2O layout: [B, C, Z, H, W]
Legacy PQD / current detector boundary: [B, C, H, W, Z]
Adapter: F_occ_legacy = F_occ_zyx.permute(0, 1, 3, 4, 2).contiguous()
```

## 建議閱讀順序

第一輪不要直接做 full TGDE。建議先看：

```text
01 -> 02 -> 03
```

這三個是 replacement DBE 的基本骨架。如果只想先測 map 能不能救回來，優先：

```text
01 + 02
```

如果要測互相幫助，再看：

```text
05 -> 06 -> 07
```

`04_geometry_register_bank_plan.md` 是 paper story 最有辨識度的模組，但不一定要第一版就做。它比較適合在 `01 + 02 + 03` 有基本正向結果後加入。

## 決策原則

每個模組都要回答一個明確問題：

```text
Does this module improve task-specific feature quality under fixed decoders?
```

如果某個模組只讓 map 變好但 OCC 明顯掉，先標成 map-protection ablation，不要直接放進 main method。

如果某個模組只增加參數但沒有 task split，也必須做 same-params no-split 對照，否則不能宣稱是 task-specific geometry decomposition。
