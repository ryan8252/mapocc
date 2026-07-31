# All Experiments

- Source: `/home/robot/Desktop/Ryan/ProtoOcc/work_dirs`
- Counted files: `118` `result.md` files; `111` have at least one parsed Map/OCC metric.
- Ranking: sort by `Map mean` descending first, then `OCC mIoU` descending. Missing values are ranked last.
- Scale: `Map mean` and `Thin avg` are normalized to 0-1; `OCC mIoU` stays in percent as reported.
- Category precedence: `_bevfusion_aligned_` -> run name starts with `ProtoOcc_multi_cnn_head_map_neck` -> `smoke` -> remaining.
- Bold score cells mark the overall top 5 by the ranking rule above.

## 1. _bevfusion_aligned_

Count: 31

| Map mean | OCC mIoU | Thin avg | Run | Work dir |
| ---: | ---: | ---: | --- | --- |
| **0.490911** | n/a | **0.382416** | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_lss2d_noaux_nano4_h200` | `work_dirs/noaux/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_lss2d_noaux_nano4_h200` |
| 0.457396 | 39.61 | 0.359974 | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8_nano4_h200` |
| 0.450566 | n/a | 0.344270 | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_bevseg256_noaux_nano4_h200` | `work_dirs/noaux/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_bevseg256_noaux_nano4_h200` |
| 0.448953 | n/a | 0.337615 | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_focal_no_depth_no_seg_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_focal_no_depth_no_seg_nano4_h200` |
| 0.444227 | n/a | 0.331088 | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_nano4_h200_2` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_nano4_h200_2` |
| 0.442974 | n/a | 0.332264 | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_dbound60_noaux_nano4_h200` | `work_dirs/noaux/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_dbound60_noaux_nano4_h200` |
| 0.437185 | n/a | 0.326413 | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_secondfpn_noaux_nano4_h200` | `work_dirs/noaux/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_secondfpn_noaux_nano4_h200` |
| 0.435057 | 39.85 | 0.337026 | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_nano4_h200` |
| 0.434867 | n/a | 0.334743 | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_bce_dice_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_bce_dice_nano4_h200` |
| 0.428283 | 39.49 | 0.338224 | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_map_lss2d256_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_map_lss2d256_nano4_h200` |
| 0.416970 | 39.75 | 0.295371 | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_focal_weight128_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_focal_weight128_nano4_h200` |
| 0.389968 | n/a | 0.286794 | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_bevfusion_recipe_noaux_nano4_h200` | `work_dirs/noaux/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_bevfusion_recipe_noaux_nano4_h200` |
| 0.351353 | 36.82 | 0.231925 | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8_lss2d_lightcue_nano4_h200` | `work_dirs/LSS2D/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8_lss2d_lightcue_nano4_h200` |
| 0.320000 | n/a | 0.226770 | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_generalized_lssfpn_noaux_nano4_h200` | `work_dirs/noaux/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_generalized_lssfpn_noaux_nano4_h200` |
| 0.146092 | 26.70 | 0.041004 | `smoke_bevfusion_aligned_focal_weight128_dbound50_1quarter_4090` | `work_dirs/smoke_bevfusion_aligned_focal_weight128_dbound50_1quarter_4090` |
| 0.134452 | n/a | 0.037643 | `smoke_bevfusion_aligned_map_only_focal128_1quarter_nano4_h200` | `work_dirs/smoke_bevfusion_aligned_map_only_focal128_1quarter_nano4_h200` |
| 0.134282 | n/a | 0.037704 | `smoke_bevfusion_aligned_map_only_classwise_bevfusion_focal_1quarter_4090` | `work_dirs/smoke_bevfusion_aligned_map_only_classwise_bevfusion_focal_1quarter_4090` |
| 0.133768 | 5.04 | 0.063183 | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8_lss2d_plain_occ_nano4_h200` | `work_dirs/LSS2D/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8_lss2d_plain_occ_nano4_h200` |
| 0.131582 | 26.19 | 0.029180 | `smoke_bevfusion_aligned_dbound60_coarse_map_1quarter_4090` | `work_dirs/smoke_bevfusion_aligned_dbound60_coarse_map_1quarter_4090` |
| 0.129672 | 25.24 | 0.028100 | `smoke_bevfusion_aligned_dbound60_1quarter_4090` | `work_dirs/smoke_bevfusion_aligned_dbound60_1quarter_4090` |
| 0.110842 | n/a | 0.025729 | `smoke_bevfusion_aligned_map_only_1quarter_nano4_h200` | `work_dirs/smoke_bevfusion_aligned_map_only_1quarter_nano4_h200` |
| 0.109697 | n/a | 0.031442 | `smoke_bevfusion_aligned_map_only_bevfusion_decoder_1quarter_nano4_h200` | `work_dirs/smoke_bevfusion_aligned_map_only_bevfusion_decoder_1quarter_nano4_h200` |
| 0.107345 | 23.29 | 0.020182 | `smoke_ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_zsum_weight4_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_zsum_weight4_1quarter_4090` |
| 0.097711 | 3.18 | 0.039232 | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_occ2map_bevfusion_aligned_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_occ2map_bevfusion_aligned_nano4_h200` |
| 0.056780 | n/a | 0.027120 | `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_nano4_h200` |
| 0.005571 | 13.75 | 0.001105 | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8_zaware_residual_k1_nano4_h200` | `work_dirs/LSS2D/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8_zaware_residual_k1_nano4_h200` |
| n/a | 37.11 | n/a | `smoke_bevfusion_aligned_focal_weight128_4090` | `work_dirs/smoke_bevfusion_aligned_focal_weight128_4090` |
| n/a | 28.18 | n/a | `smoke_bevfusion_aligned_focal_weight128_1quarter_4090` | `work_dirs/smoke_bevfusion_aligned_focal_weight128_1quarter_4090` |
| n/a | 27.44 | n/a | `smoke_bevfusion_aligned_focal_weight4_1quarter_4090` | `work_dirs/smoke_bevfusion_aligned_focal_weight4_1quarter_4090` |
| n/a | 27.27 | n/a | `smoke_bevfusion_aligned_weight4_1quarter_4090` | `work_dirs/smoke_bevfusion_aligned_weight4_1quarter_4090` |
| n/a | n/a | n/a | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_shared_lss2d256_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_shared_lss2d256_nano4_h200` |

## 2. ProtoOcc_multi_cnn_head_map_neck...

Count: 22

| Map mean | OCC mIoU | Thin avg | Run | Work dir |
| ---: | ---: | ---: | --- | --- |
| **0.501100** | **39.15** | **0.400040** | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_mapw10_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_mapw10_nano4_h200` |
| **0.499736** | **39.40** | **0.393091** | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_mapw8_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_mapw8_nano4_h200` |
| **0.498943** | **39.46** | **0.390921** | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a2_progressive_mapw4to8_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a2_progressive_mapw4to8_nano4_h200` |
| **0.480266** | **39.58** | **0.368676** | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200` |
| 0.477798 | 39.60 | 0.366713 | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_nano4_h200` |
| 0.477016 | 38.73 | 0.372227 | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a3_mapfirst12_joint12_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a3_mapfirst12_joint12_nano4_h200` |
| 0.040544 | 39.22 | 0.003424 | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_uncertainty_weighting_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_uncertainty_weighting_nano4_h200` |
| n/a | 39.82 | n/a | `ProtoOcc_multi_cnn_head_map_neck_TWCC` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_TWCC` |
| n/a | 39.64 | n/a | `ProtoOcc_multi_cnn_head_map_neck_adapter` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_adapter` |
| n/a | 39.63 | n/a | `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter` |
| n/a | 39.60 | n/a | `ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted_nano4_h200` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted_nano4_h200` |
| n/a | 39.56 | n/a | `ProtoOcc_multi_cnn_head_map_neck_vami_TWCC` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_vami_TWCC` |
| n/a | 39.55 | n/a | `ProtoOcc_multi_cnn_head_map_neck_weight4_map_hfm` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_weight4_map_hfm` |
| n/a | 39.52 | n/a | `ProtoOcc_multi_cnn_head_map_neck_overlay_dynamic` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_overlay_dynamic` |
| n/a | 38.33 | n/a | `ProtoOcc_multi_cnn_head_map_neck_detach` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_detach` |
| n/a | 38.32 | n/a | `ProtoOcc_multi_cnn_head_map_neck_tgde_htg_mte_catz` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_tgde_htg_mte_catz` |
| n/a | 38.02 | n/a | `ProtoOcc_multi_cnn_head_map_neck_vami_4090` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_vami_4090` |
| n/a | 37.99 | n/a | `ProtoOcc_multi_cnn_head_map_neck_vami_nodetach_4090` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_vami_nodetach_4090` |
| n/a | 37.93 | n/a | `ProtoOcc_multi_cnn_head_map_neck_lgmg_m2o_masked_relation` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_lgmg_m2o_masked_relation` |
| n/a | 37.74 | n/a | `ProtoOcc_multi_cnn_head_map_neck_cfv_proto_tsfg_4090` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_cfv_proto_tsfg_4090` |
| n/a | 33.09 | n/a | `ProtoOcc_multi_cnn_head_map_neck_weight_4_1quarter` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_weight_4_1quarter` |
| n/a | n/a | n/a | `ProtoOcc_multi_cnn_head_map_neck_map_only` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_map_only` |

## 3. smoke

Count: 55

| Map mean | OCC mIoU | Thin avg | Run | Work dir |
| ---: | ---: | ---: | --- | --- |
| 0.229182 | 25.22 | 0.121078 | `smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate_lr1e-4_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate_lr1e-4_1quarter_4090` |
| 0.227815 | 26.89 | 0.117563 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_1quarter_4090` |
| 0.227403 | 26.60 | 0.117972 | `smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate_mapw8_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate_mapw8_1quarter_4090` |
| 0.227109 | 26.28 | 0.117256 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_tsfg_supp_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_tsfg_supp_1quarter_4090` |
| 0.224719 | 27.96 | 0.115991 | `smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate_1quarter_4090` |
| 0.224395 | 25.28 | 0.115449 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a4_thin_dilation3_1quarter_nano4_2gpu` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_1quarter_nano4_2gpu/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a4_thin_dilation3_1quarter_nano4_2gpu` |
| 0.224310 | 25.24 | 0.114858 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a0_1quarter_nano4_2gpu` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_1quarter_nano4_2gpu/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a0_1quarter_nano4_2gpu` |
| 0.222704 | 25.31 | 0.115282 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a3_gatew03_1quarter_nano4_2gpu` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_1quarter_nano4_2gpu/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a3_gatew03_1quarter_nano4_2gpu` |
| 0.221435 | 25.37 | 0.113496 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a2_beta10_1quarter_nano4_2gpu` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_1quarter_nano4_2gpu/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a2_beta10_1quarter_nano4_2gpu` |
| 0.220928 | 25.41 | 0.112538 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a1_beta075_1quarter_nano4_2gpu` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_1quarter_nano4_2gpu/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a1_beta075_1quarter_nano4_2gpu` |
| 0.220669 | 27.79 | 0.112686 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_thin_boundary_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_thin_boundary_1quarter_4090` |
| 0.220636 | 26.61 | 0.111086 | `smoke_ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted_1quarter_4090` |
| 0.219773 | 25.56 | 0.110837 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a5_3ch_1quarter_nano4_2gpu` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_1quarter_nano4_2gpu/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a5_3ch_1quarter_nano4_2gpu` |
| 0.219697 | 27.54 | 0.111283 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_1quarter_4090` |
| 0.218582 | 25.19 | 0.108898 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_thin_roi_refine_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_thin_roi_refine_1quarter_4090` |
| 0.218158 | 25.96 | 0.107423 | `smoke_ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted_tcs_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted_tcs_1quarter_4090` |
| 0.206450 | 26.53 | 0.100369 | `smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_residual_gates_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_residual_gates_1quarter_4090` |
| 0.198112 | 24.45 | 0.091152 | `smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate_lr4e-4_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate_lr4e-4_1quarter_4090` |
| 0.191608 | 27.49 | 0.095760 | `smoke_ProtoOcc_multi_cnn_head_map_neck_focal_dice_weighted_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_focal_dice_weighted_1quarter_4090` |
| 0.186274 | 26.86 | 0.042081 | `smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_hfm_fpn_lateral_aspp_plain128_residual_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_hfm_fpn_lateral_aspp_plain128_residual_1quarter_4090` |
| 0.185434 | 24.83 | 0.038316 | `smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_hfm_fpn_lateral_aspp_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_hfm_fpn_lateral_aspp_1quarter_4090` |
| 0.174073 | 27.28 | 0.042894 | `smoke_map_neck_focal_weight128_1quarter_4090` | `work_dirs/smoke_map_neck_focal_weight128_1quarter_4090` |
| 0.171397 | 27.31 | 0.035202 | `smoke_ProtoOcc_multi_cnn_head_map_neck_aux_loss_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_aux_loss_1quarter_4090` |
| 0.167525 | 26.22 | n/a | `smoke_ProtoOcc_multi_cnn_head_map_neck_dual_area_line_head_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_dual_area_line_head_1quarter_4090` |
| 0.167216 | 25.13 | n/a | `smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_1quarter_4090` |
| 0.166692 | 26.80 | 0.032207 | `smoke_ProtoOcc_multi_cnn_head_map_neck_coordconv_xy_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_coordconv_xy_1quarter_4090` |
| 0.166120 | 27.44 | 0.032470 | `smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_1quarter_4090` |
| 0.161818 | 27.05 | n/a | `smoke_ProtoOcc_multi_cnn_head_map_neck_bevseg_res_refine_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_bevseg_res_refine_1quarter_4090` |
| 0.161200 | n/a | 0.032633 | `smoke_ProtoOcc_multi_cnn_head_map_neck_highres_gated_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_highres_gated_1quarter_4090` |
| 0.159900 | n/a | 0.027167 | `smoke_ProtoOcc_multi_cnn_head_map_neck_zaware_residual_k4_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_zaware_residual_k4_1quarter_4090` |
| 0.154764 | 25.82 | n/a | `smoke_ProtoOcc_multi_cnn_head_map_neck_bevseg_convnext_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_bevseg_convnext_1quarter_4090` |
| 0.154400 | n/a | 0.029533 | `smoke_ProtoOcc_multi_cnn_head_map_neck_pre_adapter_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_pre_adapter_1quarter_4090` |
| 0.153700 | n/a | 0.027200 | `smoke_ProtoOcc_multi_cnn_head_map_neck_zlite_residual_catz_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_zlite_residual_catz_1quarter_4090` |
| 0.151625 | n/a | 0.032150 | `smoke_ProtoOcc_multi_cnn_head_map_neck_map_only_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_map_only_1quarter_4090` |
| 0.131500 | n/a | 0.026333 | `smoke_ProtoOcc_multi_cnn_head_map_neck_zaware_compression_conv3d_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_zaware_compression_conv3d_1quarter_4090` |
| 0.129629 | n/a | 0.041248 | `smoke_ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_1quarter_4090` | `work_dirs/smoke_ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_1quarter_4090` |
| 0.122544 | n/a | 0.036334 | `smoke_ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_train_depth_mapw4_depthw03_1quarter_4090` | `work_dirs/smoke_ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_train_depth_mapw4_depthw03_1quarter_4090` |
| 0.112520 | n/a | 0.029365 | `smoke_ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_train_depth_mapw4_1quarter_4090` | `work_dirs/smoke_ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_train_depth_mapw4_1quarter_4090` |
| 0.076000 | n/a | 0.005200 | `smoke_ProtoOcc_multi_cnn_head_map_neck_zaware_compression_heightattn_k4_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_zaware_compression_heightattn_k4_1quarter_4090` |
| n/a | 35.01 | n/a | `proto_map_head_smoke_debug` | `work_dirs/proto_map_head_smoke_debug` |
| n/a | 34.91 | n/a | `proto_map_head_smoke_gt_hard` | `work_dirs/proto_map_head_smoke_gt_hard` |
| n/a | 27.08 | n/a | `smoke_map_neck_weight4_1quarter_4090` | `work_dirs/smoke_map_neck_weight4_1quarter_4090` |
| n/a | 26.76 | n/a | `smoke_ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted_query_refine_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted_query_refine_1quarter_4090` |
| n/a | 26.52 | n/a | `smoke_map_neck_highres_weight4_1quarter_4090` | `work_dirs/smoke_map_neck_highres_weight4_1quarter_4090` |
| n/a | 25.70 | n/a | `smoke_MAESTRO_2task_lss_pqd_dbe_hybrid_1quarter_4090` | `work_dirs/smoke_MAESTRO_2task_lss_pqd_dbe_hybrid_1quarter_4090` |
| n/a | 25.56 | n/a | `smoke_ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_1quarter_4090` |
| n/a | 24.12 | n/a | `smoke_MAESTRO_2task_protoocc_1quarter_4090` | `work_dirs/smoke_MAESTRO_2task_protoocc_1quarter_4090` |
| n/a | 19.06 | n/a | `smoke_ProtoOcc_multi_cnn_head_map_neck_stage123_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_stage123_1quarter_4090` |
| n/a | 13.31 | n/a | `smoke_MAESTRO_2task_1quarter_4090_2` | `work_dirs/smoke_MAESTRO_2task_1quarter_4090_2` |
| n/a | 12.98 | n/a | `smoke_MAESTRO_2task_1quarter_4090_3` | `work_dirs/smoke_MAESTRO_2task_1quarter_4090_3` |
| n/a | 3.51 | n/a | `smoke_MAESTRO_2task_pqd_cpg_no_dbe_hfm_1quarter_4090` | `work_dirs/smoke_MAESTRO_2task_pqd_cpg_no_dbe_hfm_1quarter_4090` |
| n/a | 2.94 | n/a | `smoke_MAESTRO_2task_1quarter_4090` | `work_dirs/smoke_MAESTRO_2task_1quarter_4090` |
| n/a | n/a | n/a | `smoke_ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_1quarter_4090` | `work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_1quarter_4090` |
| n/a | n/a | n/a | `smoke_ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_train_depth_1quarter_4090` | `work_dirs/smoke_ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_train_depth_1quarter_4090` |
| n/a | n/a | n/a | `smoke_ProtoOcc_maestro_map_stl_candidate_1quarter_4090` | `work_dirs/smoke_ProtoOcc_maestro_map_stl_candidate_1quarter_4090` |

## 4. remaining

Count: 10

| Map mean | OCC mIoU | Thin avg | Run | Work dir |
| ---: | ---: | ---: | --- | --- |
| 0.159370 | 27.09 | 0.028259 | `quick_test_ProtoOcc_multi_cnn_head_map_neck_1quarter_nano4_1gpu` | `work_dirs/quick_test_ProtoOcc_multi_cnn_head_map_neck_1quarter_nano4_1gpu` |
| n/a | 39.78 | n/a | `ProtoOcc_proto_map_head_map_neck_pqd_align` | `work_dirs/ProtoOcc_proto_map_head_map_neck_pqd_align` |
| n/a | 39.21 | n/a | `debug_query_tsfg_1ep` | `work_dirs/debug_query_tsfg_1ep` |
| n/a | 39.15 | n/a | `ProtoOcc_proto_map_head_v2_TWCC_w4` | `work_dirs/ProtoOcc_proto_map_head_v2_TWCC_w4` |
| n/a | 38.27 | n/a | `ProtoOcc_proto_map_head_v2_TWCC` | `work_dirs/ProtoOcc_proto_map_head_v2_TWCC` |
| n/a | 15.46 | n/a | `MAESTRO_2task_lss_occformer_TWCC` | `work_dirs/MAESTRO_2task_lss_occformer_TWCC` |
| n/a | 2.23 | n/a | `MAESTRO_2task_lss_occformer_51m2_TWCC` | `work_dirs/MAESTRO_2task_lss_occformer_51m2_TWCC` |
| n/a | 1.22 | n/a | `MAESTRO_2task_lss_occformer_51m2` | `work_dirs/MAESTRO_2task_lss_occformer_51m2` |
| n/a | n/a | n/a | `debug_feature_convention_probe` | `work_dirs/debug_feature_convention_probe` |
| n/a | n/a | n/a | `debug_bevfusion_map_gt_parity` | `work_dirs/debug_bevfusion_map_gt_parity` |

## Top 5 Class Details

### 1. `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_mapw10_nano4_h200`

- Work dir: `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_mapw10_nano4_h200`
- Scores: Map mean `0.501100`, OCC mIoU `39.15`, Thin avg `0.400040`

OCC IoU:

| Class | IoU |
| --- | ---: |
| others | 11.86 |
| barrier | 46.94 |
| bicycle | 24.90 |
| bus | 43.58 |
| car | 51.30 |
| construction_vehicle | 24.11 |
| motorcycle | 25.59 |
| pedestrian | 27.83 |
| traffic_cone | 27.36 |
| trailer | 30.71 |
| truck | 36.65 |
| driveable_surface | 81.97 |
| other_flat | 45.50 |
| sidewalk | 53.51 |
| terrain | 55.83 |
| manmade | 41.87 |
| vegetation | 36.07 |
| mIoU | 39.15 |

Map IoU:

| Class | IoU@max |
| --- | ---: |
| drivable_area | 0.802356 |
| ped_crossing | 0.481695 |
| walkway | 0.537059 |
| stop_line | 0.327481 |
| carpark_area | 0.467068 |
| divider | 0.390944 |
| mean | 0.501100 |

### 2. `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_mapw8_nano4_h200`

- Work dir: `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_mapw8_nano4_h200`
- Scores: Map mean `0.499736`, OCC mIoU `39.40`, Thin avg `0.393091`

OCC IoU:

| Class | IoU |
| --- | ---: |
| others | 12.22 |
| barrier | 46.99 |
| bicycle | 25.39 |
| bus | 44.86 |
| car | 51.56 |
| construction_vehicle | 22.71 |
| motorcycle | 25.93 |
| pedestrian | 27.79 |
| traffic_cone | 27.58 |
| trailer | 31.26 |
| truck | 36.44 |
| driveable_surface | 82.06 |
| other_flat | 46.57 |
| sidewalk | 53.53 |
| terrain | 56.19 |
| manmade | 42.38 |
| vegetation | 36.37 |
| mIoU | 39.40 |

Map IoU:

| Class | IoU@max |
| --- | ---: |
| drivable_area | 0.796735 |
| ped_crossing | 0.467771 |
| walkway | 0.528396 |
| stop_line | 0.320251 |
| carpark_area | 0.494010 |
| divider | 0.391252 |
| mean | 0.499736 |

### 3. `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a2_progressive_mapw4to8_nano4_h200`

- Work dir: `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a2_progressive_mapw4to8_nano4_h200`
- Scores: Map mean `0.498943`, OCC mIoU `39.46`, Thin avg `0.390921`

OCC IoU:

| Class | IoU |
| --- | ---: |
| others | 12.05 |
| barrier | 46.45 |
| bicycle | 24.69 |
| bus | 45.41 |
| car | 51.62 |
| construction_vehicle | 23.49 |
| motorcycle | 26.55 |
| pedestrian | 27.85 |
| traffic_cone | 27.77 |
| trailer | 31.11 |
| truck | 36.77 |
| driveable_surface | 81.93 |
| other_flat | 46.09 |
| sidewalk | 53.66 |
| terrain | 56.29 |
| manmade | 42.58 |
| vegetation | 36.49 |
| mIoU | 39.46 |

Map IoU:

| Class | IoU@max |
| --- | ---: |
| drivable_area | 0.796958 |
| ped_crossing | 0.473589 |
| walkway | 0.527848 |
| stop_line | 0.312828 |
| carpark_area | 0.496089 |
| divider | 0.386347 |
| mean | 0.498943 |

### 4. `ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_lss2d_noaux_nano4_h200`

- Work dir: `work_dirs/noaux/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_lss2d_noaux_nano4_h200`
- Scores: Map mean `0.490911`, OCC mIoU `n/a`, Thin avg `0.382416`

OCC IoU:

| Class | IoU |
| --- | ---: |
| not available in result.md | n/a |

Map IoU:

| Class | IoU@max |
| --- | ---: |
| drivable_area | 0.798479 |
| ped_crossing | 0.428640 |
| walkway | 0.517495 |
| stop_line | 0.350647 |
| carpark_area | 0.482244 |
| divider | 0.367961 |
| mean | 0.490911 |

### 5. `ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200`

- Work dir: `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200`
- Scores: Map mean `0.480266`, OCC mIoU `39.58`, Thin avg `0.368676`

OCC IoU:

| Class | IoU |
| --- | ---: |
| others | 12.19 |
| barrier | 47.10 |
| bicycle | 25.22 |
| bus | 44.72 |
| car | 51.76 |
| construction_vehicle | 22.67 |
| motorcycle | 26.70 |
| pedestrian | 27.98 |
| traffic_cone | 27.51 |
| trailer | 32.29 |
| truck | 37.30 |
| driveable_surface | 82.16 |
| other_flat | 46.05 |
| sidewalk | 53.61 |
| terrain | 56.11 |
| manmade | 42.81 |
| vegetation | 36.75 |
| mIoU | 39.58 |

Map IoU:

| Class | IoU@max |
| --- | ---: |
| drivable_area | 0.788812 |
| ped_crossing | 0.440598 |
| walkway | 0.515463 |
| stop_line | 0.292826 |
| carpark_area | 0.471296 |
| divider | 0.372603 |
| mean | 0.480266 |

