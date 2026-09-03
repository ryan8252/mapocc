# README_UniMapOcc_v2

本文件說明 **UniMapOcc 專案**的目錄架構、各檔案功能，以及訓練 / 測試 / 視覺化的使用方式。

> **ProtoOcc** = 3D semantic occupancy baseline（AAAI 2025）。
>
> **BEVFusion** = 本專案使用的 camera-only BEV map segmentation baseline，程式位於另一個 checkout：`/media/robot/16TB/ARTC2026/Ryan/bevfusion`。
>
> **UniMapOcc** = 在 ProtoOcc 上加入 BEV map segmentation，單一模型同時預測 **3D occupancy + 2D BEV map**。
>
> 目前 Ryan/ProtoOcc repo 內主要使用兩版：
>
> - **ProtoOcc baseline** → config `ProtoOcc_1key.py`，detector `ProtoOcc`
> - **UniMapOcc final** → config `unimapocc_final.py`，detector `ProtoOccCnnSegHead`
>
> 重新訓練的提醒: 如果你想要讓occ變得比較好，那你就用國網，8個gpu去重新訓練;那如果說你想要讓map訓練的比較好，那你就直接用4090然後去訓練，但是用4090可能要等2天，國網的話12小時吧

---

## 1. 目錄架構

```text
ProtoOcc/                                           # repo 根目錄
│                                                  # /media/robot/16TB/ARTC2026/Ryan/ProtoOcc
│
├── README.md                                      # 原始 ProtoOcc 官方 README
├── README_UniMapOcc.md                            # 舊版交接文件（保留）
├── README_UniMapOcc_v2.md                         # ← 本文件
├── train_eval_record_runbook.md                   # train / eval / result.md 紀錄流程
├── demo_simple.py                                 # scene-0268 六相機 + OCC + Map demo
│
├── projects/                                      # ★ UniMapOcc 核心程式碼
│   ├── configs/
│   │   ├── unimapocc/
│   │   │   └── unimapocc_final.py                 # ★ UniMapOcc 最終獨立 config
│   │   │
│   │   └── ProtoOcc/
│   │       ├── ProtoOcc_1key.py                   # 原始 ProtoOcc OCC-only baseline
│   │       ├── ProtoOcc_multi_cnn_head.py         # 加入 Map head 的起點
│   │       ├── ProtoOcc_multi_cnn_head_map_neck.py
│   │       └── ProtoOcc_multi_..._active_enhance_gate.py
│   │                                                # 歷史繼承式 UniMapOcc final config
│   │
│   └── mmdet3d_plugin/
│       ├── models/
│       │   ├── detectors/
│       │   │   ├── ProtoOcc.py                    # baseline detector
│       │   │   └── ProtoOccCnnSegHead.py          # ★ UniMapOcc OCC + Map detector
│       │   │
│       │   ├── backbones/
│       │   │   ├── resnet.py                      # CustomBEVBackbone
│       │   │   └── dual_branch_encoder.py         # ★ DBE + Map-HFM + Map adapter
│       │   ├── necks/                              # image FPN / depth / LSS / Map FPN-ASPP
│       │   ├── dense_heads/                        # OCC coarse decoder / BEV Map head
│       │   ├── OccHead/                            # ProtoOcc prototype-query decoder
│       │   └── losses/                             # Map focal loss
│       │
│       ├── datasets/
│       │   ├── nuscenes_dataset_multitask.py      # ★ OCC + Map evaluation
│       │   └── pipelines/                          # image / Occ3D / panoptic / Map GT
│       │
│       └── core/hook/ema.py                       # EMA checkpoint hook
│
├── tools/                                         # 官方訓練 / 測試與視覺化入口
│   ├── dist_train.sh / train.py
│   ├── dist_test.sh / test.py
│   ├── visualize_map_sample.py                    # 單一 sample Map prediction / GT
│   └── analysis_tools/vis_occ.py                  # exported OCC prediction renderer
│
├── TWCC/                                          # ★ 國網 Slurm 訓練腳本
│   ├── bash_script_example.sh                     # TWCC sbatch 格式範例
│   ├── train_multi_cnn_head_map_neck.sh           # OCC + Map 早期版本訓練
│   └── train_multi_cnn_head_map_neck_map_only.sh  # Map-only 實驗訓練
│
├── TWCC_nano4/                                    # ★ 國網 Nano4 / H200 訓練與評估腳本
│   ├── quick_test_nano4.sh                        # Singularity train / eval 共用 launcher
│   ├── eval_nano4.sh                              # 通用 checkpoint 評估 launcher
│   ├── test_singularity_unimapocc_nano4.sh        # Nano4 container / environment 測試
│   └── train_eval_record_*_nano4.sh               # 各實驗的 train + eval + result 紀錄腳本
│
├── mmdetection3d/                                 # mmdet3d v1.0.0rc4 dependency
├── doc/                                           # 環境安裝文件
│
├── ckpts/                                         # ☁ pretrained weights
│   └── bevdet-r50-4d-depth-cbgs_depthnet_modify.pth
│
├── data/                                          # ☁ nuScenes / Occ3D / panoptic / Map data
│   └── nuscenes/
│       ├── v1.0-trainval/ maps/
│       ├── samples/ sweeps/
│       ├── gts/                                   # Occ3D labels.npz
│       ├── panoptic/ pc_panoptic/
│       ├── bevdetv2-nuscenes_infos_train.pkl
│       ├── bevdetv2-nuscenes_infos_val.pkl
│       └── bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl
│
└── work_dirs/                                     # ☁ logs / checkpoints / evaluation results
    ├── ProtoOcc_1key/
    │   └── ProtoOcc_1key.pth                      # ProtoOcc baseline checkpoint
    │
    └── ProtoOcc_multi_..._active_enhance_gate_nano4_h200/
        ├── epoch_24_ema.pth                       # ★ UniMapOcc final checkpoint
        ├── codex_train_stdout.log
        ├── codex_eval_stdout.log
        └── result.md
```

> `★` = UniMapOcc 最核心的 config 或程式。
>
> `☁` = 本機大型資料或訓練產物，位於 `.gitignore` 範圍；交接時需另外提供資料與 checkpoint。

---

## 2. 各資料夾 / 檔案功能

### 2.1 Config（`projects/configs/`）

| 路徑 | 功能 |
|------|------|
| `unimapocc/unimapocc_final.py` | **UniMapOcc 最終獨立 config**：完整寫出 model、dataset、pipeline、optimizer、EMA、evaluation 與 checkpoint 設定，後續閱讀與訓練使用這份 |
| `ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py` | 歷史最終繼承式 config；既有 H200 final checkpoint 的原始實驗入口 |
| `ProtoOcc/ProtoOcc_1key.py` | 原始 ProtoOcc baseline：只輸出 3D occupancy，不含 BEV Map branch |

### 2.2 核心模型（`projects/mmdet3d_plugin/models/`）

| 路徑 | 主要 class / 功能 |
|------|-------------------|
| `detectors/ProtoOccCnnSegHead.py` | `ProtoOccCnnSegHead`：整合 image encoder、OCC branch 與 Map branch，訓練及推論都同時輸出兩個任務 |
| `detectors/ProtoOcc.py` | `ProtoOcc`：原始 OCC-only baseline detector |
| `necks/fpn.py` | `CustomFPN`：融合 ResNet image features |
| `necks/depth_net.py` | `CM_DepthNet`：預測 depth distribution 與 image context feature |
| `necks/view_transformer.py` | `LSSViewTransformer_depthGT`：將六視角影像特徵轉成 `200 × 200 × 16` voxel feature |
| `backbones/resnet.py` | `CustomBEVBackbone`：建立三尺度 BEV feature pyramid |
| `backbones/dual_branch_encoder.py` | `Dual_Branch_Encoder`：OCC / Map 共用 encoder；同檔包含 `MapHFMFusionLayer` 與 `PerScaleMapResidualAdapter` |
| `necks/lss_fpn.py` | `Custom_FPN_LSS`：OCC neck 與 Map neck；Map 路徑使用 lateral projection、top-down fusion 與 residual ASPP |
| `dense_heads/cnn3d_decoder.py` | `cnn3d_decoder`：產生 coarse occupancy 與 PQD 使用的 voxel feature |
| `OccHead/Prototype_Query_Decoder_nuScenes.py` | `Prototype_Query_Decoder_nuScenes`：ProtoOcc prototype-query occupancy decoder |
| `dense_heads/bev_seg_head.py` | `BEVSegHead`：輸出六類 Map logits，包含 weighted focal-Dice loss 與 two-channel active enhancement gate |
| `losses/focal_loss.py` | `BinaryMaskFocalLoss`：Map segmentation 使用的 binary focal loss |

UniMapOcc 的 Map branch 順序為 `shared multi-scale BEV → Map-HFM → PerScaleMapResidualAdapter → Custom_FPN_LSS (lateral + ASPP) → BEVSegHead → six BEV map masks`。

六個 Map class 的固定順序為 `drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider`。

### 2.3 Dataset 與評估（`projects/mmdet3d_plugin/datasets/`）

| 路徑 | 功能 |
|------|------|
| `nuscenes_dataset_multitask.py` | `NuScenesDatasetMultitask`：同時 dispatch OCC mIoU 與 Map IoU evaluation，並可輸出 `map.npz` |
| `pipelines/loading.py` | 載入六相機影像、LiDAR-derived depth、Occ3D GT 與 panoptic auxiliary GT |
| `pipelines/loading_bev_seg.py` | `LoadBEVSegmentation`：使用 nuScenes Map API 即時 rasterize 六類 BEV Map GT |
| `pipelines/formating_multitask.py` | 將 occupancy 與 Map 欄位整理成 model input |
| `core/hook/ema.py` | `MEGVIIEMAHook`：訓練時儲存 `epoch_<N>_ema.pth` |

### 2.4 其他

| 路徑 | 功能 |
|------|------|
| `tools/dist_train.sh` / `tools/train.py` | 單機多 GPU distributed training 入口 |
| `tools/dist_test.sh` / `tools/test.py` | OCC + Map distributed evaluation 入口 |
| `tools/visualize_map_sample.py` | 指定 sample index 或 token，輸出 Map prediction / GT PNG |
| `TWCC/` | 國網 Slurm 訓練腳本；包含早期 OCC + Map 與 Map-only 實驗的 `sbatch` launcher |
| `TWCC_nano4/` | 國網 Nano4 / H200 Singularity + Slurm 腳本；負責 quick test、完整訓練、checkpoint 評估與 `result.md` 紀錄 |
| `demo_simple.py` | 將 scene-0268 的六相機、OCC prediction 與 Map prediction 組成 demo frames |
| `train_eval_record_runbook.md` | 新實驗的 train、eval、checkpoint 與 `result.md` 紀錄格式 |
| `data/nuscenes/` | nuScenes 影像、Occ3D GT、panoptic、Map metadata 與 preprocess info PKL |
| `ckpts/` | image / depth backbone pretrained checkpoint |
| `work_dirs/` | 每次訓練的 config snapshot、log、checkpoint 與 evaluation result |

---

## 3. 使用方式

> 所有指令一律從 repo 根目錄執行：`/media/robot/16TB/ARTC2026/Ryan/ProtoOcc`。
>
> conda 環境：`mapocc`。

### 3.1 進入環境

```bash
cd /media/robot/16TB/ARTC2026/Ryan/ProtoOcc

source /home/robot/miniconda3/etc/profile.d/conda.sh
conda activate mapocc

export PYTHONNOUSERSITE=1
unset PIP_USER
export PYTHONPATH="$PWD:$PWD/mmdetection3d:$PWD/projects:${PYTHONPATH:-}"

python -c "import mmdet3d, projects; print(mmdet3d.__file__); print(projects.__file__)"
```

### 3.2 訓練 UniMapOcc

```bash
CONFIG=projects/configs/unimapocc/unimapocc_final.py
WORK_DIR=work_dirs/unimapocc_final

# 單卡
CUDA_VISIBLE_DEVICES=0 PORT=29500 bash tools/dist_train.sh \
  "$CONFIG" \
  1 \
  --work-dir "$WORK_DIR"

# 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh \
  "$CONFIG" \
  4 \
  --work-dir "$WORK_DIR"
```

- config：`projects/configs/unimapocc/unimapocc_final.py`
- pretrained：`ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth`
- config default：24 epochs、AdamW、learning rate `2e-4`、`samples_per_gpu=4`
- 輸出：`work_dirs/unimapocc_final/`

### 3.3 測試 UniMapOcc（OCC + Map）


#### map_loss_weight=4

```bash
CONFIG=projects/configs/unimapocc/unimapocc_final.py
CKPT=work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200/epoch_24_ema.pth

CUDA_VISIBLE_DEVICES=0 PORT=29501 bash tools/dist_test.sh \
  "$CONFIG" \
  "$CKPT" \
  1 \
  --eval miou map-miou
```

- checkpoint：`epoch_24_ema.pth`
- OCC 指標：`mIoU of 6019 samples`
- Map 指標：`map/mean/iou@max`
- `result.md` 記錄：OCC mIoU `39.58`，Map mean IoU `0.480266`（`48.03%`）
- 完整紀錄：同一 work directory 下的 `result.md`、`codex_train_stdout.log`、`codex_eval_stdout.log`

#### map_loss_weight=10

```bash
CONFIG=projects/configs/unimapocc/unimapocc_final.py
CKPT=work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_mapw10_4090/epoch_24_ema.pth
CUDA_VISIBLE_DEVICES=0 PORT=29501 bash tools/dist_test.sh \
  "$CONFIG" \
  "$CKPT" \
  1 \
  --eval miou map-miou
```

- checkpoint：`epoch_24_ema.pth`
- OCC 指標：`mIoU of 6019 samples`
- Map 指標：`map/mean/iou@max`
- `result.md` 記錄：OCC mIoU `39.00`，Map mean IoU `0.517210`（`51.72%`）
- 完整紀錄：同一 work directory 下的 `result.md`、`codex_train_stdout.log`、`codex_eval_stdout.log`

### 3.4 單一 sample Map 視覺化

```bash
CUDA_VISIBLE_DEVICES=0 python tools/visualize_map_sample.py \
  projects/configs/unimapocc/unimapocc_final.py \
  work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200/epoch_24_ema.pth \
  --sample-index 0 \
  --threshold 0.5 \
  --save-gt \
  --out-dir viz/unimapocc_map
```

常用參數：

| 參數 | 說明 | 預設 |
|------|------|------|
| `--sample-index` | validation set index | `0` |
| `--sample-token` | 直接指定 nuScenes sample token | – |
| `--threshold` | Map sigmoid threshold | `0.5` |
| `--render-classes` | 只畫指定 Map classes | 全六類 |
| `--render-axis` | `bevfusion` / `protoocc` 顯示座標 | `bevfusion` |
| `--save-gt` | 同時輸出 Map GT PNG | off |
| `--out-dir` | 圖片輸出目錄 | `viz/map_preview` |

輸出：

```text
viz/unimapocc_map/<sample_token>_pred.png
viz/unimapocc_map/<sample_token>_gt.png
```

### 3.5 匯出完整 validation prediction

```bash
CONFIG=projects/configs/unimapocc/unimapocc_final.py
CKPT=work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200/epoch_24_ema.pth

CUDA_VISIBLE_DEVICES=0 PORT=29501 bash tools/dist_test.sh \
  "$CONFIG" \
  "$CKPT" \
  1 \
  --eval miou map-miou \
  --eval-options \
  show_dir=work_dirs/qualitative_occ/ours_predictions \
  map_show_dir=work_dirs/qualitative_map/ours_map
```

- OCC：`work_dirs/qualitative_occ/ours_predictions/<scene>/<sample_token>/pred.npz`
- Map：`work_dirs/qualitative_map/ours_map/<scene>/<sample_token>/map.npz`

### 3.6 產生 scene-0268 OCC + Map demo frames

完成第 3.5 節的 OCC / Map prediction 匯出後：

```bash
python demo_simple.py
```

腳本讀取：

```text
data/nuscenes/bevdetv2-nuscenes_infos_demo0268.pkl
work_dirs/qualitative_occ/ours_predictions/
work_dirs/qualitative_map/ours_map/
```

輸出：

```text
demo_out/frames_occ_map0268_simple/frame_XXX.png
```

---

## 4. 快速對照：ProtoOcc、BEVFusion、UniMapOcc

以下比較的是本專案實際使用的三份 config；BEVFusion 欄位指 camera-only Map baseline，不代表 BEVFusion framework 的所有模式。

| | ProtoOcc | BEVFusion | UniMapOcc |
|--|----------|-----------|-----------|
| repo | `Ryan/ProtoOcc` | `Ryan/bevfusion` | `Ryan/ProtoOcc` |
| 主要任務 | 3D semantic occupancy | 2D BEV map segmentation | 3D occupancy + 2D BEV map |
| config | `projects/configs/ProtoOcc/ProtoOcc_1key.py` | `configs/nuscenes/seg/camera-r50-bev200-protoocc.yaml` | `projects/configs/unimapocc/unimapocc_final.py` |
| detector / model | `ProtoOcc` | `BEVFusion` | `ProtoOccCnnSegHead` |
| 推論輸入 | 六相機 | 六相機（LiDAR encoder = `null`） | 六相機 |
| image backbone | ResNet-50 + `CustomFPN` | ResNet-50 + `SECONDFPN` | ResNet-50 + `CustomFPN` |
| view transform | `CM_DepthNet` + `LSSViewTransformer_depthGT` | `LSSTransform` | `CM_DepthNet` + `LSSViewTransformer_depthGT` |
| 中間表示 | 3D voxel feature，`200 × 200 × 16` | 2D BEV feature，`200 × 200` | shared 3D voxel + multi-scale BEV features |
| OCC branch | DBE + `cnn3d_decoder` + PQD | ✗ | 保留 ProtoOcc DBE + `cnn3d_decoder` + PQD |
| Map branch | ✗ | `GeneralizedResNet` + `LSSFPN` + `BEVSegmentationHead` | Map-HFM + residual adapter + lateral FPN-ASPP + `BEVSegHead` |
| Map classes | ✗ | 六類 | 六類 |
| Map loss | ✗ | per-class focal loss | class-weighted focal + Dice，`map_loss_weight=4` + active-gate loss |
| 輸出 | occupancy grid | six BEV map masks | occupancy grid + six BEV map masks |
| 評估 | OCC mIoU | Map mean IoU | OCC mIoU + Map mean IoU |
| 代表 checkpoint | `work_dirs/ProtoOcc_1key/ProtoOcc_1key.pth` | `work_dirs/camera-r50-bev200-protoocc_no_cbgs_nano4_ep24/epoch_24.pth` | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200/epoch_24_ema.pth` |
| 訓練 / 測試入口 | `tools/dist_train.sh` / `tools/dist_test.sh` | `tools/train.py` / `tools/test.py`（torchpack） | `tools/dist_train.sh` / `tools/dist_test.sh` |

