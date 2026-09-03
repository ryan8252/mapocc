# README_UniMapOcc

本文件說明 **UniMapOcc（Ryan）** 在 ProtoOcc checkout 中的專案定位、最終設定、核心程式、資料流，以及訓練、評估、視覺化與交接方式。

> **ProtoOcc**：AAAI 2025 的 camera-only 3D occupancy baseline。
>
> **UniMapOcc**：本專案在 ProtoOcc 上加入 BEV semantic map segmentation，讓單一 camera-only 模型共同預測 3D occupancy 與 2D BEV map。
>
> **本文核對日期**：2026-08-03。路徑、環境、checkpoint 或程式若有異動，請同步更新本文件。

---

## 0. 接手時先看這裡

| 項目 | 交接時的指定檔案 |
| --- | --- |
| 建議閱讀、重現與後續開發的主 config | `projects/configs/unimapocc/unimapocc_final.py` |
| 原始繼承式最終 config | `projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py` |
| 原始 ProtoOcc baseline config | `projects/configs/ProtoOcc/ProtoOcc_1key.py` |
| 多任務 detector | `projects/mmdet3d_plugin/models/detectors/ProtoOccCnnSegHead.py` |
| OCC／Map shared encoder、Map-HFM、adapter | `projects/mmdet3d_plugin/models/backbones/dual_branch_encoder.py` |
| 最終 H200 EMA checkpoint | `work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200/epoch_24_ema.pth` |
| 最終實驗紀錄 | 同一 work directory 下的 `result.md`、`codex_train_stdout.log`、`codex_eval_stdout.log` |
| 詳細 train／eval／result.md 操作規範 | `train_eval_record_runbook.md` |

接手後第一輪不要急著重訓，先完成以下檢查：

```bash
cd /media/robot/16TB/ARTC2026/Ryan/ProtoOcc

source /home/robot/miniconda3/etc/profile.d/conda.sh
conda activate mapocc
export PYTHONNOUSERSITE=1
unset PIP_USER
export PYTHONPATH="$PWD:$PWD/mmdetection3d:$PWD/projects:${PYTHONPATH:-}"

# 目前 config 是否能解析
python -m py_compile projects/configs/unimapocc/unimapocc_final.py
python -c "from mmcv import Config; c=Config.fromfile('projects/configs/unimapocc/unimapocc_final.py'); print(c.model.type, c.data.train.type, c.evaluation.metric)"

# dependency import 位置
python -c "import mmdet3d, projects; print(mmdet3d.__file__); print(projects.__file__)"

# 資料與 pretrained 權重是否存在
readlink -f data
test -f data/nuscenes/bevdetv2-nuscenes_infos_train.pkl
test -f data/nuscenes/bevdetv2-nuscenes_infos_val.pkl
test -f data/nuscenes/v1.0-trainval/scene.json
test -d data/nuscenes/maps
test -d data/nuscenes/gts
test -d data/nuscenes/pc_panoptic
test -f ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth
```

---

## 1. 專案定位

UniMapOcc 是一個 **single-frame、camera-only、multi-task** 模型：

- 輸入為 nuScenes 六顆環景相機影像。
- 主要任務一是 3D semantic occupancy prediction。
- 主要任務二是六類 BEV semantic map segmentation。
- 推論模型不使用 LiDAR；訓練 pipeline 仍會讀取 LiDAR、Occ3D GT 與 nuScenes panoptic／map 資訊來建立 depth、perspective semantic、occupancy 與 map supervision。
- OCC 與 Map 共用 image backbone、depth network、LSS view transformer 與 Dual Branch Encoder 的主幹特徵，之後才分流到各自的 decoder／head；這不是兩個彼此獨立的模型。

六個 Map class 的固定順序如下，後續 class weight、active gate group 與 metric 都依賴此順序：

```text
0 drivable_area
1 ped_crossing
2 walkway
3 stop_line
4 carpark_area
5 divider
```

---

## 2. 兩份最終 config 的角色

### 2.1 建議交接主檔：獨立版

```text
projects/configs/unimapocc/unimapocc_final.py
```

這份 config：

- 只保留 MMDetection3D 的 dataset 與 default runtime base。
- 不再依賴長串 ProtoOcc experiment config。
- 將 model、data pipeline、optimizer、runner、EMA、evaluation 與 checkpoint 設定完整寫在單一檔案。
- 在每個自訂模組旁標出對應 implementation path，適合閱讀、封存、重現與後續開發。
- 顯式寫出 `map_hfm_order='hfm_then_adapter'`、`map_hfm_fusion_mode='residual_concat'`、`map_hfm_voxel_residual_scale=1.0`，避免未來 class default 改動後行為默默漂移。

### 2.2 歷史原始檔：繼承版

```text
projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py
```

此檔只有 25 行，最後一層主要負責啟用 two-channel map-active enhancement gate。完整模型行為還來自下列繼承鏈：

```text
active_enhance_gate
  -> hfm_adapter_fpn_lateral_aspp_focal_dice_weighted
  -> fpn_lateral_aspp_focal_dice_weighted
  -> fpn_lateral_aspp
  -> fpn_lateral
  -> multi_cnn_head_map_neck
  -> multi_cnn_head
  -> MMDetection3D dataset/runtime bases
```

因此不要只讀最上層 25 行就判斷整個架構，也不要在移動專案時只複製這一個檔案。

### 2.3 兩份 config 的等價範圍

在 2026-08-02 以本機 `mapocc` 環境重新解析後：

- data、optimizer、runner、hook、evaluation、checkpoint 與 log 設定一致。
- model dict 唯一的 raw 差異，是獨立版多寫了上述三個 HFM runtime default。
- 三個值與目前 `Dual_Branch_Encoder.__init__` 的 class default 相同，所以目前執行語意相同；但不應描述成兩個 Python dict 逐字、逐項完全一致。

修改任一 config 或 encoder default 後，可用以下檢查確認沒有漂移：

```bash
python - <<'PY'
from copy import deepcopy
from mmcv import Config

legacy = Config.fromfile(
    'projects/configs/ProtoOcc/'
    'ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_'
    'focal_dice_weighted_active_enhance_gate.py'
)
standalone = Config.fromfile('projects/configs/unimapocc/unimapocc_final.py')

legacy_model = deepcopy(legacy.model.to_dict())
legacy_model['dual_branch_encoder'].update(
    map_hfm_order='hfm_then_adapter',
    map_hfm_fusion_mode='residual_concat',
    map_hfm_voxel_residual_scale=1.0,
)

runtime_keys = [
    'data', 'optimizer', 'optimizer_config', 'lr_config', 'runner',
    'custom_hooks', 'load_from', 'evaluation', 'checkpoint_config',
    'log_config',
]

assert legacy_model == standalone.model.to_dict()
assert all(legacy.get(k) == standalone.get(k) for k in runtime_keys)
print('UniMapOcc config parity: OK')
PY
```

### 2.4 Source config 與歷史 H200 job 設定

目前兩份 source config 的預設值是：

```text
AdamW lr = 2e-4
samples_per_gpu = 4
workers_per_gpu = 4
runner.max_epochs = 24
evaluation.interval = 1, start = 23
checkpoint_config.interval = 3
```

歷史最終 H200 job 則透過 CLI 覆寫為：

```text
8 GPUs
samples_per_gpu = 4
workers_per_gpu = 4
AdamW lr = 4e-4
runner.max_epochs = 24
evaluation.interval = 999
checkpoint_config.interval = 24
```

主 config 保存最終架構與一般 runtime default；重現既有 H200 checkpoint 時，使用第 8.3 節的 historical overrides。

---

## 3. 模型資料流

```text
6 cameras, each 3 x 256 x 704
              |
              v
      ResNet-50 + CustomFPN
              |
              v
 CM_DepthNet + LSSViewTransformer_depthGT
              |
              v
 shared voxel feature: B x 80 x 16 x 200 x 200
              |
              v
        Dual_Branch_Encoder
              |
      +-------+-----------------------------------+
      |                                           |
      | OCC branch                                | Map branch
      |                                           |
      | shared 3-scale BEV pyramid                 | same shared pyramid
      | + voxel hierarchical fusion                | [160@100², 320@50², 640@25²]
      |                                           |     |
      |                                           | Map-HFM
      |                                           | mean-Z + max-Z geometry
      |                                           |     |
      |                                           | PerScaleMapResidualAdapter
      |                                           |     |
      |                                           | FPN lateral projection
      |                                           | + residual ASPP context
      |                                           |     |
      v                                           v
48-channel 200 x 200 x 16 voxel feature     128-channel 200 x 200 map feature
      |                                           |
      +-> cnn3d_decoder                           +-> BEVSegHead
      |                                           |     |
      +-> Prototype Query Decoder                 | two-channel active gate
                                                  |     |
                                                  +-> 6 independent map logits
```

### 3.1 OCC branch

- 保留原 ProtoOcc 的 Dual Branch Encoder、3D CNN prototype generator 與 Prototype Query Decoder。
- 最終 occupancy grid 為 `200 x 200 x 16`，範圍為 `x/y = [-40, 40] m`、`z = [-1, 5.4] m`，voxel size 為 `0.4 m`。
- `cnn3d_decoder` 先產生 coarse semantic/prototype feature，PQD 再輸出 occupancy prediction。

### 3.2 Map-HFM

- 對三個 voxel source 沿高度維分別做 mean-Z 與 max-Z，再串接成 2D geometry cue。
- cue 與相同尺度的 BEV feature 融合，輸出以 zero-init residual 加回原 feature。
- 三個尺度皆保留 channel 數 `[160, 320, 640]`。
- 最終順序固定為 `HFM -> adapter`。

### 3.3 PerScaleMapResidualAdapter

- 每個尺度使用 `1x1 -> depthwise 3x3 -> 1x1` 做 map-specific refinement。
- 最後一層 1x1 convolution 為 zero-init，初始狀態接近 identity residual。
- `detach_input=False`，Map loss 可以經由此路徑回傳至 shared feature。

### 3.4 Map FPN lateral + ASPP

- 三個 BEV pyramid level 先各自投影到 256 channels。
- 在最低解析度的 `640 x 25 x 25` level 先加入 residual ASPP global context。
- top-down fusion 後輸出 `B x 128 x 200 x 200` 的 map feature。

### 3.5 Weighted focal-Dice + active enhancement gate

- 最終 Map loss 為 focal + Dice。
- 六類 static weights 為 `[1, 2, 2, 2, 4, 4]`。
- detector 的 `map_loss_weight=4.0` 會乘上 BEVSegHead 回傳的所有 Map loss，包含 active-gate loss；它是 loss／gradient multiplier，不是「Map 任務占 4%」或「占 80%」。
- active gate 先從 decoded map feature 預測兩個 spatial channels：
  - area group：`[drivable_area, walkway, carpark_area] = [0, 2, 4]`
  - thin／overlay group：`[ped_crossing, stop_line, divider] = [1, 3, 5]`
- thin group 的 GT target 會用 dilation `2`，area group dilation `0`。
- gate probability 形成 spatial gain 後增強 decoded feature，再由獨立的六類 1x1 predictor 輸出 map logits。
- gate predictor 的初始 bias 是 `-4.0`，所以初始增益很小，但不是嚴格的零。
- 訓練時 GT 只負責監督 gate focal／Dice loss；推論時 gate 完全由 feature 自己預測，不會偷看 GT。
- 六個 map channel 使用 sigmoid，類別可以空間重疊，不是 softmax 互斥分類。

---

## 4. 目錄架構

```text
ProtoOcc/
├── README.md                              # 原始 ProtoOcc 官方 README
├── README_UniMapOcc.md                    # 本交接文件
├── train_eval_record_runbook.md           # train -> eval -> result.md 詳細流程
│
├── projects/
│   ├── configs/
│   │   ├── unimapocc/
│   │   │   └── unimapocc_final.py         # ★ 建議使用的最終獨立 config
│   │   └── ProtoOcc/
│   │       ├── ProtoOcc_1key.py            # 原始 OCC-only baseline
│   │       ├── ProtoOcc_multi_cnn_head.py  # multi-task detector + Map head 起點
│   │       ├── ProtoOcc_multi_cnn_head_map_neck.py
│   │       └── ProtoOcc_multi_..._active_enhance_gate.py
│   │                                           # 歷史繼承式最終 config
│   │
│   └── mmdet3d_plugin/
│       ├── models/
│       │   ├── detectors/ProtoOccCnnSegHead.py
│       │   ├── backbones/dual_branch_encoder.py
│       │   ├── backbones/resnet.py
│       │   ├── necks/depth_net.py
│       │   ├── necks/view_transformer.py
│       │   ├── necks/lss_fpn.py
│       │   ├── dense_heads/cnn3d_decoder.py
│       │   ├── dense_heads/bev_seg_head.py
│       │   ├── OccHead/Prototype_Query_Decoder_nuScenes.py
│       │   └── losses/focal_loss.py
│       ├── datasets/
│       │   ├── nuscenes_dataset_multitask.py
│       │   └── pipelines/
│       │       ├── loading.py
│       │       ├── loading_bev_seg.py
│       │       └── formating_multitask.py
│       └── core/hook/ema.py
│
├── tools/
│   ├── dist_train.sh / train.py
│   ├── dist_test.sh / test.py
│   ├── visualize_map_sample.py            # 單 sample Map prediction／GT
│   └── analysis_tools/vis_occ.py          # exported OCC prediction 視覺化
│
├── doc/
│   ├── install_4090.md                    # 本機 CUDA 11.8／Torch 2.0.1 安裝方式
│   └── install_TWCC 補充：unimapocc 環境重建流程.md
│
├── mmdetection3d/                         # ☁ v1.0.0rc4 editable dependency
├── data/                                  # ☁ nuScenes／Occ3D／panoptic 共用資料連結
├── ckpts/                                 # ☁ pretrained weights
└── work_dirs/                             # ☁ logs、config dump、checkpoint、result.md
```

> `☁` 代表目前被 `.gitignore` 排除或屬於本機大型 artifact；只 push 主 repository 並不會自動把它們交給下一屆。

---

## 5. 核心檔案功能對照

| 路徑 | 主要 class／功能 | 對應 config 欄位 |
| --- | --- | --- |
| `models/detectors/ProtoOccCnnSegHead.py` | 串接 image-to-voxel、OCC branch、Map branch；縮放 Map losses；train/test output | `model.type='ProtoOccCnnSegHead'`、`map_loss_weight` |
| `models/backbones/dual_branch_encoder.py` | ProtoOcc DBE、`MapHFMFusionLayer`、`PerScaleMapResidualAdapter`、三尺度 Map feature route | `dual_branch_encoder` |
| `models/backbones/resnet.py` | `CustomBEVBackbone`，產生 100／50／25 三尺度 BEV feature | `bev_encoder_backbone` |
| `models/necks/depth_net.py` | `CM_DepthNet`，預測 depth distribution 與 80-channel context | `depth_net` |
| `models/necks/view_transformer.py` | `LSSViewTransformer_depthGT`，把六視角影像 lift 到 3D voxel grid | `img_view_transformer` |
| `models/necks/lss_fpn.py` | `Custom_FPN_LSS`；OCC neck 與 128-channel Map neck；lateral projection／ASPP | `bev_encoder_neck`、`map_bev_encoder_neck` |
| `models/dense_heads/cnn3d_decoder.py` | OCC coarse prediction 與 prototype mask feature | `cnn3d_decoder` |
| `models/OccHead/Prototype_Query_Decoder_nuScenes.py` | ProtoOcc 的 PQD occupancy decoder | `prototype_query_decoder` |
| `models/dense_heads/bev_seg_head.py` | 六類 Map decoder、weighted losses、two-channel active enhancement gate | `bev_seg_head` |
| `models/losses/focal_loss.py` | 本地 `BinaryMaskFocalLoss` | `map_loss_type='focal_dice'` |
| `datasets/pipelines/loading_bev_seg.py` | 用官方 nuScenes map API 即時 rasterize `gt_masks_bev` | `LoadBEVSegmentation` |
| `datasets/pipelines/loading.py` | 影像、BDA、Occ3D GT、depth 與 panoptic auxiliary GT | train/test pipeline |
| `datasets/nuscenes_dataset_multitask.py` | 同時 dispatch OCC mIoU 與 Map IoU；可輸出 `map.npz` | `dataset_type='NuScenesDatasetMultitask'` |
| `core/hook/ema.py` | 每個 epoch 儲存 `epoch_<n>_ema.pth` | `MEGVIIEMAHook` |

### Dataset GT 對照

- `LoadOccGTFromFile` 讀取 Occ3D 的 `labels.npz`，提供 3D occupancy GT。
- `LoadBEVSegmentation` 從 nuScenes map API 即時 rasterize 2D BEV map GT。
- `LoadLidarsegFromFile` 讀取 `pc_panoptic`，提供 perspective semantic auxiliary supervision。
- 三者分別提供不同訓練目標，完整多任務訓練都會用到。

---

## 6. 環境

### 6.1 本機已驗證環境快照

```text
conda env       mapocc
Python          3.8.20
PyTorch         2.0.1+cu118
torch CUDA      11.8
MMCV            1.7.2
MMDetection     2.28.2
MMSegmentation  0.30.0
MMDetection3D   1.0.0rc4
```

完整安裝請先看 `doc/install_4090.md`。`mmdetection3d/` 是獨立、被主 repo 忽略的 checkout；本機使用 tag `v1.0.0rc4`，`mmdet3d/__init__.py` 的 MMCV upper bound 配合目前的 MMCV 1.7.2。

安裝或搬移 repo 後，重新執行 editable install：

```bash
cd /media/robot/16TB/ARTC2026/Ryan/ProtoOcc/mmdetection3d
python -m pip install -v -e .

cd ../projects
python -m pip install -v -e .

cd ..
python -c "import mmdet3d, projects; print(mmdet3d.__file__); print(projects.__file__)"
```

---

## 7. 資料與 pretrained checkpoint

config 預期 repo 根目錄下存在：

```text
data/nuscenes/
├── v1.0-trainval/
├── maps/
├── samples/
├── sweeps/
├── panoptic/
├── pc_panoptic/
├── gts/
├── bevdetv2-nuscenes_infos_train.pkl
├── bevdetv2-nuscenes_infos_val.pkl
└── bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl  # smoke test 用，若有

ckpts/
└── bevdet-r50-4d-depth-cbgs_depthnet_modify.pth
```

本機資料配置（2026-08-03 核對）：

```text
data -> /media/robot/16TB/ARTC2026/MPOcc/data
```

下列資料已確認可讀：nuScenes metadata、`samples/`、`sweeps/`、`maps/`、Occ3D `gts/`、`panoptic/`、`pc_panoptic/`、train／val info PKL，以及 quarter-train smoke-test PKL。

使用方式：

- `data_root` 在 config 中是相對路徑 `data/nuscenes/`，所有正式命令都應從 repo 根目錄執行。
- info pkl 內的 `occ_path` 也必須能解析到實際 `labels.npz`。
- Map GT 依賴 nuScenes map files 與四個 location metadata；只有影像與 Occ3D `gts/` 仍不足以跑 Map task。
- `data/`、`ckpts/`、`work_dirs/` 都在 `.gitignore`，交接時需另外複製、建立共享連結或提供下載位置。

接手後可用以下命令確認資料連結與必要檔案：

```bash
readlink -f data
test -f data/nuscenes/v1.0-trainval/scene.json
test -d data/nuscenes/maps
test -d data/nuscenes/gts
test -d data/nuscenes/pc_panoptic
test -f data/nuscenes/bevdetv2-nuscenes_infos_train.pkl
test -f data/nuscenes/bevdetv2-nuscenes_infos_val.pkl
```

---

## 8. 訓練

所有命令都從 repo 根目錄執行：

```bash
cd /media/robot/16TB/ARTC2026/Ryan/ProtoOcc
source /home/robot/miniconda3/etc/profile.d/conda.sh
conda activate mapocc
export PYTHONNOUSERSITE=1
unset PIP_USER
export PYTHONPATH="$PWD:$PWD/mmdetection3d:$PWD/projects:${PYTHONPATH:-}"
```

### 8.1 一般完整訓練模板

```bash
CONFIG=projects/configs/unimapocc/unimapocc_final.py
WORK_DIR=work_dirs/unimapocc_final
GPUS=4  # 依實際可用 GPU 數修改

PORT=29500 bash tools/dist_train.sh \
  "$CONFIG" \
  "$GPUS" \
  --work-dir "$WORK_DIR"
```

這個命令使用獨立 config 內的 default：24 epochs、AdamW `2e-4`、`samples_per_gpu=4`。若 GPU 記憶體不足，可降低 `data.samples_per_gpu`；batch size、GPU 數或 learning rate 改變後，請建立新的實驗名稱並記錄完整設定。

### 8.2 單卡 smoke test

```bash
CONFIG=projects/configs/unimapocc/unimapocc_final.py
WORK_DIR=work_dirs/smoke_unimapocc_final_1gpu

CUDA_VISIBLE_DEVICES=0 PORT=29511 bash tools/dist_train.sh \
  "$CONFIG" \
  1 \
  --work-dir "$WORK_DIR" \
  --cfg-options \
  data.train.ann_file=data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl \
  data.samples_per_gpu=1 \
  data.workers_per_gpu=2 \
  runner.max_epochs=1 \
  evaluation.interval=999 \
  checkpoint_config.interval=1 \
  checkpoint_config.max_keep_ckpts=-1
```

這只用來確認 dataset、forward、backward、loss 與 checkpoint pipeline 能工作，不能拿 smoke metric 與 full training 結果比較。

### 8.3 歷史最終 H200 設定

既有 `epoch_24_ema.pth` 的原始 job 使用繼承式 config，並在 8 GPUs 上套用下列 overrides：

```bash
CONFIG=projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate.py
WORK_DIR=work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200

PORT=29500 bash tools/dist_train.sh \
  "$CONFIG" \
  8 \
  --work-dir "$WORK_DIR" \
  --cfg-options \
  data.samples_per_gpu=4 \
  data.workers_per_gpu=4 \
  optimizer.lr=4e-4 \
  runner.max_epochs=24 \
  evaluation.interval=999 \
  checkpoint_config.interval=24
```

歷史 job 的實際生效設定記錄在該 work directory 的 `codex_train_stdout.log` Config dump；新實驗以 `unimapocc_final.py` 為主 config。

### 8.4 Checkpoint 與 resume

- regular checkpoint：由 `checkpoint_config.interval` 決定，含 model、optimizer、epoch 等 runner state，可用於 strict resume。
- EMA checkpoint：`MEGVIIEMAHook` 每個 epoch 另存 `epoch_<n>_ema.pth`，適合 evaluation；單獨使用時不含完整 optimizer／runner state。
- strict resume 應同時保存 regular 與相同 epoch 的 EMA：

```bash
CONFIG=projects/configs/unimapocc/unimapocc_final.py
WORK_DIR=work_dirs/resume_example  # 改成原本的 work directory
RESUME_EPOCH=12
MAX_EPOCHS=24
GPUS=4

PORT=29500 bash tools/dist_train.sh \
  "$CONFIG" \
  "$GPUS" \
  --work-dir "$WORK_DIR" \
  --resume-from "$WORK_DIR/epoch_${RESUME_EPOCH}.pth" \
  --cfg-options \
  custom_hooks.0.resume="$WORK_DIR/epoch_${RESUME_EPOCH}_ema.pth" \
  runner.max_epochs="$MAX_EPOCHS"
```

---

## 9. 評估

### 9.1 OCC + Map 完整評估

```bash
CONFIG=projects/configs/unimapocc/unimapocc_final.py
CKPT=work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200/epoch_24_ema.pth

CUDA_VISIBLE_DEVICES=0 PORT=29521 bash tools/dist_test.sh \
  "$CONFIG" \
  "$CKPT" \
  1 \
  --eval miou map-miou
```

上面使用交接後建議的獨立 config。若目的是逐字對照歷史 `result.md` 內的 command provenance，可把 `CONFIG` 換回第 2.2 節的繼承式 final config；在目前程式版本下兩者的 model runtime 語意相同。

一定要使用 `map-miou`（hyphen），不是 `map_miou`（underscore）。目前 dataset 接受的 Map aliases 為 `map`、`map-miou`、`bev-seg`、`bev-seg-miou`。

### 9.2 保存可重用 inference output

完整 val inference 很花時間，建議第一次就加 `--out`：

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29521 bash tools/dist_test.sh \
  "$CONFIG" \
  "$CKPT" \
  1 \
  --out work_dirs/unimapocc_epoch24_ema_outputs.pkl \
  --eval miou map-miou
```

若要同時匯出 OCC 與 Map prediction：

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29521 bash tools/dist_test.sh \
  "$CONFIG" \
  "$CKPT" \
  1 \
  --eval miou map-miou \
  --eval-options \
  show_dir=work_dirs/qualitative_occ/unimapocc_epoch24 \
  map_show_dir=work_dirs/qualitative_map/unimapocc_epoch24
```

輸出格式：

- OCC：`<show_dir>/<scene>/<sample_token>/pred.npz`
- Map：`<map_show_dir>/<scene>/<sample_token>/map.npz`

### 9.3 Metric 解讀

- OCC 主指標：log 中的 `mIoU of 6019 samples`。
- Map 主指標：`map/mean/iou@max`。
- Map evaluator 會對 thresholds `0.35, 0.40, ..., 0.65` 計算每類 IoU；`map/mean/iou@max` 是「每類先取自己的最佳 threshold，再對六類平均」，不是單一固定 threshold 的 global mIoU。

### 9.4 單一 sample 的 Map 視覺化

```bash
CUDA_VISIBLE_DEVICES=0 python tools/visualize_map_sample.py \
  projects/configs/unimapocc/unimapocc_final.py \
  work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200/epoch_24_ema.pth \
  --sample-index 0 \
  --threshold 0.5 \
  --save-gt \
  --out-dir work_dirs/qualitative_map/unimapocc_single_sample
```

可改用 `--sample-token <token>` 精準指定 frame；此工具需要 CUDA。

---

## 10. 歷史最終 artifact 與結果

### 10.1 Artifact

```text
work_dirs/
└── ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_nano4_h200/
    ├── 20260605_031208.log
    ├── codex_train_stdout.log
    ├── codex_eval_stdout.log
    ├── epoch_12_ema.pth
    ├── epoch_19_ema.pth
    ├── epoch_24_ema.pth             # 主要交接 checkpoint
    └── result.md
```

### 10.2 `epoch_24_ema.pth` 記錄數字

`result.md` 的主 summary 記錄：

| Metric | Value |
| --- | ---: |
| OCC mIoU | 39.58 |
| Map mean `iou@max` | 0.480266 |
| drivable_area | 0.788812 |
| ped_crossing | 0.440598 |
| walkway | 0.515463 |
| stop_line | 0.292826 |
| carpark_area | 0.471296 |
| divider | 0.372603 |

表中數字以 `result.md` 的主 summary 為準；後續評估請另存執行命令、config、checkpoint 與完整輸出。

### 10.3 交接 checksum（2026-08-03 快照）

| Artifact | SHA-256 |
| --- | --- |
| `unimapocc_final.py` | `c069ef31136f03079d61f7284af1187f70baf3bdc907a743f1a3179d1e311aa8` |
| 原始繼承式 final config | `9de1e87fc3c6c2ba669d2c1acf6b1439fe3ca0dbf09fbd073c63f669cb63e2f9` |
| `epoch_24_ema.pth` | `7e18ec2d727b22b48b1e2e536459886a720117ad71be21c3ac22922053bd004d` |
| pretrained depth checkpoint | `489de932c5f11f27fcc9482ad57ab0b11f90582d2585fe7e030b886ce84c3aa8` |
| `result.md` | `f8891e8c90402c492bce30e48d7e38a15ea3812ac725afc5e52151d26ea45429` |

檔案只要修改，checksum 就會改變；它的用途是確認交接複製過程沒有損壞，不是永久版本號。

---

## 11. Baseline 與 UniMapOcc 快速對照

| 項目 | 原始 ProtoOcc baseline | UniMapOcc final |
| --- | --- | --- |
| 主 config | `ProtoOcc_1key.py` | `unimapocc_final.py` |
| detector | `ProtoOcc` | `ProtoOccCnnSegHead` |
| 推論輸入 | 六相機 | 六相機 |
| OCC DBE／PQD | 有 | 保留 |
| BEV Map head | 無 | `BEVSegHead`，六類 |
| Map-specific geometry | 無 | Map-HFM |
| Map-specific refinement | 無 | per-scale residual adapter + FPN lateral + ASPP |
| Map loss | 無 | class-weighted focal + Dice，外層 `map_loss_weight=4` |
| Map active gate | 無 | two-channel group-aware gate |
| 輸出 | OCC | OCC + Map |

此表只比較目前最終 code path，不包含其他 ablation config。

---

## 12. 修改程式時的最低驗證

### 12.1 只改 config

```bash
python -m py_compile projects/configs/unimapocc/unimapocc_final.py
python -c "from mmcv import Config; Config.fromfile('projects/configs/unimapocc/unimapocc_final.py'); print('parse OK')"
```

再跑第 2.3 節 parity check。

### 12.2 改 detector／encoder／head

至少完成：

1. import／registry check；
2. config parse；
3. one-batch forward smoke；
4. one-epoch smoke train；
5. EMA checkpoint eval，且同時看到 OCC 與 Map metric；
6. 在新的 `work_dirs/<run_name>/result.md` 記錄 config、命令、checkpoint、OCC mIoU、Map mean 與六類 Map IoU。

不要覆寫舊 `result.md`；新增帶日期的 section，保留先前 evidence。

---

## 13. 交接打包清單

### 必須進 Git／code archive

- `README_UniMapOcc.md`
- `projects/configs/unimapocc/unimapocc_final.py`
- 原始繼承式 config 與其 base chain
- `projects/mmdet3d_plugin/` 內本專案使用到的 detector、encoder、neck、head、loss、dataset、pipeline、hook
- `tools/dist_train.sh`、`tools/dist_test.sh`、`tools/train.py`、`tools/test.py`
- `tools/visualize_map_sample.py`
- `train_eval_record_runbook.md`
- 環境安裝文件

### 必須另外保存或提供下載位置

- nuScenes、Occ3D GT、nuScenes panoptic 與 preprocess pkl
- `ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth`
- `epoch_24_ema.pth`
- `result.md` 與原始 train／eval logs
- 可重現環境的 package list 或 container
- 被主 repo 忽略的 `mmdetection3d v1.0.0rc4` checkout，以及必要 local patch

### 下一屆完成接手的判定

- [ ] 能從 `unimapocc_final.py` 成功 parse config。
- [ ] `mmdet3d` 與 `projects` import 都指到正確 checkout。
- [ ] data link、train/val pkl、Occ3D GT、panoptic 與 maps 都可讀。
- [ ] pretrained depth checkpoint 與 final EMA checksum 正確。
- [ ] 能完成一次 one-epoch smoke train。
- [ ] 能用 final EMA 跑出 OCC mIoU 與 `map/mean/iou@max`。
- [ ] 能用 `tools/visualize_map_sample.py` 輸出 prediction 與 GT。
- [ ] 新實驗會建立新的 work directory 並更新 `result.md`。

完成以上項目，才算是程式、資料、環境、權重與實驗 evidence 都真正交接完成。
