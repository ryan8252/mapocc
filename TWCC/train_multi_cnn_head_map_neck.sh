#!/bin/bash

# TWCC sbatch script for ProtoOcc + CNN BEVSegHead + 128-channel map neck
# Usage:
#   sbatch train_multi_cnn_head_map_neck.sh


#SBATCH -J multi_cnn_head_map_neck        # 任務名稱
#SBATCH --account=MST113104                          # 計畫帳號
#SBATCH -p gp4d                                      # 用可跑 2 天的分區
#SBATCH -N 1                                         # 申請 1 台主機
#SBATCH --ntasks-per-node=4                          # 每台機器 4 個任務
#SBATCH --gres=gpu:4                                 # 申請 4 顆 V100 GPU
#SBATCH --cpus-per-task=4                            # 每一顆 GPU 配 4 核 CPU
#SBATCH --mem=192G                                   # 申請 256GB 系統記憶體
#SBATCH -o %j.log                                    # 訓練 Log 輸出位置
#SBATCH -e %j.log                                    # 錯誤 Log 輸出位置

set -euo pipefail

source /home/u2336262/miniconda3/etc/profile.d/conda.sh
conda activate /home/u2336262/.conda/envs/unimapocc

module purge
module load cuda/11.7

export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROTOOCC_DIR="/home/u2336262/Desktop/artc_2026/mapocc"

CONFIG="projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py"
WORK_DIR="${PROTOOCC_DIR}/work_dirs/ProtoOcc_multi_cnn_head_map_neck"
PRETRAIN_CKPT="${PROTOOCC_DIR}/ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth"
GPUS=4
SAMPLES_PER_GPU=4
WORKERS_PER_GPU=1

# Original ProtoOcc LR is 2e-4 for 4 GPUs * 4 samples/GPU.
# TWCC uses 8 GPUs * 4 samples/GPU, so use linear scaling: 2e-4 * 2 = 4e-4.
LR=2e-4

if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "[ERROR] Pretrained checkpoint not found:"
    echo "        ${PRETRAIN_CKPT}"
    exit 1
fi

cd "${PROTOOCC_DIR}"
mkdir -p "${WORK_DIR}"

export PORT="$(shuf -i 20000-30000 -n 1)"

echo "[INFO] Working directory: ${PROTOOCC_DIR}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Work dir: ${WORK_DIR}"
echo "[INFO] GPUs: ${GPUS}"
echo "[INFO] Samples per GPU: ${SAMPLES_PER_GPU}"
echo "[INFO] Workers per GPU: ${WORKERS_PER_GPU}"
echo "[INFO] Global batch size: $((GPUS * SAMPLES_PER_GPU))"
echo "[INFO] Learning rate: ${LR}"
echo "[INFO] PORT: ${PORT}"

bash tools/dist_train.sh "${CONFIG}" "${GPUS}" --work-dir "${WORK_DIR}" \
    --cfg-options \
    data.samples_per_gpu="${SAMPLES_PER_GPU}" \
    data.workers_per_gpu="${WORKERS_PER_GPU}" \
    optimizer.lr="${LR}"


# 目前最強 
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 12.17
# ===> barrier - IoU = 47.95
# ===> bicycle - IoU = 26.34
# ===> bus - IoU = 44.81
# ===> car - IoU = 51.91
# ===> construction_vehicle - IoU = 23.47
# ===> motorcycle - IoU = 26.54
# ===> pedestrian - IoU = 27.64
# ===> traffic_cone - IoU = 28.26
# ===> trailer - IoU = 33.31
# ===> truck - IoU = 37.16
# ===> driveable_surface - IoU = 81.92
# ===> other_flat - IoU = 46.13
# ===> sidewalk - IoU = 53.28
# ===> terrain - IoU = 56.46
# ===> manmade - IoU = 42.89
# ===> vegetation - IoU = 36.67
# ===> mIoU of 6019 samples: 39.82
# {'mIoU': array([0.122, 0.48 , 0.263, 0.448, 0.519, 0.235, 0.265, 0.276, 0.283,
#        0.333, 0.372, 0.819, 0.461, 0.533, 0.565, 0.429, 0.367, 0.895]), 'TP': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0.]), 'FP': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0.]), 'FN': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0.]), 'GT_counts': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0.]), 'map/drivable_area/iou@max': 0.7636419534683228, 'map/drivable_area/iou@0.35': 0.761790931224823, 'map/drivable_area/iou@0.40': 0.7636066675186157, 'map/drivable_area/iou@0.45': 0.7636419534683228, 'map/drivable_area/iou@0.50': 0.7619627714157104, 'map/drivable_area/iou@0.55': 0.7584928274154663, 'map/drivable_area/iou@0.60': 0.7530829906463623, 'map/drivable_area/iou@0.65': 0.7454871535301208, 
# 'map/ped_crossing/iou@max': 0.30520492792129517, 'map/ped_crossing/iou@0.35': 0.30520492792129517, 'map/ped_crossing/iou@0.40': 0.30351799726486206, 'map/ped_crossing/iou@0.45': 0.300645112991333, 'map/ped_crossing/iou@0.50': 0.2964698076248169, 'map/ped_crossing/iou@0.55': 0.2911423146724701, 'map/ped_crossing/iou@0.60': 0.28424331545829773, 'map/ped_crossing/iou@0.65': 0.2754683196544647, 
# 'map/walkway/iou@max': 0.451495498418808, 'map/walkway/iou@0.35': 0.451495498418808, 'map/walkway/iou@0.40': 0.44879841804504395, 'map/walkway/iou@0.45': 0.4436418116092682, 'map/walkway/iou@0.50': 0.43621301651000977, 'map/walkway/iou@0.55': 0.4263114035129547, 'map/walkway/iou@0.60': 0.413851797580719, 'map/walkway/iou@0.65': 0.3986535370349884, 
# 'map/stop_line/iou@max': 0.20730510354042053, 'map/stop_line/iou@0.35': 0.20730510354042053, 'map/stop_line/iou@0.40': 0.20076099038124084, 'map/stop_line/iou@0.45': 0.193232923746109, 'map/stop_line/iou@0.50': 0.18507887423038483, 'map/stop_line/iou@0.55': 0.17600110173225403, 'map/stop_line/iou@0.60': 0.16621719300746918, 'map/stop_line/iou@0.65': 0.1556234210729599, 
# 'map/carpark_area/iou@max': 0.3924679756164551, 'map/carpark_area/iou@0.35': 0.3924679756164551, 'map/carpark_area/iou@0.40': 0.3907131552696228, 'map/carpark_area/iou@0.45': 0.3872010111808777, 'map/carpark_area/iou@0.50': 0.3821965456008911, 'map/carpark_area/iou@0.55': 0.375283807516098, 'map/carpark_area/iou@0.60': 0.3668643534183502, 'map/carpark_area/iou@0.65': 0.35714200139045715, 
# 'map/divider/iou@max': 0.27630800008773804, 'map/divider/iou@0.35': 0.27630800008773804, 'map/divider/iou@0.40': 0.2705865502357483, 'map/divider/iou@0.45': 0.2636789381504059, 'map/divider/iou@0.50': 0.256011426448822, 'map/divider/iou@0.55': 0.2475271373987198, 'map/divider/iou@0.60': 0.23803828656673431, 'map/divider/iou@0.65': 0.22739987075328827, 'map/mean/iou@max': 0.3994038999080658}
