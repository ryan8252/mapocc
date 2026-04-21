#!/bin/bash

# TWCC sbatch script for ProtoOcc + prototype-based map head + map BEV neck (with PGBR)
# Usage:
#   sbatch train_proto_map_head_map_neck.sh


#SBATCH -J mapocc_proto_map_head_map_neck           # 任務名稱 (隨便取)
#SBATCH --account=MST113104      # 計畫帳號 (從教授的帳號中查)
#SBATCH -p gp4d                             # 用可跑 2 天的分區
#SBATCH -N 1                                # 申請 1 台主機
#SBATCH --ntasks-per-node=8                 # 建議與 -N 一起使用，代表每台機器 8 個任務
#SBATCH --gres=gpu:8                        # 申請 8 顆 V100 GPU
#SBATCH --cpus-per-task=4                   # 每一顆 GPU 配 4 核 CPU
#SBATCH --mem=256G                           # 申請 256GB 系統記憶體
#SBATCH -o %j.log                           # 訓練 Log 輸出位置
#SBATCH -e %j.log                           # 錯誤 Log 輸出位置

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
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROTOOCC_DIR="/home/u2336262/Desktop/artc_2026/mapocc"

CONFIG="projects/configs/ProtoOcc/ProtoOcc_proto_map_head_map_neck.py"
WORK_DIR="${PROTOOCC_DIR}/work_dirs/ProtoOcc_proto_map_head_map_neck_pgbr"
PRETRAIN_CKPT="${PROTOOCC_DIR}/ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth"
GPUS=8
SAMPLES_PER_GPU=4
WORKERS_PER_GPU=1

# Original ProtoOcc LR is 2e-4 for 4 GPUs * 4 samples/GPU.
# TWCC uses 8 GPUs * 4 samples/GPU, so use linear scaling: 2e-4 * 2 = 4e-4.
LR=4e-4

if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "[ERROR] Pretrained checkpoint not found:"
    echo "        ${PRETRAIN_CKPT}"
    echo "        Please prepare ProtoOcc_1key.pth before training stage 1."
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
