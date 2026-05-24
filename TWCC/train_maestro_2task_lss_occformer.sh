#!/bin/bash

# TWCC sbatch script for MAESTRO 2-task LSS + OccFormer reproduction.
# Default is the paper-range 51.2m shared-canvas variant.
# Usage:
#   sbatch train_maestro_2task_lss_occformer_51m2.sh

#SBATCH -J maestro_2task
#SBATCH --account=MST113104
#SBATCH -p gp4d
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH -o %j.log
#SBATCH -e %j.log

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

CONFIG="projects/configs/MAESTRO/MAESTRO_2task_lss_occformer.py"
WORK_DIR="${PROTOOCC_DIR}/work_dirs/MAESTRO_2task_lss_occformer"
PRETRAIN_CKPT="${PROTOOCC_DIR}/ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth"

GPUS=4

# MAESTRO reports batch size 8 across 4 RTX 3090 GPUs.
# On 4 TWCC GPUs, this is samples_per_gpu=2.
SAMPLES_PER_GPU=2
WORKERS_PER_GPU=2
LR=1e-4

if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "[ERROR] Pretrained checkpoint not found:"
    echo "        ${PRETRAIN_CKPT}"
    exit 1
fi

cd "${PROTOOCC_DIR}"
mkdir -p "${WORK_DIR}"

export PORT="$(shuf -i 20000-30000 -n 1)"

echo "[INFO] Script dir: ${SCRIPT_DIR}"
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
    optimizer.lr="${LR}" \
    optimizer.weight_decay=0.01
