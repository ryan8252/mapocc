#!/bin/bash

# Nano4 / 25a-lgn01 sbatch launcher for:
# ProtoOcc + FPN lateral ASPP map neck + focal-Dice weighted map loss.
#
# Submit from Nano4 login node:
#   sbatch train_fpn_lateral_aspp_focal_dice_weighted_nano4.sh
#
# Aggressive H200 regime:
#   8 GPUs * 4 samples/GPU = 32, workers/GPU = 1, lr = 4e-4.
# Nano4 H200 nodes have 2TB host memory; this launcher requests 1.5TB because
# ProtoOcc map loading is host-memory sensitive.

#SBATCH -J fpn_aspp_fdw
#SBATCH --account=MST113104
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=1536G
#SBATCH --time=48:00:00
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.log

set -euo pipefail

module purge
module load gcc/11.5
module load openmpi/5.0.10-cuda13.0

CONDA_SH="${CONDA_SH:-/home/u2336262/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/home/u2336262/.conda/envs/unimapocc}"
PROTOOCC_DIR="${PROTOOCC_DIR:-/home/u2336262/Desktop/artc_2026/mapocc}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER

CONFIG="projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted.py"
WORK_DIR="${PROTOOCC_DIR}/work_dirs/ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted_nano4_h200"
PRETRAIN_CKPT="${PROTOOCC_DIR}/ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth"

GPUS="${GPUS:-8}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
LR="${LR:-4e-4}"

if [ ! -d "${PROTOOCC_DIR}" ]; then
    echo "[ERROR] ProtoOcc directory not found: ${PROTOOCC_DIR}"
    exit 1
fi

if [ ! -f "${PROTOOCC_DIR}/${CONFIG}" ]; then
    echo "[ERROR] Config not found: ${PROTOOCC_DIR}/${CONFIG}"
    exit 1
fi

if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "[ERROR] Pretrained checkpoint not found: ${PRETRAIN_CKPT}"
    exit 1
fi

cd "${PROTOOCC_DIR}"
mkdir -p "${WORK_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PORT="${PORT:-$(shuf -i 20000-30000 -n 1)}"

echo "[INFO] Host: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-n/a}"
echo "[INFO] SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-n/a}"
echo "[INFO] ProtoOcc dir: ${PROTOOCC_DIR}"
echo "[INFO] Conda env: ${CONDA_ENV}"
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
