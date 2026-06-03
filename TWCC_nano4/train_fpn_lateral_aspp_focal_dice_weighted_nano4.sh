#!/bin/bash

# Nano4 / 25a-lgn01 Singularity sbatch launcher for:
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
module load singularity/4.3.7

CONDA_ENV="${CONDA_ENV:-/home/u2336262/.conda/envs/unimapocc}"
PROTOOCC_DIR="${PROTOOCC_DIR:-/home/u2336262/Desktop/artc_2026/mapocc}"
SIF="${SIF:-/home/u2336262/Desktop/artc_2026/containers/cuda118-cudnn8-devel-ubuntu20.04.sif}"

CONFIG="projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted.py"
WORK_DIR="${PROTOOCC_DIR}/work_dirs/ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted_nano4_h200"
PRETRAIN_CKPT="${PROTOOCC_DIR}/ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth"

GPUS="${GPUS:-8}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
LR="${LR:-4e-4}"
TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-}"

if [ ! -d "${PROTOOCC_DIR}" ]; then
    echo "[ERROR] ProtoOcc directory not found: ${PROTOOCC_DIR}"
    exit 1
fi

if [ ! -d "${CONDA_ENV}" ]; then
    echo "[ERROR] Conda env not found: ${CONDA_ENV}"
    exit 1
fi

if [ ! -f "${SIF}" ]; then
    echo "[ERROR] Singularity image not found: ${SIF}"
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

export PORT="${PORT:-$(shuf -i 20000-30000 -n 1)}"

echo "[INFO] Host: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-n/a}"
echo "[INFO] SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-n/a}"
echo "[INFO] ProtoOcc dir: ${PROTOOCC_DIR}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Singularity image: ${SIF}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Work dir: ${WORK_DIR}"
echo "[INFO] GPUs: ${GPUS}"
echo "[INFO] Samples per GPU: ${SAMPLES_PER_GPU}"
echo "[INFO] Workers per GPU: ${WORKERS_PER_GPU}"
echo "[INFO] Global batch size: $((GPUS * SAMPLES_PER_GPU))"
echo "[INFO] Learning rate: ${LR}"
echo "[INFO] PORT: ${PORT}"
if [ -n "${TRAIN_ANN_FILE}" ]; then
    echo "[INFO] Train ann_file override: ${TRAIN_ANN_FILE}"
fi

SINGULARITY_BIND_ARGS=(--bind /home/u2336262:/home/u2336262)
if [ -n "${EXTRA_BINDS:-}" ]; then
    SINGULARITY_BIND_ARGS+=(--bind "${EXTRA_BINDS}")
fi

singularity exec --cleanenv --nv \
    "${SINGULARITY_BIND_ARGS[@]}" \
    "${SIF}" \
    bash -c '
set -euo pipefail

export ENV_PATH="$1"
export PROTOOCC_DIR="$2"
export CONFIG="$3"
export WORK_DIR="$4"
export GPUS="$5"
export SAMPLES_PER_GPU="$6"
export WORKERS_PER_GPU="$7"
export LR="$8"
export PORT="$9"
export TRAIN_ANN_FILE="${10}"

export CUDA_HOME=/usr/local/cuda
export PATH="${ENV_PATH}/bin:${CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "${PROTOOCC_DIR}"

echo "[INFO] Container CUDA_HOME: ${CUDA_HOME}"
echo "[INFO] Python: $(python -c "import sys; print(sys.executable)")"
python -c "import torch; print(\"[INFO] Torch:\", torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.version.cuda == \"11.8\", torch.version.cuda; print(torch.ones(1, device=\"cuda\"))"
python -c "import mmcv, mmdet, mmseg; print(\"[INFO] OpenMMLab:\", mmcv.__version__, mmdet.__version__, mmseg.__version__)"
nvcc --version

pwd
ls -ld data data/nuscenes
ls -lh data/nuscenes/*infos*.pkl
python -c "import os; from mmcv import Config; cfg = Config.fromfile(os.environ[\"CONFIG\"]); ann = os.environ.get(\"TRAIN_ANN_FILE\") or cfg.data.train.ann_file; print(\"[INFO] Train ann_file:\", ann); assert os.path.exists(ann), \"train ann_file not found: \" + ann"
python -c "import os; from mmcv import Config; cfg = Config.fromfile(os.environ[\"CONFIG\"]); ann = cfg.data.val.ann_file; print(\"[INFO] Val ann_file:\", ann); assert os.path.exists(ann), \"val ann_file not found: \" + ann"

CFG_OPTIONS=(
    data.samples_per_gpu="${SAMPLES_PER_GPU}"
    data.workers_per_gpu="${WORKERS_PER_GPU}"
    optimizer.lr="${LR}"
)

if [ -n "${TRAIN_ANN_FILE}" ]; then
    CFG_OPTIONS+=(data.train.ann_file="${TRAIN_ANN_FILE}")
fi

bash tools/dist_train.sh "${CONFIG}" "${GPUS}" --work-dir "${WORK_DIR}" \
    --cfg-options "${CFG_OPTIONS[@]}"
' _ \
    "${CONDA_ENV}" \
    "${PROTOOCC_DIR}" \
    "${CONFIG}" \
    "${WORK_DIR}" \
    "${GPUS}" \
    "${SAMPLES_PER_GPU}" \
    "${WORKERS_PER_GPU}" \
    "${LR}" \
    "${PORT}" \
    "${TRAIN_ANN_FILE}"
