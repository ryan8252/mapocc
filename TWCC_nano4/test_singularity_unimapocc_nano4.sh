#!/bin/bash

# Nano4 / 25a-lgn01 preflight test for the Singularity + unimapocc env.
# This job does not start training. It only checks container GPU access,
# Python package versions, CUDA 11.8 alignment, OpenMMLab imports, mmcv CUDA ops,
# project plugin import, config loading, and key file paths.
#
# Submit with the Nano4 GPU dev partition:
#   sbatch test_singularity_unimapocc_nano4.sh
#
# `ngstest` has no GPU GRES on 25a-lgn01, so do not use it for --nv tests.

#SBATCH -J nano4_env_test
#SBATCH --account=MST113104
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.log

set -euo pipefail

module purge
module load singularity/4.3.7

CONDA_ENV="${CONDA_ENV:-/home/u2336262/.conda/envs/unimapocc}"
PROTOOCC_DIR="${PROTOOCC_DIR:-/home/u2336262/Desktop/artc_2026/mapocc}"
SIF="${SIF:-/home/u2336262/Desktop/artc_2026/containers/cuda118-cudnn8-devel-ubuntu20.04.sif}"
CONFIG="${CONFIG:-projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp_focal_dice_weighted.py}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${PROTOOCC_DIR}/ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth}"
TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-}"

for path in "${CONDA_ENV}" "${PROTOOCC_DIR}"; do
    if [ ! -d "${path}" ]; then
        echo "[ERROR] Directory not found: ${path}"
        exit 1
    fi
done

for path in "${SIF}" "${PROTOOCC_DIR}/${CONFIG}" "${PRETRAIN_CKPT}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] File not found: ${path}"
        exit 1
    fi
done

echo "[INFO] Host: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-n/a}"
echo "[INFO] SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-n/a}"
echo "[INFO] ProtoOcc dir: ${PROTOOCC_DIR}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Singularity image: ${SIF}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Pretrain checkpoint: ${PRETRAIN_CKPT}"
if [ -n "${TRAIN_ANN_FILE}" ]; then
    echo "[INFO] Train ann_file override: ${TRAIN_ANN_FILE}"
fi

SINGULARITY_BIND_ARGS=(
    --bind /home/u2336262:/home/u2336262
    --bind /work:/work
)

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
export PRETRAIN_CKPT="$4"
export TRAIN_ANN_FILE="$5"

export CUDA_HOME=/usr/local/cuda
export PATH="${ENV_PATH}/bin:${CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER

cd "${PROTOOCC_DIR}"

echo "[CHECK] nvidia-smi"
nvidia-smi

echo "[CHECK] nvcc"
nvcc --version

echo "[CHECK] Python executable"
python -c "import sys; print(sys.executable)"

echo "[CHECK] Torch CUDA"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); assert torch.version.cuda == \"11.8\", torch.version.cuda; print(torch.ones(1, device=\"cuda\"))"

echo "[CHECK] NumPy / NetworkX"
python -c "import numpy as np; print(np.__version__); assert np.__version__.startswith(\"1.23.\"), np.__version__"
python -c "import networkx as nx; print(nx.__version__); assert nx.__version__ == \"2.2\", nx.__version__"

echo "[CHECK] OpenCV headless"
python -c "import cv2; print(cv2.__version__)"

echo "[CHECK] OpenMMLab imports"
python -c "import mmcv, mmdet, mmseg, mmdet3d; print(mmcv.__version__, mmdet.__version__, mmseg.__version__, mmdet3d.__version__)"

echo "[CHECK] MMCV CUDA ops"
python -c "from mmcv.ops import get_compiler_version, get_compiling_cuda_version; print(get_compiler_version()); print(get_compiling_cuda_version())"

echo "[CHECK] ProtoOcc plugin import"
python -c "import importlib; importlib.import_module(\"projects.mmdet3d_plugin\"); print(\"projects.mmdet3d_plugin OK\")"

echo "[CHECK] Config load"
pwd
ls -ld data data/nuscenes
ls -lh data/nuscenes/*infos*.pkl
python -c "import os; from mmcv import Config; cfg = Config.fromfile(os.environ[\"CONFIG\"]); ann = os.environ.get(\"TRAIN_ANN_FILE\") or cfg.data.train.ann_file; print(cfg.model.type); print(\"train ann_file\", ann); assert os.path.exists(ann), \"train ann_file not found: \" + ann"
python -c "import os; from mmcv import Config; cfg = Config.fromfile(os.environ[\"CONFIG\"]); ann = cfg.data.val.ann_file; print(\"val ann_file\", ann); assert os.path.exists(ann), \"val ann_file not found: \" + ann"

echo "[CHECK] Key files"
python -c "import os; print(os.path.exists(os.environ[\"PRETRAIN_CKPT\"])); print(os.path.exists(\"data/nuscenes\"))"

echo "[OK] Nano4 Singularity unimapocc preflight passed."
' _ \
    "${CONDA_ENV}" \
    "${PROTOOCC_DIR}" \
    "${CONFIG}" \
    "${PRETRAIN_CKPT}" \
    "${TRAIN_ANN_FILE}"
