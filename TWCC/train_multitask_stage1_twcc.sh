#!/bin/bash

# TWCC sbatch script for unimapocc / ProtoOcc multitask stage 1
# Usage:
#   sbatch train_multitask_stage1_twcc.sh

#SBATCH -J unimapocc_s1
#SBATCH --account=MST113104
#SBATCH -p gp4d
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o %j.log
#SBATCH -e %j.log

set -euo pipefail

source /home/u2336262/miniconda3/etc/profile.d/conda.sh
conda activate /home/u2336262/.conda/envs/unimapocc

module purge
module load cuda/11.7
module load openmpi/4.1.6_ucx1.14.1_cuda12.3

export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Avoid picking up ~/.local packages or unrelated project paths.
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROTOOCC_DIR="${REPO_ROOT}/ProtoOcc"

CONFIG="projects/configs/ProtoOcc/ProtoOcc_multitask_stage1.py"
WORK_DIR="${PROTOOCC_DIR}/work_dirs/ProtoOcc_multitask_stage1"
PRETRAIN_CKPT="${PROTOOCC_DIR}/work_dirs/ProtoOcc_1key/ProtoOcc_1key.pth"
GPUS=8

if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "[ERROR] Pretrained checkpoint not found:"
    echo "        ${PRETRAIN_CKPT}"
    echo "        Please prepare ProtoOcc_1key.pth before training stage 1."
    exit 1
fi

cd "${PROTOOCC_DIR}"
mkdir -p "${WORK_DIR}"

export MASTER_PORT="$(shuf -i 20000-30000 -n 1)"

echo "[INFO] Working directory: ${PROTOOCC_DIR}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Work dir: ${WORK_DIR}"
echo "[INFO] GPUs: ${GPUS}"
echo "[INFO] MASTER_PORT: ${MASTER_PORT}"

bash tools/dist_train.sh "${CONFIG}" "${GPUS}" --work-dir "${WORK_DIR}"
