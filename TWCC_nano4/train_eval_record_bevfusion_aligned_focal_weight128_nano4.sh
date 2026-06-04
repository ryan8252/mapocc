#!/bin/bash

# Nano4 / 25a-lgn01 Singularity sbatch launcher for:
# ProtoOcc BEVFusion-aligned focal-only map loss, map_loss_weight=128.
#
# Default regime:
#   full nuScenes train split, 24 epochs, train -> eval epoch_24_ema.pth
#   -> result.md.
#
# Submit from the ProtoOcc checkout on Nano4:
#   sbatch TWCC_nano4/train_eval_record_bevfusion_aligned_focal_weight128_nano4.sh
#
# This wraps quick_test_nano4.sh so the container setup and result recorder stay
# identical to the known-good Nano4 workflow.

#SBATCH -J aligned_f128_24
#SBATCH --account=MST113104
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1536G
#SBATCH --time=48:00:00
#SBATCH -o %j.log
#SBATCH -e %j.log

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export PROTOOCC_DIR="${PROTOOCC_DIR:-/home/u2336262/Desktop/artc_2026/mapocc}"

export RUN_MODE="${RUN_MODE:-full}"
export CONFIG="${CONFIG:-projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_focal_weight128.py}"
export WORK_DIR="${WORK_DIR:-${PROTOOCC_DIR}/work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_focal_weight128_nano4_h200}"

# Aggressive H200 regime: 8 GPUs * 4 samples/GPU = 32, workers/GPU = 4.
# LR is scaled to 4e-4 for this doubled global batch.
export GPUS="${GPUS:-8}"
export SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
export LR="${LR:-4e-4}"
export EPOCHS="${EPOCHS:-24}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-24}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-999}"
export EVAL_TIMEOUT_MIN="${EVAL_TIMEOUT_MIN:-180}"
export TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-}"

if [ -f "${SUBMIT_DIR}/TWCC_nano4/quick_test_nano4.sh" ]; then
    QUICK_TEST="${SUBMIT_DIR}/TWCC_nano4/quick_test_nano4.sh"
elif [ -f "${PROTOOCC_DIR}/TWCC_nano4/quick_test_nano4.sh" ]; then
    QUICK_TEST="${PROTOOCC_DIR}/TWCC_nano4/quick_test_nano4.sh"
else
    echo "[ERROR] quick_test_nano4.sh not found."
    echo "[ERROR] Checked: ${SUBMIT_DIR}/TWCC_nano4/quick_test_nano4.sh"
    echo "[ERROR] Checked: ${PROTOOCC_DIR}/TWCC_nano4/quick_test_nano4.sh"
    exit 1
fi

exec bash "${QUICK_TEST}" "${CONFIG}"
