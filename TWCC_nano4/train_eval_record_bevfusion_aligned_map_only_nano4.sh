#!/bin/bash

# Nano4 / 25a-lgn01 Singularity sbatch launcher for:
# ProtoOcc BEVFusion-aligned map-only upper-bound diagnostic.
#
# Default regime:
#   full nuScenes train split, 24 epochs, train -> eval epoch_24_ema.pth
#   -> result.md.
#
# Submit from the ProtoOcc checkout on Nano4:
#   sbatch TWCC_nano4/train_eval_record_bevfusion_aligned_map_only_nano4.sh
#
# Smoke example:
#   RUN_MODE=smoke sbatch --export=ALL TWCC_nano4/train_eval_record_bevfusion_aligned_map_only_nano4.sh
#
# This wraps quick_test_nano4.sh so the container setup, checkpoint selection,
# and result recorder stay identical to the known-good Nano4 workflow.

#SBATCH -J aligned_maponly
#SBATCH --account=MST113104
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1536G
#SBATCH --time=48:00:00
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.log

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export PROTOOCC_DIR="${PROTOOCC_DIR:-/home/u2336262/Desktop/artc_2026/mapocc}"

export RUN_MODE="${RUN_MODE:-full}"
export CONFIG="${CONFIG:-projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only.py}"

case "${RUN_MODE}" in
    smoke)
        DEFAULT_EPOCHS=1
        DEFAULT_TRAIN_ANN_FILE="data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl"
        DEFAULT_WORK_DIR="${PROTOOCC_DIR}/work_dirs/smoke_bevfusion_aligned_map_only_1quarter_nano4_h200"
        DEFAULT_EVAL_TIMEOUT_MIN=90
        ;;
    full)
        DEFAULT_EPOCHS=24
        DEFAULT_TRAIN_ANN_FILE=""
        DEFAULT_WORK_DIR="${PROTOOCC_DIR}/work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_nano4_h200"
        DEFAULT_EVAL_TIMEOUT_MIN=180
        ;;
    *)
        echo "[ERROR] RUN_MODE must be smoke or full, got: ${RUN_MODE}"
        exit 1
        ;;
esac

# H200 full-regime defaults: 8 GPUs * 4 samples/GPU = 32, workers/GPU = 4.
# LR matches the aligned focal128 full run for an apples-to-apples upper bound.
export GPUS="${GPUS:-8}"
export SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
export LR="${LR:-4e-4}"
export EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-${EPOCHS}}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-999}"
export EVAL_TIMEOUT_MIN="${EVAL_TIMEOUT_MIN:-${DEFAULT_EVAL_TIMEOUT_MIN}}"
export TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-${DEFAULT_TRAIN_ANN_FILE}}"
export WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"

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
