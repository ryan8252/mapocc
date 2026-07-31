#!/bin/bash

# Nano4 / 25a-lgn01 Singularity sbatch launcher for the adapter-first
# additive Map-HFM smoke ablation.
#
# Default regime:
#   smoke: 1 epoch, 1quarter train split, train -> eval epoch_1_ema.pth
#   -> result.md.
#
# Submit from the ProtoOcc checkout on Nano4:
#   sbatch TWCC_nano4/train_eval_record_adapter_first_additive_hfm_nano4.sh
#
# Full-run override:
#   RUN_MODE=full sbatch --export=ALL \
#     TWCC_nano4/train_eval_record_adapter_first_additive_hfm_nano4.sh

#SBATCH -J adapthfm_smoke
#SBATCH --account=MST113104
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=4:00:00
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.log

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export PROTOOCC_DIR="${PROTOOCC_DIR:-/home/u2336262/Desktop/artc_2026/mapocc}"

export RUN_MODE="${RUN_MODE:-smoke}"
export CONFIG="${CONFIG:-projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm.py}"

case "${RUN_MODE}" in
    smoke)
        DEFAULT_EPOCHS=1
        DEFAULT_TRAIN_ANN_FILE="data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl"
        DEFAULT_WORK_DIR="${PROTOOCC_DIR}/work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_1quarter_nano4_2gpu"
        DEFAULT_EVAL_TIMEOUT_MIN=90
        DEFAULT_GPUS=2
        DEFAULT_SAMPLES_PER_GPU=2
        DEFAULT_WORKERS_PER_GPU=1
        DEFAULT_LR=2e-4
        ;;
    full)
        DEFAULT_EPOCHS=24
        DEFAULT_TRAIN_ANN_FILE=""
        DEFAULT_WORK_DIR="${PROTOOCC_DIR}/work_dirs/ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_nano4_h200"
        DEFAULT_EVAL_TIMEOUT_MIN=180
        DEFAULT_GPUS=8
        DEFAULT_SAMPLES_PER_GPU=4
        DEFAULT_WORKERS_PER_GPU=4
        DEFAULT_LR=2e-4
        ;;
    *)
        echo "[ERROR] RUN_MODE must be smoke or full, got: ${RUN_MODE}"
        exit 1
        ;;
esac

export GPUS="${GPUS:-${DEFAULT_GPUS}}"
export SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-${DEFAULT_SAMPLES_PER_GPU}}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-${DEFAULT_WORKERS_PER_GPU}}"
export LR="${LR:-${DEFAULT_LR}}"
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
