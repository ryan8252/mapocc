#!/bin/bash

# Nano4 full train+eval launcher for A2:
# current best active-gate config + progressive map_loss_weight 4 -> 8.
#
# Submit from the ProtoOcc checkout on Nano4:
#   sbatch TWCC_nano4/train_eval_record_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a2_progressive_mapw4to8_nano4.sh

#SBATCH -J a2_mapw4to8
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

export RUN_MODE="full"
export CONFIG="${CONFIG:-projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a2_progressive_mapw4to8.py}"
export WORK_DIR="${WORK_DIR:-${PROTOOCC_DIR}/work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a2_progressive_mapw4to8_nano4_h200}"

export GPUS="${GPUS:-8}"
export SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
export LR="${LR:-4e-4}"
export EPOCHS="${EPOCHS:-24}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-12}"
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
