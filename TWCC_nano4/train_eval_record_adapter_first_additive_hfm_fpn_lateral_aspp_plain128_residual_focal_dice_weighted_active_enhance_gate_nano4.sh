#!/bin/bash

# Nano4 / 25a-lgn01 Singularity sbatch launcher for:
# ProtoOcc + adapter-first additive Map-HFM + FPN lateral ASPP map neck
# + plain 128ch FPN residual + focal-Dice weighted map loss
# + map-active feature enhancement gate.
#
# Default regime:
#   full nuScenes train split, 24 epochs, train -> eval epoch_24_ema.pth
#   -> result.md.
#
# Submit from the ProtoOcc checkout on Nano4:
#   sbatch TWCC_nano4/train_eval_record_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate_nano4.sh
#
# This wraps quick_test_nano4.sh so the container setup, checkpoint selection,
# eval, and result.md recorder stay identical to the known Nano4 workflow.

#SBATCH -J addhfm_res
#SBATCH --account=MST113104
#SBATCH -p 8gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024G
#SBATCH --time=48:00:00
#SBATCH -o %j.log
#SBATCH -e %j.log

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export PROTOOCC_DIR="${PROTOOCC_DIR:-/home/u2336262/Desktop/artc_2026/mapocc}"

export RUN_MODE="${RUN_MODE:-full}"
export CONFIG="${CONFIG:-projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate.py}"
export WORK_DIR="${WORK_DIR:-${PROTOOCC_DIR}/work_dirs/ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate_nano4_h200}"

export GPUS="${GPUS:-8}"
export SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
export LR="${LR:-1e-4}"
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
