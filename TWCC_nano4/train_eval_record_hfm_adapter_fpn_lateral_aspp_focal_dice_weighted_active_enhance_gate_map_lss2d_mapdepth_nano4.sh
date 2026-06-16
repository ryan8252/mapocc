#!/bin/bash

# Nano4 / 25a-lgn01 Singularity sbatch launcher for:
#   方案1 (native +-40/0.4 grid): active-gate MTL + separate 2D-LSS map branch
#   fed by its OWN soft, unsupervised depth (MapDepthNet). OCC keeps its 3D
#   voxel + sharp supervised depth.
#
# Config: ..._active_enhance_gate_map_lss2d_mapdepth.py
# Apples-to-apples baseline (same native grid, map from 3D-voxel compress +
# shared sharp depth):
#   ..._active_enhance_gate.py
# Run the baseline at the SAME LR for a clean comparison.
#
# Submit from the ProtoOcc checkout on Nano4:
#   sbatch TWCC_nano4/train_eval_record_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_map_lss2d_mapdepth_nano4.sh
#
# 1-GPU smoke example:
#   RUN_MODE=smoke GPUS=1 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=1 \
#   sbatch --export=ALL -p dev --gres=gpu:1 --cpus-per-task=8 --mem=256G \
#     --time=03:59:00 TWCC_nano4/train_eval_record_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_map_lss2d_mapdepth_nano4.sh
#
# This wraps quick_test_nano4.sh so container setup, checkpoint selection,
# eval, and result.md recording stay identical to the known Nano4 workflow.

#SBATCH -J map_l2d_md
#SBATCH --account=MST113104
#SBATCH -p 16gpus
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
export CONFIG="${CONFIG:-projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_map_lss2d_mapdepth.py}"

case "${RUN_MODE}" in
    smoke)
        DEFAULT_EPOCHS=1
        DEFAULT_TRAIN_ANN_FILE="data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl"
        DEFAULT_WORK_DIR="${PROTOOCC_DIR}/work_dirs/smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_map_lss2d_mapdepth_1quarter_nano4_h200"
        DEFAULT_EVAL_TIMEOUT_MIN=90
        DEFAULT_GPUS=1
        DEFAULT_SAMPLES_PER_GPU=2
        DEFAULT_WORKERS_PER_GPU=1
        DEFAULT_LR=2e-4
        ;;
    full)
        DEFAULT_EPOCHS=24
        DEFAULT_TRAIN_ANN_FILE=""
        DEFAULT_WORK_DIR="${PROTOOCC_DIR}/work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_map_lss2d_mapdepth_nano4_h200"
        DEFAULT_EVAL_TIMEOUT_MIN=180
        DEFAULT_GPUS=8
        DEFAULT_SAMPLES_PER_GPU=4
        DEFAULT_WORKERS_PER_GPU=4
        # 2e-4 matches the LSS2D map branch sibling (map_lss2d256) and the
        # config's optimizer default. If your active_enhance_gate baseline was
        # trained at 4e-4, re-run it at 2e-4 too (or set LR=4e-4 here) so the
        # comparison is LR-matched.
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
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-3}"
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
