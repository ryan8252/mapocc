#!/bin/bash

# Nano4 / 25a-lgn01 dev-partition queue for active-gate ablation smoke tests.
#
# Regime:
#   2 GPUs, 1 epoch, 1/4 nuScenes train split, samples/GPU=2,
#   train -> eval epoch_1_ema.pth -> result.md for each config.
#
# Default order:
#   A1 beta=0.75
#   A4 thin dilation=3
#   A5 3-channel gate
#   A2 beta=1.0
#   A3 gate loss weight=0.3
#
# Submit from the ProtoOcc checkout on Nano4:
#   sbatch TWCC_nano4/smoke_active_gate_ablation_dev_2gpu_nano4.sh
#
# To run a subset:
#   CONFIGS="..._a1_beta075 ..._a4_thin_dilation3" \
#     sbatch --export=ALL TWCC_nano4/smoke_active_gate_ablation_dev_2gpu_nano4.sh

#SBATCH -J actgate_abla2g
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

DEFAULT_CONFIGS=(
    ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a1_beta075
    ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a4_thin_dilation3
    ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a5_3ch
    ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a2_beta10
    ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_a3_gatew03
)

if [ -n "${CONFIGS:-}" ]; then
    # shellcheck disable=SC2206
    ABLATION_CONFIGS=(${CONFIGS})
else
    ABLATION_CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

export RUN_MODE="${RUN_MODE:-smoke}"
export GPUS="${GPUS:-2}"
export SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-2}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
export LR="${LR:-2e-4}"
export EPOCHS="${EPOCHS:-1}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-999}"
export EVAL_TIMEOUT_MIN="${EVAL_TIMEOUT_MIN:-90}"
export TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl}"
export FORCE_TRAIN="${FORCE_TRAIN:-1}"

if [ "${RUN_MODE}" != "smoke" ]; then
    echo "[ERROR] This ablation queue is intended for RUN_MODE=smoke only."
    echo "[ERROR] Got RUN_MODE=${RUN_MODE}"
    exit 1
fi

if [ "${GPUS}" != "2" ]; then
    echo "[ERROR] This script is a 2-GPU dev smoke queue. Got GPUS=${GPUS}"
    exit 1
fi

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

echo "[INFO] Active-gate ablation smoke queue"
echo "[INFO] Partition: dev"
echo "[INFO] ProtoOcc dir: ${PROTOOCC_DIR}"
echo "[INFO] Quick-test helper: ${QUICK_TEST}"
echo "[INFO] GPUs: ${GPUS}"
echo "[INFO] Samples per GPU: ${SAMPLES_PER_GPU}"
echo "[INFO] Workers per GPU: ${WORKERS_PER_GPU}"
echo "[INFO] Global batch size: $((GPUS * SAMPLES_PER_GPU))"
echo "[INFO] Epochs: ${EPOCHS}"
echo "[INFO] Train split: ${TRAIN_ANN_FILE}"
echo "[INFO] Config count: ${#ABLATION_CONFIGS[@]}"

for CONFIG_NAME in "${ABLATION_CONFIGS[@]}"; do
    CONFIG_INPUT="${CONFIG_NAME%.py}"
    case "${CONFIG_INPUT}" in
        */*) CONFIG_PATH="${CONFIG_INPUT}.py" ;;
        *) CONFIG_PATH="projects/configs/ProtoOcc/${CONFIG_INPUT}.py" ;;
    esac
    CONFIG_STEM="$(basename "${CONFIG_PATH}" .py)"
    export WORK_DIR="${PROTOOCC_DIR}/work_dirs/smoke_${CONFIG_STEM}_1quarter_nano4_2gpu"
    export TRAIN_PORT="$(shuf -i 20000-26000 -n 1)"
    export EVAL_PORT="$(shuf -i 26001-32000 -n 1)"

    echo
    echo "======================================================================"
    echo "[INFO] Running config: ${CONFIG_PATH}"
    echo "[INFO] Work dir: ${WORK_DIR}"
    echo "[INFO] Train port: ${TRAIN_PORT}"
    echo "[INFO] Eval port: ${EVAL_PORT}"
    echo "======================================================================"

    bash "${QUICK_TEST}" "${CONFIG_PATH}"
done

echo "[INFO] Active-gate ablation smoke queue finished."
