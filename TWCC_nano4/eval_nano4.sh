#!/bin/bash

# Nano4 / 25a-lgn01 Singularity eval wrapper.
#
# Use the same argument shape as tools/dist_test.sh:
#
#   bash TWCC_nano4/eval_nano4.sh <config> <checkpoint> <gpus> [dist_test args...]
#
# Example:
#
#   bash TWCC_nano4/eval_nano4.sh \
#     projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8.py \
#     work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8_nano4_h200/epoch_5_ema.pth \
#     1 \
#     --eval miou map-miou
#
# Running with bash on a login node auto-submits this script via sbatch.
# Direct sbatch also works:
#
#   sbatch TWCC_nano4/eval_nano4.sh <config> <checkpoint> <gpus> --eval miou map-miou

#SBATCH -J eval
#SBATCH --account=MST113104
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.log

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash TWCC_nano4/eval_nano4.sh <config> <checkpoint> <gpus> [dist_test args...]

Example:
  bash TWCC_nano4/eval_nano4.sh \
    projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8.py \
    work_dirs/ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_focal_dice_weighted_active_enhance_gate_bevfusion_aligned_mapw8_nano4_h200/epoch_5_ema.pth \
    1 \
    --eval miou map-miou

Useful overrides:
  PROTOOCC_DIR=/home/u2336262/Desktop/artc_2026/mapocc
  CONDA_ENV=/home/u2336262/.conda/envs/unimapocc
  SIF=/home/u2336262/Desktop/artc_2026/containers/cuda118-cudnn8-devel-ubuntu20.04.sif
  EVAL_LOG=work_dirs/.../codex_eval_epoch5_stdout.log
  SBATCH_PARTITION=dev
  SBATCH_TIME=04:00:00
  SBATCH_MEM=256G
USAGE
}

if [ "$#" -lt 3 ]; then
    usage
    exit 2
fi

CONFIG_INPUT="$1"
CKPT="$2"
GPUS="$3"
shift 3
DIST_TEST_ARGS=("$@")

if ! [[ "${GPUS}" =~ ^[0-9]+$ ]] || [ "${GPUS}" -lt 1 ]; then
    echo "[ERROR] <gpus> must be a positive integer, got: ${GPUS}" >&2
    exit 2
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "[ERROR] Not inside a Slurm job and sbatch was not found." >&2
        echo "[HINT] Run this on Nano4 login node, or submit with sbatch manually." >&2
        exit 127
    fi

    SCRIPT_PATH="$(readlink -f "$0")"
    echo "[INFO] Submitting Nano4 eval job via sbatch:"
    echo "       ${SCRIPT_PATH} ${CONFIG_INPUT} ${CKPT} ${GPUS} ${DIST_TEST_ARGS[*]:-}"

    exec sbatch \
        --job-name="${SBATCH_JOB_NAME:-eval}" \
        --account="${SBATCH_ACCOUNT:-MST113104}" \
        --partition="${SBATCH_PARTITION:-dev}" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gres="gpu:${GPUS}" \
        --cpus-per-task="${SBATCH_CPUS_PER_TASK:-8}" \
        --mem="${SBATCH_MEM:-256G}" \
        --time="${SBATCH_TIME:-04:00:00}" \
        --output="${SBATCH_OUTPUT:-%x-%j.log}" \
        --error="${SBATCH_ERROR:-%x-%j.log}" \
        --export=ALL \
        "${SCRIPT_PATH}" "${CONFIG_INPUT}" "${CKPT}" "${GPUS}" "${DIST_TEST_ARGS[@]}"
fi

module purge
module load singularity/4.3.7

CONDA_ENV="${CONDA_ENV:-/home/u2336262/.conda/envs/unimapocc}"
PROTOOCC_DIR="${PROTOOCC_DIR:-/home/u2336262/Desktop/artc_2026/mapocc}"
SIF="${SIF:-/home/u2336262/Desktop/artc_2026/containers/cuda118-cudnn8-devel-ubuntu20.04.sif}"

CONFIG_INPUT="${CONFIG_INPUT%.py}"
case "${CONFIG_INPUT}" in
    */*) CONFIG="${CONFIG_INPUT}.py" ;;
    *)   CONFIG="projects/configs/ProtoOcc/${CONFIG_INPUT}.py" ;;
esac

CKPT_STEM="$(basename "${CKPT}" .pth)"
CKPT_DIR="$(dirname "${CKPT}")"
EVAL_LOG="${EVAL_LOG:-${CKPT_DIR}/codex_eval_${CKPT_STEM}_stdout.log}"
PORT="${PORT:-$(shuf -i 26001-32000 -n 1)}"

if [ ! -d "${PROTOOCC_DIR}" ]; then
    echo "[ERROR] ProtoOcc directory not found: ${PROTOOCC_DIR}" >&2
    exit 1
fi

if [ ! -d "${CONDA_ENV}" ]; then
    echo "[ERROR] Conda env not found: ${CONDA_ENV}" >&2
    exit 1
fi

if [ ! -f "${SIF}" ]; then
    echo "[ERROR] Singularity image not found: ${SIF}" >&2
    exit 1
fi

SINGULARITY_BIND_ARGS=(
    --bind /home/u2336262:/home/u2336262
    --bind /work:/work
)
if [ -n "${EXTRA_BINDS:-}" ]; then
    SINGULARITY_BIND_ARGS+=(--bind "${EXTRA_BINDS}")
fi

echo "[INFO] Host: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-n/a}"
echo "[INFO] ProtoOcc dir: ${PROTOOCC_DIR}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Checkpoint: ${CKPT}"
echo "[INFO] GPUs: ${GPUS}"
echo "[INFO] Port: ${PORT}"
echo "[INFO] Eval log: ${EVAL_LOG}"
echo "[INFO] Dist-test args: ${DIST_TEST_ARGS[*]:-}"

srun --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gres="gpu:${GPUS}" \
    singularity exec --cleanenv --nv \
    "${SINGULARITY_BIND_ARGS[@]}" \
    "${SIF}" \
    bash -c '
set -euo pipefail

ENV_PATH="$1"
PROTOOCC_DIR="$2"
CONFIG="$3"
CKPT="$4"
GPUS="$5"
PORT="$6"
EVAL_LOG="$7"
shift 7
DIST_TEST_ARGS=("$@")

export CUDA_HOME=/usr/local/cuda
export PATH="${ENV_PATH}/bin:${CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CONFIG
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "${PROTOOCC_DIR}"
test -f "${CONFIG}"
test -f "${CKPT}"
mkdir -p "$(dirname "${EVAL_LOG}")"

echo "[CHECK] Container nvidia-smi"
nvidia-smi
python -c "import torch; print(\"[CHECK] Torch:\", torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
python -c "import mmcv, mmdet, mmseg; print(\"[CHECK] OpenMMLab:\", mmcv.__version__, mmdet.__version__, mmseg.__version__)"
python -c "import importlib; importlib.import_module(\"projects.mmdet3d_plugin\"); print(\"[CHECK] projects.mmdet3d_plugin OK\")"
python -c "import os; from mmcv import Config; cfg = Config.fromfile(os.environ[\"CONFIG\"]); ann = cfg.data.val.ann_file; print(\"[CHECK] val ann_file:\", ann); assert os.path.exists(ann), \"val ann_file not found: \" + ann"

echo "[INFO] Running:"
printf "  PORT=%q bash tools/dist_test.sh %q %q %q" "${PORT}" "${CONFIG}" "${CKPT}" "${GPUS}"
printf " %q" "${DIST_TEST_ARGS[@]}"
printf "\n"

PORT="${PORT}" bash tools/dist_test.sh "${CONFIG}" "${CKPT}" "${GPUS}" "${DIST_TEST_ARGS[@]}" 2>&1 | tee "${EVAL_LOG}"
' bash \
    "${CONDA_ENV}" \
    "${PROTOOCC_DIR}" \
    "${CONFIG}" \
    "${CKPT}" \
    "${GPUS}" \
    "${PORT}" \
    "${EVAL_LOG}" \
    "${DIST_TEST_ARGS[@]}"
