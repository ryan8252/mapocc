#!/bin/bash

# =====================================================================
# Nano4 / 25a-lgn01 Singularity sbatch generic QUICK-TEST launcher.
#
# 用法 (你唯一要改的就是 config 名稱)：
#
#   方法 A — 直接當參數丟給 sbatch（推薦，不用改檔案）：
#       sbatch TWCC_nano4/quick_test_nano4.sh ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp
#
#   方法 B — 改下面的 CONFIG_NAME_DEFAULT 這一行，然後：
#       sbatch TWCC_nano4/quick_test_nano4.sh
#
# config 名稱可以是：
#   - 純名稱            : ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp
#   - 加 .py            : ProtoOcc_multi_cnn_head_map_neck_fpn_lateral_aspp.py
#   - 完整相對路徑      : projects/configs/ProtoOcc/xxx.py
# 三種都會自動正規化。
#
# 預設為 smoke 快測：1 epoch + nuScenes train 1/4 split，train -> eval -> result.md。
# 每個 config 會自動產生獨立 work_dir，互不覆蓋。
#
# 跑完整訓練：
#   RUN_MODE=full sbatch --export=ALL TWCC_nano4/quick_test_nano4.sh <config>
#
# Quick-test 預設資源：8 GPUs * 2 samples/GPU = 16, workers/GPU = 1, lr = 2e-4。
# =====================================================================

#SBATCH -J quick_test
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

module purge
module load singularity/4.3.7

# ---------------------------------------------------------------------
# 只要改這一行（或用 sbatch quick_test_nano4.sh <config> 當參數覆蓋它）
# ---------------------------------------------------------------------
CONFIG_NAME_DEFAULT="ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_dbound60_coarse_map"

CONDA_ENV="${CONDA_ENV:-/home/u2336262/.conda/envs/unimapocc}"
PROTOOCC_DIR="${PROTOOCC_DIR:-/home/u2336262/Desktop/artc_2026/mapocc}"
SIF="${SIF:-/home/u2336262/Desktop/artc_2026/containers/cuda118-cudnn8-devel-ubuntu20.04.sif}"

# 解析 config：可吃 $1 參數 -> 環境變數 CONFIG -> 上面的預設值。
CONFIG_INPUT="${1:-${CONFIG:-${CONFIG_NAME_DEFAULT}}}"
CONFIG_INPUT="${CONFIG_INPUT%.py}"           # 去掉結尾 .py（如果有）
case "${CONFIG_INPUT}" in
    */*) CONFIG="${CONFIG_INPUT}.py" ;;                                   # 已含路徑
    *)   CONFIG="projects/configs/ProtoOcc/${CONFIG_INPUT}.py" ;;         # 純名稱
esac
CONFIG_TAG="$(basename "${CONFIG}" .py)"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-${PROTOOCC_DIR}/ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth}"

RUN_MODE="${RUN_MODE:-smoke}"
case "${RUN_MODE}" in
    smoke)
        DEFAULT_EPOCHS=1
        DEFAULT_TRAIN_ANN_FILE="data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl"
        DEFAULT_WORK_DIR="${PROTOOCC_DIR}/work_dirs/quick_test_${CONFIG_TAG}_1quarter_nano4"
        ;;
    full)
        DEFAULT_EPOCHS=24
        DEFAULT_TRAIN_ANN_FILE=""
        DEFAULT_WORK_DIR="${PROTOOCC_DIR}/work_dirs/${CONFIG_TAG}_nano4_full"
        ;;
    *)
        echo "[ERROR] RUN_MODE must be smoke or full, got: ${RUN_MODE}"
        exit 1
        ;;
esac

WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"
GPUS="${GPUS:-8}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-2}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-${DEFAULT_TRAIN_ANN_FILE}}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-${EPOCHS}}"
EVAL_INTERVAL="${EVAL_INTERVAL:-999}"
EVAL_TIMEOUT_MIN="${EVAL_TIMEOUT_MIN:-90}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

TRAIN_PORT="${TRAIN_PORT:-$(shuf -i 20000-26000 -n 1)}"
EVAL_PORT="${EVAL_PORT:-$(shuf -i 26001-32000 -n 1)}"

TRAIN_STDOUT="${WORK_DIR}/codex_train_stdout.log"
EVAL_STDOUT="${WORK_DIR}/codex_eval_stdout.log"
RESULT_MD="${WORK_DIR}/result.md"

if [ ! -d "${PROTOOCC_DIR}" ]; then
    echo "[ERROR] ProtoOcc directory not found: ${PROTOOCC_DIR}"
    exit 1
fi

if [ ! -d "${CONDA_ENV}" ]; then
    echo "[ERROR] Conda env not found: ${CONDA_ENV}"
    exit 1
fi

if [ ! -f "${SIF}" ]; then
    echo "[ERROR] Singularity image not found: ${SIF}"
    exit 1
fi

if [ ! -f "${PROTOOCC_DIR}/${CONFIG}" ]; then
    echo "[ERROR] Config not found: ${PROTOOCC_DIR}/${CONFIG}"
    echo "[HINT] 可用的 config 列在: ${PROTOOCC_DIR}/projects/configs/ProtoOcc/"
    exit 1
fi

if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "[ERROR] Pretrained checkpoint not found: ${PRETRAIN_CKPT}"
    exit 1
fi

mkdir -p "${WORK_DIR}"

echo "[INFO] Host: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-n/a}"
echo "[INFO] SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-n/a}"
echo "[INFO] ProtoOcc dir: ${PROTOOCC_DIR}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Singularity image: ${SIF}"
echo "[INFO] Run mode: ${RUN_MODE}"
echo "[INFO] Config tag: ${CONFIG_TAG}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Work dir: ${WORK_DIR}"
echo "[INFO] GPUs: ${GPUS}"
echo "[INFO] Samples per GPU: ${SAMPLES_PER_GPU}"
echo "[INFO] Workers per GPU: ${WORKERS_PER_GPU}"
echo "[INFO] Global batch size: $((GPUS * SAMPLES_PER_GPU))"
echo "[INFO] Learning rate: ${LR}"
echo "[INFO] Epochs: ${EPOCHS}"
echo "[INFO] Checkpoint interval: ${CHECKPOINT_INTERVAL}"
echo "[INFO] Eval interval during train: ${EVAL_INTERVAL}"
echo "[INFO] Train port: ${TRAIN_PORT}"
echo "[INFO] Eval port: ${EVAL_PORT}"
echo "[INFO] Eval timeout: ${EVAL_TIMEOUT_MIN}m"
if [ -n "${TRAIN_ANN_FILE}" ]; then
    echo "[INFO] Train ann_file override: ${TRAIN_ANN_FILE}"
else
    echo "[INFO] Train ann_file override: disabled, using config default"
fi
echo "[CHECK] Host nvidia-smi before container"
nvidia-smi

SINGULARITY_BIND_ARGS=(
    --bind /home/u2336262:/home/u2336262
    --bind /work:/work
)
if [ -n "${EXTRA_BINDS:-}" ]; then
    SINGULARITY_BIND_ARGS+=(--bind "${EXTRA_BINDS}")
fi

srun --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
    --gres="gpu:${GPUS}" \
    singularity exec --cleanenv --nv \
    "${SINGULARITY_BIND_ARGS[@]}" \
    "${SIF}" \
    bash -c '
set -euo pipefail

export ENV_PATH="$1"
export PROTOOCC_DIR="$2"
export CONFIG="$3"
export WORK_DIR="$4"
export GPUS="$5"
export SAMPLES_PER_GPU="$6"
export WORKERS_PER_GPU="$7"
export LR="$8"
export EPOCHS="$9"
export TRAIN_ANN_FILE="${10}"
export CHECKPOINT_INTERVAL="${11}"
export EVAL_INTERVAL="${12}"
export TRAIN_PORT="${13}"
export EVAL_PORT="${14}"
export EVAL_TIMEOUT_MIN="${15}"
export TRAIN_STDOUT="${16}"
export EVAL_STDOUT="${17}"
export RESULT_MD="${18}"
export FORCE_TRAIN="${19}"
export RUN_MODE="${20}"
export PRETRAIN_CKPT="${21}"

export CUDA_HOME=/usr/local/cuda
export PATH="${ENV_PATH}/bin:${CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "${PROTOOCC_DIR}"
mkdir -p "${WORK_DIR}"

echo "[CHECK] Container nvidia-smi"
nvidia-smi
echo "[CHECK] Container CUDA_HOME: ${CUDA_HOME}"
echo "[CHECK] Python: $(python -c "import sys; print(sys.executable)")"
python -c "import torch; print(\"[CHECK] Torch:\", torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.version.cuda == \"11.8\", torch.version.cuda; print(torch.ones(1, device=\"cuda\"))"
python -c "import mmcv, mmdet, mmseg; print(\"[CHECK] OpenMMLab:\", mmcv.__version__, mmdet.__version__, mmseg.__version__)"
python -c "import importlib; importlib.import_module(\"projects.mmdet3d_plugin\"); print(\"[CHECK] projects.mmdet3d_plugin OK\")"

python -c "import os; from mmcv import Config; cfg = Config.fromfile(os.environ[\"CONFIG\"]); ann = os.environ.get(\"TRAIN_ANN_FILE\") or cfg.data.train.ann_file; print(\"[CHECK] train ann_file:\", ann); assert os.path.exists(ann), \"train ann_file not found: \" + ann"
python -c "import os; from mmcv import Config; cfg = Config.fromfile(os.environ[\"CONFIG\"]); ann = cfg.data.val.ann_file; print(\"[CHECK] val ann_file:\", ann); assert os.path.exists(ann), \"val ann_file not found: \" + ann"
python -c "import os; assert os.path.exists(os.environ[\"PRETRAIN_CKPT\"]), os.environ[\"PRETRAIN_CKPT\"]; print(\"[CHECK] pretrain ckpt OK:\", os.environ[\"PRETRAIN_CKPT\"])"

CFG_OPTIONS=(
    data.samples_per_gpu="${SAMPLES_PER_GPU}"
    data.workers_per_gpu="${WORKERS_PER_GPU}"
    optimizer.lr="${LR}"
    runner.max_epochs="${EPOCHS}"
    evaluation.interval="${EVAL_INTERVAL}"
    checkpoint_config.interval="${CHECKPOINT_INTERVAL}"
)

if [ -n "${TRAIN_ANN_FILE}" ]; then
    CFG_OPTIONS+=(data.train.ann_file="${TRAIN_ANN_FILE}")
fi

select_ckpt() {
    local exact_ema="${WORK_DIR}/epoch_${EPOCHS}_ema.pth"
    local exact_regular="${WORK_DIR}/epoch_${EPOCHS}.pth"
    if [ -f "${exact_ema}" ]; then
        echo "${exact_ema}"
        return 0
    fi
    if [ -f "${exact_regular}" ]; then
        echo "${exact_regular}"
        return 0
    fi
    if compgen -G "${WORK_DIR}/epoch_*_ema.pth" > /dev/null; then
        ls -1v "${WORK_DIR}"/epoch_*_ema.pth | tail -n 1
        return 0
    fi
    if compgen -G "${WORK_DIR}/epoch_*.pth" > /dev/null; then
        ls -1v "${WORK_DIR}"/epoch_*.pth | tail -n 1
        return 0
    fi
    return 1
}

write_result() {
    local train_status="$1"
    local eval_status="$2"
    local ckpt_path="${3:-}"
    export RESULT_TRAIN_STATUS="${train_status}"
    export RESULT_EVAL_STATUS="${eval_status}"
    export RESULT_CKPT="${ckpt_path}"
    python - <<PY
import os
import re
from datetime import datetime
from pathlib import Path

work_dir = Path(os.environ["WORK_DIR"])
result_md = Path(os.environ["RESULT_MD"])
train_stdout = Path(os.environ["TRAIN_STDOUT"])
eval_stdout = Path(os.environ["EVAL_STDOUT"])
config = os.environ["CONFIG"]
ckpt = os.environ.get("RESULT_CKPT", "")
train_status = os.environ.get("RESULT_TRAIN_STATUS", "unknown")
eval_status = os.environ.get("RESULT_EVAL_STATUS", "unknown")
na = "n/a"
mean_key = "mean"
ped_key = "ped_crossing"
stop_key = "stop_line"
divider_key = "divider"
date_fmt = "%Y-%m-%d"
time_fmt = "%Y-%m-%d %H:%M:%S"
nl = chr(10)
cr = chr(13)

def read(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

def parse_train(text):
    final = None
    pat = re.compile("Epoch \\[{}\\]\\[\\d+/\\d+\\]".format(re.escape(os.environ["EPOCHS"])))
    for line in text.replace(cr, nl).splitlines():
        if pat.search(line):
            final = line
    metrics = dict(re.findall("([A-Za-z0-9_]+):\\s*([0-9.+-eE]+)", final or ""))
    return final, metrics

def parse_eval(text):
    occ_miou = None
    occ = {}
    for line in text.replace(cr, nl).splitlines():
        m = re.match("^===>\\s+(.+?)\\s+-\\s+IoU\\s+=\\s+([0-9.]+)", line)
        if m:
            occ[m.group(1)] = m.group(2)
            continue
        m = re.match("^===>\\s+mIoU of .*?:\\s+([0-9.]+)", line)
        if m:
            occ_miou = m.group(1)
    maps = {}
    quote = chr(39)
    pattern = quote + "map/([^" + quote + "]+)/iou@max" + quote + ":\\s*([0-9.+\\-eE]+)"
    for key, value in re.findall(pattern, text):
        try:
            maps[key] = float(value)
        except ValueError:
            pass
    return occ_miou, occ, maps

def table(rows):
    if not rows:
        rows = [("not available", "n/a")]
    out = ["| Item | Value |", "| --- | ---: |"]
    out.extend(f"| {k} | {v} |" for k, v in rows)
    return nl.join(out) + nl

def indent_block(text):
    if not text:
        return "    not available" + nl
    return nl.join("    " + line for line in text.replace(cr, nl).splitlines()) + nl

train_text = read(train_stdout)
eval_text = read(eval_stdout)
train_final, train_metrics = parse_train(train_text)
occ_miou, occ, maps = parse_eval(eval_text)

train_keys = [
    "loss", "loss_map_bce", "loss_map_focal", "loss_map_dice",
    "loss_segmentation", "loss_depth", "grad_norm",
]
train_rows = [(k, train_metrics[k]) for k in train_keys if k in train_metrics]
occ_rows = list(occ.items())
if occ_miou is not None:
    occ_rows.append(("mIoU", f"**{occ_miou}**"))
map_rows = [(k, f"{v:.6f}") for k, v in sorted(maps.items()) if k != mean_key]
if mean_key in maps:
    map_rows.append(("**mean**", f"**{maps[mean_key]:.6f}**"))

title = f"# Eval Result - {work_dir.name}"
if result_md.exists():
    title = f"## Rerun - {datetime.now().strftime(time_fmt)}"

summary = [
    f"- OCC mIoU: {occ_miou if occ_miou is not None else na}",
    f"- Map mean iou@max: {maps.get(mean_key, na)}",
    "- Thin classes: "
    f"ped_crossing={maps.get(ped_key, na)}, "
    f"stop_line={maps.get(stop_key, na)}, "
    f"divider={maps.get(divider_key, na)}",
]

body = f"""{title}

Date: {datetime.now().strftime(date_fmt)}

## Run

- Config: {config}
- Work dir: {os.environ["WORK_DIR"]}
- Run mode: {os.environ["RUN_MODE"]}
- Protocol: quick test ({os.environ["RUN_MODE"]} mode); train -> eval -> result.md
- Train status: {train_status}
- Eval status: {eval_status}
- Train split override: {os.environ.get("TRAIN_ANN_FILE") or "config default"}
- Train schedule: {os.environ["EPOCHS"]} epoch, samples_per_gpu={os.environ["SAMPLES_PER_GPU"]}, workers_per_gpu={os.environ["WORKERS_PER_GPU"]}, lr={os.environ["LR"]}
- Train stdout: {train_stdout}
- Eval checkpoint: {ckpt or "n/a"}
- Eval stdout: {eval_stdout}

## Commands

Train command:

    {os.environ.get("TRAIN_CMD_TEXT", "n/a")}

Eval command:

    {os.environ.get("EVAL_CMD_TEXT", "eval skipped")}

## Train Result

{table(train_rows)}
Final train line:

    {train_final or "not available"}

## Summary

{chr(10).join(summary)}

## OCC IoU

{table(occ_rows)}
## Map IoU

{table(map_rows)}
## Raw Eval Output

{indent_block(eval_text)}
"""

result_md.parent.mkdir(parents=True, exist_ok=True)
if result_md.exists():
    with result_md.open("a", encoding="utf-8") as f:
        f.write(nl + nl + "---" + nl + nl)
        f.write(body)
else:
    result_md.write_text(body, encoding="utf-8")
print(f"[INFO] wrote result: {result_md}")
PY
}

CKPT=""
if [ "${FORCE_TRAIN}" != "1" ] && CKPT="$(select_ckpt)"; then
    TRAIN_STATUS="skipped_existing_checkpoint"
    echo "[INFO] Training skipped, existing checkpoint found: ${CKPT}"
else
    export PORT="${TRAIN_PORT}"
    TRAIN_CMD=(bash tools/dist_train.sh "${CONFIG}" "${GPUS}" --work-dir "${WORK_DIR}" --cfg-options "${CFG_OPTIONS[@]}")
    printf -v TRAIN_CMD_TEXT "%q " "${TRAIN_CMD[@]}"
    export TRAIN_CMD_TEXT

    echo "[INFO] Train command: ${TRAIN_CMD_TEXT}"
    set +e
    "${TRAIN_CMD[@]}" 2>&1 | tee "${TRAIN_STDOUT}"
    TRAIN_CODE=${PIPESTATUS[0]}
    set -e
    TRAIN_STATUS="success"
    if [ "${TRAIN_CODE}" -ne 0 ]; then
        TRAIN_STATUS="failed_return_code_${TRAIN_CODE}"
        write_result "${TRAIN_STATUS}" "skipped_train_failed" ""
        exit "${TRAIN_CODE}"
    fi

    if ! CKPT="$(select_ckpt)"; then
        write_result "${TRAIN_STATUS}" "skipped_no_checkpoint" ""
        echo "[ERROR] No checkpoint found under ${WORK_DIR}"
        exit 1
    fi
fi

export PORT="${EVAL_PORT}"
EVAL_CMD=(timeout --kill-after=60s "${EVAL_TIMEOUT_MIN}m" bash tools/dist_test.sh "${CONFIG}" "${CKPT}" "${GPUS}" --eval miou map-miou)
printf -v EVAL_CMD_TEXT "%q " "${EVAL_CMD[@]}"
export EVAL_CMD_TEXT

echo "[INFO] Eval checkpoint: ${CKPT}"
echo "[INFO] Eval command: ${EVAL_CMD_TEXT}"
set +e
"${EVAL_CMD[@]}" 2>&1 | tee "${EVAL_STDOUT}"
EVAL_CODE=${PIPESTATUS[0]}
set -e

EVAL_STATUS="success"
if [ "${EVAL_CODE}" -ne 0 ]; then
    EVAL_STATUS="failed_return_code_${EVAL_CODE}"
    if [ "${EVAL_CODE}" -eq 124 ] || [ "${EVAL_CODE}" -eq 137 ] || [ "${EVAL_CODE}" -eq 143 ]; then
        EVAL_STATUS="timeout_or_killed_${EVAL_CODE}"
    fi
fi

write_result "${TRAIN_STATUS}" "${EVAL_STATUS}" "${CKPT}"
exit "${EVAL_CODE}"
' _ \
    "${CONDA_ENV}" \
    "${PROTOOCC_DIR}" \
    "${CONFIG}" \
    "${WORK_DIR}" \
    "${GPUS}" \
    "${SAMPLES_PER_GPU}" \
    "${WORKERS_PER_GPU}" \
    "${LR}" \
    "${EPOCHS}" \
    "${TRAIN_ANN_FILE}" \
    "${CHECKPOINT_INTERVAL}" \
    "${EVAL_INTERVAL}" \
    "${TRAIN_PORT}" \
    "${EVAL_PORT}" \
    "${EVAL_TIMEOUT_MIN}" \
    "${TRAIN_STDOUT}" \
    "${EVAL_STDOUT}" \
    "${RESULT_MD}" \
    "${FORCE_TRAIN}" \
    "${RUN_MODE}" \
    "${PRETRAIN_CKPT}"
