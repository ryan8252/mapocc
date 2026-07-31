#!/usr/bin/env python3
"""Run 4090 smoke ablations for the adapter-first additive Map-HFM candidate.

Each job runs:
    1. train for 1 epoch on the 1-quarter nuScenes train split
    2. evaluate the epoch_1 EMA checkpoint when available
    3. append a result.md entry in that job's work dir

Jobs:
    base:
        original config
    mapw8:
        original config + model.map_loss_weight=8.0
    lr1e-4:
        original config + optimizer.lr=1e-4
    lr4e-4:
        original config + optimizer.lr=4e-4
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    "projects/configs/ProtoOcc/"
    "ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_"
    "fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate.py"
)
BASE_STEM = (
    "ProtoOcc_multi_cnn_head_map_neck_adapter_first_additive_hfm_"
    "fpn_lateral_aspp_plain128_residual_focal_dice_weighted_active_enhance_gate"
)
TRAIN_ANN = "data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl"
SAMPLES_PER_GPU = "2"
WORKERS_PER_GPU = "1"
EPOCHS = 1
GPUS = "1"
EVAL_METRICS = ["miou", "map-miou"]


@dataclass(frozen=True)
class JobSpec:
    name: str
    work_dir: str
    cfg_options: Tuple[str, ...]
    note: str
    train_port: int
    eval_port: int


JOBS: Dict[str, JobSpec] = {
    "base": JobSpec(
        name="base",
        work_dir=f"work_dirs/smoke_{BASE_STEM}_1quarter_4090",
        cfg_options=(),
        note="Original config. Expected static map_loss_weight is inherited as 4.0.",
        train_port=0,
        eval_port=0,
    ),
    "mapw8": JobSpec(
        name="mapw8",
        work_dir=f"work_dirs/smoke_{BASE_STEM}_mapw8_1quarter_4090",
        cfg_options=("model.map_loss_weight=8.0",),
        note="Only increase the outer map loss weight from 4.0 to 8.0.",
        train_port=0,
        eval_port=0,
    ),
    "lr1e-4": JobSpec(
        name="lr1e-4",
        work_dir=f"work_dirs/smoke_{BASE_STEM}_lr1e-4_1quarter_4090",
        cfg_options=("optimizer.lr=1e-4",),
        note="Only lower optimizer.lr from the inherited 2e-4 default to 1e-4.",
        train_port=0,
        eval_port=0,
    ),
    "lr4e-4": JobSpec(
        name="lr4e-4",
        work_dir=f"work_dirs/smoke_{BASE_STEM}_lr4e-4_1quarter_4090",
        cfg_options=("optimizer.lr=4e-4",),
        note="Only raise optimizer.lr from the inherited 2e-4 default to 4e-4.",
        train_port=0,
        eval_port=0,
    ),
}

DEFAULT_QUEUE = ["base", "mapw8", "lr1e-4", "lr4e-4"]
MAP_CLASSES = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sh(cmd: Iterable[str]) -> str:
    return shlex.join(list(cmd))


def generated_port(offset: int) -> int:
    return 20000 + ((time.time_ns() + os.getpid() * 97 + offset * 7919) % 25000)


def common_cfg_options() -> List[str]:
    return [
        f"data.train.ann_file={TRAIN_ANN}",
        f"data.samples_per_gpu={SAMPLES_PER_GPU}",
        f"data.workers_per_gpu={WORKERS_PER_GPU}",
        f"runner.max_epochs={EPOCHS}",
        "evaluation.interval=999",
        "checkpoint_config.interval=1",
    ]


def latest_train_log(work_dir: Path) -> Optional[Path]:
    logs = [
        path
        for path in work_dir.glob("*.log")
        if not path.name.startswith("codex_") and path.name != "result.md"
    ]
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def checkpoint_for_eval(work_dir: Path) -> Optional[Path]:
    ema = work_dir / f"epoch_{EPOCHS}_ema.pth"
    if ema.exists():
        return ema
    regular = work_dir / f"epoch_{EPOCHS}.pth"
    if regular.exists():
        return regular
    return None


def train_command(job: JobSpec, conda_env: str, timeout_min: int) -> List[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "timeout",
        "--kill-after=60s",
        f"{timeout_min}m",
        "env",
        f"PORT={job.train_port}",
        "bash",
        "tools/dist_train.sh",
        BASE_CONFIG,
        GPUS,
        "--work-dir",
        job.work_dir,
        "--cfg-options",
        *common_cfg_options(),
        *job.cfg_options,
    ]


def eval_command(
    job: JobSpec,
    conda_env: str,
    ckpt: Path,
    timeout_min: int,
) -> List[str]:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "timeout",
        "--kill-after=60s",
        f"{timeout_min}m",
        "env",
        f"PORT={job.eval_port}",
        "bash",
        "tools/dist_test.sh",
        BASE_CONFIG,
        rel(ckpt),
        GPUS,
        "--eval",
        *EVAL_METRICS,
    ]
    if job.cfg_options:
        cmd.extend(["--cfg-options", *job.cfg_options])
    return cmd


def run_command(cmd: List[str], stdout_path: Path, dry_run: bool) -> int:
    log(f"command: {sh(cmd)}")
    log(f"stdout: {rel(stdout_path)}")
    if dry_run:
        return 0
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def parse_train_summary(train_log: Optional[Path]) -> Tuple[Optional[str], Dict[str, str]]:
    if train_log is None or not train_log.exists():
        return None, {}
    pat = re.compile(rf"Epoch \[{EPOCHS}\]\[\d+/\d+\]")
    final_line = None
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if pat.search(line):
            final_line = line
    if final_line is None:
        return None, {}
    metrics: Dict[str, str] = {}
    for key, raw_value in re.findall(r"([A-Za-z0-9_]+):\s*([^,\s]+)", final_line):
        try:
            float(raw_value)
        except ValueError:
            continue
        metrics[key] = raw_value
    return final_line, metrics


def parse_eval_output(eval_log: Path) -> Tuple[Optional[str], Dict[str, str], Dict[str, float]]:
    if not eval_log.exists():
        return None, {}, {}
    text = eval_log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    occ: Dict[str, str] = {}
    occ_miou: Optional[str] = None
    for line in text.splitlines():
        match = re.match(r"^===>\s+(.+?)\s+-\s+IoU\s+=\s+([0-9.]+)", line)
        if match:
            occ[match.group(1)] = match.group(2)
            continue
        match = re.match(r"^===>\s+mIoU of .*?:\s+([0-9.]+)", line)
        if match:
            occ_miou = match.group(1)

    map_metrics: Dict[str, float] = {}
    for key, value in re.findall(
        r"'map/([^']+)/iou@max':\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    ):
        try:
            map_metrics[key] = float(value)
        except ValueError:
            pass
    return occ_miou, occ, map_metrics


def status_from_return(code: int, stage: str) -> str:
    if code == 0:
        return "success"
    if stage in {"train", "eval"} and code in {124, 137, 143}:
        return "timeout_or_killed"
    return f"failed_return_code_{code}"


def format_table(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return "| Item | Value |\n| --- | ---: |\n| not available | n/a |\n"
    lines = ["| Item | Value |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines) + "\n"


def metric_float(metrics: Dict[str, str], key: str) -> Optional[float]:
    if key not in metrics:
        return None
    try:
        return float(metrics[key])
    except ValueError:
        return None


def train_tables(metrics: Dict[str, str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    total = metric_float(metrics, "loss")
    map_keys = sorted(key for key in metrics if key.startswith("loss_map_"))
    map_total = sum(metric_float(metrics, key) or 0.0 for key in map_keys)
    depth = metric_float(metrics, "loss_depth") or 0.0
    other = max((total or 0.0) - map_total - depth, 0.0)

    balance_rows: List[Tuple[str, str]] = [
        ("loss_map_total", f"{map_total:.4f}"),
        ("loss_depth", f"{depth:.4f}"),
        ("loss_other", f"{other:.4f}"),
    ]
    if total is not None:
        denom = max(total, 1e-12)
        balance_rows.extend(
            [
                ("loss_total", f"{total:.4f}"),
                ("map_total / loss_total", f"{map_total / denom:.4f}"),
                ("depth / loss_total", f"{depth / denom:.4f}"),
                ("other / loss_total", f"{other / denom:.4f}"),
            ]
        )

    detail_keys = [
        "loss_cls",
        "loss_mask",
        "loss_dice",
        "loss_cls_RPL",
        "loss_mask_RPL",
        "loss_dice_RPL",
        "loss_CE_prototype",
        "lovasz_softmax_loss_prototype",
        "loss_segmentation",
        "loss_depth",
        *map_keys,
        "loss",
        "grad_norm",
    ]
    detail_rows = [(key, metrics[key]) for key in detail_keys if key in metrics]
    return balance_rows, detail_rows


def write_result(
    job: JobSpec,
    train_status: str,
    eval_status: str,
    train_cmd: List[str],
    eval_cmd: Optional[List[str]],
    ckpt: Optional[Path],
    train_stdout: Path,
    eval_stdout: Path,
    dry_run: bool = False,
) -> None:
    work_dir = ROOT / job.work_dir
    result_md = work_dir / "result.md"
    if dry_run:
        log(f"dry-run: would write {rel(result_md)}")
        return

    train_log = latest_train_log(work_dir)
    final_train_line, train_metrics = parse_train_summary(train_log)
    occ_miou, occ, map_metrics = parse_eval_output(eval_stdout)
    balance_rows, detail_rows = train_tables(train_metrics)

    map_rows = [(name, f"{map_metrics[name]:.6f}") for name in MAP_CLASSES if name in map_metrics]
    if "mean" in map_metrics:
        map_rows.append(("mean", f"{map_metrics['mean']:.6f}"))

    occ_rows = list(occ.items())
    if occ_miou is not None:
        occ_rows.append(("mIoU", f"**{occ_miou}**"))

    title = f"# Smoke Result - {Path(job.work_dir).name}"
    if result_md.exists():
        title = f"## Rerun - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    body = f"""{title}

Date: {datetime.now().strftime('%Y-%m-%d')}

## Run

- Base config: `{BASE_CONFIG}`
- Job: `{job.name}`
- Work dir: `{job.work_dir}`
- Note: {job.note}
- Extra cfg-options: `{sh(job.cfg_options) if job.cfg_options else 'none'}`
- Train status: `{train_status}`
- Eval status: `{eval_status}`
- Train split override: `{TRAIN_ANN}`
- Train schedule: `{EPOCHS} epoch`, samples_per_gpu=`{SAMPLES_PER_GPU}`, workers_per_gpu=`{WORKERS_PER_GPU}`
- Train log: `{rel(train_log) if train_log else 'n/a'}`
- Train stdout: `{rel(train_stdout)}`
- Eval checkpoint: `{rel(ckpt) if ckpt else 'n/a'}`
- Eval stdout: `{rel(eval_stdout)}`

## Commands

```bash
{sh(train_cmd)}
```

```bash
{sh(eval_cmd) if eval_cmd else 'eval skipped: no checkpoint available'}
```

## Loss Balance

{format_table(balance_rows)}
## Train Loss Detail

{format_table(detail_rows)}
Final train line:

```text
{final_train_line if final_train_line else 'not available'}
```

## Summary

- OCC mIoU: `{occ_miou if occ_miou is not None else 'n/a'}`
- Map mean iou@max: `{map_metrics.get('mean', 'n/a')}`
- Thin classes: `ped_crossing={map_metrics.get('ped_crossing', 'n/a')}`, `stop_line={map_metrics.get('stop_line', 'n/a')}`, `divider={map_metrics.get('divider', 'n/a')}`

## OCC IoU

{format_table(occ_rows)}
## Map IoU

{format_table(map_rows)}
"""

    result_md.parent.mkdir(parents=True, exist_ok=True)
    if result_md.exists():
        with result_md.open("a", encoding="utf-8") as stream:
            stream.write("\n\n---\n\n")
            stream.write(body)
        log(f"appended rerun to {rel(result_md)}")
    else:
        result_md.write_text(body, encoding="utf-8")
        log(f"wrote {rel(result_md)}")


def run_job(args: argparse.Namespace, job: JobSpec) -> None:
    config_path = ROOT / BASE_CONFIG
    work_dir = ROOT / job.work_dir
    train_stdout = work_dir / "codex_train_stdout.log"
    eval_stdout = work_dir / "codex_eval_stdout.log"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        log(f"[{job.name}] missing config: {BASE_CONFIG}")
        return

    ckpt = checkpoint_for_eval(work_dir)
    train_cmd = train_command(job, args.conda_env, args.train_timeout_min)
    should_train = args.force_train or ckpt is None or not args.reuse_existing_checkpoint

    train_status = "skipped_existing_checkpoint"
    if should_train:
        log(f"[{job.name}] train start")
        code = run_command(train_cmd, train_stdout, args.dry_run)
        train_status = status_from_return(code, "train")
        log(f"[{job.name}] train status: {train_status}")
        ckpt = checkpoint_for_eval(work_dir)
        if args.dry_run and ckpt is None:
            ckpt = work_dir / f"epoch_{EPOCHS}_ema.pth"
        if code != 0:
            write_result(
                job,
                train_status,
                "skipped_train_failed",
                train_cmd,
                None,
                ckpt,
                train_stdout,
                eval_stdout,
                args.dry_run,
            )
            return
    else:
        log(f"[{job.name}] train skipped; checkpoint exists: {rel(ckpt)}")

    if ckpt is None:
        log(f"[{job.name}] no checkpoint available for eval")
        write_result(
            job,
            train_status,
            "skipped_no_checkpoint",
            train_cmd,
            None,
            ckpt,
            train_stdout,
            eval_stdout,
            args.dry_run,
        )
        return

    eval_cmd = eval_command(job, args.conda_env, ckpt, args.eval_timeout_min)
    log(f"[{job.name}] eval start")
    eval_code = run_command(eval_cmd, eval_stdout, args.dry_run)
    eval_status = status_from_return(eval_code, "eval")
    log(f"[{job.name}] eval status: {eval_status}")

    write_result(
        job,
        train_status,
        eval_status,
        train_cmd,
        eval_cmd,
        ckpt,
        train_stdout,
        eval_stdout,
        args.dry_run,
    )


def resolve_jobs(args: argparse.Namespace) -> Dict[str, JobSpec]:
    jobs = dict(JOBS)
    port_base = args.port_base if args.port_base is not None else generated_port(0)
    for index, name in enumerate(DEFAULT_QUEUE):
        job = jobs[name]
        jobs[name] = replace(
            job,
            train_port=port_base + index * 2,
            eval_port=port_base + index * 2 + 1,
        )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential 4090 smoke runner for adapter-first additive Map-HFM + "
            "FPN lateral ASPP + plain128 residual + focal/Dice weighted + "
            "active enhance gate ablations."
        )
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        choices=sorted(JOBS),
        default=DEFAULT_QUEUE,
        help="Jobs to run in order. Default: base mapw8 lr1e-4 lr4e-4.",
    )
    parser.add_argument("--conda-env", default="mapocc")
    parser.add_argument("--train-timeout-min", type=int, default=180)
    parser.add_argument("--eval-timeout-min", type=int, default=90)
    parser.add_argument(
        "--reuse-existing-checkpoint",
        action="store_true",
        help="Skip training when epoch_1_ema.pth or epoch_1.pth already exists.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Train even if --reuse-existing-checkpoint is also set.",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=None,
        help="Optional deterministic port base. Jobs use base/base+1, base+2/base+3, ...",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running train/eval or writing result.md.",
    )
    return parser.parse_args()


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    jobs = resolve_jobs(args)
    log(f"repo: {ROOT}")
    log(f"queue: {' -> '.join(args.jobs)}")
    for name in args.jobs:
        run_job(args, jobs[name])
    log("queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
