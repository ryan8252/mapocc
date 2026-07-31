#!/usr/bin/env python3
"""Run 4090 smoke tests for MAESTRO map-STL depth-loss balance ablations.

Base config:
    projects/configs/ProtoOcc/ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_train_depth.py

Jobs:
    mapw4:
        model.map_loss_weight=4.0
    mapw4_depthw03:
        model.map_loss_weight=4.0
        model.depth_net.loss_depth_weight=0.3

Each job runs one 1-quarter / 1-epoch train, evaluates epoch_1_ema.pth when
available, and writes result.md in that job's work dir.
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
    "ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_train_depth.py"
)
TRAIN_ANN = "data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl"
SAMPLES_PER_GPU = "2"
WORKERS_PER_GPU = "1"
EPOCHS = 1
GPUS = "1"
EVAL_METRICS = ["miou", "map-miou"]

REFERENCE_NOTE = (
    "Reference same-ruler smoke: train_depth=False depth118 map mean=0.129629; "
    "train_depth=True depth118 map mean=0.107748."
)


@dataclass(frozen=True)
class JobSpec:
    name: str
    work_dir: str
    cfg_options: Tuple[str, ...]
    note: str
    train_port: int
    eval_port: int


JOBS: Dict[str, JobSpec] = {
    "mapw4": JobSpec(
        name="mapw4",
        work_dir=(
            "work_dirs/"
            "smoke_ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_"
            "train_depth_mapw4_1quarter_4090"
        ),
        cfg_options=("model.map_loss_weight=4.0",),
        note="Increase map focal loss scale while keeping the default PV depth loss.",
        train_port=0,
        eval_port=0,
    ),
    "mapw4_depthw03": JobSpec(
        name="mapw4_depthw03",
        work_dir=(
            "work_dirs/"
            "smoke_ProtoOcc_maestro_map_stl_candidate_bevfusion_depth118_"
            "train_depth_mapw4_depthw03_1quarter_4090"
        ),
        cfg_options=(
            "model.map_loss_weight=4.0",
            "model.depth_net.loss_depth_weight=0.3",
        ),
        note="Increase map focal loss scale and reduce PV depth loss weight.",
        train_port=0,
        eval_port=0,
    ),
}

DEFAULT_QUEUE = ["mapw4", "mapw4_depthw03"]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


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
        p
        for p in work_dir.glob("*.log")
        if not p.name.startswith("codex_") and p.name != "result.md"
    ]
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def checkpoint_for_eval(work_dir: Path) -> Optional[Path]:
    ema = work_dir / f"epoch_{EPOCHS}_ema.pth"
    if ema.exists():
        return ema
    regular = work_dir / f"epoch_{EPOCHS}.pth"
    if regular.exists():
        return regular
    return None


def train_command(job: JobSpec, conda_env: str, timeout_min: int) -> List[str]:
    cfg_options = common_cfg_options() + list(job.cfg_options)
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
        *cfg_options,
    ]


def eval_command(
    job: JobSpec,
    conda_env: str,
    ckpt: Path,
    timeout_min: int,
) -> List[str]:
    # Loss-scale cfg-options do not change inference, but passing them keeps
    # test-time config metadata aligned with the checkpoint that was trained.
    cfg_options = list(job.cfg_options)
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
        f"PORT={job.eval_port}",
        "bash",
        "tools/dist_test.sh",
        BASE_CONFIG,
        rel(ckpt),
        GPUS,
        "--eval",
        *EVAL_METRICS,
        "--cfg-options",
        *cfg_options,
    ]


def run_command(cmd: List[str], stdout_path: Path, dry_run: bool) -> int:
    log(f"command: {sh(cmd)}")
    log(f"stdout: {rel(stdout_path)}")
    if dry_run:
        return 0
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def parse_train_summary(train_log: Optional[Path]) -> Tuple[Optional[str], Dict[str, str]]:
    if train_log is None or not train_log.exists():
        return None, {}
    text = train_log.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(rf"Epoch \[{EPOCHS}\]\[\d+/\d+\]")
    final_line = None
    for line in text.splitlines():
        if pat.search(line):
            final_line = line
    if final_line is None:
        return None, {}
    metrics = dict(re.findall(r"([A-Za-z0-9_]+):\s*([0-9.+-eE]+)", final_line))
    return final_line, metrics


def parse_eval_output(eval_log: Path) -> Tuple[Optional[str], Dict[str, str], Dict[str, float], str]:
    if not eval_log.exists():
        return None, {}, {}, ""
    text = eval_log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    occ: Dict[str, str] = {}
    occ_miou: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"^===>\s+(.+?)\s+-\s+IoU\s+=\s+([0-9.]+)", line)
        if m:
            occ[m.group(1)] = m.group(2)
            continue
        m = re.match(r"^===>\s+mIoU of .*?:\s+([0-9.]+)", line)
        if m:
            occ_miou = m.group(1)

    map_metrics: Dict[str, float] = {}
    for key, value in re.findall(r"'map/([^']+)/iou@max':\s*([0-9.+\-eE]+)", text):
        try:
            map_metrics[key] = float(value)
        except ValueError:
            pass
    return occ_miou, occ, map_metrics, text


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


def train_rows(metrics: Dict[str, str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    map_keys = sorted(k for k in metrics if k.startswith("loss_map_"))
    map_total = sum(metric_float(metrics, key) or 0.0 for key in map_keys)
    total = metric_float(metrics, "loss")
    pv_depth = metric_float(metrics, "loss_depth")
    pv_seg = metric_float(metrics, "loss_segmentation")

    balance_rows: List[Tuple[str, str]] = []
    if pv_seg is not None:
        balance_rows.append(("loss_segmentation", f"{pv_seg:.4f}"))
    if pv_depth is not None:
        balance_rows.append(("loss_depth", f"{pv_depth:.4f}"))
    balance_rows.append(("loss_map_total", f"{map_total:.4f}"))
    if total is not None:
        balance_rows.append(("loss_total", f"{total:.4f}"))
        balance_rows.append(("map_total / loss_total", f"{map_total / max(total, 1e-12):.4f}"))
        pv_total = (pv_seg or 0.0) + (pv_depth or 0.0)
        balance_rows.append(("pv_total / loss_total", f"{pv_total / max(total, 1e-12):.4f}"))

    detail_keys = [
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
    occ_miou, occ, map_metrics, _ = parse_eval_output(eval_stdout)
    balance, details = train_rows(train_metrics)

    map_rows = [
        (key, f"{value:.6f}")
        for key, value in sorted(map_metrics.items())
        if key != "mean"
    ]
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
- Extra cfg-options: `{sh(job.cfg_options)}`
- Train status: `{train_status}`
- Eval status: `{eval_status}`
- Train split override: `{TRAIN_ANN}`
- Train schedule: `{EPOCHS} epoch`, samples_per_gpu=`{SAMPLES_PER_GPU}`, workers_per_gpu=`{WORKERS_PER_GPU}`
- Train log: `{rel(train_log) if train_log else 'n/a'}`
- Train stdout: `{rel(train_stdout)}`
- Eval checkpoint: `{rel(ckpt) if ckpt else 'n/a'}`
- Eval stdout: `{rel(eval_stdout)}`
- {REFERENCE_NOTE}

## Commands

```bash
{sh(train_cmd)}
```

```bash
{sh(eval_cmd) if eval_cmd else 'eval skipped: no checkpoint available'}
```

## Loss Balance

{format_table(balance)}
## Train Loss Detail

{format_table(details)}
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
        with result_md.open("a", encoding="utf-8") as f:
            f.write("\n\n---\n\n")
            f.write(body)
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
    if args.port_base is not None:
        base = args.port_base
    else:
        base = generated_port(0)
    for index, name in enumerate(DEFAULT_QUEUE):
        job = jobs[name]
        jobs[name] = replace(
            job,
            train_port=base + index * 2,
            eval_port=base + index * 2 + 1,
        )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential 4090 smoke runner for MAESTRO map-STL depth-loss "
            "balance ablations."
        )
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        choices=sorted(JOBS),
        default=DEFAULT_QUEUE,
        help="Jobs to run in order.",
    )
    parser.add_argument("--conda-env", default="mapocc")
    parser.add_argument("--train-timeout-min", type=int, default=100)
    parser.add_argument("--eval-timeout-min", type=int, default=70)
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
        help="Optional deterministic port base. Jobs use base/base+1, base+2/base+3.",
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
