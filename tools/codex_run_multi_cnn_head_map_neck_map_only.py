#!/usr/bin/env python3
"""4090 smoke runner for ProtoOcc_multi_cnn_head_map_neck_map_only.py.

Workflow:
    train 1 epoch on nuScenes 1/4 split
    -> eval epoch_1_ema.pth with map-miou
    -> write work-dir result.md

Usage:
    conda run -n mapocc python tools/codex_run_multi_cnn_head_map_neck_map_only.py
    python tools/codex_run_multi_cnn_head_map_neck_map_only.py --dry-run
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
TRAIN_ANN = "data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl"
SAMPLES_PER_GPU = "2"
WORKERS_PER_GPU = "1"
EPOCHS = 1
GPUS = "1"
LR = "2e-4"
EVAL_METRICS = ["map-miou"]

PROTOCOL_NOTE = (
    "ProtoOcc map-only native grid: map supervision +-40/0.4 (200), "
    "BCE loss_weight=5 + Dice loss_weight=1, map_loss_weight=1, "
    "depth supervision kept enabled"
)
REFERENCE_NOTE = (
    "Same-ruler 4090 smoke comparison should use other "
    "work_dirs/smoke_*_1quarter_4090 map-neck runs. This is not the "
    "BEVFusion/MAESTRO aligned +-50/0.5 protocol."
)


@dataclass(frozen=True)
class JobSpec:
    name: str
    config: str
    work_dir: str
    train_port: int
    eval_port: int


JOBS: Dict[str, JobSpec] = {
    "map_only": JobSpec(
        name="map_only",
        config=(
            "projects/configs/ProtoOcc/"
            "ProtoOcc_multi_cnn_head_map_neck_map_only.py"
        ),
        work_dir=(
            "work_dirs/"
            "smoke_ProtoOcc_multi_cnn_head_map_neck_map_only_1quarter_4090"
        ),
        train_port=29671,
        eval_port=29672,
    ),
}

DEFAULT_QUEUE = ["map_only"]


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


def generated_port(salt: int) -> int:
    return 20000 + ((time.time_ns() + os.getpid() * 97 + salt) % 25000)


def latest_train_log(work_dir: Path) -> Optional[Path]:
    logs = [
        path for path in work_dir.glob("*.log")
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


def train_command(job: JobSpec, conda_env: str) -> List[str]:
    return [
        "conda", "run", "-n", conda_env,
        "env", f"PORT={job.train_port}",
        "bash", "tools/dist_train.sh",
        job.config,
        GPUS,
        "--work-dir", job.work_dir,
        "--cfg-options",
        f"data.train.ann_file={TRAIN_ANN}",
        f"data.samples_per_gpu={SAMPLES_PER_GPU}",
        f"data.workers_per_gpu={WORKERS_PER_GPU}",
        f"optimizer.lr={LR}",
        f"runner.max_epochs={EPOCHS}",
        "evaluation.interval=999",
        "checkpoint_config.interval=1",
    ]


def eval_command(
    job: JobSpec, conda_env: str, ckpt: Path, eval_timeout_min: int
) -> List[str]:
    return [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "timeout", "--kill-after=60s", f"{eval_timeout_min}m",
        "env", f"PORT={job.eval_port}",
        "bash", "tools/dist_test.sh",
        job.config,
        rel(ckpt),
        GPUS,
        "--eval", *EVAL_METRICS,
    ]


def run_command(cmd: List[str], stdout_path: Path, dry_run: bool) -> int:
    log(f"command: {sh(cmd)}")
    log(f"stdout: {rel(stdout_path)}")
    if dry_run:
        return 0
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def parse_train_summary(
    train_log: Optional[Path],
) -> Tuple[Optional[str], Dict[str, str]]:
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


def parse_eval_output(
    eval_log: Path,
) -> Tuple[Dict[str, float], str]:
    if not eval_log.exists():
        return {}, ""
    text = eval_log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    map_metrics: Dict[str, float] = {}
    for key, value in re.findall(r"'map/([^']+)/iou@max':\s*([0-9.+\-eE]+)", text):
        try:
            map_metrics[key] = float(value)
        except ValueError:
            pass
    return map_metrics, text


def status_from_return(code: int, stage: str) -> str:
    if code == 0:
        return "success"
    if stage == "eval" and code in {124, 137, 143}:
        return "timeout_or_killed"
    return f"failed_return_code_{code}"


def format_table(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return "| Item | Value |\n| --- | ---: |\n| not available | n/a |\n"
    lines = ["| Item | Value |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines) + "\n"


def indent_block(text: str) -> str:
    if not text:
        return "    not available\n"
    return "\n".join("    " + line for line in text.splitlines()) + "\n"


def train_metric_rows(train_metrics: Dict[str, str]) -> List[Tuple[str, str]]:
    keys = [
        "loss",
        "loss_map_total",
        "loss_map_focal",
        "loss_map_bce",
        "loss_map_dice",
        "loss_segmentation",
        "loss_depth",
        "grad_norm",
    ]
    rows = [(key, train_metrics[key]) for key in keys if key in train_metrics]
    seen = {key for key, _ in rows}
    for key in sorted(train_metrics):
        if key.startswith("loss_map_") and key not in seen:
            rows.append((key, train_metrics[key]))
            seen.add(key)
    return rows


def write_result(
    job: JobSpec,
    train_status: str,
    eval_status: str,
    train_cmd: List[str],
    eval_cmd: Optional[List[str]],
    ckpt: Optional[Path],
    train_stdout: Path,
    eval_stdout: Path,
    force_result: bool,
    dry_run: bool = False,
) -> None:
    work_dir = ROOT / job.work_dir
    result_md = work_dir / "result.md"
    if dry_run:
        log(f"dry-run: would write {rel(result_md)}")
        return

    train_log = latest_train_log(work_dir)
    final_train_line, train_metrics = parse_train_summary(train_log)
    map_metrics, eval_text = parse_eval_output(eval_stdout)

    map_rows = [
        (key, f"{value:.6f}")
        for key, value in sorted(map_metrics.items())
        if key != "mean"
    ]
    if "mean" in map_metrics:
        map_rows.append(("mean", f"{map_metrics['mean']:.6f}"))

    title = f"# Eval Result - {Path(job.work_dir).name}"
    if result_md.exists():
        title = f"## Rerun - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    summary_lines = [
        f"- Map mean iou@max: `{map_metrics.get('mean', 'n/a')}`",
        "- Thin classes: "
        f"`ped_crossing={map_metrics.get('ped_crossing', 'n/a')}`, "
        f"`stop_line={map_metrics.get('stop_line', 'n/a')}`, "
        f"`divider={map_metrics.get('divider', 'n/a')}`",
        f"- {REFERENCE_NOTE}",
    ]

    body = f"""{title}

Date: {datetime.now().strftime('%Y-%m-%d')}

## Run

- Config: `{job.config}`
- Work dir: `{job.work_dir}`
- Protocol: `{PROTOCOL_NOTE}`
- Train status: `{train_status}`
- Eval status: `{eval_status}`
- Train split override: `{TRAIN_ANN}`
- Train schedule: `{EPOCHS} epoch`, samples_per_gpu=`{SAMPLES_PER_GPU}`, workers_per_gpu=`{WORKERS_PER_GPU}`, lr=`{LR}`
- Eval metrics: `{' '.join(EVAL_METRICS)}`
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

## Train Result

{format_table(train_metric_rows(train_metrics))}
Final train line:

```text
{final_train_line if final_train_line else 'not available'}
```

## Summary

{chr(10).join(summary_lines)}

## Map IoU

{format_table(map_rows)}
## Raw Eval Output

{indent_block(eval_text)}
"""

    result_md.parent.mkdir(parents=True, exist_ok=True)
    if result_md.exists() and not force_result:
        log(f"result exists, leaving unchanged: {rel(result_md)}")
        return
    if result_md.exists():
        with result_md.open("a", encoding="utf-8") as stream:
            stream.write("\n\n---\n\n")
            stream.write(body)
        log(f"appended rerun to {rel(result_md)}")
    else:
        result_md.write_text(body, encoding="utf-8")
        log(f"wrote {rel(result_md)}")


def run_job(args: argparse.Namespace, job: JobSpec) -> None:
    config_path = ROOT / job.config
    work_dir = ROOT / job.work_dir
    result_md = work_dir / "result.md"
    train_stdout = work_dir / "codex_train_stdout.log"
    eval_stdout = work_dir / "codex_eval_stdout.log"
    if not args.dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f"missing config: {job.config}")

    if result_md.exists() and not args.force_result:
        log(f"[{job.name}] result.md already exists; skip. Use --force-result to append rerun.")
        return

    train_cmd = train_command(job, args.conda_env)
    ckpt = checkpoint_for_eval(work_dir)

    should_train = args.force_train or ckpt is None
    train_status = "skipped_existing_checkpoint"
    if should_train:
        log(f"[{job.name}] train start")
        train_code = run_command(train_cmd, train_stdout, args.dry_run)
        train_status = status_from_return(train_code, "train")
        log(f"[{job.name}] train status: {train_status}")
        ckpt = checkpoint_for_eval(work_dir)
        if args.dry_run and ckpt is None and train_code == 0:
            ckpt = work_dir / f"epoch_{EPOCHS}_ema.pth"
        if train_code != 0:
            write_result(
                job, train_status, "skipped_train_failed", train_cmd, None,
                ckpt, train_stdout, eval_stdout, args.force_result, args.dry_run)
            return
    else:
        log(f"[{job.name}] train skipped; checkpoint exists: {rel(ckpt)}")

    if ckpt is None:
        log(f"[{job.name}] no checkpoint available for eval")
        write_result(
            job, train_status, "skipped_no_checkpoint", train_cmd, None,
            ckpt, train_stdout, eval_stdout, args.force_result, args.dry_run)
        return

    eval_cmd = eval_command(job, args.conda_env, ckpt, args.eval_timeout_min)
    log(f"[{job.name}] eval start")
    eval_code = run_command(eval_cmd, eval_stdout, args.dry_run)
    eval_status = status_from_return(eval_code, "eval")
    log(f"[{job.name}] eval status: {eval_status}")

    write_result(
        job, train_status, eval_status, train_cmd, eval_cmd, ckpt,
        train_stdout, eval_stdout, args.force_result, args.dry_run)


def resolve_jobs(args: argparse.Namespace) -> Dict[str, JobSpec]:
    jobs = dict(JOBS)
    job = jobs["map_only"]
    jobs[job.name] = replace(
        job,
        train_port=args.train_port if args.train_port is not None else generated_port(0),
        eval_port=args.eval_port if args.eval_port is not None else generated_port(7919),
    )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/eval/record a 4090 1-quarter smoke run for "
            "ProtoOcc_multi_cnn_head_map_neck_map_only.py."
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
    parser.add_argument("--eval-timeout-min", type=int, default=40)
    parser.add_argument("--train-port", type=int, default=None)
    parser.add_argument("--eval-port", type=int, default=None)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if epoch_1_ema.pth or epoch_1.pth already exists.",
    )
    parser.add_argument(
        "--force-result",
        action="store_true",
        help="Append a rerun section if result.md already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them or writing result.md.",
    )
    return parser.parse_args()


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    jobs = resolve_jobs(args)
    log(f"repo: {ROOT}")
    log(f"queue: {' -> '.join(args.jobs)}")
    for job_name in args.jobs:
        run_job(args, jobs[job_name])
    log("queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
