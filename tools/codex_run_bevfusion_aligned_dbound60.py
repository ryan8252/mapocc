#!/usr/bin/env python3
"""1-quarter train/eval/result runner for the BEVFusion-aligned dbound60 config.

This mirrors tools/codex_run_map_config_queue.py and train_eval_record_runbook.md,
but targets the depth-aligned BEVFusion parity config:

    ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_dbound60.py

It is the clean A/B partner of the existing 45m run
    work_dirs/smoke_bevfusion_aligned_weight4_1quarter_4090   (map mean iou@max = 0.1335)
with ONLY the LSS depth range changed (45m / 88 bins -> 60m / 118 bins). Both use
map_loss_weight=4.0 and the BEVFusion map protocol ([-50, 50] / 0.5m, 200x200).

Workflow per job: train 1 epoch on the 1-quarter split -> eval epoch_1_ema.pth with
miou + map-miou -> write result.md. It never starts eval until train has exited and a
checkpoint exists, and never overwrites an existing result.md unless --force-result.

Usage:
    conda run -n mapocc python tools/codex_run_bevfusion_aligned_dbound60.py            # run it
    python tools/codex_run_bevfusion_aligned_dbound60.py --dry-run                      # print only
    python tools/codex_run_bevfusion_aligned_dbound60.py --force-train --force-result   # redo
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ANN = "data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl"
SAMPLES_PER_GPU = "2"
EPOCHS = 1
GPUS = "1"
EVAL_METRICS = ["miou", "map-miou"]

# 45m reference recorded for context in result.md (same protocol, depth=45m).
REFERENCE_NOTE = (
    "45m A/B partner: work_dirs/smoke_bevfusion_aligned_weight4_1quarter_4090 "
    "(map mean iou@max = 0.1335, occ mIoU = 27.27)"
)


@dataclass(frozen=True)
class JobSpec:
    name: str
    config: str
    work_dir: str
    train_port: int
    eval_port: int
    eval_first: bool = False


JOBS: Dict[str, JobSpec] = {
    "bevfusion_aligned_dbound60": JobSpec(
        name="bevfusion_aligned_dbound60",
        config="projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_dbound60.py",
        work_dir="work_dirs/smoke_bevfusion_aligned_dbound60_1quarter_4090",
        train_port=29621,
        eval_port=29622,
    ),
}

DEFAULT_QUEUE = ["bevfusion_aligned_dbound60"]


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
    with stdout_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True
        )
    return proc.returncode


def parse_train_summary(
    train_log: Optional[Path],
) -> Tuple[Optional[str], Dict[str, str]]:
    """Return the last `Epoch [EPOCHS][N/M]` log line and its key:value metrics.

    Iter count varies with split/batch size, so match any N/M rather than a fixed
    one (the original queue script hard-coded 3500/3513).
    """
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
) -> Tuple[Optional[str], Dict[str, str], Dict[str, float]]:
    if not eval_log.exists():
        return None, {}, {}
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
    return occ_miou, occ, map_metrics


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
    occ_miou, occ, map_metrics = parse_eval_output(eval_stdout)

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

    train_metric_keys = [
        "loss",
        "loss_map_bce",
        "loss_map_dice",
        "loss_segmentation",
        "loss_depth",
        "grad_norm",
    ]
    train_rows = [
        (key, train_metrics[key]) for key in train_metric_keys if key in train_metrics
    ]

    title = f"# Eval Result - {Path(job.work_dir).name}"
    if result_md.exists() and force_result:
        title = f"## Rerun - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    summary_lines = [
        f"- OCC mIoU: `{occ_miou if occ_miou is not None else 'n/a'}`",
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
- Protocol: `BEVFusion-aligned: canvas +-51.2/0.4 (256), map +-50/0.5 (200), LSS depth 60m/118 bins`
- Train status: `{train_status}`
- Eval status: `{eval_status}`
- Train split override: `{TRAIN_ANN}`
- Train schedule: `{EPOCHS} epoch`, samples_per_gpu=`{SAMPLES_PER_GPU}`
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

{format_table(train_rows)}
Final train line:

```text
{final_train_line if final_train_line else 'not available'}
```

## Summary

{chr(10).join(summary_lines)}

## OCC IoU

{format_table(occ_rows)}
## Map IoU

{format_table(map_rows)}
"""

    result_md.parent.mkdir(parents=True, exist_ok=True)
    if result_md.exists() and force_result:
        with result_md.open("a", encoding="utf-8") as f:
            f.write("\n\n---\n\n")
            f.write(body)
        log(f"appended rerun to {rel(result_md)}")
    elif result_md.exists():
        log(f"result exists, leaving unchanged: {rel(result_md)}")
    else:
        result_md.write_text(body, encoding="utf-8")
        log(f"wrote {rel(result_md)}")


def run_job(args: argparse.Namespace, job: JobSpec) -> None:
    config_path = ROOT / job.config
    work_dir = ROOT / job.work_dir
    result_md = work_dir / "result.md"
    train_stdout = work_dir / "codex_train_stdout.log"
    eval_stdout = work_dir / "codex_eval_stdout.log"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        log(f"[{job.name}] missing config: {job.config}")
        return

    if result_md.exists() and not args.force_result:
        log(f"[{job.name}] result.md already exists; skip. Use --force-result to append rerun.")
        return

    train_cmd = train_command(job, args.conda_env)
    ckpt = checkpoint_for_eval(work_dir)

    should_train = args.force_train or ckpt is None
    if job.eval_first and ckpt is not None and not args.force_train:
        should_train = False

    train_status = "skipped_existing_checkpoint"
    if should_train:
        log(f"[{job.name}] train start")
        code = run_command(train_cmd, train_stdout, args.dry_run)
        train_status = status_from_return(code, "train")
        log(f"[{job.name}] train status: {train_status}")
        ckpt = checkpoint_for_eval(work_dir)
        if code != 0:
            write_result(
                job, train_status, "skipped_train_failed", train_cmd, None,
                ckpt, train_stdout, eval_stdout, args.force_result, args.dry_run,
            )
            return
    else:
        log(f"[{job.name}] train skipped; checkpoint exists: {rel(ckpt)}")

    if ckpt is None:
        log(f"[{job.name}] no checkpoint available for eval")
        write_result(
            job, train_status, "skipped_no_checkpoint", train_cmd, None,
            ckpt, train_stdout, eval_stdout, args.force_result, args.dry_run,
        )
        return

    eval_cmd = eval_command(job, args.conda_env, ckpt, args.eval_timeout_min)
    log(f"[{job.name}] eval start")
    eval_code = run_command(eval_cmd, eval_stdout, args.dry_run)
    eval_status = status_from_return(eval_code, "eval")
    log(f"[{job.name}] eval status: {eval_status}")

    write_result(
        job, train_status, eval_status, train_cmd, eval_cmd,
        ckpt, train_stdout, eval_stdout, args.force_result, args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/eval/record runner for the BEVFusion-aligned dbound60 config."
    )
    parser.add_argument(
        "--jobs", nargs="+", choices=sorted(JOBS), default=DEFAULT_QUEUE,
        help="Jobs to run in order. Default: bevfusion_aligned_dbound60.",
    )
    parser.add_argument("--conda-env", default="mapocc")
    parser.add_argument("--eval-timeout-min", type=int, default=40)
    parser.add_argument(
        "--force-train", action="store_true",
        help="Retrain even if epoch_1_ema.pth or epoch_1.pth already exists.",
    )
    parser.add_argument(
        "--force-result", action="store_true",
        help="Append a rerun section if result.md already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running them or writing result.md.",
    )
    return parser.parse_args()


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    log(f"repo: {ROOT}")
    log(f"queue: {' -> '.join(args.jobs)}")
    for job_name in args.jobs:
        run_job(args, JOBS[job_name])
    log("queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
