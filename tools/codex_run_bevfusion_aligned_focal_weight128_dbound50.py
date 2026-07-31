#!/usr/bin/env python3
"""1-quarter train/eval/result runner for aligned focal weight128 + dbound50.

This smoke ablation keeps the BEVFusion-aligned focal weight128 setup fixed and
changes only the LSS depth range:

    dbound45: [1.0, 45.0, 0.5] -> 88 bins
    dbound50: [1.0, 50.0, 0.5] -> 98 bins

The 98-bin depth head cannot directly load the normal 88-bin depth final layer.
Before training, this runner creates a filtered warm-start checkpoint under the
work dir by dropping depth_net.depth_conv.4.{weight,bias}, then passes that path
through cfg-options as load_from.

Usage:
    conda run -n mapocc python tools/codex_run_bevfusion_aligned_focal_weight128_dbound50.py
    python tools/codex_run_bevfusion_aligned_focal_weight128_dbound50.py --dry-run
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
EVAL_METRICS = ["miou", "map-miou"]
DEFAULT_PRETRAIN = "ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth"

PROTOCOL_NOTE = (
    "BEVFusion-aligned focal weight128 dbound50: canvas +-51.2/0.4 (256), "
    "map +-50/0.5 (200), LSS depth 50m/98 bins"
)
REFERENCE_NOTE = (
    "A/B reference: aligned focal weight128 dbound45 smoke "
    "work_dirs/smoke_bevfusion_aligned_focal_weight128_1quarter_4090 "
    "(map mean iou@max = 0.1502, occ mIoU = 28.18)"
)


@dataclass(frozen=True)
class JobSpec:
    name: str
    config: str
    work_dir: str
    train_port: int
    eval_port: int


JOBS: Dict[str, JobSpec] = {
    "bevfusion_aligned_focal_weight128_dbound50": JobSpec(
        name="bevfusion_aligned_focal_weight128_dbound50",
        config=(
            "projects/configs/ProtoOcc/"
            "ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_focal_weight128_dbound50.py"
        ),
        work_dir="work_dirs/smoke_bevfusion_aligned_focal_weight128_dbound50_1quarter_4090",
        train_port=29641,
        eval_port=29642,
    ),
}

DEFAULT_QUEUE = ["bevfusion_aligned_focal_weight128_dbound50"]


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


def filtered_pretrain_path(work_dir: Path) -> Path:
    return work_dir / "bevdet_depthnet_no_depth_conv4_for_dbound50.pth"


def prepare_filtered_pretrain(src: Path, dst: Path, dry_run: bool) -> Path:
    log(f"filtered warm-start source: {rel(src)}")
    log(f"filtered warm-start target: {rel(dst)}")
    if dry_run:
        return dst
    if not src.exists():
        raise FileNotFoundError(f"pretrain checkpoint not found: {src}")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        log("filtered warm-start checkpoint already exists")
        return dst

    import torch

    ckpt = torch.load(str(src), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    drop_suffixes = (
        "depth_net.depth_conv.4.weight",
        "depth_net.depth_conv.4.bias",
    )
    dropped = []
    for key in list(state.keys()):
        bare = key[7:] if key.startswith("module.") else key
        if bare.endswith(drop_suffixes):
            dropped.append(key)
            state.pop(key)
    if not dropped:
        log("[WARN] no depth_net.depth_conv.4 keys found to drop")
    else:
        log("dropped warm-start keys: " + ", ".join(dropped))

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt["state_dict"] = state
        meta = ckpt.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["codex_filtered_for"] = "aligned_focal_weight128_dbound50"
            meta["codex_dropped_keys"] = dropped
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(dst))
    return dst


def train_command(job: JobSpec, conda_env: str, load_from: Path) -> List[str]:
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
        f"load_from={rel(load_from)}",
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
) -> Tuple[Optional[str], Dict[str, str], Dict[str, float], str]:
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
    occ_miou, occ, map_metrics, eval_text = parse_eval_output(eval_stdout)

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
        "loss_map_focal",
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
    if result_md.exists():
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
- Protocol: `{PROTOCOL_NOTE}`
- Train status: `{train_status}`
- Eval status: `{eval_status}`
- Train split override: `{TRAIN_ANN}`
- Train schedule: `{EPOCHS} epoch`, samples_per_gpu=`{SAMPLES_PER_GPU}`, workers_per_gpu=`{WORKERS_PER_GPU}`, lr=`{LR}`
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
## Raw Eval Output

{indent_block(eval_text)}
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
    config_path = ROOT / job.config
    work_dir = ROOT / job.work_dir
    train_stdout = work_dir / "codex_train_stdout.log"
    eval_stdout = work_dir / "codex_eval_stdout.log"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        log(f"[{job.name}] missing config: {job.config}")
        return

    pretrain_src = ROOT / args.pretrain
    filtered_load_from = prepare_filtered_pretrain(
        pretrain_src, filtered_pretrain_path(work_dir), args.dry_run
    )

    train_cmd = train_command(job, args.conda_env, filtered_load_from)
    ckpt = checkpoint_for_eval(work_dir)

    should_train = args.force_train or ckpt is None
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
                ckpt, train_stdout, eval_stdout, args.dry_run,
            )
            return
    else:
        log(f"[{job.name}] train skipped; checkpoint exists: {rel(ckpt)}")

    if ckpt is None:
        log(f"[{job.name}] no checkpoint available for eval")
        write_result(
            job, train_status, "skipped_no_checkpoint", train_cmd, None,
            ckpt, train_stdout, eval_stdout, args.dry_run,
        )
        return

    eval_cmd = eval_command(job, args.conda_env, ckpt, args.eval_timeout_min)
    log(f"[{job.name}] eval start")
    eval_code = run_command(eval_cmd, eval_stdout, args.dry_run)
    eval_status = status_from_return(eval_code, "eval")
    log(f"[{job.name}] eval status: {eval_status}")

    write_result(
        job, train_status, eval_status, train_cmd, eval_cmd,
        ckpt, train_stdout, eval_stdout, args.dry_run,
    )


def resolve_jobs(args: argparse.Namespace) -> Dict[str, JobSpec]:
    jobs = dict(JOBS)
    job = jobs["bevfusion_aligned_focal_weight128_dbound50"]
    jobs[job.name] = replace(
        job,
        train_port=args.train_port if args.train_port is not None else generated_port(0),
        eval_port=args.eval_port if args.eval_port is not None else generated_port(7919),
    )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/eval/record runner for aligned focal weight128 + dbound50 "
            "on the 1-quarter nuScenes split."
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
    parser.add_argument("--pretrain", default=DEFAULT_PRETRAIN)
    parser.add_argument("--eval-timeout-min", type=int, default=40)
    parser.add_argument(
        "--reuse-existing-checkpoint",
        action="store_true",
        help="Skip training when epoch_1_ema.pth or epoch_1.pth already exists.",
    )
    parser.add_argument("--train-port", type=int, default=None)
    parser.add_argument("--eval-port", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them or writing result.md.",
    )
    args = parser.parse_args()
    args.force_train = not args.reuse_existing_checkpoint
    return args


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
