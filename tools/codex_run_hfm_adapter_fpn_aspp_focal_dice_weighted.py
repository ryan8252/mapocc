#!/usr/bin/env python3
"""Train, evaluate, and record the HFM-adapter + FPN-ASPP smoke run."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    "projects/configs/ProtoOcc/"
    "ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_"
    "focal_dice_weighted.py"
)
WORK_DIR = (
    "work_dirs/"
    "smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_"
    "focal_dice_weighted_1quarter_4090"
)
TRAIN_ANN = "data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl"
SAMPLES_PER_GPU = "2"
EPOCHS = 1
GPUS = "1"
MAP_CLASSES = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
]


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def quote(cmd: Iterable[str]) -> str:
    return shlex.join(list(cmd))


def train_command(args: argparse.Namespace) -> List[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "env",
        f"PORT={args.train_port}",
        "bash",
        "tools/dist_train.sh",
        CONFIG,
        GPUS,
        "--work-dir",
        WORK_DIR,
        "--cfg-options",
        f"data.train.ann_file={TRAIN_ANN}",
        f"data.samples_per_gpu={SAMPLES_PER_GPU}",
        f"runner.max_epochs={EPOCHS}",
        "evaluation.interval=999",
        "checkpoint_config.interval=1",
    ]


def eval_command(args: argparse.Namespace, ckpt: Path) -> List[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "timeout",
        "--kill-after=60s",
        f"{args.eval_timeout_min}m",
        "env",
        f"PORT={args.eval_port}",
        "bash",
        "tools/dist_test.sh",
        CONFIG,
        rel(ckpt),
        GPUS,
        "--eval",
        "miou",
        "map-miou",
    ]


def checkpoint_for_eval(work_dir: Path) -> Optional[Path]:
    ema = work_dir / f"epoch_{EPOCHS}_ema.pth"
    if ema.exists():
        return ema
    regular = work_dir / f"epoch_{EPOCHS}.pth"
    if regular.exists():
        return regular
    return None


def latest_train_log(work_dir: Path) -> Optional[Path]:
    logs = [
        path
        for path in work_dir.glob("*.log")
        if not path.name.startswith("codex_") and path.name != "result.md"
    ]
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def run_command(cmd: List[str], stdout_path: Path, dry_run: bool) -> int:
    log(f"command: {quote(cmd)}")
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
    final_line = None
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if f"Epoch [{EPOCHS}][3500/3513]" in line:
            final_line = line
    if final_line is None:
        return None, {}
    metrics = dict(re.findall(r"([A-Za-z0-9_]+):\s*([0-9.+-eE]+)", final_line))
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
    if stage == "eval" and code in {124, 137, 143}:
        return "timeout_or_killed"
    return f"failed_return_code_{code}"


def table(rows: List[Tuple[str, str]], header: Tuple[str, str] = ("Item", "Value")) -> str:
    if not rows:
        rows = [("not available", "n/a")]
    lines = [f"| {header[0]} | {header[1]} |", "| --- | ---: |"]
    lines.extend(f"| {left} | {right} |" for left, right in rows)
    return "\n".join(lines) + "\n"


def write_result(
    args: argparse.Namespace,
    train_status: str,
    eval_status: str,
    train_cmd: List[str],
    eval_cmd: Optional[List[str]],
    ckpt: Optional[Path],
    train_stdout: Path,
    eval_stdout: Path,
) -> Path:
    work_dir = ROOT / WORK_DIR
    result_md = work_dir / "result.md"
    if args.dry_run:
        log(f"dry-run: would write {rel(result_md)}")
        return result_md
    train_log = latest_train_log(work_dir)
    final_train_line, train_metrics = parse_train_summary(train_log)
    occ_miou, occ, map_metrics = parse_eval_output(eval_stdout)

    train_keys = [
        "loss",
        "loss_map_bce",
        "loss_map_focal",
        "loss_map_dice",
        "loss_segmentation",
        "loss_depth",
        "grad_norm",
    ]
    train_rows = [(key, train_metrics[key]) for key in train_keys if key in train_metrics]
    occ_rows = list(occ.items())
    if occ_miou is not None:
        occ_rows.append(("mIoU", f"**{occ_miou}**"))

    map_rows = [(name, f"{map_metrics[name]:.6f}") for name in MAP_CLASSES if name in map_metrics]
    if "mean" in map_metrics:
        map_rows.append(("**mean**", f"**{map_metrics['mean']:.6f}**"))

    title = f"# Eval Result - {Path(WORK_DIR).name}"
    if result_md.exists():
        title = f"## Rerun - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    summary_lines = [
        f"- OCC mIoU: `{occ_miou if occ_miou is not None else 'n/a'}`",
        f"- Map mean iou@max: `{map_metrics.get('mean', 'n/a')}`",
        "- Thin classes: "
        f"`ped_crossing={map_metrics.get('ped_crossing', 'n/a')}`, "
        f"`stop_line={map_metrics.get('stop_line', 'n/a')}`, "
        f"`divider={map_metrics.get('divider', 'n/a')}`",
    ]

    body = f"""{title}

Date: {datetime.now().strftime('%Y-%m-%d')}

## Run

- Config: `{CONFIG}`
- Work dir: `{WORK_DIR}`
- Train status: `{train_status}`
- Eval status: `{eval_status}`
- Train split override: `{TRAIN_ANN}`
- Train schedule: `{EPOCHS} epoch`
- Train log: `{rel(train_log) if train_log else 'n/a'}`
- Train stdout: `{rel(train_stdout)}`
- Eval checkpoint: `{rel(ckpt) if ckpt else 'n/a'}`
- Eval stdout: `{rel(eval_stdout)}`

## Commands

```bash
{quote(train_cmd)}
```

```bash
{quote(eval_cmd) if eval_cmd else 'eval skipped: no checkpoint available'}
```

## Train Result

{table(train_rows)}
Final train line:

```text
{final_train_line if final_train_line else 'not available'}
```

## Summary

{chr(10).join(summary_lines)}

## OCC IoU

{table(occ_rows, header=("Class", "IoU"))}
## Map IoU

{table(map_rows, header=("Class", "IoU@max"))}
"""
    result_md.parent.mkdir(parents=True, exist_ok=True)
    if result_md.exists():
        with result_md.open("a", encoding="utf-8") as stream:
            stream.write("\n\n---\n\n")
            stream.write(body)
    else:
        result_md.write_text(body, encoding="utf-8")
    log(f"wrote {rel(result_md)}")
    return result_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the HFM-adapter + FPN-ASPP + focal-Dice weighted "
            "1quarter smoke train/eval."
        )
    )
    parser.add_argument("--conda-env", default="mapocc")
    parser.add_argument("--train-port", type=int, default=29617)
    parser.add_argument("--eval-port", type=int, default=29618)
    parser.add_argument("--eval-timeout-min", type=int, default=60)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if epoch_1_ema.pth or epoch_1.pth already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    work_dir = ROOT / WORK_DIR
    config_path = ROOT / CONFIG
    train_stdout = work_dir / "codex_train_stdout.log"
    eval_stdout = work_dir / "codex_eval_stdout.log"
    work_dir.mkdir(parents=True, exist_ok=True)

    log(f"repo: {ROOT}")
    log(f"config: {CONFIG}")
    log(f"work_dir: {WORK_DIR}")
    if not config_path.exists():
        log(f"missing config: {CONFIG}")
        return 2

    train_cmd = train_command(args)
    ckpt = checkpoint_for_eval(work_dir)
    train_status = "skipped_existing_checkpoint"
    if args.force_train or ckpt is None:
        log("train start")
        train_code = run_command(train_cmd, train_stdout, args.dry_run)
        train_status = status_from_return(train_code, "train")
        log(f"train status: {train_status}")
        if args.dry_run:
            log("dry-run finished")
            return 0
        ckpt = checkpoint_for_eval(work_dir)
        if train_code != 0:
            write_result(
                args,
                train_status,
                "skipped_train_failed",
                train_cmd,
                None,
                ckpt,
                train_stdout,
                eval_stdout,
            )
            return train_code
    else:
        log(f"train skipped; checkpoint exists: {rel(ckpt)}")

    if ckpt is None:
        log("no checkpoint available for eval")
        write_result(
            args,
            train_status,
            "skipped_no_checkpoint",
            train_cmd,
            None,
            ckpt,
            train_stdout,
            eval_stdout,
        )
        return 3

    eval_cmd = eval_command(args, ckpt)
    log("eval start")
    eval_code = run_command(eval_cmd, eval_stdout, args.dry_run)
    eval_status = status_from_return(eval_code, "eval")
    log(f"eval status: {eval_status}")
    write_result(
        args,
        train_status,
        eval_status,
        train_cmd,
        eval_cmd,
        ckpt,
        train_stdout,
        eval_stdout,
    )
    return eval_code


if __name__ == "__main__":
    sys.exit(main())
