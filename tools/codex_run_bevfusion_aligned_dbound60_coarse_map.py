#!/usr/bin/env python3
"""1-quarter train/eval/result runner for the coarse-map dbound60 ablation.

This reuses the BEVFusion-aligned dbound60 runner's train/eval/result parser,
but targets:

    ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_dbound60_coarse_map.py

The ablation keeps the final map supervision/output at [-50, 50] / 0.5m
(200x200), while changing only the internal map-neck feature before
`_align_map_feature` from 256x256 to 128x128.

Usage:
    conda run -n mapocc python tools/codex_run_bevfusion_aligned_dbound60_coarse_map.py
    python tools/codex_run_bevfusion_aligned_dbound60_coarse_map.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from typing import Dict

import codex_run_bevfusion_aligned_dbound60 as base


COARSE_NOTE = (
    "Coarse-map ablation: internal map-neck output is 128x128 "
    "(extra_upsample=1, about 0.8m/cell) before _align_map_feature; "
    "final BEVFusion-aligned map supervision/output remains 200x200. "
    "A/B baseline: work_dirs/smoke_bevfusion_aligned_dbound60_1quarter_4090 "
    "(same dbound60 config, internal map feature 256x256)."
)

JOBS: Dict[str, base.JobSpec] = {
    "bevfusion_aligned_dbound60_coarse_map": base.JobSpec(
        name="bevfusion_aligned_dbound60_coarse_map",
        config=(
            "projects/configs/ProtoOcc/"
            "ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_dbound60_coarse_map.py"
        ),
        work_dir="work_dirs/smoke_bevfusion_aligned_dbound60_coarse_map_1quarter_4090",
        train_port=29631,
        eval_port=29632,
    ),
}

DEFAULT_QUEUE = ["bevfusion_aligned_dbound60_coarse_map"]


def generated_port(salt: int) -> int:
    # Do not probe sockets here: some managed shells block socket creation before
    # torch.distributed gets a chance to launch. Generate a different high port
    # every run, and let --train-port/--eval-port override it when needed.
    return 20000 + ((time.time_ns() + os.getpid() * 97 + salt) % 25000)


def resolve_jobs(args: argparse.Namespace) -> Dict[str, base.JobSpec]:
    jobs = dict(JOBS)
    job = jobs["bevfusion_aligned_dbound60_coarse_map"]
    jobs[job.name] = replace(
        job,
        train_port=args.train_port if args.train_port is not None else generated_port(0),
        eval_port=args.eval_port if args.eval_port is not None else generated_port(7919),
    )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/eval/record runner for the BEVFusion-aligned dbound60 "
            "coarse-map ablation."
        )
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        choices=sorted(JOBS),
        default=DEFAULT_QUEUE,
        help="Jobs to run in order. Default: bevfusion_aligned_dbound60_coarse_map.",
    )
    parser.add_argument("--conda-env", default="mapocc")
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
    args.force_result = True
    return args


def main() -> int:
    os.chdir(base.ROOT)
    base.REFERENCE_NOTE = COARSE_NOTE
    args = parse_args()
    jobs = resolve_jobs(args)
    base.log(f"repo: {base.ROOT}")
    base.log(f"queue: {' -> '.join(args.jobs)}")
    for job_name in args.jobs:
        base.run_job(args, jobs[job_name])
    base.log("queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
