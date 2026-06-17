#!/usr/bin/env python3
"""Run the native Map-TSFG suppression A 4090 smoke train/eval."""

from __future__ import annotations

import argparse
import os
import sys

from codex_run_map_config_queue import JobSpec, ROOT, log, run_job


JOB = JobSpec(
    name="map_tsfg_supp_a",
    config=(
        "projects/configs/ProtoOcc/"
        "ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_"
        "focal_dice_weighted_active_enhance_gate_tsfg_supp.py"
    ),
    work_dir=(
        "work_dirs/"
        "smoke_ProtoOcc_multi_cnn_head_map_neck_hfm_adapter_fpn_lateral_aspp_"
        "focal_dice_weighted_active_enhance_gate_tsfg_supp_1quarter_4090"
    ),
    train_port=29681,
    eval_port=29682,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/eval/record the native Map-TSFG suppression A "
            "1quarter smoke run on a single 4090."
        )
    )
    parser.add_argument("--conda-env", default="mapocc")
    parser.add_argument("--eval-timeout-min", type=int, default=60)
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
        help="Print commands without running train/eval or writing result.md.",
    )
    return parser.parse_args()


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    log(f"repo: {ROOT}")
    log(f"job: {JOB.name}")
    log(f"config: {JOB.config}")
    log(f"work_dir: {JOB.work_dir}")
    run_job(args, JOB)
    log("Map-TSFG suppression A smoke runner finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
