#!/usr/bin/env python3
"""Run Adapter+Map-HFM+FPN-Lateral-ASPP 4090 smoke ablations."""

from __future__ import annotations

import argparse
import os
import sys

from codex_run_map_config_queue import JobSpec, ROOT, log, run_job


JOBS = {
    "direct": JobSpec(
        name="direct",
        config=(
            "projects/configs/ProtoOcc/"
            "ProtoOcc_multi_cnn_head_map_neck_adapter_hfm_fpn_lateral_aspp.py"
        ),
        work_dir=(
            "work_dirs/"
            "smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_hfm_"
            "fpn_lateral_aspp_1quarter_4090"
        ),
        train_port=29641,
        eval_port=29642,
    ),
    "plain128_residual": JobSpec(
        name="plain128_residual",
        config=(
            "projects/configs/ProtoOcc/"
            "ProtoOcc_multi_cnn_head_map_neck_adapter_hfm_fpn_lateral_aspp_"
            "plain128_residual.py"
        ),
        work_dir=(
            "work_dirs/"
            "smoke_ProtoOcc_multi_cnn_head_map_neck_adapter_hfm_"
            "fpn_lateral_aspp_plain128_residual_1quarter_4090"
        ),
        train_port=29643,
        eval_port=29644,
    ),
}

DEFAULT_JOBS = ["direct", "plain128_residual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run the Adapter+Map-HFM+FPN-Lateral-ASPP "
            "1quarter smoke ablations on a single 4090."
        )
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        choices=sorted(JOBS),
        default=DEFAULT_JOBS,
        help="Jobs to run in order. Default: direct plain128_residual.",
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
    log(f"queue: {' -> '.join(args.jobs)}")
    for job_name in args.jobs:
        run_job(args, JOBS[job_name])
    log("Adapter+Map-HFM+FPN-Lateral-ASPP ablation queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
