#!/usr/bin/env python3
"""Render every validation frame from one or more nuScenes scenes."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from render_occ_qualitative_comparison import (
    index_prediction_dir,
    load_gt,
    load_infos,
    load_prediction,
    render_example,
    resolve_gt_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render GT | ProtoOcc | Ours for all frames in scenes.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--ours-dir", required=True)
    parser.add_argument("--scene-names", nargs="+", required=True)
    parser.add_argument(
        "--info-pkl",
        default="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl")
    parser.add_argument("--nuscenes-root", default="data/nuscenes")
    parser.add_argument(
        "--out-dir",
        default="work_dirs/qualitative_occ/scene_frames")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    return args


def render_frame(path, scene_name, sequence_index, dataset_index, info,
                 baseline_path, ours_path, nuscenes_root, dpi):
    baseline = load_prediction(baseline_path)
    ours = load_prediction(ours_path)
    gt, valid = load_gt(resolve_gt_path(info, nuscenes_root))
    if not (baseline.shape == ours.shape == gt.shape == valid.shape):
        raise ValueError(
            f"Shape mismatch at index {dataset_index}: "
            f"baseline={baseline.shape}, ours={ours.shape}, "
            f"gt={gt.shape}, valid={valid.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), squeeze=False)
    render_example(axes[0], gt, valid, baseline, ours, callouts=[])
    fig.suptitle(
        f"{scene_name}  |  frame {sequence_index:02d}  |  "
        f"validation index {dataset_index}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    infos = load_infos(args.info_pkl)
    baseline_paths = index_prediction_dir(args.baseline_dir)
    ours_paths = index_prediction_dir(args.ours_dir)
    requested = set(args.scene_names)
    found_scenes = set()
    csv_rows = []
    scene_positions = {scene_name: 0 for scene_name in args.scene_names}

    for dataset_index, info in enumerate(infos):
        scene_name = str(
            info.get("scene_name", info.get("scene_token", "")))
        if scene_name not in requested:
            continue
        token = str(info["token"])
        if token not in baseline_paths or token not in ours_paths:
            continue
        found_scenes.add(scene_name)
        scene_positions[scene_name] += 1
        sequence_index = scene_positions[scene_name]
        scene_dir = out_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"frame{sequence_index:02d}_idx{dataset_index:05d}_{token}.png")
        output = scene_dir / filename
        render_frame(
            output,
            scene_name,
            sequence_index,
            dataset_index,
            info,
            baseline_paths[token],
            ours_paths[token],
            args.nuscenes_root,
            args.dpi,
        )
        csv_rows.append(
            dict(
                scene_name=scene_name,
                frame_in_scene=sequence_index,
                dataset_index=dataset_index,
                sample_token=token,
                image_path=str(output),
            ))
        print(
            f"{scene_name} frame={sequence_index:02d} "
            f"index={dataset_index} -> {output}")

    missing = requested - found_scenes
    if missing:
        raise KeyError(
            f"Scenes not found with matching predictions: {sorted(missing)}")

    csv_path = out_dir / "scene_frames.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scene_name",
                "frame_in_scene",
                "dataset_index",
                "sample_token",
                "image_path",
            ),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Rendered {len(csv_rows)} frames from {len(found_scenes)} scenes")
    print(f"Output: {out_dir}")
    print(f"Index: {csv_path}")


if __name__ == "__main__":
    main()
