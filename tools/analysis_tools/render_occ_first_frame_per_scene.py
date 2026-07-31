#!/usr/bin/env python3
"""Render the first validation frame of every nuScenes scene.

This tool performs no improvement-based ranking.  It walks the validation
info PKL in its original order, keeps the first frame for each scene that has
both baseline and ours predictions, and writes one clean
Ground Truth | ProtoOcc | Ours comparison PNG per scene.
"""

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
        description="Render the first validation frame from every scene.")
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="ProtoOcc prediction root containing nested pred.npz files.")
    parser.add_argument(
        "--ours-dir",
        required=True,
        help="Ours prediction root containing nested pred.npz files.")
    parser.add_argument(
        "--info-pkl",
        default="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl")
    parser.add_argument(
        "--nuscenes-root",
        default="data/nuscenes")
    parser.add_argument(
        "--out-dir",
        default="work_dirs/qualitative_occ/first_frame_per_scene")
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output PNG resolution.")
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Optional limit for a quick preview; default renders every scene.")
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if args.max_scenes is not None and args.max_scenes < 1:
        parser.error("--max-scenes must be positive")
    return args


def select_first_frames(infos, baseline_paths, ours_paths):
    selected = []
    seen_scenes = set()
    for index, info in enumerate(infos):
        scene_name = str(
            info.get("scene_name", info.get("scene_token", "")))
        if not scene_name or scene_name in seen_scenes:
            continue
        token = str(info["token"])
        if token not in baseline_paths or token not in ours_paths:
            continue
        seen_scenes.add(scene_name)
        selected.append(
            dict(
                index=index,
                scene_name=scene_name,
                token=token,
            ))
    return selected


def render_comparison(path, row, info, baseline_path, ours_path,
                      nuscenes_root, dpi):
    baseline = load_prediction(baseline_path)
    ours = load_prediction(ours_path)
    gt, camera_mask = load_gt(resolve_gt_path(info, nuscenes_root))
    if not (baseline.shape == ours.shape == gt.shape == camera_mask.shape):
        raise ValueError(
            f"Shape mismatch for {row['token']}: baseline={baseline.shape}, "
            f"ours={ours.shape}, gt={gt.shape}, mask={camera_mask.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), squeeze=False)
    render_example(
        axes[0],
        gt,
        camera_mask,
        baseline,
        ours,
        callouts=[],
    )
    fig.suptitle(
        f"{row['scene_name']}  |  validation index {row['index']}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    infos = load_infos(args.info_pkl)
    baseline_paths = index_prediction_dir(args.baseline_dir)
    ours_paths = index_prediction_dir(args.ours_dir)
    selected = select_first_frames(infos, baseline_paths, ours_paths)
    if args.max_scenes is not None:
        selected = selected[:args.max_scenes]
    if not selected:
        raise RuntimeError(
            "No scene has matching baseline and ours predictions.")

    csv_rows = []
    for position, row in enumerate(selected, start=1):
        filename = (
            f"{position:03d}_{row['scene_name']}_idx{row['index']:05d}_"
            f"{row['token']}.png")
        output = images_dir / filename
        render_comparison(
            output,
            row,
            infos[row["index"]],
            baseline_paths[row["token"]],
            ours_paths[row["token"]],
            args.nuscenes_root,
            args.dpi,
        )
        csv_rows.append(
            dict(
                order=position,
                scene_name=row["scene_name"],
                dataset_index=row["index"],
                sample_token=row["token"],
                image_path=str(output),
            ))
        print(
            f"[{position}/{len(selected)}] {row['scene_name']} "
            f"index={row['index']} -> {output}")

    csv_path = out_dir / "first_frame_per_scene.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "order",
                "scene_name",
                "dataset_index",
                "sample_token",
                "image_path",
            ),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Rendered {len(selected)} scenes")
    print(f"Images: {images_dir}")
    print(f"Index: {csv_path}")


if __name__ == "__main__":
    main()
