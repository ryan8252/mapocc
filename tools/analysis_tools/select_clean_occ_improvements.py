#!/usr/bin/env python3
"""Select clean multi-object OCC qualitative improvements.

A clean improvement frame has multiple spatially separate GT components for
which Ours corrects voxels missed by ProtoOcc, while ProtoOcc has no correct
target-class voxel that Ours gets wrong in the frame.  The output contains no
automatic callouts so the final regions can be marked manually.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from render_occ_qualitative_comparison import (
    CLASS_NAMES,
    index_prediction_dir,
    load_gt,
    load_infos,
    load_prediction,
    render_example,
    resolve_gt_path,
)


DEFAULT_CLASS_IDS = (2, 3, 4, 5, 6, 7, 8, 9, 10)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find frames with multiple clean Ours-correct/ProtoOcc-wrong "
            "object components and no target-class regression."))
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--ours-dir", required=True)
    parser.add_argument(
        "--info-pkl",
        default="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl")
    parser.add_argument("--nuscenes-root", default="data/nuscenes")
    parser.add_argument(
        "--out-dir",
        default="work_dirs/qualitative_occ/clean_multi_object_candidates")
    parser.add_argument(
        "--class-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_CLASS_IDS),
        help="Classes on which clean improvements/regressions are checked.")
    parser.add_argument(
        "--min-components",
        type=int,
        default=2,
        help="Minimum number of separate clean GT components in a frame.")
    parser.add_argument(
        "--max-components",
        type=int,
        default=3,
        help="Maximum number of qualifying components; use 0 for no limit.")
    parser.add_argument(
        "--min-component-cells",
        type=int,
        default=3,
        help="Minimum BEV footprint of a GT component in grid cells.")
    parser.add_argument(
        "--min-component-rescued-voxels",
        type=int,
        default=2,
        help="Minimum Ours-correct/ProtoOcc-wrong voxels per component.")
    parser.add_argument(
        "--max-regressed-voxels",
        type=int,
        default=0,
        help=(
            "Maximum target-class ProtoOcc-correct/Ours-wrong voxels in the "
            "entire frame. Keep 0 for strictly clean examples."))
    parser.add_argument(
        "--rank-start",
        type=int,
        default=1,
        help=(
            "One-based starting rank. For example, --rank-start 13 "
            "--top-k 8 selects ranks 13 through 20 when available."))
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if args.min_components < 1:
        parser.error("--min-components must be positive")
    if args.max_components < 0:
        parser.error("--max-components must be non-negative")
    if args.max_components and args.max_components < args.min_components:
        parser.error("--max-components must be zero or >= --min-components")
    if args.min_component_cells < 1:
        parser.error("--min-component-cells must be positive")
    if args.min_component_rescued_voxels < 1:
        parser.error("--min-component-rescued-voxels must be positive")
    if args.max_regressed_voxels < 0:
        parser.error("--max-regressed-voxels must be non-negative")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.rank_start < 1:
        parser.error("--rank-start must be positive")
    invalid = [
        class_id for class_id in args.class_ids
        if class_id < 0 or class_id >= len(CLASS_NAMES) - 1
    ]
    if invalid:
        parser.error(f"invalid class ids: {invalid}")
    return args


def analyze_frame(
        gt,
        valid,
        baseline,
        ours,
        class_ids,
        min_component_cells,
        min_component_rescued_voxels):
    target = valid & np.isin(gt, class_ids)
    baseline_correct = target & (baseline == gt)
    ours_correct = target & (ours == gt)
    rescued = ours_correct & ~baseline_correct
    regressed = baseline_correct & ~ours_correct

    components = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_id in class_ids:
        gt_class = target & (gt == class_id)
        gt_bev = gt_class.any(axis=2)
        labels, count = ndimage.label(gt_bev, structure=structure)
        for component_id in range(1, count + 1):
            component_bev = labels == component_id
            bev_cells = int(component_bev.sum())
            if bev_cells < min_component_cells:
                continue
            component_3d = component_bev[:, :, None] & gt_class
            gt_voxels = int(component_3d.sum())
            component_rescued = int((rescued & component_3d).sum())
            component_regressed = int((regressed & component_3d).sum())
            baseline_hits = int((baseline_correct & component_3d).sum())
            ours_hits = int((ours_correct & component_3d).sum())
            gain = ours_hits - baseline_hits
            if (
                    component_rescued < min_component_rescued_voxels
                    or component_regressed != 0
                    or gain <= 0):
                continue
            components.append(
                dict(
                    class_id=class_id,
                    class_name=CLASS_NAMES[class_id],
                    bev_cells=bev_cells,
                    gt_voxels=gt_voxels,
                    rescued=component_rescued,
                    regressed=component_regressed,
                    baseline_hits=baseline_hits,
                    ours_hits=ours_hits,
                    gain=gain,
                ))

    components.sort(
        key=lambda item: (item["bev_cells"], item["rescued"]),
        reverse=True,
    )
    return dict(
        rescued=int(rescued.sum()),
        regressed=int(regressed.sum()),
        net=int(rescued.sum() - regressed.sum()),
        components=components,
        component_count=len(components),
        component_cells=sum(item["bev_cells"] for item in components),
        component_rescued=sum(item["rescued"] for item in components),
    )


def find_candidates(args, infos, baseline_paths, ours_paths):
    rows = []
    for index, info in enumerate(infos):
        token = str(info["token"])
        if token not in baseline_paths or token not in ours_paths:
            continue
        baseline = load_prediction(baseline_paths[token])
        ours = load_prediction(ours_paths[token])
        gt, valid = load_gt(resolve_gt_path(info, args.nuscenes_root))
        if not (baseline.shape == ours.shape == gt.shape == valid.shape):
            raise ValueError(
                f"Shape mismatch at index {index}, token {token}: "
                f"baseline={baseline.shape}, ours={ours.shape}, "
                f"gt={gt.shape}, valid={valid.shape}")
        stats = analyze_frame(
            gt,
            valid,
            baseline,
            ours,
            args.class_ids,
            args.min_component_cells,
            args.min_component_rescued_voxels,
        )
        if stats["regressed"] > args.max_regressed_voxels:
            continue
        if stats["component_count"] < args.min_components:
            continue
        if (
                args.max_components
                and stats["component_count"] > args.max_components):
            continue
        rows.append(
            dict(
                index=index,
                scene_name=str(
                    info.get("scene_name", info.get("scene_token", ""))),
                token=token,
                **stats,
            ))

    rows.sort(
        key=lambda row: (
            row["component_count"],
            row["component_cells"],
            row["component_rescued"],
            row["net"],
        ),
        reverse=True,
    )
    return rows


def component_summary(components):
    return "; ".join(
        f"{item['class_name']}:cells={item['bev_cells']},rescued={item['rescued']}"
        for item in components)


def write_csv(path, rows, rank_start=1):
    fields = (
        "rank",
        "dataset_index",
        "scene_name",
        "sample_token",
        "component_count",
        "component_cells",
        "component_rescued",
        "rescued",
        "regressed",
        "net",
        "components",
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=rank_start):
            writer.writerow(
                dict(
                    rank=rank,
                    dataset_index=row["index"],
                    scene_name=row["scene_name"],
                    sample_token=row["token"],
                    component_count=row["component_count"],
                    component_cells=row["component_cells"],
                    component_rescued=row["component_rescued"],
                    rescued=row["rescued"],
                    regressed=row["regressed"],
                    net=row["net"],
                    components=component_summary(row["components"]),
                ))


def render_montage(
        path, rows, infos, baseline_paths, ours_paths, nuscenes_root, dpi):
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12.5, 4.0 * len(rows)),
        squeeze=False,
    )
    for row_index, row in enumerate(rows):
        baseline = load_prediction(baseline_paths[row["token"]])
        ours = load_prediction(ours_paths[row["token"]])
        gt, valid = load_gt(
            resolve_gt_path(infos[row["index"]], nuscenes_root))
        label = (
            f"idx={row['index']} {row['scene_name']}\n"
            f"components={row['component_count']} "
            f"rescued={row['rescued']} regressed={row['regressed']}")
        render_example(
            axes[row_index],
            gt,
            valid,
            baseline,
            ours,
            callouts=[],
            row_label=label,
        )
    fig.suptitle(
        "Clean multi-object occupancy improvements",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = load_infos(args.info_pkl)
    baseline_paths = index_prediction_dir(args.baseline_dir)
    ours_paths = index_prediction_dir(args.ours_dir)

    candidates = find_candidates(
        args, infos, baseline_paths, ours_paths)
    write_csv(out_dir / "all_clean_candidates.csv", candidates)
    selection_offset = args.rank_start - 1
    selected = candidates[
        selection_offset:selection_offset + args.top_k]
    if not selected:
        raise RuntimeError(
            "No clean multi-component candidate met the requested filters. "
            "Try lowering --min-component-cells or allowing a small "
            "--max-regressed-voxels value.")
    write_csv(
        out_dir / "selected_clean_candidates.csv",
        selected,
        rank_start=args.rank_start,
    )
    render_montage(
        out_dir / "clean_multi_object_comparison.png",
        selected,
        infos,
        baseline_paths,
        ours_paths,
        args.nuscenes_root,
        args.dpi,
    )

    print(f"Eligible clean frames: {len(candidates)}")
    print(f"Selected frames: {len(selected)}")
    print(f"CSV: {out_dir / 'all_clean_candidates.csv'}")
    print(f"Figure: {out_dir / 'clean_multi_object_comparison.png'}")


if __name__ == "__main__":
    main()
