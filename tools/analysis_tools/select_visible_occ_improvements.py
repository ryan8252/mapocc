#!/usr/bin/env python3
"""Select visually salient OCC improvements in rendered 2D BEV space.

Unlike voxel-space selectors, this tool scores the exact top-down semantic
images shown in the qualitative figure.  It finds spatially separate regions
where Ours matches GT and ProtoOcc does not, rejects regions with nearby
visible regressions, and limits regression elsewhere in the frame.
"""

import argparse
import csv
import math
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
    topdown_semantic,
)


DEFAULT_CLASS_IDS = (2, 3, 4, 5, 6, 7, 8, 9, 10)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select visible Ours-correct/ProtoOcc-wrong BEV regions.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--ours-dir", required=True)
    parser.add_argument(
        "--info-pkl",
        default="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl")
    parser.add_argument("--nuscenes-root", default="data/nuscenes")
    parser.add_argument(
        "--out-dir",
        default="work_dirs/qualitative_occ/visible_improvement_candidates")
    parser.add_argument(
        "--class-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_CLASS_IDS),
        help="GT classes included in visible rescued/regressed comparison.")
    parser.add_argument(
        "--min-components",
        type=int,
        default=2,
        help="Minimum number of separate clean visible rescued regions.")
    parser.add_argument(
        "--max-components",
        type=int,
        default=3,
        help="Number of largest clean regions used by ranking; 0 uses all.")
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=6,
        help="Minimum rendered BEV area of each rescued region.")
    parser.add_argument(
        "--local-regression-radius",
        type=int,
        default=2,
        help="BEV dilation radius around a rescued region for local checks.")
    parser.add_argument(
        "--max-local-regressed-pixels",
        type=int,
        default=0,
        help="Maximum visible regressions near each selected region.")
    parser.add_argument(
        "--max-visible-regressed-pixels",
        type=int,
        default=20,
        help="Maximum target-class visible regressions in the whole frame.")
    parser.add_argument(
        "--min-rescue-regress-ratio",
        type=float,
        default=2.0,
        help="Require visible rescued >= ratio * visible regressed.")
    parser.add_argument(
        "--regression-penalty",
        type=float,
        default=1.0,
        help="Penalty per visible regressed pixel in the ranking score.")
    parser.add_argument(
        "--rank-start",
        type=int,
        default=1,
        help="One-based starting rank after optional scene deduplication.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--allow-same-scene",
        action="store_true",
        help="Allow multiple frames from the same nuScenes scene.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if args.min_components < 1:
        parser.error("--min-components must be positive")
    if args.max_components < 0:
        parser.error("--max-components must be non-negative")
    if args.min_component_pixels < 1:
        parser.error("--min-component-pixels must be positive")
    if args.local_regression_radius < 0:
        parser.error("--local-regression-radius must be non-negative")
    if args.max_local_regressed_pixels < 0:
        parser.error("--max-local-regressed-pixels must be non-negative")
    if args.max_visible_regressed_pixels < 0:
        parser.error("--max-visible-regressed-pixels must be non-negative")
    if args.min_rescue_regress_ratio < 0:
        parser.error("--min-rescue-regress-ratio must be non-negative")
    if args.regression_penalty < 0:
        parser.error("--regression-penalty must be non-negative")
    if args.rank_start < 1:
        parser.error("--rank-start must be positive")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    invalid = [
        class_id for class_id in args.class_ids
        if class_id < 0 or class_id >= len(CLASS_NAMES) - 1
    ]
    if invalid:
        parser.error(f"invalid class ids: {invalid}")
    return args


def project_frame(gt, valid, baseline, ours):
    return (
        topdown_semantic(gt, valid),
        topdown_semantic(baseline, valid),
        topdown_semantic(ours, valid),
    )


def analyze_visible_frame(gt_bev, baseline_bev, ours_bev, args):
    target = np.isin(gt_bev, args.class_ids)
    baseline_correct = target & (baseline_bev == gt_bev)
    ours_correct = target & (ours_bev == gt_bev)
    rescued = ours_correct & ~baseline_correct
    regressed = baseline_correct & ~ours_correct
    rescued_count = int(rescued.sum())
    regressed_count = int(regressed.sum())

    if regressed_count > args.max_visible_regressed_pixels:
        return None
    if (
            regressed_count > 0
            and rescued_count < args.min_rescue_regress_ratio * regressed_count):
        return None

    structure = np.ones((3, 3), dtype=np.uint8)
    components = []
    for class_id in args.class_ids:
        class_rescued = rescued & (gt_bev == class_id)
        labels, count = ndimage.label(class_rescued, structure=structure)
        for component_id in range(1, count + 1):
            mask = labels == component_id
            pixels = int(mask.sum())
            if pixels < args.min_component_pixels:
                continue
            if args.local_regression_radius:
                local_region = ndimage.binary_dilation(
                    mask,
                    structure=structure,
                    iterations=args.local_regression_radius,
                )
            else:
                local_region = mask
            local_regressed = int((regressed & local_region).sum())
            if local_regressed > args.max_local_regressed_pixels:
                continue
            coords = np.argwhere(mask)
            center = coords.mean(axis=0)
            components.append(
                dict(
                    class_id=class_id,
                    class_name=CLASS_NAMES[class_id],
                    pixels=pixels,
                    local_regressed=local_regressed,
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                ))

    components.sort(key=lambda item: item["pixels"], reverse=True)
    if len(components) < args.min_components:
        return None
    selected_components = (
        components
        if args.max_components == 0
        else components[:args.max_components]
    )
    component_pixels = sum(item["pixels"] for item in selected_components)
    score = component_pixels - args.regression_penalty * regressed_count
    ratio = (
        math.inf
        if regressed_count == 0
        else rescued_count / regressed_count
    )
    return dict(
        score=float(score),
        visible_rescued=rescued_count,
        visible_regressed=regressed_count,
        visible_net=rescued_count - regressed_count,
        rescue_regress_ratio=ratio,
        component_count=len(components),
        ranked_component_count=len(selected_components),
        component_pixels=component_pixels,
        components=selected_components,
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
        stats = analyze_visible_frame(
            *project_frame(gt, valid, baseline, ours), args)
        if stats is None:
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
            row["score"],
            row["component_pixels"],
            row["visible_net"],
            row["visible_rescued"],
        ),
        reverse=True,
    )
    return rows


def selection_pool(candidates, allow_same_scene):
    if allow_same_scene:
        return candidates
    selected = []
    scenes = set()
    for row in candidates:
        if row["scene_name"] in scenes:
            continue
        selected.append(row)
        scenes.add(row["scene_name"])
    return selected


def component_summary(components):
    return "; ".join(
        f"{item['class_name']}:pixels={item['pixels']},"
        f"center=({item['center_x']:.1f},{item['center_y']:.1f}),"
        f"local_regressed={item['local_regressed']}"
        for item in components)


def write_csv(path, rows, rank_start=1):
    fields = (
        "rank",
        "dataset_index",
        "scene_name",
        "sample_token",
        "score",
        "visible_rescued",
        "visible_regressed",
        "visible_net",
        "rescue_regress_ratio",
        "component_count",
        "ranked_component_count",
        "component_pixels",
        "components",
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=rank_start):
            ratio = row["rescue_regress_ratio"]
            writer.writerow(
                dict(
                    rank=rank,
                    dataset_index=row["index"],
                    scene_name=row["scene_name"],
                    sample_token=row["token"],
                    score=row["score"],
                    visible_rescued=row["visible_rescued"],
                    visible_regressed=row["visible_regressed"],
                    visible_net=row["visible_net"],
                    rescue_regress_ratio=(
                        "inf" if math.isinf(ratio) else ratio),
                    component_count=row["component_count"],
                    ranked_component_count=row["ranked_component_count"],
                    component_pixels=row["component_pixels"],
                    components=component_summary(row["components"]),
                ))


def render_triptych(path, row, info, baseline_path, ours_path,
                    nuscenes_root, dpi):
    baseline = load_prediction(baseline_path)
    ours = load_prediction(ours_path)
    gt, valid = load_gt(resolve_gt_path(info, nuscenes_root))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), squeeze=False)
    render_example(
        axes[0], gt, valid, baseline, ours, callouts=[])
    fig.suptitle(
        f"index={row['index']}  {row['scene_name']}  |  "
        f"visible rescued={row['visible_rescued']}  "
        f"regressed={row['visible_regressed']}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_montage(path, rows, infos, baseline_paths, ours_paths,
                   nuscenes_root, dpi):
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(12.5, 4.0 * len(rows)), squeeze=False)
    for row_index, row in enumerate(rows):
        baseline = load_prediction(baseline_paths[row["token"]])
        ours = load_prediction(ours_paths[row["token"]])
        gt, valid = load_gt(
            resolve_gt_path(infos[row["index"]], nuscenes_root))
        label = (
            f"idx={row['index']} {row['scene_name']}\n"
            f"visible rescued={row['visible_rescued']} "
            f"regressed={row['visible_regressed']} "
            f"regions={row['ranked_component_count']}")
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
        "Visible 2D BEV occupancy improvements",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
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
    candidates = find_candidates(args, infos, baseline_paths, ours_paths)
    write_csv(out_dir / "all_visible_candidates.csv", candidates)

    pool = selection_pool(candidates, args.allow_same_scene)
    offset = args.rank_start - 1
    selected = pool[offset:offset + args.top_k]
    if not selected:
        raise RuntimeError(
            "No visible candidate met the requested selection range and "
            "filters. Try reducing --min-component-pixels or increasing "
            "--max-visible-regressed-pixels.")
    write_csv(
        out_dir / "selected_visible_candidates.csv",
        selected,
        rank_start=args.rank_start,
    )

    for rank, row in enumerate(selected, start=args.rank_start):
        output = images_dir / (
            f"rank{rank:03d}_idx{row['index']:05d}_{row['scene_name']}.png")
        render_triptych(
            output,
            row,
            infos[row["index"]],
            baseline_paths[row["token"]],
            ours_paths[row["token"]],
            args.nuscenes_root,
            args.dpi,
        )
    render_montage(
        out_dir / "visible_improvement_comparison.png",
        selected,
        infos,
        baseline_paths,
        ours_paths,
        args.nuscenes_root,
        args.dpi,
    )

    print(f"Eligible visible frames: {len(candidates)}")
    print(f"Scene-distinct selection pool: {len(pool)}")
    print(f"Selected frames: {len(selected)}")
    print(f"CSV: {out_dir / 'selected_visible_candidates.csv'}")
    print(f"Images: {images_dir}")
    print(f"Figure: {out_dir / 'visible_improvement_comparison.png'}")


if __name__ == "__main__":
    main()
