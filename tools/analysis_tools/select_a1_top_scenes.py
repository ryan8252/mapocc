#!/usr/bin/env python3
"""Select top scenes for pedestrian/car/traffic-cone A1 improvements.

Scene ranking is based on camera-observed occupancy IoU:

    score(scene) = mean_c [IoU_A1(scene, c) - IoU_ProtoOcc(scene, c)]

where c is pedestrian, car, or traffic_cone. TP/FP/FN are accumulated over
every validation frame in a scene before IoU is computed. For each selected
scene, one representative frame is chosen by class coverage first and
sample-level IoU gain second. The script writes exactly three standalone OCC
panels per selected scene: GT, ProtoOcc, and A1.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy import ndimage

from render_occ_qualitative_comparison import (
    CALLOUT_COLORS,
    CLASS_COLORS,
    CLASS_NAMES,
    index_prediction_dir,
    load_gt,
    load_infos,
    load_prediction,
    resolve_gt_path,
    topdown_semantic,
)


DEFAULT_CLASS_IDS = (7, 4, 8)  # pedestrian, car, traffic_cone


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rank nuScenes scenes by pedestrian/car/traffic-cone IoU gain and "
            "render GT, ProtoOcc, and A1 panels."
        ))
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--ours-dir", required=True)
    parser.add_argument(
        "--info-pkl",
        default="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl")
    parser.add_argument(
        "--nuscenes-root",
        default="data/nuscenes")
    parser.add_argument(
        "--out-dir",
        default="work_dirs/qualitative_protoocc_vs_a1/top5_scenes")
    parser.add_argument(
        "--class-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_CLASS_IDS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-scene-gt-voxels",
        type=int,
        default=20,
        help=(
            "Each target class must have at least this many camera-observed "
            "GT voxels across the scene to enter the ranking."))
    parser.add_argument(
        "--max-callouts",
        type=int,
        default=5)
    parser.add_argument(
        "--allow-class-regression",
        action="store_true",
        help=(
            "Rank by macro gain even if one target class regresses. By "
            "default A1 must beat ProtoOcc on every target class."))
    parser.add_argument(
        "--scene-names",
        nargs="+",
        help=(
            "Render these scene names in the given order without applying "
            "ranking, class-improvement, or GT-support filters."))
    parser.add_argument(
        "--no-callouts",
        action="store_true",
        help="Do not draw circles or class labels on the OCC panels.")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.min_scene_gt_voxels < 1:
        parser.error("--min-scene-gt-voxels must be positive")
    invalid = [
        class_id for class_id in args.class_ids
        if class_id < 0 or class_id >= len(CLASS_NAMES) - 1
    ]
    if invalid:
        parser.error(f"invalid class ids: {invalid}")
    return args


def empty_class_stats(class_ids):
    return {
        class_id: dict(
            gt=0,
            baseline_tp=0,
            baseline_fp=0,
            baseline_fn=0,
            ours_tp=0,
            ours_fp=0,
            ours_fn=0,
        )
        for class_id in class_ids
    }


def accumulate_stats(stats, gt, valid, baseline, ours, class_ids):
    for class_id in class_ids:
        gt_class = (gt == class_id) & valid
        baseline_class = (baseline == class_id) & valid
        ours_class = (ours == class_id) & valid
        stats[class_id]["gt"] += int(gt_class.sum())
        stats[class_id]["baseline_tp"] += int(
            (baseline_class & gt_class).sum())
        stats[class_id]["baseline_fp"] += int(
            (baseline_class & ~gt_class & valid).sum())
        stats[class_id]["baseline_fn"] += int(
            (~baseline_class & gt_class).sum())
        stats[class_id]["ours_tp"] += int((ours_class & gt_class).sum())
        stats[class_id]["ours_fp"] += int(
            (ours_class & ~gt_class & valid).sum())
        stats[class_id]["ours_fn"] += int((~ours_class & gt_class).sum())


def class_iou(stats, prefix):
    tp = stats[f"{prefix}_tp"]
    fp = stats[f"{prefix}_fp"]
    fn = stats[f"{prefix}_fn"]
    union = tp + fp + fn
    return float(tp / union) if union > 0 else np.nan


def compute_iou_summary(stats, class_ids):
    per_class = {}
    deltas = []
    for class_id in class_ids:
        baseline_iou = class_iou(stats[class_id], "baseline")
        ours_iou = class_iou(stats[class_id], "ours")
        delta = ours_iou - baseline_iou
        per_class[class_id] = dict(
            gt=stats[class_id]["gt"],
            baseline_iou=baseline_iou,
            ours_iou=ours_iou,
            delta=delta,
        )
        if np.isfinite(delta):
            deltas.append(delta)
    macro_delta = float(np.mean(deltas)) if deltas else -np.inf
    return per_class, macro_delta


def sample_summary(gt, valid, baseline, ours, class_ids):
    stats = empty_class_stats(class_ids)
    accumulate_stats(stats, gt, valid, baseline, ours, class_ids)
    per_class, macro_delta = compute_iou_summary(stats, class_ids)
    coverage = sum(item["gt"] > 0 for item in per_class.values())
    rescued = int(
        (((ours == gt) & (baseline != gt) & valid)
         & np.isin(gt, class_ids)).sum())
    regressed = int(
        (((baseline == gt) & (ours != gt) & valid)
         & np.isin(gt, class_ids)).sum())
    return dict(
        per_class=per_class,
        macro_delta=macro_delta,
        coverage=coverage,
        rescued=rescued,
        regressed=regressed,
        net=rescued - regressed,
    )


def rank_scenes(
        infos,
        baseline_paths,
        ours_paths,
        nuscenes_root,
        class_ids,
        min_scene_gt_voxels,
        require_all_positive,
        requested_scene_names=None):
    requested = (
        None
        if requested_scene_names is None
        else set(requested_scene_names))
    scenes = {}
    for index, info in enumerate(infos):
        token = str(info["token"])
        if token not in baseline_paths or token not in ours_paths:
            continue
        scene_name = str(info.get("scene_name", info.get("scene_token", "")))
        if requested is not None and scene_name not in requested:
            continue
        scene = scenes.setdefault(
            scene_name,
            dict(
                scene_name=scene_name,
                stats=empty_class_stats(class_ids),
                samples=[],
            ))
        baseline = load_prediction(baseline_paths[token])
        ours = load_prediction(ours_paths[token])
        gt, valid = load_gt(resolve_gt_path(info, nuscenes_root))
        if not (baseline.shape == ours.shape == gt.shape == valid.shape):
            raise ValueError(
                f"Shape mismatch at index {index}, token {token}: "
                f"{baseline.shape}, {ours.shape}, {gt.shape}, {valid.shape}")
        accumulate_stats(
            scene["stats"], gt, valid, baseline, ours, class_ids)
        frame_summary = sample_summary(
            gt, valid, baseline, ours, class_ids)
        frame_summary.update(index=index, token=token)
        scene["samples"].append(frame_summary)

    ranked = []
    for scene in scenes.values():
        per_class, macro_delta = compute_iou_summary(
            scene["stats"], class_ids)
        if requested is None:
            if any(
                    per_class[class_id]["gt"] < min_scene_gt_voxels
                    for class_id in class_ids):
                continue
            if (
                    require_all_positive
                    and any(
                        per_class[class_id]["delta"] <= 0.0
                        for class_id in class_ids)):
                continue
        # The representative frame should show as many target classes as
        # possible; within equal coverage, prefer the strongest macro IoU gain,
        # then the largest net number of corrected target voxels.
        representative = max(
            scene["samples"],
            key=lambda item: (
                item["coverage"],
                item["macro_delta"],
                item["net"],
            ))
        ranked.append(
            dict(
                scene_name=scene["scene_name"],
                macro_delta=macro_delta,
                per_class=per_class,
                representative=representative,
            ))
    if requested_scene_names is None:
        ranked.sort(
            key=lambda item: (
                item["macro_delta"],
                min(
                    item["per_class"][class_id]["delta"]
                    for class_id in class_ids),
            ),
            reverse=True,
        )
    else:
        order = {
            scene_name: index
            for index, scene_name in enumerate(requested_scene_names)
        }
        missing = requested.difference(
            scene["scene_name"] for scene in ranked)
        if missing:
            raise KeyError(
                f"Requested scenes not found in prediction caches: "
                f"{sorted(missing)}")
        ranked.sort(key=lambda item: order[item["scene_name"]])
    return ranked


def build_diverse_callouts(
        gt, valid, baseline, ours, class_ids, max_callouts):
    rescued = (
        (ours == gt)
        & (baseline != gt)
        & valid
        & np.isin(gt, class_ids)
    )
    structure = np.ones((3, 3), dtype=np.uint8)
    by_class = {}
    all_components = []
    for class_id in class_ids:
        rescue_bev = (rescued & (gt == class_id)).any(axis=2)
        labels, count = ndimage.label(rescue_bev, structure=structure)
        components = []
        for component_id in range(1, count + 1):
            coords = np.argwhere(labels == component_id)
            if not len(coords):
                continue
            low = coords.min(axis=0)
            high = coords.max(axis=0)
            center = 0.5 * (low + high)
            radius = max(5.0, 0.7 * float(np.max(high - low + 1)))
            component = dict(
                class_id=class_id,
                size=int(len(coords)),
                center=(float(center[0]), float(center[1])),
                radius=radius,
            )
            components.append(component)
            all_components.append(component)
        components.sort(key=lambda item: item["size"], reverse=True)
        by_class[class_id] = components

    selected = []
    selected_ids = set()
    # Ensure class coverage before filling with additional large components.
    for class_id in class_ids:
        if by_class[class_id]:
            component = by_class[class_id][0]
            selected.append(component)
            selected_ids.add(id(component))
            if len(selected) == max_callouts:
                return selected
    all_components.sort(key=lambda item: item["size"], reverse=True)
    for component in all_components:
        if id(component) in selected_ids:
            continue
        selected.append(component)
        if len(selected) == max_callouts:
            break
    return selected


def render_panel(
        path,
        volume,
        valid,
        title,
        callouts,
        class_ids):
    bev = topdown_semantic(volume, valid)
    rgb = CLASS_COLORS[np.clip(bev, 0, len(CLASS_COLORS) - 1)]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.imshow(
        rgb.transpose(1, 0, 2),
        origin="lower",
        interpolation="nearest",
    )
    ax.scatter(
        [(bev.shape[0] - 1) / 2.0],
        [(bev.shape[1] - 1) / 2.0],
        marker="+",
        s=75,
        linewidths=1.4,
        color="black",
        zorder=5,
    )
    color_by_class = {
        class_id: CALLOUT_COLORS[index % len(CALLOUT_COLORS)]
        for index, class_id in enumerate(class_ids)
    }
    for callout in callouts:
        center_x, center_y = callout["center"]
        color = color_by_class[callout["class_id"]]
        ax.add_patch(
            Circle(
                (center_x, center_y),
                callout["radius"],
                fill=False,
                edgecolor=color,
                linewidth=2.8,
                zorder=6,
            ))
        ax.text(
            center_x + callout["radius"],
            center_y + callout["radius"],
            CLASS_NAMES[callout["class_id"]],
            color=color,
            fontsize=9,
            fontweight="bold",
            zorder=7,
        )
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_ranking_csv(path, ranked, class_ids):
    fields = ["rank", "scene_name", "macro_iou_delta",
              "representative_index", "representative_token"]
    for class_id in class_ids:
        name = CLASS_NAMES[class_id]
        fields.extend([
            f"{name}_gt_voxels",
            f"{name}_protoocc_iou",
            f"{name}_a1_iou",
            f"{name}_iou_delta",
        ])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, scene in enumerate(ranked, start=1):
            row = dict(
                rank=rank,
                scene_name=scene["scene_name"],
                macro_iou_delta=scene["macro_delta"],
                representative_index=scene["representative"]["index"],
                representative_token=scene["representative"]["token"],
            )
            for class_id in class_ids:
                name = CLASS_NAMES[class_id]
                stats = scene["per_class"][class_id]
                row[f"{name}_gt_voxels"] = stats["gt"]
                row[f"{name}_protoocc_iou"] = stats["baseline_iou"]
                row[f"{name}_a1_iou"] = stats["ours_iou"]
                row[f"{name}_iou_delta"] = stats["delta"]
            writer.writerow(row)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = load_infos(args.info_pkl)
    baseline_paths = index_prediction_dir(args.baseline_dir)
    ours_paths = index_prediction_dir(args.ours_dir)

    ranked = rank_scenes(
        infos,
        baseline_paths,
        ours_paths,
        args.nuscenes_root,
        args.class_ids,
        args.min_scene_gt_voxels,
        require_all_positive=not args.allow_class_regression,
        requested_scene_names=args.scene_names,
    )
    if args.scene_names is None and len(ranked) < args.top_k:
        raise RuntimeError(
            f"Only {len(ranked)} scenes meet the target-class support "
            f"threshold; requested top-k={args.top_k}.")
    write_ranking_csv(
        out_dir / "scene_iou_ranking.csv",
        ranked,
        args.class_ids,
    )

    selected = ranked if args.scene_names is not None else ranked[:args.top_k]
    write_ranking_csv(
        out_dir / (
            "selected_scene_iou.csv"
            if args.scene_names is not None
            else "top5_scene_iou.csv"),
        selected,
        args.class_ids,
    )
    for rank, scene in enumerate(selected, start=1):
        sample = scene["representative"]
        index = sample["index"]
        token = sample["token"]
        info = infos[index]
        baseline = load_prediction(baseline_paths[token])
        ours = load_prediction(ours_paths[token])
        gt, valid = load_gt(resolve_gt_path(info, args.nuscenes_root))
        callouts = (
            []
            if args.no_callouts
            else build_diverse_callouts(
                gt,
                valid,
                baseline,
                ours,
                args.class_ids,
                args.max_callouts,
            ))
        prefix = f"{scene['scene_name']}_rank{rank:02d}"
        render_panel(
            out_dir / f"{prefix}_gt.png",
            gt,
            valid,
            f"{scene['scene_name']} — Ground Truth",
            callouts,
            args.class_ids,
        )
        render_panel(
            out_dir / f"{prefix}_protoocc.png",
            baseline,
            valid,
            f"{scene['scene_name']} — ProtoOcc",
            callouts,
            args.class_ids,
        )
        render_panel(
            out_dir / f"{prefix}_a1.png",
            ours,
            valid,
            f"{scene['scene_name']} — A1",
            callouts,
            args.class_ids,
        )

    print(f"Full ranking: {out_dir / 'scene_iou_ranking.csv'}")
    selected_csv_name = (
        "selected_scene_iou.csv" if args.scene_names else "top5_scene_iou.csv"
    )
    print(f"Selected scenes: {out_dir / selected_csv_name}")
    print(f"Rendered {len(selected) * 3} panels in {out_dir}")


if __name__ == "__main__":
    main()
