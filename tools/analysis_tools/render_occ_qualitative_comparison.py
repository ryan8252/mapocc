#!/usr/bin/env python3
"""Render paper-style GT / ProtoOcc / Ours qualitative comparisons.

The input prediction directories must use the layout produced by the
NuScenesDatasetOccpancy ``show_dir`` evaluator:

    <prediction_dir>/<scene-name>/<sample-token>/pred.npz

Each ``pred.npz`` must contain a uint8-like ``pred`` array with shape
``[200, 200, 16]``. The script ranks scenes before rendering them:

    rescued  = ours correct AND baseline wrong
    regressed = baseline correct AND ours wrong

Only camera-observed GT voxels are scored. The final panels use the same
semantic palette and the same automatically generated callout circles.
"""

import argparse
import csv
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch
from scipy import ndimage


CLASS_NAMES = (
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
)

# Palette follows the qualitative style commonly used by nuScenes occupancy
# papers: object classes are saturated, surfaces/stuff are muted, free is white.
CLASS_COLORS = np.asarray(
    [
        [0, 0, 0],
        [255, 127, 0],
        [251, 180, 174],
        [255, 235, 0],
        [0, 145, 220],
        [0, 190, 220],
        [190, 190, 0],
        [230, 0, 0],
        [255, 225, 145],
        [125, 55, 45],
        [135, 35, 165],
        [155, 155, 155],
        [100, 0, 80],
        [145, 0, 95],
        [145, 240, 90],
        [220, 220, 235],
        [0, 160, 0],
        [255, 255, 255],
    ],
    dtype=np.float32,
) / 255.0

CALLOUT_COLORS = (
    "#e41a1c",
    "#ffbf00",
    "#00a6d6",
    "#8c8c8c",
    "#984ea3",
    "#00a651",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select and render qualitative occupancy improvements as "
            "Ground Truth | ProtoOcc | Ours."
        ))
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="ProtoOcc prediction root containing nested pred.npz files.")
    parser.add_argument(
        "--ours-dir",
        required=True,
        help="Ours/A1 prediction root containing nested pred.npz files.")
    parser.add_argument(
        "--info-pkl",
        default="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl",
        help="nuScenes validation info PKL.")
    parser.add_argument(
        "--nuscenes-root",
        default="data/nuscenes",
        help="nuScenes data root used to resolve occupancy GT paths.")
    parser.add_argument(
        "--out-dir",
        default="work_dirs/qualitative_protoocc_vs_a1",
        help="Output directory.")
    parser.add_argument(
        "--target-classes",
        type=int,
        nargs="+",
        default=[2, 6, 7, 8, 3, 9, 10, 4, 5],
        help=(
            "Class IDs used for candidate ranking. Defaults to object classes "
            "with visible range/intra-class variation."))
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of rows in the main comparison montage.")
    parser.add_argument(
        "--rank-start",
        type=int,
        default=1,
        help=(
            "One-based starting rank for score/net selection. For example, "
            "--rank-start 10 --top-k 21 renders ranks 10 through 30."))
    parser.add_argument(
        "--selection-mode",
        choices=("diverse", "score", "net"),
        default="diverse",
        help=(
            "How to choose the final top-k: class-diverse examples, the "
            "class-balanced score, or the largest rescued-minus-regressed "
            "voxel count."))
    parser.add_argument(
        "--max-callouts",
        type=int,
        default=4,
        help="Maximum rescued components circled in each example.")
    parser.add_argument(
        "--no-callouts",
        action="store_true",
        help="Do not draw rescued-component circles or class labels.")
    parser.add_argument(
        "--min-gt-voxels",
        type=int,
        default=5,
        help="Minimum observed GT voxels for a class to affect ranking.")
    parser.add_argument(
        "--min-rescued-voxels",
        type=int,
        default=2,
        help="Minimum rescued target voxels required for an eligible scene.")
    parser.add_argument(
        "--allow-same-scene",
        action="store_true",
        help="Allow multiple frames from the same nuScenes scene.")
    parser.add_argument(
        "--selected-indices",
        type=int,
        nargs="+",
        help=(
            "Bypass automatic top-k selection and render these validation "
            "indices in the specified order."))
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.rank_start < 1:
        parser.error("--rank-start must be positive")
    if args.selection_mode == "diverse" and args.rank_start != 1:
        parser.error("--rank-start is only supported with score/net selection")
    if args.max_callouts < 1:
        parser.error("--max-callouts must be positive")
    invalid = [
        class_id for class_id in args.target_classes
        if class_id < 0 or class_id >= len(CLASS_NAMES) - 1
    ]
    if invalid:
        parser.error(f"invalid --target-classes: {invalid}")
    return args


def load_infos(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "infos" in payload:
        return payload["infos"]
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported info PKL structure in {path}")


def index_prediction_dir(root):
    root = Path(root)
    paths = {}
    for path in root.rglob("pred.npz"):
        token = path.parent.name
        if token in paths:
            raise RuntimeError(
                f"Duplicate sample token {token} under {root}: "
                f"{paths[token]} and {path}")
        paths[token] = path
    if not paths:
        raise FileNotFoundError(f"No nested pred.npz files found under {root}")
    return paths


def resolve_gt_path(info, nuscenes_root):
    occ_path = Path(info["occ_path"])
    candidates = [
        occ_path / "labels.npz",
        Path(nuscenes_root) / occ_path / "labels.npz",
    ]
    normalized = str(occ_path).replace("\\", "/")
    marker = "data/nuscenes/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        candidates.append(Path(nuscenes_root) / relative / "labels.npz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve GT labels for occ_path={info['occ_path']}; "
        f"tried {[str(path) for path in candidates]}")


def load_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        if "pred" not in data:
            raise KeyError(f"{path} does not contain `pred`")
        pred = data["pred"].astype(np.uint8, copy=False)
    if pred.ndim != 3:
        raise ValueError(f"Expected [X,Y,Z] prediction in {path}, got {pred.shape}")
    return pred


def load_gt(path):
    with np.load(path, allow_pickle=False) as data:
        semantics = data["semantics"].astype(np.uint8, copy=False)
        camera_mask = data["mask_camera"].astype(bool, copy=False)
    return semantics, camera_mask


def radial_masks(shape):
    x_size, y_size, _ = shape
    xs = np.arange(x_size, dtype=np.float32)
    ys = np.arange(y_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    cx = (x_size - 1) / 2.0
    cy = (y_size - 1) / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    distance /= distance.max() + 1e-6
    near = distance < 0.5
    return near[:, :, None], (~near)[:, :, None]


def score_sample(gt, camera_mask, baseline, ours, target_classes):
    valid = camera_mask & (gt < len(CLASS_NAMES))
    baseline_correct = (baseline == gt) & valid
    ours_correct = (ours == gt) & valid
    rescued = ours_correct & ~baseline_correct
    regressed = baseline_correct & ~ours_correct
    near_mask, far_mask = radial_masks(gt.shape)

    per_class = {}
    score = 0.0
    total_rescued = 0
    total_regressed = 0
    active_classes = 0
    for class_id in target_classes:
        gt_class = valid & (gt == class_id)
        gt_count = int(gt_class.sum())
        rescue_count = int((rescued & gt_class).sum())
        regress_count = int((regressed & gt_class).sum())
        near_rescue = int((rescued & gt_class & near_mask).sum())
        far_rescue = int((rescued & gt_class & far_mask).sum())
        per_class[class_id] = dict(
            gt=gt_count,
            rescued=rescue_count,
            regressed=regress_count,
            near_rescued=near_rescue,
            far_rescued=far_rescue,
        )
        if gt_count > 0:
            # Class-balanced net recovery with a mild support term. This keeps
            # tiny classes relevant without letting a one-voxel accident rank
            # above a well-supported improvement.
            net_rate = (rescue_count - regress_count) / gt_count
            support = math.log1p(rescue_count)
            score += net_rate * support
            total_rescued += rescue_count
            total_regressed += regress_count
            active_classes += 1

    return dict(
        score=score,
        rescued=total_rescued,
        regressed=total_regressed,
        net=total_rescued - total_regressed,
        active_classes=active_classes,
        per_class=per_class,
    )


def evaluate_candidates(
        infos,
        baseline_paths,
        ours_paths,
        nuscenes_root,
        target_classes,
        min_gt_voxels,
        min_rescued_voxels):
    rows = []
    global_counts = {
        class_id: dict(gt=0, rescued=0, regressed=0,
                       near_rescued=0, far_rescued=0)
        for class_id in target_classes
    }
    for index, info in enumerate(infos):
        token = str(info["token"])
        if token not in baseline_paths or token not in ours_paths:
            continue
        baseline = load_prediction(baseline_paths[token])
        ours = load_prediction(ours_paths[token])
        gt, camera_mask = load_gt(resolve_gt_path(info, nuscenes_root))
        if not (baseline.shape == ours.shape == gt.shape == camera_mask.shape):
            raise ValueError(
                f"Shape mismatch for {token}: baseline={baseline.shape}, "
                f"ours={ours.shape}, gt={gt.shape}, mask={camera_mask.shape}")
        stats = score_sample(
            gt, camera_mask, baseline, ours, target_classes)
        eligible_rescued = 0
        for class_id, class_stats in stats["per_class"].items():
            for key in global_counts[class_id]:
                global_counts[class_id][key] += class_stats[key]
            if class_stats["gt"] >= min_gt_voxels:
                eligible_rescued += class_stats["rescued"]
        if eligible_rescued < min_rescued_voxels:
            continue
        rows.append(
            dict(
                index=index,
                token=token,
                scene_name=str(info.get("scene_name", info.get("scene_token", ""))),
                score=float(stats["score"]),
                rescued=int(stats["rescued"]),
                regressed=int(stats["regressed"]),
                net=int(stats["net"]),
                per_class=stats["per_class"],
            ))
    rows.sort(key=lambda row: (row["score"], row["net"]), reverse=True)
    return rows, global_counts


def choose_diverse_rows(rows, target_classes, top_k, allow_same_scene):
    selected = []
    used_tokens = set()
    used_scenes = set()

    # First pass: try to cover distinct target classes with positive net gain.
    for class_id in target_classes:
        best = None
        best_value = None
        for row in rows:
            if row["token"] in used_tokens:
                continue
            if not allow_same_scene and row["scene_name"] in used_scenes:
                continue
            stats = row["per_class"][class_id]
            value = stats["rescued"] - stats["regressed"]
            if stats["rescued"] <= 0 or value <= 0:
                continue
            rank_value = (value / max(stats["gt"], 1), stats["rescued"])
            if best_value is None or rank_value > best_value:
                best = row
                best_value = rank_value
        if best is not None:
            selected.append(best)
            used_tokens.add(best["token"])
            used_scenes.add(best["scene_name"])
        if len(selected) >= top_k:
            return selected

    # Second pass: fill remaining slots by the global deterministic score.
    for row in rows:
        if row["token"] in used_tokens:
            continue
        if not allow_same_scene and row["scene_name"] in used_scenes:
            continue
        selected.append(row)
        used_tokens.add(row["token"])
        used_scenes.add(row["scene_name"])
        if len(selected) >= top_k:
            break
    return selected


def choose_ranked_rows(
        rows, top_k, allow_same_scene, ranking_key, rank_start=1):
    """Take the highest-ranked rows, optionally keeping scenes distinct."""
    ranked = sorted(
        rows,
        key=lambda row: (row[ranking_key], row["score"], row["net"]),
        reverse=True,
    )
    offset = rank_start - 1
    if allow_same_scene:
        return ranked[offset:offset + top_k]

    distinct_rows = []
    used_scenes = set()
    for row in ranked:
        if row["scene_name"] in used_scenes:
            continue
        distinct_rows.append(row)
        used_scenes.add(row["scene_name"])
        if len(distinct_rows) >= offset + top_k:
            break
    return distinct_rows[offset:offset + top_k]


def topdown_semantic(semantics, camera_mask):
    """Project [X,Y,Z] labels to BEV, keeping the highest observed occupied voxel."""
    if semantics.shape != camera_mask.shape:
        raise ValueError("semantics and camera_mask must have identical shapes")
    x_size, y_size, z_size = semantics.shape
    bev = np.full((x_size, y_size), 17, dtype=np.uint8)
    observed = np.zeros((x_size, y_size), dtype=bool)
    for z_index in range(z_size):
        valid = camera_mask[:, :, z_index]
        observed |= valid
        occupied = valid & (semantics[:, :, z_index] != 17)
        bev[occupied] = semantics[:, :, z_index][occupied]
    bev[~observed] = 17
    return bev


def build_callouts(
        gt,
        camera_mask,
        baseline,
        ours,
        target_classes,
        max_callouts):
    valid = camera_mask & (gt < len(CLASS_NAMES))
    rescued = (ours == gt) & (baseline != gt) & valid
    candidates = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_id in target_classes:
        rescue_bev = (rescued & (gt == class_id)).any(axis=2)
        labels, component_count = ndimage.label(rescue_bev, structure=structure)
        for component_id in range(1, component_count + 1):
            coords = np.argwhere(labels == component_id)
            if len(coords) == 0:
                continue
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            center_x = 0.5 * (x_min + x_max)
            center_y = 0.5 * (y_min + y_max)
            radius = max(5.0, 0.65 * max(x_max - x_min + 1, y_max - y_min + 1))
            candidates.append(
                dict(
                    class_id=class_id,
                    size=int(len(coords)),
                    center=(center_x, center_y),
                    radius=float(radius),
                ))
    candidates.sort(key=lambda item: item["size"], reverse=True)
    return candidates[:max_callouts]


def render_example(
        axes,
        gt,
        camera_mask,
        baseline,
        ours,
        callouts,
        row_label=None):
    panels = (
        ("Ground Truth", gt),
        ("ProtoOcc", baseline),
        ("Ours", ours),
    )
    for column, (title, volume) in enumerate(panels):
        ax = axes[column]
        bev = topdown_semantic(volume, camera_mask)
        rgb = CLASS_COLORS[np.clip(bev, 0, len(CLASS_COLORS) - 1)]
        ax.imshow(rgb.transpose(1, 0, 2), origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.scatter(
            [(bev.shape[0] - 1) / 2.0],
            [(bev.shape[1] - 1) / 2.0],
            marker="+",
            s=60,
            linewidths=1.2,
            color="black",
            zorder=5,
        )
        for callout_index, callout in enumerate(callouts):
            center_x, center_y = callout["center"]
            color = CALLOUT_COLORS[callout_index % len(CALLOUT_COLORS)]
            circle = Circle(
                (center_x, center_y),
                callout["radius"],
                fill=False,
                edgecolor=color,
                linewidth=2.2,
                zorder=6,
            )
            ax.add_patch(circle)
            ax.text(
                center_x + callout["radius"],
                center_y + callout["radius"],
                CLASS_NAMES[callout["class_id"]],
                color=color,
                fontsize=7,
                fontweight="bold",
                zorder=7,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        if row_label is not None and column == 0:
            ax.set_ylabel(row_label, fontsize=8)


def render_montage(
        path,
        selected_rows,
        infos,
        baseline_paths,
        ours_paths,
        nuscenes_root,
        target_classes,
        max_callouts,
        no_callouts=False):
    row_count = len(selected_rows)
    fig, axes = plt.subplots(
        row_count,
        3,
        figsize=(12.8, 4.0 * row_count),
        squeeze=False,
    )
    for row_index, row in enumerate(selected_rows):
        info = infos[row["index"]]
        baseline = load_prediction(baseline_paths[row["token"]])
        ours = load_prediction(ours_paths[row["token"]])
        gt, camera_mask = load_gt(resolve_gt_path(info, nuscenes_root))
        callouts = (
            []
            if no_callouts
            else build_callouts(
                gt,
                camera_mask,
                baseline,
                ours,
                target_classes,
                max_callouts,
            )
        )
        label = (
            f"idx={row['index']}  scene={row['scene_name']}\n"
            f"rescued={row['rescued']} regressed={row['regressed']}")
        render_example(
            axes[row_index],
            gt,
            camera_mask,
            baseline,
            ours,
            callouts,
            row_label=label,
        )

    legend = [
        Patch(facecolor=CLASS_COLORS[index], edgecolor="black", linewidth=0.2,
              label=name)
        for index, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(
        handles=legend,
        loc="center left",
        bbox_to_anchor=(0.91, 0.5),
        fontsize=8,
        ncol=1,
        frameon=False,
    )
    fig.suptitle(
        "Qualitative occupancy comparison: ProtoOcc vs. Ours",
        fontsize=16,
        fontweight="bold",
    )
    if not no_callouts:
        fig.text(
            0.5,
            0.012,
            "Callouts mark GT components where Ours predicts the correct class and "
            "ProtoOcc does not.",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0.025, 0.90, 0.965))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_selection_csv(path, rows, target_classes):
    fields = [
        "rank",
        "dataset_index",
        "scene_name",
        "sample_token",
        "score",
        "rescued",
        "regressed",
        "net",
    ]
    for class_id in target_classes:
        name = CLASS_NAMES[class_id]
        fields.extend([
            f"{name}_gt",
            f"{name}_rescued",
            f"{name}_regressed",
            f"{name}_near_rescued",
            f"{name}_far_rescued",
        ])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            output = dict(
                rank=rank,
                dataset_index=row["index"],
                scene_name=row["scene_name"],
                sample_token=row["token"],
                score=row["score"],
                rescued=row["rescued"],
                regressed=row["regressed"],
                net=row["net"],
            )
            for class_id in target_classes:
                name = CLASS_NAMES[class_id]
                stats = row["per_class"][class_id]
                for key, value in stats.items():
                    output[f"{name}_{key}"] = value
            writer.writerow(output)


def write_global_csv(path, global_counts, target_classes):
    fields = [
        "class_id",
        "class_name",
        "gt",
        "rescued",
        "regressed",
        "net",
        "near_rescued",
        "far_rescued",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for class_id in target_classes:
            counts = global_counts[class_id]
            writer.writerow(
                dict(
                    class_id=class_id,
                    class_name=CLASS_NAMES[class_id],
                    gt=counts["gt"],
                    rescued=counts["rescued"],
                    regressed=counts["regressed"],
                    net=counts["rescued"] - counts["regressed"],
                    near_rescued=counts["near_rescued"],
                    far_rescued=counts["far_rescued"],
                ))


def rows_from_selected_indices(
        selected_indices,
        infos,
        baseline_paths,
        ours_paths,
        nuscenes_root,
        target_classes):
    rows = []
    for index in selected_indices:
        if index < 0 or index >= len(infos):
            raise IndexError(f"Selected index {index} outside [0, {len(infos)})")
        info = infos[index]
        token = str(info["token"])
        if token not in baseline_paths or token not in ours_paths:
            raise KeyError(f"Missing prediction for selected token {token}")
        baseline = load_prediction(baseline_paths[token])
        ours = load_prediction(ours_paths[token])
        gt, camera_mask = load_gt(resolve_gt_path(info, nuscenes_root))
        stats = score_sample(gt, camera_mask, baseline, ours, target_classes)
        rows.append(
            dict(
                index=index,
                token=token,
                scene_name=str(info.get("scene_name", info.get("scene_token", ""))),
                score=float(stats["score"]),
                rescued=int(stats["rescued"]),
                regressed=int(stats["regressed"]),
                net=int(stats["net"]),
                per_class=stats["per_class"],
            ))
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = load_infos(args.info_pkl)
    baseline_paths = index_prediction_dir(args.baseline_dir)
    ours_paths = index_prediction_dir(args.ours_dir)

    candidates, global_counts = evaluate_candidates(
        infos,
        baseline_paths,
        ours_paths,
        args.nuscenes_root,
        args.target_classes,
        args.min_gt_voxels,
        args.min_rescued_voxels,
    )
    write_selection_csv(
        out_dir / "all_ranked_candidates.csv",
        candidates,
        args.target_classes,
    )
    write_global_csv(
        out_dir / "global_rescue_regression.csv",
        global_counts,
        args.target_classes,
    )

    if args.selected_indices is not None:
        selected = rows_from_selected_indices(
            args.selected_indices,
            infos,
            baseline_paths,
            ours_paths,
            args.nuscenes_root,
            args.target_classes,
        )
    elif args.selection_mode == "diverse":
        selected = choose_diverse_rows(
            candidates,
            args.target_classes,
            args.top_k,
            args.allow_same_scene,
        )
    else:
        selected = choose_ranked_rows(
            candidates,
            args.top_k,
            args.allow_same_scene,
            ranking_key=args.selection_mode,
            rank_start=args.rank_start,
        )
    if not selected:
        raise RuntimeError(
            "No eligible scenes were selected. Check prediction coverage or "
            "lower --min-rescued-voxels.")

    write_selection_csv(
        out_dir / "selected_examples.csv",
        selected,
        args.target_classes,
    )
    render_montage(
        out_dir / "qualitative_comparison.png",
        selected,
        infos,
        baseline_paths,
        ours_paths,
        args.nuscenes_root,
        args.target_classes,
        args.max_callouts,
        args.no_callouts,
    )
    print(f"Ranked candidates: {out_dir / 'all_ranked_candidates.csv'}")
    print(f"Selected examples: {out_dir / 'selected_examples.csv'}")
    print(f"Figure: {out_dir / 'qualitative_comparison.png'}")


if __name__ == "__main__":
    main()
