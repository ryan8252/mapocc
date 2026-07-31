#!/usr/bin/env python3
"""Select and render thesis-ready BEV map segmentation improvements.

The inputs are token-indexed ``map.npz`` trees exported by the BEVFusion and
ProtoOcc datasets.  Map segmentation is multi-label, so all comparisons are
performed independently per channel; no argmax is used.  BEVFusion tensors are
also converted to ProtoOcc's canonical ``(class, x, y)`` axis convention before
any pixel-wise comparison.

By default, ``drivable_area`` is excluded because the two local loaders use
different nuScenes layer definitions for that channel.  The five remaining GT
channels must agree almost exactly.  Each model uses one fixed set of global
per-class validation thresholds matching the reported IoU@max evaluation;
thresholds are never tuned per frame.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402


MAP_CLASSES = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)

DISPLAY_NAMES = {
    "drivable_area": "Drivable area",
    "ped_crossing": "Pedestrian crossing",
    "walkway": "Walkway",
    "stop_line": "Stop line",
    "carpark_area": "Carpark area",
    "divider": "Divider",
}

# Match BEVFusion's stock map palette.  Later entries overwrite earlier ones
# where independent map channels overlap, keeping thin structures visible.
MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "divider": (106, 61, 154),
}
DRAW_ORDER = (
    "drivable_area",
    "carpark_area",
    "walkway",
    "ped_crossing",
    "stop_line",
    "divider",
)

DEFAULT_RENDER_CLASSES = (
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)
DEFAULT_TARGET_CLASSES = ("ped_crossing", "stop_line", "divider")
DEFAULT_BOUND = (-40.0, 40.0, 0.4)
# Fixed global thresholds from the two exact thesis checkpoints' result.md
# files (class order is MAP_CLASSES).  Override both flags for other models.
DEFAULT_BASELINE_THRESHOLDS = (0.45, 0.45, 0.45, 0.40, 0.35, 0.40)
DEFAULT_OURS_THRESHOLDS = (0.35, 0.35, 0.40, 0.35, 0.35, 0.60)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rank multi-label BEV map improvements and render clean "
            "Ground Truth | BEVFusion | Ours comparisons."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="BEVFusion export root containing <scene>/<token>/map.npz.",
    )
    parser.add_argument(
        "--ours-dir",
        required=True,
        help="MapOcc/ProtoOcc export root containing <scene>/<token>/map.npz.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/qualitative_map/bevfusion_vs_ours",
    )
    parser.add_argument(
        "--render-classes",
        nargs="+",
        choices=MAP_CLASSES,
        default=list(DEFAULT_RENDER_CLASSES),
        help=(
            "Channels rendered and used for per-frame mean IoU. "
            "drivable_area is intentionally omitted by default."
        ),
    )
    parser.add_argument(
        "--target-classes",
        nargs="+",
        choices=MAP_CLASSES,
        default=list(DEFAULT_TARGET_CLASSES),
        help="Thin classes used for rescued-region selection.",
    )
    parser.add_argument(
        "--baseline-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_BASELINE_THRESHOLDS),
        help=(
            "One shared threshold or six values in MAP_CLASSES order. "
            "Default: BEVFusion's fixed validation IoU@max thresholds."
        ),
    )
    parser.add_argument(
        "--ours-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_OURS_THRESHOLDS),
        help=(
            "One shared threshold or six values in MAP_CLASSES order. "
            "Default: Ours' fixed validation IoU@max thresholds."
        ),
    )
    parser.add_argument("--min-components", type=int, default=1)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--min-component-pixels", type=int, default=6)
    parser.add_argument("--local-regression-radius", type=int, default=2)
    parser.add_argument("--max-local-regressed-pixels", type=int, default=4)
    parser.add_argument("--max-regressed-errors", type=int, default=120)
    parser.add_argument("--min-fix-regress-ratio", type=float, default=1.25)
    parser.add_argument("--min-mean-iou-gain", type=float, default=0.0)
    parser.add_argument(
        "--min-per-class-iou-gain",
        type=float,
        default=-0.01,
        help=(
            "Reject a frame if any rendered class falls below this IoU gain. "
            "Default allows at most a 0.01 per-class drop."
        ),
    )
    parser.add_argument("--regression-penalty", type=float, default=1.0)
    parser.add_argument("--rank-start", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--allow-same-scene",
        action="store_true",
        help="Allow multiple selected frames from one nuScenes scene.",
    )
    parser.add_argument(
        "--no-class-diversity",
        action="store_true",
        help="Do not reserve one selected example for each target class.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Use only the intersection if the two token sets differ.",
    )
    parser.add_argument(
        "--min-gt-parity",
        type=float,
        default=1.0,
        help="Minimum per-channel GT IoU after axis alignment.",
    )
    parser.add_argument(
        "--allow-gt-mismatch",
        action="store_true",
        help="Continue when an included GT channel fails the parity check.",
    )
    parser.add_argument(
        "--expected-xbound",
        type=float,
        nargs=3,
        default=list(DEFAULT_BOUND),
    )
    parser.add_argument(
        "--expected-ybound",
        type=float,
        nargs=3,
        default=list(DEFAULT_BOUND),
    )
    parser.add_argument(
        "--allow-any-geometry",
        action="store_true",
        help="Accept any geometry as long as both exports match exactly.",
    )
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Add ranking statistics to rendered row labels.",
    )
    args = parser.parse_args()

    for name in ("render_classes", "target_classes"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name.replace('_', '-')} contains duplicates")
    missing_targets = set(args.target_classes) - set(args.render_classes)
    if missing_targets:
        parser.error(
            "--target-classes must be included in --render-classes; missing "
            + ", ".join(sorted(missing_targets))
        )
    for option in ("baseline_thresholds", "ours_thresholds"):
        values = getattr(args, option)
        if len(values) not in (1, len(MAP_CLASSES)):
            parser.error(
                f"--{option.replace('_', '-')} needs one value or "
                f"{len(MAP_CLASSES)} values"
            )
        if any(value < 0.0 or value > 1.0 for value in values):
            parser.error(f"--{option.replace('_', '-')} must be in [0, 1]")
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
    if args.max_regressed_errors < 0:
        parser.error("--max-regressed-errors must be non-negative")
    if args.min_fix_regress_ratio < 0:
        parser.error("--min-fix-regress-ratio must be non-negative")
    if not -1.0 <= args.min_per_class_iou_gain <= 1.0:
        parser.error("--min-per-class-iou-gain must be in [-1, 1]")
    if args.regression_penalty < 0:
        parser.error("--regression-penalty must be non-negative")
    if args.rank_start < 1 or args.top_k < 1:
        parser.error("--rank-start and --top-k must be positive")
    if not 0.0 <= args.min_gt_parity <= 1.0:
        parser.error("--min-gt-parity must be in [0, 1]")
    if args.progress_interval < 0 or args.dpi < 1:
        parser.error("--progress-interval must be non-negative and --dpi positive")
    return args


def scalar_text(value, key, path):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{path}: {key} must contain exactly one string")
    return str(array.reshape(-1)[0])


def index_export(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Map export directory does not exist: {root}")

    paths = {}
    for path in root.rglob("map.npz"):
        token = path.parent.name
        if token in paths:
            raise ValueError(
                f"Duplicate sample token {token}: {paths[token]} and {path}"
            )
        paths[token] = path
    if not paths:
        raise FileNotFoundError(f"No <scene>/<token>/map.npz files under {root}")
    return paths


def convert_to_protoocc_axis(array, axis_convention, path):
    if axis_convention == "protoocc":
        return np.asarray(array).copy()
    if axis_convention == "bevfusion":
        # Inverse of ProtoOcc -> BEVFusion:
        #   p.transpose(0, 2, 1)[:, ::-1, :]
        return np.asarray(array).transpose(0, 2, 1)[:, :, ::-1].copy()
    raise ValueError(
        f"{path}: unknown axis_convention={axis_convention!r}; "
        "expected 'bevfusion' or 'protoocc'"
    )


def load_map_sample(path):
    with np.load(path, allow_pickle=False) as data:
        required = {
            "probs",
            "gt",
            "classes",
            "xbound",
            "ybound",
            "sample_token",
            "scene_name",
            "axis_convention",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path}: missing NPZ keys {sorted(missing)}")

        probs = np.asarray(data["probs"], dtype=np.float32)
        gt = np.asarray(data["gt"], dtype=np.uint8)
        classes = tuple(str(value) for value in np.asarray(data["classes"]).tolist())
        xbound = np.asarray(data["xbound"], dtype=np.float64)
        ybound = np.asarray(data["ybound"], dtype=np.float64)
        token = scalar_text(data["sample_token"], "sample_token", path)
        scene_name = scalar_text(data["scene_name"], "scene_name", path)
        axis = scalar_text(data["axis_convention"], "axis_convention", path)

    if token != path.parent.name:
        raise ValueError(
            f"{path}: metadata token {token} != directory {path.parent.name}"
        )
    if len(classes) != len(set(classes)):
        raise ValueError(f"{path}: duplicate class names {classes}")
    if probs.ndim != 3 or gt.ndim != 3 or probs.shape != gt.shape:
        raise ValueError(
            f"{path}: expected matching (C,H,W) arrays, got "
            f"probs={probs.shape}, gt={gt.shape}"
        )
    if probs.shape[0] != len(classes):
        raise ValueError(
            f"{path}: {probs.shape[0]} channels but {len(classes)} classes"
        )
    if not np.isfinite(probs).all():
        raise ValueError(f"{path}: probabilities contain NaN or Inf")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError(f"{path}: probabilities are outside [0, 1]")
    if xbound.shape != (3,) or ybound.shape != (3,):
        raise ValueError(f"{path}: xbound and ybound must each have 3 values")

    probs = convert_to_protoocc_axis(probs, axis, path)
    gt = convert_to_protoocc_axis(gt, axis, path).astype(bool, copy=False)
    missing_classes = set(MAP_CLASSES) - set(classes)
    if missing_classes:
        raise ValueError(f"{path}: missing map classes {sorted(missing_classes)}")
    order = [classes.index(name) for name in MAP_CLASSES]
    probs = probs[order]
    gt = gt[order]

    cells_x = int(round((xbound[1] - xbound[0]) / xbound[2]))
    cells_y = int(round((ybound[1] - ybound[0]) / ybound[2]))
    expected_shape = (len(MAP_CLASSES), cells_x, cells_y)
    if probs.shape != expected_shape:
        raise ValueError(
            f"{path}: geometry implies {expected_shape}, got {probs.shape} "
            "after conversion to ProtoOcc axes"
        )

    return {
        "probs": probs,
        "gt": gt,
        "token": token,
        "scene_name": scene_name,
        "xbound": xbound,
        "ybound": ybound,
    }


def expand_thresholds(values):
    if len(values) == 1:
        return np.full(len(MAP_CLASSES), values[0], dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def binary_iou(first, second):
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def per_class_iou(prediction, target, indices):
    return np.asarray(
        [binary_iou(prediction[index], target[index]) for index in indices],
        dtype=np.float64,
    )


def validate_pair(baseline, ours, args, baseline_path, ours_path):
    if baseline["token"] != ours["token"]:
        raise ValueError(
            f"Token mismatch: {baseline_path}={baseline['token']} vs "
            f"{ours_path}={ours['token']}"
        )
    if baseline["scene_name"] != ours["scene_name"]:
        raise ValueError(
            f"Scene mismatch for {ours['token']}: "
            f"{baseline['scene_name']} vs {ours['scene_name']}"
        )
    for name in ("xbound", "ybound"):
        if not np.allclose(baseline[name], ours[name], rtol=0.0, atol=1e-6):
            raise ValueError(
                f"Geometry mismatch for {ours['token']} ({name}): "
                f"BEVFusion={baseline[name].tolist()} vs "
                f"Ours={ours[name].tolist()}"
            )
    if not args.allow_any_geometry:
        expected = {
            "xbound": np.asarray(args.expected_xbound),
            "ybound": np.asarray(args.expected_ybound),
        }
        for name, values in expected.items():
            if not np.allclose(ours[name], values, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"Unexpected thesis grid for {ours['token']} ({name}): "
                    f"got {ours[name].tolist()}, expected {values.tolist()}. "
                    "Use the aligned checkpoints or explicitly pass "
                    "--allow-any-geometry."
                )

    render_indices = [MAP_CLASSES.index(name) for name in args.render_classes]
    gt_ious = per_class_iou(baseline["gt"], ours["gt"], render_indices)
    minimum = float(gt_ious.min())
    if minimum < args.min_gt_parity and not args.allow_gt_mismatch:
        details = ", ".join(
            f"{MAP_CLASSES[index]}={iou:.6f}"
            for index, iou in zip(render_indices, gt_ious)
        )
        raise ValueError(
            f"GT protocol mismatch for {ours['token']}: {details}. "
            "Do not rank incomparable channels; exclude the mismatched class "
            "or pass --allow-gt-mismatch after documenting the protocol."
        )
    return minimum


def analyze_frame(gt, baseline_pred, ours_pred, args):
    render_indices = [MAP_CLASSES.index(name) for name in args.render_classes]
    target_indices = [MAP_CLASSES.index(name) for name in args.target_classes]

    baseline_iou = per_class_iou(baseline_pred, gt, render_indices)
    ours_iou = per_class_iou(ours_pred, gt, render_indices)
    # Average only over classes present in this frame's GT.  Otherwise removing
    # one false-positive pixel from an empty class changes its IoU from 0 to 1
    # and can dominate a qualitative ranking.  Such FP fixes/regressions remain
    # represented by the explicit pixel counts below.
    present_classes = np.asarray(
        [bool(gt[index].any()) for index in render_indices]
    )
    if not present_classes.any():
        return None
    mean_baseline_iou = float(baseline_iou[present_classes].mean())
    mean_ours_iou = float(ours_iou[present_classes].mean())
    per_class_iou_gain = ours_iou - baseline_iou
    mean_iou_gain = mean_ours_iou - mean_baseline_iou
    if mean_iou_gain < args.min_mean_iou_gain:
        return None
    if float(per_class_iou_gain.min()) < args.min_per_class_iou_gain:
        return None

    # Count fixes/regressions over every rendered class.  Connected components
    # are still required to come from the requested thin target classes below.
    rescued_pos = gt & ours_pred & ~baseline_pred
    lost_pos = gt & baseline_pred & ~ours_pred
    removed_fp = ~gt & baseline_pred & ~ours_pred
    new_fp = ~gt & ours_pred & ~baseline_pred

    rescued_count = int(rescued_pos[render_indices].sum())
    lost_count = int(lost_pos[render_indices].sum())
    removed_fp_count = int(removed_fp[render_indices].sum())
    new_fp_count = int(new_fp[render_indices].sum())
    fixed_errors = rescued_count + removed_fp_count
    regressed_errors = lost_count + new_fp_count
    if regressed_errors > args.max_regressed_errors:
        return None
    if (
        regressed_errors > 0
        and fixed_errors < args.min_fix_regress_ratio * regressed_errors
    ):
        return None

    regressed_visible = np.any(
        (lost_pos | new_fp)[render_indices], axis=0
    )
    structure = np.ones((3, 3), dtype=np.uint8)
    components = []
    for class_index in target_indices:
        labels, count = ndimage.label(
            rescued_pos[class_index], structure=structure
        )
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
            local_regressed = int((regressed_visible & local_region).sum())
            if local_regressed > args.max_local_regressed_pixels:
                continue
            center = np.argwhere(mask).mean(axis=0)
            components.append(
                {
                    "class_name": MAP_CLASSES[class_index],
                    "pixels": pixels,
                    "local_regressed": local_regressed,
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                }
            )

    components.sort(key=lambda item: item["pixels"], reverse=True)
    if len(components) < args.min_components:
        return None
    ranked_components = (
        components if args.max_components == 0 else components[: args.max_components]
    )
    component_pixels = sum(item["pixels"] for item in ranked_components)
    primary_class = ranked_components[0]["class_name"]
    ratio = math.inf if regressed_errors == 0 else fixed_errors / regressed_errors
    score = (
        1000.0 * mean_iou_gain
        + component_pixels
        + 0.25 * removed_fp_count
        - args.regression_penalty * regressed_errors
    )

    return {
        "score": float(score),
        "mean_baseline_iou": mean_baseline_iou,
        "mean_ours_iou": mean_ours_iou,
        "mean_iou_gain": mean_iou_gain,
        "min_per_class_iou_gain": float(per_class_iou_gain.min()),
        "per_class_iou_gain": {
            MAP_CLASSES[index]: float(gain)
            for index, gain in zip(render_indices, per_class_iou_gain)
        },
        "rescued_positive": rescued_count,
        "removed_false_positive": removed_fp_count,
        "fixed_errors": fixed_errors,
        "lost_positive": lost_count,
        "new_false_positive": new_fp_count,
        "regressed_errors": regressed_errors,
        "fix_regress_ratio": ratio,
        "component_count": len(components),
        "ranked_component_count": len(ranked_components),
        "component_pixels": component_pixels,
        "primary_class": primary_class,
        "components": ranked_components,
    }


def find_candidates(args, baseline_paths, ours_paths):
    baseline_tokens = set(baseline_paths)
    ours_tokens = set(ours_paths)
    if baseline_tokens != ours_tokens and not args.allow_partial:
        only_baseline = sorted(baseline_tokens - ours_tokens)
        only_ours = sorted(ours_tokens - baseline_tokens)
        raise ValueError(
            "The two exports do not contain identical token sets: "
            f"BEVFusion-only={len(only_baseline)} {only_baseline[:3]}, "
            f"Ours-only={len(only_ours)} {only_ours[:3]}. "
            "Complete both exports or pass --allow-partial."
        )
    tokens = sorted(baseline_tokens & ours_tokens)
    if not tokens:
        raise RuntimeError("The two map exports have no common sample tokens")

    baseline_thresholds = expand_thresholds(args.baseline_thresholds)
    ours_thresholds = expand_thresholds(args.ours_thresholds)
    rows = []
    for position, token in enumerate(tokens, start=1):
        baseline = load_map_sample(baseline_paths[token])
        ours = load_map_sample(ours_paths[token])
        gt_parity = validate_pair(
            baseline, ours, args, baseline_paths[token], ours_paths[token]
        )
        baseline_pred = baseline["probs"] >= baseline_thresholds[:, None, None]
        ours_pred = ours["probs"] >= ours_thresholds[:, None, None]
        stats = analyze_frame(ours["gt"], baseline_pred, ours_pred, args)
        if stats is not None:
            rows.append(
                {
                    "scene_name": ours["scene_name"],
                    "token": token,
                    "gt_parity_min_iou": gt_parity,
                    **stats,
                }
            )
        if args.progress_interval and position % args.progress_interval == 0:
            print(
                f"Compared {position}/{len(tokens)} samples; "
                f"eligible={len(rows)}",
                flush=True,
            )

    rows.sort(
        key=lambda row: (
            row["score"],
            row["mean_iou_gain"],
            row["component_pixels"],
            row["fixed_errors"],
        ),
        reverse=True,
    )
    for global_rank, row in enumerate(rows, start=1):
        row["global_rank"] = global_rank
    return rows, len(tokens)


def select_candidates(candidates, args):
    candidates = candidates[args.rank_start - 1:]
    selected = []
    selected_tokens = set()
    selected_scenes = set()

    def can_add(row):
        return (
            row["token"] not in selected_tokens
            and (args.allow_same_scene or row["scene_name"] not in selected_scenes)
        )

    def add(row):
        selected.append(row)
        selected_tokens.add(row["token"])
        selected_scenes.add(row["scene_name"])

    if not args.no_class_diversity:
        for class_name in args.target_classes:
            match = next(
                (
                    row
                    for row in candidates
                    if row["primary_class"] == class_name and can_add(row)
                ),
                None,
            )
            if match is not None:
                add(match)
            if len(selected) == args.top_k:
                break

    if len(selected) < args.top_k:
        for row in candidates:
            if can_add(row):
                add(row)
            if len(selected) == args.top_k:
                break

    selected.sort(key=lambda row: row["score"], reverse=True)
    return selected


def component_summary(components):
    return "; ".join(
        f"{item['class_name']}:pixels={item['pixels']},"
        f"center=({item['center_x']:.1f},{item['center_y']:.1f}),"
        f"local_regressed={item['local_regressed']}"
        for item in components
    )


def class_gain_summary(gains):
    return "; ".join(f"{name}={gain:+.6f}" for name, gain in gains.items())


def write_csv(path, rows, rank_start=1):
    fields = (
        "rank",
        "global_rank",
        "scene_name",
        "sample_token",
        "primary_class",
        "score",
        "gt_parity_min_iou",
        "mean_baseline_iou",
        "mean_ours_iou",
        "mean_iou_gain",
        "min_per_class_iou_gain",
        "per_class_iou_gain",
        "rescued_positive",
        "removed_false_positive",
        "fixed_errors",
        "lost_positive",
        "new_false_positive",
        "regressed_errors",
        "fix_regress_ratio",
        "component_count",
        "ranked_component_count",
        "component_pixels",
        "components",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=rank_start):
            ratio = row["fix_regress_ratio"]
            writer.writerow(
                {
                    "rank": rank,
                    "global_rank": row["global_rank"],
                    "scene_name": row["scene_name"],
                    "sample_token": row["token"],
                    "primary_class": row["primary_class"],
                    "score": f"{row['score']:.6f}",
                    "gt_parity_min_iou": f"{row['gt_parity_min_iou']:.6f}",
                    "mean_baseline_iou": f"{row['mean_baseline_iou']:.6f}",
                    "mean_ours_iou": f"{row['mean_ours_iou']:.6f}",
                    "mean_iou_gain": f"{row['mean_iou_gain']:.6f}",
                    "min_per_class_iou_gain": (
                        f"{row['min_per_class_iou_gain']:.6f}"
                    ),
                    "per_class_iou_gain": class_gain_summary(
                        row["per_class_iou_gain"]
                    ),
                    "rescued_positive": row["rescued_positive"],
                    "removed_false_positive": row["removed_false_positive"],
                    "fixed_errors": row["fixed_errors"],
                    "lost_positive": row["lost_positive"],
                    "new_false_positive": row["new_false_positive"],
                    "regressed_errors": row["regressed_errors"],
                    "fix_regress_ratio": (
                        "inf" if math.isinf(ratio) else f"{ratio:.6f}"
                    ),
                    "component_count": row["component_count"],
                    "ranked_component_count": row["ranked_component_count"],
                    "component_pixels": row["component_pixels"],
                    "components": component_summary(row["components"]),
                }
            )


def masks_to_rgb(masks, render_classes):
    canvas = np.full((*masks.shape[-2:], 3), 240, dtype=np.uint8)
    render_set = set(render_classes)
    for class_name in DRAW_ORDER:
        if class_name not in render_set:
            continue
        class_index = MAP_CLASSES.index(class_name)
        canvas[masks[class_index]] = MAP_PALETTE[class_name]
    return canvas


def draw_map_panel(ax, masks, render_classes, xbound, ybound, title):
    rgb = masks_to_rgb(masks, render_classes)
    ax.imshow(
        rgb.transpose(1, 0, 2),
        origin="lower",
        interpolation="nearest",
        extent=(xbound[0], xbound[1], ybound[0], ybound[1]),
    )
    ax.plot(0.0, 0.0, marker="+", color="black", markersize=5, markeredgewidth=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def load_render_masks(row, baseline_paths, ours_paths, args):
    token = row["token"]
    baseline = load_map_sample(baseline_paths[token])
    ours = load_map_sample(ours_paths[token])
    baseline_thresholds = expand_thresholds(args.baseline_thresholds)
    ours_thresholds = expand_thresholds(args.ours_thresholds)
    return (
        ours["gt"],
        baseline["probs"] >= baseline_thresholds[:, None, None],
        ours["probs"] >= ours_thresholds[:, None, None],
        ours["xbound"],
        ours["ybound"],
    )


def render_triptych(path, row, baseline_paths, ours_paths, args):
    gt, baseline, ours, xbound, ybound = load_render_masks(
        row, baseline_paths, ours_paths, args
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), squeeze=False)
    for ax, masks, title in zip(
        axes[0],
        (gt, baseline, ours),
        ("Ground Truth", "BEVFusion", "Ours"),
    ):
        draw_map_panel(ax, masks, args.render_classes, xbound, ybound, title)
    if args.show_stats:
        fig.suptitle(
            f"{row['scene_name']} | mean IoU gain={row['mean_iou_gain']:+.3f} | "
            f"fixed={row['fixed_errors']} regressed={row['regressed_errors']}",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.5)
    else:
        fig.tight_layout(w_pad=0.5)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_montage(path, rows, baseline_paths, ours_paths, args):
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(11.2, 3.65 * len(rows)),
        squeeze=False,
    )
    for row_index, row in enumerate(rows):
        gt, baseline, ours, xbound, ybound = load_render_masks(
            row, baseline_paths, ours_paths, args
        )
        for column, (masks, title) in enumerate(
            zip((gt, baseline, ours), ("Ground Truth", "BEVFusion", "Ours"))
        ):
            panel_title = title if row_index == 0 else ""
            draw_map_panel(
                axes[row_index, column],
                masks,
                args.render_classes,
                xbound,
                ybound,
                panel_title,
            )
        row_label = row["scene_name"]
        if args.show_stats:
            row_label += (
                f"\nIoU {row['mean_iou_gain']:+.3f}; "
                f"fix/reg {row['fixed_errors']}/{row['regressed_errors']}"
            )
        axes[row_index, 0].set_ylabel(
            row_label,
            fontsize=9,
            rotation=90,
            labelpad=5,
        )
    fig.tight_layout(h_pad=0.35, w_pad=0.35)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_legend(path, render_classes, dpi):
    handles = [
        Patch(
            facecolor=np.asarray(MAP_PALETTE[name]) / 255.0,
            edgecolor="black",
            linewidth=0.3,
            label=DISPLAY_NAMES[name],
        )
        for name in DRAW_ORDER
        if name in set(render_classes)
    ]
    fig, ax = plt.subplots(figsize=(2.9 * len(handles), 0.75))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=10,
        handlelength=1.5,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    baseline_root = Path(args.baseline_dir).expanduser().resolve()
    ours_root = Path(args.ours_dir).expanduser().resolve()
    if baseline_root == ours_root:
        raise ValueError("--baseline-dir and --ours-dir must be different")

    out_dir = Path(args.out_dir).expanduser()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    baseline_paths = index_export(baseline_root)
    ours_paths = index_export(ours_root)
    print(
        f"Indexed BEVFusion={len(baseline_paths)} and Ours={len(ours_paths)} "
        "token-indexed map files"
    )
    candidates, compared = find_candidates(args, baseline_paths, ours_paths)
    write_csv(out_dir / "all_visible_map_candidates.csv", candidates)

    selected = select_candidates(candidates, args)
    if not selected:
        raise RuntimeError(
            "No map frame passed the requested filters. Try lowering "
            "--min-component-pixels or --min-fix-regress-ratio, or inspect "
            "all_visible_map_candidates.csv."
        )
    write_csv(
        out_dir / "selected_visible_map_candidates.csv",
        selected,
        rank_start=args.rank_start,
    )

    settings = vars(args).copy()
    settings.update(
        {
            "baseline_dir": str(baseline_root),
            "ours_dir": str(ours_root),
            "out_dir": str(out_dir.resolve()),
            "compared_samples": compared,
            "eligible_candidates": len(candidates),
            "selected_samples": len(selected),
            "axis_for_comparison": "protoocc_(class,x,y)",
            "gt_used_for_rendering": "ours",
            "threshold_policy": "fixed_global_per_model_per_class",
        }
    )
    with (out_dir / "selection_settings.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(settings, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for rank, row in enumerate(selected, start=args.rank_start):
        output = images_dir / (
            f"rank{rank:03d}_{row['scene_name']}_{row['token'][:8]}.png"
        )
        render_triptych(output, row, baseline_paths, ours_paths, args)
    render_montage(
        out_dir / "visible_map_improvement_comparison.png",
        selected,
        baseline_paths,
        ours_paths,
        args,
    )
    render_legend(out_dir / "map_class_legend.png", args.render_classes, args.dpi)

    print(f"Compared samples: {compared}")
    print(f"Eligible candidates: {len(candidates)}")
    print(f"Selected frames: {len(selected)}")
    print(f"Selected CSV: {out_dir / 'selected_visible_map_candidates.csv'}")
    print(f"Settings: {out_dir / 'selection_settings.json'}")
    print(f"Individual images: {images_dir}")
    print(f"Montage: {out_dir / 'visible_map_improvement_comparison.png'}")
    print(f"Legend: {out_dir / 'map_class_legend.png'}")


if __name__ == "__main__":
    main()
