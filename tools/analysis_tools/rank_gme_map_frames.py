#!/usr/bin/env python3
"""Rank per-frame GME changes for sparse/thin BEV map classes.

This tool compares two ProtoOcc ``map.npz`` export trees produced with
``map_show_dir``.  Map segmentation is multi-label, so pedestrian crossing,
stop line, and divider are evaluated independently (never with argmax).

The ranking is intended for qualitative case discovery.  In addition to
per-frame IoU/precision/recall, it measures tolerant thin-structure coverage,
GT-component recovery, fragmentation, and accidental merging.  The latter
metrics help reject examples where GME merely makes a thin prediction thicker
or joins neighboring structures.

For a defensible ablation figure, keep the same threshold for both models.  The
default is a shared threshold of 0.5, matching ``visualize_map_sample.py``.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage


MAP_CLASSES = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)
TARGET_CLASSES = ("ped_crossing", "stop_line", "divider")
DISPLAY_NAMES = {
    "ped_crossing": "Pedestrian Crossing",
    "stop_line": "Stop Line",
    "divider": "Divider",
}
DEFAULT_BOUND = (-40.0, 40.0, 0.4)
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


CSV_FIELDS = (
    "class_rank",
    "eligible",
    "eligibility_reason",
    "scene_name",
    "sample_token",
    "class_name",
    "display_name",
    "without_threshold",
    "with_threshold",
    "ranking_score",
    "gt_pixels",
    "without_pred_pixels",
    "with_pred_pixels",
    "without_iou",
    "with_iou",
    "iou_gain",
    "without_precision",
    "with_precision",
    "precision_gain",
    "without_recall",
    "with_recall",
    "recall_gain",
    "without_f1",
    "with_f1",
    "f1_gain",
    "without_tolerant_precision",
    "with_tolerant_precision",
    "tolerant_precision_gain",
    "without_tolerant_recall",
    "with_tolerant_recall",
    "tolerant_recall_gain",
    "without_tolerant_f1",
    "with_tolerant_f1",
    "tolerant_f1_gain",
    "rescued_gt_pixels",
    "lost_gt_pixels",
    "removed_false_positive_pixels",
    "new_false_positive_pixels",
    "fixed_errors",
    "regressed_errors",
    "net_fixed_errors",
    "gt_component_count",
    "without_component_recall_mean",
    "with_component_recall_mean",
    "component_recall_gain",
    "without_recovered_components",
    "with_recovered_components",
    "recovered_components_gain",
    "without_fragmentation_surplus",
    "with_fragmentation_surplus",
    "continuity_gain",
    "without_merge_surplus",
    "with_merge_surplus",
    "separation_gain",
    "topology_gain",
    "without_isolated_pred_components",
    "with_isolated_pred_components",
    "without_near_gt_thickness_ratio",
    "with_near_gt_thickness_ratio",
    "thickness_fidelity_gain",
    "largest_fixed_component_pixels",
    "change_center_x_index",
    "change_center_y_index",
    "change_center_x_m",
    "change_center_y_m",
    "dominant_improvement",
    "gt_parity_iou",
    "without_npz",
    "with_npz",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--without-dir",
        required=True,
        help="W/o GME export root containing <scene>/<token>/map.npz.",
    )
    parser.add_argument(
        "--with-dir",
        required=True,
        help="W/ GME export root containing <scene>/<token>/map.npz.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/qualitative_map/gme_ablation/ranked",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=MAP_CLASSES,
        default=list(TARGET_CLASSES),
        help="Classes ranked independently (default: the GME sparse/thin group).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Shared threshold for both models; recommended for qualitative use.",
    )
    parser.add_argument(
        "--without-thresholds",
        type=float,
        nargs="+",
        help="Optional one or six W/o GME thresholds in MAP_CLASSES order.",
    )
    parser.add_argument(
        "--with-thresholds",
        type=float,
        nargs="+",
        help="Optional one or six W/ GME thresholds in MAP_CLASSES order.",
    )
    parser.add_argument(
        "--tolerance-pixels",
        type=int,
        default=2,
        help="Alignment tolerance in BEV pixels (2 pixels = 0.8 m here).",
    )
    parser.add_argument("--min-gt-pixels", type=int, default=8)
    parser.add_argument("--min-gt-component-pixels", type=int, default=4)
    parser.add_argument("--min-pred-component-pixels", type=int, default=2)
    parser.add_argument(
        "--component-recall-threshold",
        type=float,
        default=0.5,
        help="Tolerant GT-component recall needed to count it as recovered.",
    )
    parser.add_argument(
        "--min-iou-gain",
        type=float,
        default=0.0,
        help="Minimum per-frame IoU gain for a top-candidate row.",
    )
    parser.add_argument(
        "--max-precision-drop",
        type=float,
        default=0.03,
        help="Largest allowed precision drop for a top-candidate row.",
    )
    parser.add_argument(
        "--top-k-per-class",
        type=int,
        default=30,
        help="Number of eligible rows written to each per-class top CSV.",
    )
    parser.add_argument(
        "--unique-scenes",
        action="store_true",
        help="Keep at most one top candidate from each scene per class.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Compare only common tokens when the two exports are incomplete.",
    )
    parser.add_argument(
        "--min-gt-parity",
        type=float,
        default=1.0,
        help="Minimum GT IoU between paired NPZ files for every ranked class.",
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
        help="Accept matching geometry other than the native [-40,40] @ 0.4 m grid.",
    )
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Debug only: compare at most this many common tokens (0 means all).",
    )
    args = parser.parse_args()

    if len(args.classes) != len(set(args.classes)):
        parser.error("--classes contains duplicates")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    for option in ("without_thresholds", "with_thresholds"):
        values = getattr(args, option)
        if values is None:
            continue
        if len(values) not in (1, len(MAP_CLASSES)):
            parser.error(
                "--{} needs one value or {} values".format(
                    option.replace("_", "-"), len(MAP_CLASSES)
                )
            )
        if any(value < 0.0 or value > 1.0 for value in values):
            parser.error("--{} must be in [0, 1]".format(option.replace("_", "-")))
    if args.tolerance_pixels < 0:
        parser.error("--tolerance-pixels must be non-negative")
    if args.min_gt_pixels < 1:
        parser.error("--min-gt-pixels must be positive")
    if args.min_gt_component_pixels < 1 or args.min_pred_component_pixels < 1:
        parser.error("component pixel limits must be positive")
    if not 0.0 <= args.component_recall_threshold <= 1.0:
        parser.error("--component-recall-threshold must be in [0, 1]")
    if not -1.0 <= args.min_iou_gain <= 1.0:
        parser.error("--min-iou-gain must be in [-1, 1]")
    if not 0.0 <= args.max_precision_drop <= 1.0:
        parser.error("--max-precision-drop must be in [0, 1]")
    if args.top_k_per_class < 1:
        parser.error("--top-k-per-class must be positive")
    if not 0.0 <= args.min_gt_parity <= 1.0:
        parser.error("--min-gt-parity must be in [0, 1]")
    if args.progress_interval < 0 or args.max_samples < 0:
        parser.error("progress interval and max samples must be non-negative")
    return args


def scalar_text(value, key, path):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("{}: {} must contain one string".format(path, key))
    return str(array.reshape(-1)[0])


def index_export(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Map export directory does not exist: {}".format(root))
    paths = {}
    for path in root.rglob("map.npz"):
        token = path.parent.name
        if token in paths:
            raise ValueError(
                "Duplicate sample token {}: {} and {}".format(
                    token, paths[token], path
                )
            )
        paths[token] = path
    if not paths:
        raise FileNotFoundError("No <scene>/<token>/map.npz files under {}".format(root))
    return root, paths


def convert_to_protoocc_axis(array, convention, path):
    if convention == "protoocc":
        return np.asarray(array).copy()
    if convention == "bevfusion":
        # Inverse of ProtoOcc -> BEVFusion display conversion:
        # p.transpose(0, 2, 1)[:, ::-1, :]
        return np.asarray(array).transpose(0, 2, 1)[:, :, ::-1].copy()
    raise ValueError(
        "{}: unknown axis_convention={!r}; expected protoocc or bevfusion".format(
            path, convention
        )
    )


def load_map_sample(path):
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
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise KeyError("{}: missing NPZ keys {}".format(path, sorted(missing)))
        probs = np.asarray(data["probs"], dtype=np.float32)
        gt = np.asarray(data["gt"], dtype=np.uint8)
        classes = tuple(str(value) for value in np.asarray(data["classes"]).tolist())
        xbound = np.asarray(data["xbound"], dtype=np.float64)
        ybound = np.asarray(data["ybound"], dtype=np.float64)
        token = scalar_text(data["sample_token"], "sample_token", path)
        scene_name = scalar_text(data["scene_name"], "scene_name", path)
        convention = scalar_text(data["axis_convention"], "axis_convention", path)

    if token != path.parent.name:
        raise ValueError(
            "{}: metadata token {} != directory {}".format(path, token, path.parent.name)
        )
    if probs.ndim != 3 or gt.ndim != 3 or probs.shape != gt.shape:
        raise ValueError(
            "{}: expected matching (C,H,W), got probs={} gt={}".format(
                path, probs.shape, gt.shape
            )
        )
    if probs.shape[0] != len(classes) or len(classes) != len(set(classes)):
        raise ValueError("{}: invalid class metadata {}".format(path, classes))
    if not np.isfinite(probs).all() or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("{}: probabilities must be finite and in [0, 1]".format(path))
    if xbound.shape != (3,) or ybound.shape != (3,):
        raise ValueError("{}: xbound/ybound must each contain 3 values".format(path))
    missing_classes = set(MAP_CLASSES) - set(classes)
    if missing_classes:
        raise ValueError("{}: missing map classes {}".format(path, sorted(missing_classes)))

    probs = convert_to_protoocc_axis(probs, convention, path)
    gt = convert_to_protoocc_axis(gt, convention, path).astype(bool, copy=False)
    order = [classes.index(name) for name in MAP_CLASSES]
    probs = probs[order]
    gt = gt[order]

    cells_x = int(round((xbound[1] - xbound[0]) / xbound[2]))
    cells_y = int(round((ybound[1] - ybound[0]) / ybound[2]))
    expected_shape = (len(MAP_CLASSES), cells_x, cells_y)
    if probs.shape != expected_shape:
        raise ValueError(
            "{}: geometry implies {}, got {}".format(path, expected_shape, probs.shape)
        )
    return {
        "probs": probs,
        "gt": gt,
        "token": token,
        "scene_name": scene_name,
        "xbound": xbound,
        "ybound": ybound,
    }


def expand_thresholds(values, shared):
    if values is None:
        return np.full(len(MAP_CLASSES), shared, dtype=np.float32)
    if len(values) == 1:
        return np.full(len(MAP_CLASSES), values[0], dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def safe_ratio(numerator, denominator, empty_value=0.0):
    if denominator == 0:
        return float(empty_value)
    return float(numerator / denominator)


def f1_score(precision, recall):
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def binary_iou(first, second):
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def dilate(mask, iterations):
    if iterations <= 0:
        return mask
    return ndimage.binary_dilation(
        mask, structure=CONNECTIVITY, iterations=iterations
    )


def segmentation_metrics(gt, pred, tolerance):
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    precision = safe_ratio(tp, tp + fp, empty_value=1.0 if not gt.any() else 0.0)
    recall = safe_ratio(tp, tp + fn, empty_value=1.0)
    iou = safe_ratio(tp, tp + fp + fn, empty_value=1.0)

    gt_dilated = dilate(gt, tolerance)
    pred_dilated = dilate(pred, tolerance)
    tolerant_tp_pred = int((pred & gt_dilated).sum())
    tolerant_tp_gt = int((gt & pred_dilated).sum())
    tolerant_precision = safe_ratio(
        tolerant_tp_pred,
        int(pred.sum()),
        empty_value=1.0 if not gt.any() else 0.0,
    )
    tolerant_recall = safe_ratio(
        tolerant_tp_gt, int(gt.sum()), empty_value=1.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_pixels": int(pred.sum()),
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "tolerant_precision": tolerant_precision,
        "tolerant_recall": tolerant_recall,
        "tolerant_f1": f1_score(tolerant_precision, tolerant_recall),
    }


def valid_component_ids(labels, count, min_pixels):
    if count == 0:
        return []
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    return [index for index in range(1, count + 1) if sizes[index] >= min_pixels]


def component_metrics(
    gt,
    pred,
    tolerance,
    min_gt_pixels,
    min_pred_pixels,
    recovery_threshold,
):
    gt_labels, gt_count = ndimage.label(gt, structure=CONNECTIVITY)
    pred_labels, pred_count = ndimage.label(pred, structure=CONNECTIVITY)
    gt_ids = valid_component_ids(gt_labels, gt_count, min_gt_pixels)
    pred_ids = valid_component_ids(pred_labels, pred_count, min_pred_pixels)
    valid_pred = np.isin(pred_labels, pred_ids) if pred_ids else np.zeros_like(pred)
    valid_pred_labels = np.where(valid_pred, pred_labels, 0)
    valid_gt = np.isin(gt_labels, gt_ids) if gt_ids else np.zeros_like(gt)
    valid_gt_labels = np.where(valid_gt, gt_labels, 0)
    valid_pred_dilated = dilate(valid_pred, tolerance)

    recalls = []
    fragmentation_surplus = 0
    recovered = 0
    for component_id in gt_ids:
        component = gt_labels == component_id
        pixels = int(component.sum())
        recall = safe_ratio(int((component & valid_pred_dilated).sum()), pixels)
        recalls.append(recall)
        if recall >= recovery_threshold:
            recovered += 1

        search_region = dilate(component, tolerance)
        overlapping_pred_ids = np.unique(valid_pred_labels[search_region])
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]
        fragmentation_surplus += max(int(len(overlapping_pred_ids)) - 1, 0)

    merge_surplus = 0
    isolated_pred_components = 0
    valid_gt_dilated = dilate(valid_gt, tolerance)
    for component_id in pred_ids:
        component = pred_labels == component_id
        # Use the same positional tolerance as thin-structure recall. This also
        # catches a slightly offset fuzzy bridge that runs between two GT
        # components without landing on both centerlines exactly.
        search_region = dilate(component, tolerance)
        overlapping_gt_ids = np.unique(valid_gt_labels[search_region])
        overlapping_gt_ids = overlapping_gt_ids[overlapping_gt_ids > 0]
        merge_surplus += max(int(len(overlapping_gt_ids)) - 1, 0)
        if not (component & valid_gt_dilated).any():
            isolated_pred_components += 1

    near_gt_pred_pixels = int((valid_pred & valid_gt_dilated).sum())
    valid_gt_pixels = int(valid_gt.sum())
    thickness_ratio = safe_ratio(
        near_gt_pred_pixels, valid_gt_pixels, empty_value=0.0
    )
    thickness_error = min(abs(math.log(max(thickness_ratio, 1e-6))), 3.0)
    return {
        "gt_component_count": len(gt_ids),
        "component_recall_mean": float(np.mean(recalls)) if recalls else 0.0,
        "recovered_components": recovered,
        "fragmentation_surplus": fragmentation_surplus,
        "merge_surplus": merge_surplus,
        "isolated_pred_components": isolated_pred_components,
        "near_gt_thickness_ratio": thickness_ratio,
        "thickness_error": thickness_error,
    }


def largest_fixed_region(gt, without_pred, with_pred, xbound, ybound):
    rescued = gt & with_pred & ~without_pred
    removed_fp = ~gt & without_pred & ~with_pred
    fixed = rescued | removed_fp
    labels, count = ndimage.label(fixed, structure=CONNECTIVITY)
    if count == 0:
        return {
            "largest_fixed_component_pixels": 0,
            "change_center_x_index": math.nan,
            "change_center_y_index": math.nan,
            "change_center_x_m": math.nan,
            "change_center_y_m": math.nan,
        }
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    component_id = int(np.argmax(sizes[1:]) + 1)
    component = labels == component_id
    center = np.argwhere(component).mean(axis=0)
    center_x_index = float(center[0])
    center_y_index = float(center[1])
    center_x_m = float(xbound[0] + (center_x_index + 0.5) * xbound[2])
    center_y_m = float(ybound[0] + (center_y_index + 0.5) * ybound[2])
    return {
        "largest_fixed_component_pixels": int(sizes[component_id]),
        "change_center_x_index": center_x_index,
        "change_center_y_index": center_y_index,
        "change_center_x_m": center_x_m,
        "change_center_y_m": center_y_m,
    }


def validate_pair(without, with_gme, args, without_path, with_path):
    token = with_gme["token"]
    if without["token"] != token:
        raise ValueError("Token mismatch: {} vs {}".format(without_path, with_path))
    if without["scene_name"] != with_gme["scene_name"]:
        raise ValueError(
            "Scene mismatch for {}: {} vs {}".format(
                token, without["scene_name"], with_gme["scene_name"]
            )
        )
    for key in ("xbound", "ybound"):
        if not np.allclose(without[key], with_gme[key], rtol=0.0, atol=1e-6):
            raise ValueError(
                "Geometry mismatch for {} ({}): {} vs {}".format(
                    token, key, without[key].tolist(), with_gme[key].tolist()
                )
            )
        if not args.allow_any_geometry:
            expected = np.asarray(getattr(args, "expected_" + key), dtype=np.float64)
            if not np.allclose(with_gme[key], expected, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "Unexpected {} for {}: got {}, expected {}. "
                    "Use --allow-any-geometry only if intentional.".format(
                        key, token, with_gme[key].tolist(), expected.tolist()
                    )
                )

    parity = {}
    for class_name in args.classes:
        class_index = MAP_CLASSES.index(class_name)
        value = binary_iou(without["gt"][class_index], with_gme["gt"][class_index])
        parity[class_name] = value
        if value < args.min_gt_parity:
            raise ValueError(
                "GT mismatch for {} {}: IoU={:.6f} < {:.6f}".format(
                    token, class_name, value, args.min_gt_parity
                )
            )
    return parity


def eligibility(row, args):
    reasons = []
    if row["gt_pixels"] < args.min_gt_pixels:
        reasons.append("too_few_gt_pixels")
    if row["iou_gain"] <= args.min_iou_gain:
        reasons.append("no_iou_gain")
    if row["precision_gain"] < -args.max_precision_drop:
        reasons.append("precision_drop")
    if row["fixed_errors"] <= row["regressed_errors"]:
        reasons.append("fixes_not_greater_than_regressions")
    if (
        row["rescued_gt_pixels"] <= row["lost_gt_pixels"]
        and row["removed_false_positive_pixels"] <= row["new_false_positive_pixels"]
    ):
        reasons.append("no_net_recovery_or_cleanup")
    if row["with_pred_pixels"] == 0:
        reasons.append("empty_with_gme_prediction")
    return not reasons, "ok" if not reasons else ";".join(reasons)


def analyze_class(
    class_name,
    gt,
    without_pred,
    with_pred,
    parity,
    without_path,
    with_path,
    scene_name,
    token,
    xbound,
    ybound,
    args,
):
    without_stats = segmentation_metrics(gt, without_pred, args.tolerance_pixels)
    with_stats = segmentation_metrics(gt, with_pred, args.tolerance_pixels)
    without_components = component_metrics(
        gt,
        without_pred,
        args.tolerance_pixels,
        args.min_gt_component_pixels,
        args.min_pred_component_pixels,
        args.component_recall_threshold,
    )
    with_components = component_metrics(
        gt,
        with_pred,
        args.tolerance_pixels,
        args.min_gt_component_pixels,
        args.min_pred_component_pixels,
        args.component_recall_threshold,
    )

    rescued = int((gt & with_pred & ~without_pred).sum())
    lost = int((gt & without_pred & ~with_pred).sum())
    removed_fp = int((~gt & without_pred & ~with_pred).sum())
    new_fp = int((~gt & with_pred & ~without_pred).sum())
    fixed_errors = rescued + removed_fp
    regressed_errors = lost + new_fp
    net_fixed_errors = fixed_errors - regressed_errors

    gt_component_count = with_components["gt_component_count"]
    component_recall_gain = (
        with_components["component_recall_mean"]
        - without_components["component_recall_mean"]
    )
    recovered_components_gain = (
        with_components["recovered_components"]
        - without_components["recovered_components"]
    )
    component_denominator = max(gt_component_count, 1)
    recovered_fraction_gain = recovered_components_gain / component_denominator
    continuity_gain = (
        without_components["fragmentation_surplus"]
        - with_components["fragmentation_surplus"]
    ) / component_denominator
    separation_gain = (
        without_components["merge_surplus"] - with_components["merge_surplus"]
    ) / component_denominator
    topology_gain = continuity_gain + separation_gain
    thickness_fidelity_gain = (
        without_components["thickness_error"] - with_components["thickness_error"]
    )

    iou_gain = with_stats["iou"] - without_stats["iou"]
    precision_gain = with_stats["precision"] - without_stats["precision"]
    recall_gain = with_stats["recall"] - without_stats["recall"]
    f1_gain = with_stats["f1"] - without_stats["f1"]
    tolerant_precision_gain = (
        with_stats["tolerant_precision"] - without_stats["tolerant_precision"]
    )
    tolerant_recall_gain = (
        with_stats["tolerant_recall"] - without_stats["tolerant_recall"]
    )
    tolerant_f1_gain = (
        with_stats["tolerant_f1"] - without_stats["tolerant_f1"]
    )
    net_error_rate = net_fixed_errors / max(int(gt.sum()), 1)

    base_score = (
        4.0 * iou_gain
        + 2.0 * tolerant_f1_gain
        + 1.5 * component_recall_gain
        + 0.75 * recovered_fraction_gain
        + 0.50 * float(np.clip(topology_gain, -2.0, 2.0))
        + 0.25 * float(np.clip(thickness_fidelity_gain, -1.0, 1.0))
        + 0.25 * float(np.clip(net_error_rate, -1.0, 1.0))
    )
    # Prefer changes large enough to remain visible in a thesis figure while
    # retaining the normalized quality terms above.
    visibility_weight = 1.0 + 0.15 * math.log1p(int(gt.sum()))
    ranking_score = base_score * visibility_weight

    if rescued > lost and removed_fp > new_fp:
        dominant_improvement = "recovery_and_cleanup"
    elif rescued > lost:
        dominant_improvement = "recovery"
    elif removed_fp > new_fp:
        dominant_improvement = "cleanup"
    else:
        dominant_improvement = "none"

    row = {
        "class_rank": 0,
        "scene_name": scene_name,
        "sample_token": token,
        "class_name": class_name,
        "display_name": DISPLAY_NAMES.get(class_name, class_name),
        "ranking_score": ranking_score,
        "gt_pixels": int(gt.sum()),
        "without_pred_pixels": without_stats["pred_pixels"],
        "with_pred_pixels": with_stats["pred_pixels"],
        "without_iou": without_stats["iou"],
        "with_iou": with_stats["iou"],
        "iou_gain": iou_gain,
        "without_precision": without_stats["precision"],
        "with_precision": with_stats["precision"],
        "precision_gain": precision_gain,
        "without_recall": without_stats["recall"],
        "with_recall": with_stats["recall"],
        "recall_gain": recall_gain,
        "without_f1": without_stats["f1"],
        "with_f1": with_stats["f1"],
        "f1_gain": f1_gain,
        "without_tolerant_precision": without_stats["tolerant_precision"],
        "with_tolerant_precision": with_stats["tolerant_precision"],
        "tolerant_precision_gain": tolerant_precision_gain,
        "without_tolerant_recall": without_stats["tolerant_recall"],
        "with_tolerant_recall": with_stats["tolerant_recall"],
        "tolerant_recall_gain": tolerant_recall_gain,
        "without_tolerant_f1": without_stats["tolerant_f1"],
        "with_tolerant_f1": with_stats["tolerant_f1"],
        "tolerant_f1_gain": tolerant_f1_gain,
        "rescued_gt_pixels": rescued,
        "lost_gt_pixels": lost,
        "removed_false_positive_pixels": removed_fp,
        "new_false_positive_pixels": new_fp,
        "fixed_errors": fixed_errors,
        "regressed_errors": regressed_errors,
        "net_fixed_errors": net_fixed_errors,
        "gt_component_count": gt_component_count,
        "without_component_recall_mean": without_components[
            "component_recall_mean"
        ],
        "with_component_recall_mean": with_components["component_recall_mean"],
        "component_recall_gain": component_recall_gain,
        "without_recovered_components": without_components[
            "recovered_components"
        ],
        "with_recovered_components": with_components["recovered_components"],
        "recovered_components_gain": recovered_components_gain,
        "without_fragmentation_surplus": without_components[
            "fragmentation_surplus"
        ],
        "with_fragmentation_surplus": with_components["fragmentation_surplus"],
        "continuity_gain": continuity_gain,
        "without_merge_surplus": without_components["merge_surplus"],
        "with_merge_surplus": with_components["merge_surplus"],
        "separation_gain": separation_gain,
        "topology_gain": topology_gain,
        "without_isolated_pred_components": without_components[
            "isolated_pred_components"
        ],
        "with_isolated_pred_components": with_components[
            "isolated_pred_components"
        ],
        "without_near_gt_thickness_ratio": without_components[
            "near_gt_thickness_ratio"
        ],
        "with_near_gt_thickness_ratio": with_components[
            "near_gt_thickness_ratio"
        ],
        "thickness_fidelity_gain": thickness_fidelity_gain,
        "dominant_improvement": dominant_improvement,
        "gt_parity_iou": parity,
        "without_npz": str(without_path),
        "with_npz": str(with_path),
    }
    row.update(largest_fixed_region(gt, without_pred, with_pred, xbound, ybound))
    row["eligible"], row["eligibility_reason"] = eligibility(row, args)
    return row


def compare_exports(args, without_paths, with_paths):
    without_tokens = set(without_paths)
    with_tokens = set(with_paths)
    if without_tokens != with_tokens and not args.allow_partial:
        only_without = sorted(without_tokens - with_tokens)
        only_with = sorted(with_tokens - without_tokens)
        raise ValueError(
            "The exports do not contain identical token sets: "
            "without-only={} {}, with-only={} {}. Complete both exports or "
            "pass --allow-partial.".format(
                len(only_without),
                only_without[:3],
                len(only_with),
                only_with[:3],
            )
        )
    tokens = sorted(without_tokens & with_tokens)
    if args.max_samples:
        tokens = tokens[: args.max_samples]
    if not tokens:
        raise RuntimeError("The two exports have no common sample tokens")

    without_thresholds = expand_thresholds(args.without_thresholds, args.threshold)
    with_thresholds = expand_thresholds(args.with_thresholds, args.threshold)
    rows = []
    for position, token in enumerate(tokens, start=1):
        without = load_map_sample(without_paths[token])
        with_gme = load_map_sample(with_paths[token])
        parity = validate_pair(
            without,
            with_gme,
            args,
            without_paths[token],
            with_paths[token],
        )
        without_pred = without["probs"] >= without_thresholds[:, None, None]
        with_pred = with_gme["probs"] >= with_thresholds[:, None, None]
        for class_name in args.classes:
            class_index = MAP_CLASSES.index(class_name)
            row = analyze_class(
                class_name,
                with_gme["gt"][class_index],
                without_pred[class_index],
                with_pred[class_index],
                parity[class_name],
                without_paths[token],
                with_paths[token],
                with_gme["scene_name"],
                token,
                with_gme["xbound"],
                with_gme["ybound"],
                args,
            )
            row["without_threshold"] = float(without_thresholds[class_index])
            row["with_threshold"] = float(with_thresholds[class_index])
            rows.append(row)
        if args.progress_interval and position % args.progress_interval == 0:
            print(
                "Compared {}/{} samples ({} class rows)".format(
                    position, len(tokens), len(rows)
                ),
                flush=True,
            )

    for class_name in args.classes:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        class_rows.sort(
            key=lambda row: (
                row["ranking_score"],
                row["iou_gain"],
                row["tolerant_f1_gain"],
                row["gt_pixels"],
            ),
            reverse=True,
        )
        for rank, row in enumerate(class_rows, start=1):
            row["class_rank"] = rank
    rows.sort(key=lambda row: (args.classes.index(row["class_name"]), row["class_rank"]))
    return rows, len(tokens), without_thresholds, with_thresholds


def format_csv_value(value):
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return "{:.8f}".format(float(value))
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    return value


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row[key]) for key in CSV_FIELDS})


def select_top_rows(rows, class_name, args):
    candidates = [
        row for row in rows if row["class_name"] == class_name and row["eligible"]
    ]
    candidates.sort(
        key=lambda row: (
            row["ranking_score"],
            row["iou_gain"],
            row["tolerant_f1_gain"],
            row["gt_pixels"],
        ),
        reverse=True,
    )
    if not args.unique_scenes:
        return candidates[: args.top_k_per_class]
    selected = []
    scenes = set()
    for row in candidates:
        if row["scene_name"] in scenes:
            continue
        selected.append(row)
        scenes.add(row["scene_name"])
        if len(selected) == args.top_k_per_class:
            break
    return selected


def limit_rows(rows, args):
    """Apply the requested top-k and optional scene diversity to sorted rows."""
    if not args.unique_scenes:
        return rows[: args.top_k_per_class]
    selected = []
    scenes = set()
    for row in rows:
        if row["scene_name"] in scenes:
            continue
        selected.append(row)
        scenes.add(row["scene_name"])
        if len(selected) == args.top_k_per_class:
            break
    return selected


def write_morphology_rankings(out_dir, rows, args):
    """Write class-specific views matching the qualitative failure sought."""
    outputs = {}
    specifications = (
        (
            "ped_crossing",
            "top_ped_crossing_recovery.csv",
            lambda row: row["recall_gain"] > 0.0,
            lambda row: (
                row["recall_gain"],
                row["component_recall_gain"],
                row["iou_gain"],
                row["ranking_score"],
            ),
        ),
        (
            "stop_line",
            "top_stop_line_continuity.csv",
            lambda row: row["continuity_gain"] > 0.0 or row["recall_gain"] > 0.0,
            lambda row: (
                row["continuity_gain"],
                row["component_recall_gain"],
                row["recall_gain"],
                row["ranking_score"],
            ),
        ),
        (
            "divider",
            "top_divider_separation.csv",
            lambda row: row["separation_gain"] > 0.0,
            lambda row: (
                row["separation_gain"],
                row["continuity_gain"],
                row["precision_gain"],
                row["iou_gain"],
                row["ranking_score"],
            ),
        ),
    )
    for class_name, filename, predicate, sort_key in specifications:
        if class_name not in args.classes:
            continue
        candidates = [
            row
            for row in rows
            if row["class_name"] == class_name
            and row["eligible"]
            and predicate(row)
        ]
        candidates.sort(key=sort_key, reverse=True)
        candidates = limit_rows(candidates, args)
        write_csv(out_dir / filename, candidates)
        outputs[filename] = len(candidates)
    return outputs


def write_token_list(path, selected, class_names):
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(
            "# class\ttop_rank\tscene_name\tsample_token\t"
            "iou_gain\tprecision_gain\trecall_gain\tseparation_gain\n"
        )
        for class_name in class_names:
            for top_rank, row in enumerate(selected.get(class_name, ()), start=1):
                handle.write(
                    "{}\t{}\t{}\t{}\t{:+.6f}\t{:+.6f}\t{:+.6f}\t{:+.6f}\n".format(
                        class_name,
                        top_rank,
                        row["scene_name"],
                        row["sample_token"],
                        row["iou_gain"],
                        row["precision_gain"],
                        row["recall_gain"],
                        row["separation_gain"],
                    )
                )


def main():
    args = parse_args()
    without_root, without_paths = index_export(args.without_dir)
    with_root, with_paths = index_export(args.with_dir)
    if without_root == with_root:
        raise ValueError("--without-dir and --with-dir resolve to the same directory")

    print(
        "Indexed W/o GME={} and W/ GME={} map files".format(
            len(without_paths), len(with_paths)
        )
    )
    rows, compared, without_thresholds, with_thresholds = compare_exports(
        args, without_paths, with_paths
    )
    thresholds_match = bool(np.array_equal(without_thresholds, with_thresholds))
    if not thresholds_match:
        print(
            "WARNING: W/o and W/ GME thresholds differ. Do not use apparent "
            "separation as mechanism evidence without documenting this confound."
        )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "all_ranked_class_frames.csv", rows)

    selected = {}
    combined = []
    for class_name in args.classes:
        top_rows = select_top_rows(rows, class_name, args)
        selected[class_name] = top_rows
        combined.extend(top_rows)
        write_csv(out_dir / "top_{}.csv".format(class_name), top_rows)
    write_csv(out_dir / "top_candidates.csv", combined)
    morphology_outputs = write_morphology_rankings(out_dir, rows, args)
    write_token_list(out_dir / "selected_tokens.txt", selected, args.classes)

    settings = vars(args).copy()
    settings.update(
        {
            "without_dir": str(without_root),
            "with_dir": str(with_root),
            "out_dir": str(out_dir.resolve()),
            "compared_samples": compared,
            "class_rows": len(rows),
            "eligible_rows": sum(bool(row["eligible"]) for row in rows),
            "without_thresholds_expanded": without_thresholds.tolist(),
            "with_thresholds_expanded": with_thresholds.tolist(),
            "thresholds_match": thresholds_match,
            "threshold_policy": (
                "shared_fixed" if thresholds_match else "per_model_fixed_confounded"
            ),
            "axis_for_comparison": "protoocc_(class,x,y)",
            "ranking_note": (
                "Qualitative discovery only. Final examples require visual review; "
                "GME does not have instance/topology-specific supervision."
            ),
            "selected_per_class": {
                class_name: len(values) for class_name, values in selected.items()
            },
            "morphology_rankings": morphology_outputs,
        }
    )
    with (out_dir / "ranking_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Compared samples: {}".format(compared))
    print("Class rows: {}".format(len(rows)))
    print("Eligible rows: {}".format(settings["eligible_rows"]))
    for class_name in args.classes:
        print("Top {}: {}".format(class_name, len(selected[class_name])))
    for filename, count in morphology_outputs.items():
        print("Morphology ranking {}: {}".format(filename, count))
    print("All rows: {}".format(out_dir / "all_ranked_class_frames.csv"))
    print("Combined top candidates: {}".format(out_dir / "top_candidates.csv"))
    print("Tokens for visualization: {}".format(out_dir / "selected_tokens.txt"))
    print("Settings: {}".format(out_dir / "ranking_settings.json"))


if __name__ == "__main__":
    main()
