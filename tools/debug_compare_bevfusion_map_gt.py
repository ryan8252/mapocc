#!/usr/bin/env python
"""Compare ProtoOcc and BEVFusion BEV map GT raster conventions.

This is a protocol-audit utility, not a training script.  It rasterizes the
same nuScenes val samples with:

1. ProtoOcc's current LoadBEVSegmentation convention.
2. BEVFusion's original LoadBEVSegmentation convention.
3. BEVFusion's axis convention but ProtoOcc's drivable-area layer mapping.

It then applies identity / flip / transpose transforms to the ProtoOcc label and
reports aggregate per-class IoU against the BEVFusion-style references.
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion


DEFAULT_MAP_CLASSES = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)

LOCATIONS = (
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
)


def parse_bound(values: Sequence[float]) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("bounds must contain min max step")
    return float(values[0]), float(values[1]), float(values[2])


def load_infos(info_path: Path) -> List[dict]:
    with info_path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    if isinstance(data, list):
        return data
    raise TypeError(f"Unsupported info pkl format: {info_path}")


def quat_transform(rotation: Sequence[float], translation: Sequence[float]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = np.asarray(translation, dtype=np.float32)
    return mat


def lidar2global_from_info(info: dict) -> np.ndarray:
    lidar2ego = quat_transform(
        info["lidar2ego_rotation"], info["lidar2ego_translation"]
    )
    ego2global = quat_transform(
        info["ego2global_rotation"], info["ego2global_translation"]
    )
    return ego2global @ lidar2ego


def protoocc_layer_mapping(classes: Iterable[str]) -> Dict[str, List[str]]:
    mappings = {}
    for name in classes:
        if name in ("drivable_area", "drivable_area*"):
            mappings[name] = ["road_segment", "lane"]
        elif name == "divider":
            mappings[name] = ["road_divider", "lane_divider"]
        else:
            mappings[name] = [name]
    return mappings


def bevfusion_layer_mapping(classes: Iterable[str]) -> Dict[str, List[str]]:
    mappings = {}
    for name in classes:
        if name == "drivable_area*":
            mappings[name] = ["road_segment", "lane"]
        elif name == "divider":
            mappings[name] = ["road_divider", "lane_divider"]
        else:
            mappings[name] = [name]
    return mappings


def unique_preserve_order(names: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(names))


def rasterize_map(
    info: dict,
    nusc_map: NuScenesMap,
    classes: Sequence[str],
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
    mapping_mode: str,
    axis_mode: str,
) -> np.ndarray:
    patch_h = ybound[1] - ybound[0]
    patch_w = xbound[1] - xbound[0]
    canvas_h = int(round(patch_h / ybound[2]))
    canvas_w = int(round(patch_w / xbound[2]))
    if canvas_h != canvas_w:
        raise ValueError(
            "This audit intentionally mirrors the original loaders, whose "
            "boolean assignment assumes square map rasters."
        )

    lidar2global = lidar2global_from_info(info)
    map_pose = lidar2global[:2, 3]
    patch_box = (map_pose[0], map_pose[1], patch_h, patch_w)

    rotation = lidar2global[:3, :3]
    direction = rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    patch_angle = np.arctan2(direction[1], direction[0]) / np.pi * 180.0

    if mapping_mode == "protoocc":
        mappings = protoocc_layer_mapping(classes)
        layer_names = unique_preserve_order(
            layer for names in mappings.values() for layer in names
        )
    elif mapping_mode == "bevfusion":
        mappings = bevfusion_layer_mapping(classes)
        # Match the original BEVFusion code style.  The order is irrelevant for
        # correctness because labels index back into this same list by name.
        layer_names = list(set(layer for names in mappings.values() for layer in names))
    else:
        raise ValueError(f"unknown mapping_mode: {mapping_mode}")

    masks = nusc_map.get_map_mask(
        patch_box=patch_box,
        patch_angle=patch_angle,
        layer_names=layer_names,
        canvas_size=(canvas_h, canvas_w),
    )

    # Both current ProtoOcc and original BEVFusion transpose masks before label
    # assignment.
    masks = masks.transpose(0, 2, 1).astype(np.bool_)

    labels = np.zeros((len(classes), canvas_h, canvas_w), dtype=np.uint8)
    for class_index, class_name in enumerate(classes):
        for layer_name in mappings[class_name]:
            mask_index = layer_names.index(layer_name)
            labels[class_index, masks[mask_index]] = 1

    if axis_mode == "bevfusion":
        return np.ascontiguousarray(labels)
    if axis_mode == "protoocc":
        return np.ascontiguousarray(labels.transpose(0, 2, 1)[:, :, ::-1])
    raise ValueError(f"unknown axis_mode: {axis_mode}")


def candidate_transforms(label: np.ndarray) -> Dict[str, np.ndarray]:
    transposed = label.transpose(0, 2, 1)
    return {
        "identity": label,
        "flip_x": label[:, ::-1, :],
        "flip_y": label[:, :, ::-1],
        "flip_xy": label[:, ::-1, ::-1],
        "transpose_xy": transposed,
        "transpose_xy_flip_x": transposed[:, ::-1, :],
        "transpose_xy_flip_y": transposed[:, :, ::-1],
        "transpose_xy_flip_xy": transposed[:, ::-1, ::-1],
    }


def new_stats(num_classes: int) -> Dict[str, np.ndarray]:
    return {
        "intersection": np.zeros(num_classes, dtype=np.float64),
        "union": np.zeros(num_classes, dtype=np.float64),
    }


def update_iou_stats(stats: Dict[str, np.ndarray], pred: np.ndarray, target: np.ndarray) -> None:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    stats["intersection"] += np.logical_and(pred_bool, target_bool).sum(axis=(1, 2))
    stats["union"] += np.logical_or(pred_bool, target_bool).sum(axis=(1, 2))


def finalize_iou(stats: Dict[str, np.ndarray]) -> np.ndarray:
    intersection = stats["intersection"]
    union = stats["union"]
    return np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan, dtype=np.float64),
        where=union > 0,
    )


def format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.6f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_scene_location_lookup(data_root: Path, version: str) -> Dict[str, str]:
    from nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    lookup = {}
    for scene in nusc.scene:
        log = nusc.get("log", scene["log_token"])
        lookup[scene["token"]] = log["location"]
    return lookup


def resolve_location(
    info: dict,
    scene_location_lookup: Optional[Dict[str, str]],
) -> str:
    location = info.get("location")
    if location:
        return location
    if scene_location_lookup is None:
        raise KeyError(
            "Info pkl has no `location`. Provide a BEVFusion info pkl with "
            "locations or allow NuScenes scene lookup with --nusc-version."
        )
    scene_token = info["scene_token"]
    return scene_location_lookup[scene_token]


def default_info_path(repo_root: Path) -> Path:
    candidates = [
        repo_root.parent / "bevfusion/data/nuscenes/nuscenes_infos_val.pkl",
        repo_root / "data/nuscenes/nuscenes_infos_val.pkl",
        repo_root / "data/nuscenes/bevdetv2-nuscenes_infos_val.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=repo_root / "data/nuscenes")
    parser.add_argument("--info-pkl", type=Path, default=default_info_path(repo_root))
    parser.add_argument("--nusc-version", default="v1.0-trainval")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--xbound", nargs=3, type=float, default=(-50.0, 50.0, 0.5))
    parser.add_argument("--ybound", nargs=3, type=float, default=(-50.0, 50.0, 0.5))
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_MAP_CLASSES))
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "work_dirs/debug_bevfusion_map_gt_parity/result.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xbound = parse_bound(args.xbound)
    ybound = parse_bound(args.ybound)
    classes = tuple(args.classes)
    infos = load_infos(args.info_pkl)
    selected_infos = infos[args.start : args.start + args.num_samples]
    if not selected_infos:
        raise ValueError("No samples selected.")

    needs_location_lookup = any("location" not in info or not info["location"] for info in selected_infos)
    scene_location_lookup = (
        build_scene_location_lookup(args.data_root, args.nusc_version)
        if needs_location_lookup
        else None
    )

    maps = {location: NuScenesMap(str(args.data_root), location) for location in LOCATIONS}

    references = {
        "bevfusion_exact": "BEVFusion axes + BEVFusion drivable mapping",
        "bevfusion_axes_proto_layers": "BEVFusion axes + ProtoOcc drivable mapping",
    }
    transform_stats = {
        ref_name: {name: new_stats(len(classes)) for name in candidate_transforms(np.zeros((len(classes), 1, 1), dtype=np.uint8))}
        for ref_name in references
    }
    positive_counts = {
        "protoocc": np.zeros(len(classes), dtype=np.float64),
        "bevfusion_exact": np.zeros(len(classes), dtype=np.float64),
        "bevfusion_axes_proto_layers": np.zeros(len(classes), dtype=np.float64),
    }

    total_pixels = 0
    sample_lines = []

    for sample_idx, info in enumerate(selected_infos):
        location = resolve_location(info, scene_location_lookup)
        nusc_map = maps[location]

        proto_label = rasterize_map(
            info,
            nusc_map,
            classes,
            xbound,
            ybound,
            mapping_mode="protoocc",
            axis_mode="protoocc",
        )
        bevfusion_exact = rasterize_map(
            info,
            nusc_map,
            classes,
            xbound,
            ybound,
            mapping_mode="bevfusion",
            axis_mode="bevfusion",
        )
        bevfusion_axes_proto_layers = rasterize_map(
            info,
            nusc_map,
            classes,
            xbound,
            ybound,
            mapping_mode="protoocc",
            axis_mode="bevfusion",
        )

        labels = {
            "protoocc": proto_label,
            "bevfusion_exact": bevfusion_exact,
            "bevfusion_axes_proto_layers": bevfusion_axes_proto_layers,
        }
        for key, label in labels.items():
            positive_counts[key] += label.sum(axis=(1, 2))
        total_pixels += proto_label.shape[1] * proto_label.shape[2]

        transformed = candidate_transforms(proto_label)
        for transform_name, transformed_label in transformed.items():
            update_iou_stats(
                transform_stats["bevfusion_exact"][transform_name],
                transformed_label,
                bevfusion_exact,
            )
            update_iou_stats(
                transform_stats["bevfusion_axes_proto_layers"][transform_name],
                transformed_label,
                bevfusion_axes_proto_layers,
            )

        if sample_idx < 5:
            sample_lines.append(
                f"- {args.start + sample_idx}: token `{info.get('token')}`, "
                f"scene `{info.get('scene_name', info.get('scene_token'))}`, "
                f"location `{location}`"
            )

    report = []
    report.append("# BEVFusion Map GT Parity Audit")
    report.append("")
    report.append(f"- info pkl: `{args.info_pkl}`")
    report.append(f"- data root: `{args.data_root}`")
    report.append(f"- samples: `{len(selected_infos)}` starting at `{args.start}`")
    report.append(f"- xbound: `{list(xbound)}`")
    report.append(f"- ybound: `{list(ybound)}`")
    report.append(f"- classes: `{list(classes)}`")
    report.append("")
    report.append("## First Samples")
    report.append("")
    report.extend(sample_lines)
    report.append("")

    report.append("## Positive Pixel Ratios")
    report.append("")
    pos_rows = []
    denom = len(selected_infos) * total_pixels / len(selected_infos)
    for class_idx, class_name in enumerate(classes):
        row = [class_name]
        for key in ("protoocc", "bevfusion_exact", "bevfusion_axes_proto_layers"):
            row.append(format_float(positive_counts[key][class_idx] / denom))
        pos_rows.append(row)
    report.append(
        markdown_table(
            [
                "class",
                "protoocc",
                "bevfusion_exact",
                "bevfusion_axes_proto_layers",
            ],
            pos_rows,
        )
    )
    report.append("")

    for ref_name, description in references.items():
        report.append(f"## ProtoOcc Label Transforms vs `{ref_name}`")
        report.append("")
        report.append(description)
        report.append("")
        rows = []
        best_name = None
        best_mean = -1.0
        for transform_name, stats in transform_stats[ref_name].items():
            ious = finalize_iou(stats)
            mean_iou = np.nanmean(ious)
            if mean_iou > best_mean:
                best_mean = mean_iou
                best_name = transform_name
            rows.append(
                [transform_name, format_float(mean_iou)]
                + [format_float(value) for value in ious]
            )
        report.append(markdown_table(["transform", "mean"] + list(classes), rows))
        report.append("")
        report.append(f"Best transform: `{best_name}` with mean IoU `{best_mean:.6f}`")
        report.append("")

    report.append("## Interpretation Hints")
    report.append("")
    report.append(
        "- If `identity` is near 1.0 against `bevfusion_exact`, the current "
        "ProtoOcc loader already matches BEVFusion GT convention."
    )
    report.append(
        "- If a flip/transpose transform is near 1.0 but `identity` is low, "
        "there is an axis-convention mismatch."
    )
    report.append(
        "- If `bevfusion_axes_proto_layers` reaches near 1.0 but "
        "`bevfusion_exact` does not, axis is explainable but the drivable-area "
        "layer definition differs."
    )

    text = "\n".join(report) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"[WROTE] {args.out}")


if __name__ == "__main__":
    main()
