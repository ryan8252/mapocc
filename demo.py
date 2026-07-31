#!/usr/bin/env python3
"""Render six nuScenes cameras with OCC and BEV-map predictions.

Expected prediction layouts (produced by this repository's evaluators):

    <occ-dir>/<scene-name>/<sample-token>/pred.npz
    <map-dir>/<scene-name>/<sample-token>/map.npz

``pred.npz["pred"]`` is an ``(X, Y, Z)`` semantic occupancy volume.
``map.npz["probs"]`` is a multi-label ``(C, X, Y)`` probability tensor.

The BEV arrays are rendered in their native ProtoOcc axis convention without
transpose, flip, or rotation. Therefore image rows are x, image columns are y,
+x points down, and +y points right.
"""

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent

FREE_LABEL = 17
OCC_COLORS_RGB = np.asarray(
    [
        [0, 0, 0],        # 0: others
        [112, 128, 144],  # 1: barrier
        [220, 20, 60],    # 2: bicycle
        [255, 127, 80],   # 3: bus
        [255, 158, 0],    # 4: car
        [233, 150, 70],   # 5: construction vehicle
        [255, 61, 99],    # 6: motorcycle
        [0, 0, 230],      # 7: pedestrian
        [47, 79, 79],     # 8: traffic cone
        [255, 140, 0],    # 9: trailer
        [255, 99, 71],    # 10: truck
        [0, 207, 191],    # 11: driveable surface
        [175, 0, 75],     # 12: other flat
        [75, 0, 75],      # 13: sidewalk
        [112, 180, 60],   # 14: terrain
        [222, 184, 135],  # 15: manmade
        [0, 175, 0],      # 16: vegetation
    ],
    dtype=np.uint8,
)

DEFAULT_MAP_CLASSES = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)
MAP_COLORS_RGB = {
    "drivable_area": (166, 206, 227),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "divider": (106, 61, 154),
}
# Draw broad regions first so thin structures remain visible on overlaps.
MAP_DRAW_ORDER = (
    "drivable_area",
    "carpark_area",
    "walkway",
    "ped_crossing",
    "stop_line",
    "divider",
)

CAMERA_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
CAMERA_CELL_SIZE = (400, 225)  # width, height
BEV_SIZE = 600


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--info-pkl",
        type=Path,
        default=Path("data/nuscenes/bevdetv2-nuscenes_infos_demo0268.pkl"),
        help="nuScenes info pickle used to align cameras, scenes, and tokens.",
    )
    parser.add_argument(
        "--occ-dir",
        type=Path,
        default=Path("work_dirs/qualitative_occ/ours_predictions"),
        help="Root containing <scene>/<token>/pred.npz.",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("work_dirs/qualitative_map/ours_map"),
        help="Root containing <scene>/<token>/map.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("demo_out/frames_occ_map0268"),
        help="Directory in which rendered PNG frames and frames.csv are saved.",
    )
    parser.add_argument(
        "--scene-names",
        nargs="+",
        help="Optional scene filter, for example: --scene-names scene-0268.",
    )
    parser.add_argument(
        "--map-threshold",
        type=float,
        nargs="+",
        default=[0.5],
        help=(
            "One shared map threshold or one value per map class. "
            "Map channels are multi-label and thresholded independently."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of successfully rendered frames; 0 renders all.",
    )
    camera_mask = parser.add_mutually_exclusive_group()
    camera_mask.add_argument(
        "--use-camera-mask",
        dest="use_camera_mask",
        action="store_true",
        default=True,
        help="Hide OCC voxels outside the GT camera-visible region (default).",
    )
    camera_mask.add_argument(
        "--no-camera-mask",
        dest="use_camera_mask",
        action="store_false",
        help="Render OCC predictions without the GT camera visibility mask.",
    )
    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if not args.map_threshold:
        parser.error("--map-threshold needs at least one value")
    if any(value < 0.0 or value > 1.0 for value in args.map_threshold):
        parser.error("--map-threshold values must be in [0, 1]")
    return args


def resolve_repo_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_infos(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    if not isinstance(infos, (list, tuple)):
        raise TypeError(f"Expected a list of infos in {path}, got {type(infos)}")
    return list(infos)


def scene_name_from_info(info):
    scene_name = info.get("scene_name")
    if scene_name is not None:
        return str(scene_name)

    for part in Path(str(info.get("occ_path", ""))).parts:
        if part.startswith("scene-"):
            return part
    scene_token = info.get("scene_token")
    if scene_token is not None:
        return str(scene_token)
    raise KeyError(f"Cannot determine scene name for sample {info.get('token')}")


def decode_text(value):
    value = np.asarray(value)
    if value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_occ_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        if "pred" not in data:
            raise KeyError(f"{path} does not contain the 'pred' array")
        pred = np.asarray(data["pred"])

    if pred.ndim != 3:
        raise ValueError(f"Expected OCC shape (X,Y,Z), got {pred.shape} in {path}")
    if pred.size and (pred.min() < 0 or pred.max() > FREE_LABEL):
        raise ValueError(
            f"OCC labels must be in [0,{FREE_LABEL}], got "
            f"[{pred.min()},{pred.max()}] in {path}"
        )
    return pred.astype(np.uint8, copy=False)


def load_camera_mask(info, expected_shape):
    occ_path = resolve_repo_path(info["occ_path"])
    labels_path = occ_path / "labels.npz"
    with np.load(labels_path, allow_pickle=False) as data:
        if "mask_camera" not in data:
            raise KeyError(f"{labels_path} does not contain 'mask_camera'")
        mask = np.asarray(data["mask_camera"], dtype=bool)
    if mask.shape != expected_shape:
        raise ValueError(
            f"mask_camera shape {mask.shape} does not match OCC "
            f"shape {expected_shape} for {labels_path}"
        )
    return mask


def collapse_occ_to_bev(pred, camera_mask=None):
    """Choose the most frequent non-free class in every vertical column."""
    if camera_mask is not None:
        pred = np.where(camera_mask, pred, FREE_LABEL)

    class_counts = np.stack(
        [(pred == class_id).sum(axis=2) for class_id in range(FREE_LABEL)],
        axis=0,
    )
    has_occupied_voxel = (pred != FREE_LABEL).any(axis=2)
    winners = class_counts.argmax(axis=0).astype(np.uint8)
    bev = np.full(pred.shape[:2], FREE_LABEL, dtype=np.uint8)
    bev[has_occupied_voxel] = winners[has_occupied_voxel]
    return bev


def render_occ_bev(bev):
    image = np.full((*bev.shape, 3), 255, dtype=np.uint8)
    occupied = bev != FREE_LABEL
    image[occupied] = OCC_COLORS_RGB[bev[occupied]][:, ::-1]  # RGB to BGR
    return image


def load_map_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        if "probs" not in data:
            raise KeyError(f"{path} does not contain the 'probs' array")
        probs = np.asarray(data["probs"], dtype=np.float32)
        classes = (
            tuple(decode_text(item) for item in data["classes"])
            if "classes" in data
            else DEFAULT_MAP_CLASSES
        )
        axis_convention = (
            decode_text(data["axis_convention"])
            if "axis_convention" in data
            else "protoocc"
        )

    if probs.ndim != 3:
        raise ValueError(f"Expected map shape (C,X,Y), got {probs.shape} in {path}")
    if probs.shape[0] != len(classes):
        raise ValueError(
            f"Map has {probs.shape[0]} channels but {len(classes)} class names "
            f"in {path}"
        )
    if not np.isfinite(probs).all():
        raise ValueError(f"Map probabilities contain NaN/Inf in {path}")
    if probs.size and (probs.min() < -1e-6 or probs.max() > 1.0 + 1e-6):
        raise ValueError(
            f"Expected sigmoid map probabilities in [0,1], got "
            f"[{probs.min():.4f},{probs.max():.4f}] in {path}"
        )
    if axis_convention != "protoocc":
        raise ValueError(
            f"{path} uses axis_convention={axis_convention!r}; this renderer "
            "intentionally performs no axis conversion and expects 'protoocc'"
        )
    return probs, classes


def expand_map_thresholds(values, num_classes):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 1:
        return np.repeat(values, num_classes)
    if values.size != num_classes:
        raise ValueError(
            "Provide one --map-threshold value or exactly "
            f"{num_classes} values, got {values.size}"
        )
    return values


def render_map_bev(probs, classes, thresholds):
    masks = probs >= thresholds[:, None, None]
    image = np.full((*probs.shape[1:], 3), 255, dtype=np.uint8)
    class_to_index = {name: index for index, name in enumerate(classes)}

    for class_name in MAP_DRAW_ORDER:
        class_index = class_to_index.get(class_name)
        if class_index is None:
            continue
        color_rgb = MAP_COLORS_RGB[class_name]
        image[masks[class_index]] = color_rgb[::-1]  # RGB to BGR

    unknown_classes = set(classes) - set(MAP_COLORS_RGB)
    if unknown_classes:
        raise KeyError(
            "No visualization color for map classes: "
            + ", ".join(sorted(unknown_classes))
        )
    return image


def draw_text_with_outline(
    image,
    text,
    origin,
    scale=0.65,
    foreground=(255, 255, 255),
    outline=(0, 0, 0),
    thickness=1,
):
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        outline,
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        foreground,
        thickness,
        cv2.LINE_AA,
    )


def add_ego_arrow(image):
    """Draw a black ego arrow toward native +x (down in the raw x/y image)."""
    image = image.copy()
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    arrow_length = max(36, min(height, width) // 10)
    end = (center[0], center[1] + arrow_length)

    # A white halo keeps the required black arrow visible over dark classes.
    cv2.arrowedLine(
        image,
        center,
        end,
        (255, 255, 255),
        9,
        cv2.LINE_AA,
        tipLength=0.35,
    )
    cv2.arrowedLine(
        image,
        center,
        end,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
        tipLength=0.35,
    )
    draw_text_with_outline(
        image,
        "+x",
        (end[0] + 10, end[1] + 6),
        scale=0.6,
        foreground=(0, 0, 0),
        outline=(255, 255, 255),
        thickness=2,
    )
    return image


def make_bev_panel(image, title):
    panel = cv2.resize(
        image,
        (BEV_SIZE, BEV_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    panel = add_ego_arrow(panel)
    draw_text_with_outline(
        panel,
        title,
        (12, 30),
        scale=0.75,
        thickness=2,
    )
    return panel


def load_camera_image(data_path):
    path = resolve_repo_path(data_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read camera image: {path}")
    return image


def make_camera_panel(info):
    tiles = []
    for camera_name in CAMERA_ORDER:
        try:
            data_path = info["cams"][camera_name]["data_path"]
        except KeyError as error:
            raise KeyError(
                f"Missing {camera_name} metadata for sample {info.get('token')}"
            ) from error
        image = load_camera_image(data_path)
        image = cv2.resize(image, CAMERA_CELL_SIZE, interpolation=cv2.INTER_AREA)
        draw_text_with_outline(
            image,
            camera_name,
            (10, 25),
            scale=0.58,
            thickness=1,
        )
        tiles.append(image)

    top = np.concatenate(tiles[:3], axis=1)
    bottom = np.concatenate(tiles[3:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def compose_frame(camera_panel, occ_image, map_image, threshold_label):
    occ_panel = make_bev_panel(occ_image, "OCCUPANCY PREDICTION (BEV)")
    map_panel = make_bev_panel(
        map_image,
        f"MAP PREDICTION (BEV, THR={threshold_label})",
    )
    bev_row = np.concatenate([occ_panel, map_panel], axis=1)
    if camera_panel.shape[1] != bev_row.shape[1]:
        raise ValueError(
            f"Camera width {camera_panel.shape[1]} and BEV width "
            f"{bev_row.shape[1]} do not match"
        )
    return np.concatenate([camera_panel, bev_row], axis=0)


def format_threshold_label(thresholds):
    if np.allclose(thresholds, thresholds[0]):
        return f"{thresholds[0]:.2f}"
    return "PER-CLASS"


def output_path_for_frame(out_dir, scene_name, frame_index, multi_scene):
    scene_dir = out_dir / scene_name if multi_scene else out_dir
    scene_dir.mkdir(parents=True, exist_ok=True)
    return scene_dir / f"frame_{frame_index:03d}.png"


def main():
    args = parse_args()
    info_path = resolve_repo_path(args.info_pkl)
    occ_dir = resolve_repo_path(args.occ_dir)
    map_dir = resolve_repo_path(args.map_dir)
    out_dir = resolve_repo_path(args.out_dir)

    for path, description in (
        (info_path, "info pickle"),
        (occ_dir, "OCC prediction root"),
        (map_dir, "map prediction root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description} does not exist: {path}")

    infos = load_infos(info_path)
    requested_scenes = set(args.scene_names or ())
    indexed_infos = []
    for dataset_index, info in enumerate(infos):
        scene_name = scene_name_from_info(info)
        if requested_scenes and scene_name not in requested_scenes:
            continue
        indexed_infos.append((dataset_index, scene_name, info))
    indexed_infos.sort(
        key=lambda item: (item[1], int(item[2].get("timestamp", 0)))
    )
    if not indexed_infos:
        raise KeyError(
            f"No samples matched scenes {sorted(requested_scenes)} in {info_path}"
        )

    selected_scenes = {item[1] for item in indexed_infos}
    multi_scene = len(selected_scenes) > 1
    scene_frame_indices = defaultdict(int)
    rows = []
    skipped = 0

    for dataset_index, scene_name, info in indexed_infos:
        if args.max_frames and len(rows) >= args.max_frames:
            break

        sample_token = str(info["token"])
        occ_path = occ_dir / scene_name / sample_token / "pred.npz"
        map_path = map_dir / scene_name / sample_token / "map.npz"
        missing = [str(path) for path in (occ_path, map_path) if not path.is_file()]
        if missing:
            print("skip missing prediction:", ", ".join(missing))
            skipped += 1
            continue

        occ_pred = load_occ_prediction(occ_path)
        camera_mask = (
            load_camera_mask(info, occ_pred.shape)
            if args.use_camera_mask
            else None
        )
        occ_bev = collapse_occ_to_bev(occ_pred, camera_mask)
        occ_image = render_occ_bev(occ_bev)

        map_probs, map_classes = load_map_prediction(map_path)
        thresholds = expand_map_thresholds(
            args.map_threshold,
            len(map_classes),
        )
        map_image = render_map_bev(map_probs, map_classes, thresholds)

        if occ_bev.shape != map_probs.shape[1:]:
            raise ValueError(
                f"OCC BEV shape {occ_bev.shape} and map BEV shape "
                f"{map_probs.shape[1:]} differ for {scene_name}/{sample_token}"
            )

        camera_panel = make_camera_panel(info)
        frame = compose_frame(
            camera_panel,
            occ_image,
            map_image,
            format_threshold_label(thresholds),
        )

        frame_index = scene_frame_indices[scene_name]
        output_path = output_path_for_frame(
            out_dir,
            scene_name,
            frame_index,
            multi_scene,
        )
        if not cv2.imwrite(str(output_path), frame):
            raise IOError(f"Failed to save frame: {output_path}")
        scene_frame_indices[scene_name] += 1
        rows.append(
            {
                "scene_name": scene_name,
                "frame_index": frame_index,
                "dataset_index": dataset_index,
                "timestamp": info.get("timestamp", ""),
                "sample_token": sample_token,
                "image_path": str(output_path),
                "occ_path": str(occ_path),
                "map_path": str(map_path),
            }
        )

        if len(rows) == 1 or len(rows) % 10 == 0:
            print(
                f"rendered {len(rows)}: {scene_name}/{sample_token} "
                f"-> {output_path}"
            )

    if not rows:
        raise RuntimeError(
            "No frames were rendered. Check scene names and prediction roots."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "frames.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"done: rendered={len(rows)}, skipped={skipped}")
    print(f"frames: {out_dir}")
    print(f"index: {csv_path}")


if __name__ == "__main__":
    main()
