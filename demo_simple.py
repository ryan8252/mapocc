#!/usr/bin/env python3
"""Simple six-camera + OCC BEV + map BEV renderer for scene-0268."""

import pickle
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
INFO_PKL = ROOT / "data/nuscenes/bevdetv2-nuscenes_infos_demo0268.pkl"
OCC_DIR = ROOT / "work_dirs/qualitative_occ/ours_predictions"
MAP_DIR = ROOT / "work_dirs/qualitative_map/ours_map"
OUT_DIR = ROOT / "demo_out/frames_occ_map0268_simple"

FREE_LABEL = 17
MAP_THRESHOLD = 0.5
ARROW_LENGTH = 40
ARROW_THICKNESS = 13
ARROW_TIP_LENGTH = 0.35
CAMERAS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]

# RGB, occupancy class 0..16.
OCC_COLORS = np.array([
    [0, 0, 0], [112, 128, 144], [220, 20, 60], [255, 127, 80],
    [255, 158, 0], [233, 150, 70], [255, 61, 99], [0, 0, 230],
    [47, 79, 79], [255, 140, 0], [255, 99, 71], [0, 207, 191],
    [175, 0, 75], [75, 0, 75], [112, 180, 60],
    [222, 184, 135], [0, 175, 0],
], dtype=np.uint8)

# Map order: drivable, ped crossing, walkway, stop line, carpark, divider.
MAP_COLORS = np.array([
    [166, 206, 227], [251, 154, 153], [227, 26, 28],
    [253, 191, 111], [255, 127, 0], [106, 61, 154],
], dtype=np.uint8)
MAP_DRAW_ORDER = [0, 4, 2, 1, 3, 5]  # broad regions first, thin lines last


def bev_from_occ(pred, camera_mask):
    """Most frequent non-free class along Z; no x/y transform."""
    pred = np.where(camera_mask, pred, FREE_LABEL)
    counts = np.stack([
        (pred == class_id).sum(axis=2) for class_id in range(FREE_LABEL)
    ])
    bev = counts.argmax(axis=0).astype(np.uint8)
    bev[~(pred != FREE_LABEL).any(axis=2)] = FREE_LABEL
    return bev


def colorize_occ(bev):
    image = np.full((*bev.shape, 3), 255, dtype=np.uint8)
    occupied = bev != FREE_LABEL
    image[occupied] = OCC_COLORS[bev[occupied]][:, ::-1]  # RGB -> BGR
    return image


def colorize_map(probs):
    image = np.full((*probs.shape[1:], 3), 255, dtype=np.uint8)
    masks = probs >= MAP_THRESHOLD
    for class_id in MAP_DRAW_ORDER:
        image[masks[class_id]] = MAP_COLORS[class_id][::-1]
    return image


def put_text(image, text, position, scale, color):
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA,
    )


def put_camera_text(image, text, position, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        image, text, position, font, scale, (0, 0, 0), 5, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, position, font, scale, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_block_arrow(image, center):
    """Draw a filled arrow with a rectangular shaft and hard corners."""
    x, y = center
    end_x = x + ARROW_LENGTH
    head_length = max(1, round(ARROW_LENGTH * ARROW_TIP_LENGTH))
    head_base_x = end_x - head_length
    half_shaft = max(1, ARROW_THICKNESS // 2)
    half_head = max(half_shaft + 1, ARROW_THICKNESS)
    points = np.array([
        [x, y - half_shaft],
        [head_base_x, y - half_shaft],
        [head_base_x, y - half_head],
        [end_x, y],
        [head_base_x, y + half_head],
        [head_base_x, y + half_shaft],
        [x, y + half_shaft],
    ], dtype=np.int32)
    cv2.fillPoly(image, [points], (0, 0, 0), lineType=cv2.LINE_8)


def make_bev_panel(image, title):
    # Rotate the raw (x, y) BEV 90 degrees counterclockwise: +x points right.
    image = np.ascontiguousarray(np.rot90(image, k=1))
    panel = cv2.resize(image, (600, 600), interpolation=cv2.INTER_NEAREST)
    center = (300 - ARROW_LENGTH // 2, 300)
    draw_block_arrow(panel, center)
    put_text(panel, title, (12, 30), 0.75, (0, 0, 0))
    return panel


def make_camera_panel(info):
    tiles = []
    for camera in CAMERAS:
        image = cv2.imread(str(ROOT / info["cams"][camera]["data_path"]))
        image = cv2.resize(image, (400, 225))
        put_camera_text(image, camera, (10, 25), 0.58)
        tiles.append(image)
    top = np.concatenate(tiles[:3], axis=1)
    bottom = np.concatenate(tiles[3:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def main():
    with open(INFO_PKL, "rb") as handle:
        infos = pickle.load(handle)["infos"]
    infos.sort(key=lambda info: info["timestamp"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for index, info in enumerate(infos):
        scene, token = info["scene_name"], info["token"]
        occ_path = OCC_DIR / scene / token / "pred.npz"
        map_path = MAP_DIR / scene / token / "map.npz"
        if not occ_path.exists() or not map_path.exists():
            print("skip", scene, token)
            continue

        with np.load(occ_path) as data:
            occ_pred = data["pred"]
        with np.load(ROOT / info["occ_path"] / "labels.npz") as data:
            camera_mask = data["mask_camera"].astype(bool)
        with np.load(map_path) as data:
            map_probs = data["probs"]

        occ = make_bev_panel(
            colorize_occ(bev_from_occ(occ_pred, camera_mask)),
            "Occupancy Prediction",
        )
        map_bev = make_bev_panel(
            colorize_map(map_probs),
            "BEV Map Segmentation",
        )
        cameras = make_camera_panel(info)
        frame = np.concatenate(
            [cameras, np.concatenate([occ, map_bev], axis=1)], axis=0
        )
        cv2.imwrite(str(OUT_DIR / f"frame_{index:03d}.png"), frame)

        rendered += 1
        if index % 10 == 0:
            print("rendered", index)
    print("done", rendered)


if __name__ == "__main__":
    main()
