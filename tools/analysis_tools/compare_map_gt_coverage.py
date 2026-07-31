#!/usr/bin/env python3
"""Compare nuScenes BEV map GT coverage on the 40 m and 50 m grids.

By default, this script follows the validation dataset's timestamp sorting and
aggregates every frame from its first scene.  ``--scene`` selects one complete
scene by name or token, while ``--all`` aggregates the entire validation set.

The reported denominator is the number of frame-cell observations per class:
``num_frames * 200 * 200``.  Map classes are independent binary channels, so
their percentages are not expected to sum to 100%.
"""

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmdet3d_plugin.datasets.pipelines.loading_bev_seg import (  # noqa: E402
    LoadBEVSegmentation,
)


MAP_CLASSES = (
    'drivable_area',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
    'divider',
)

DISPLAY_NAMES = (
    'Drivable area',
    'Pedestrian crossing',
    'Walkway',
    'Stop line',
    'Carpark area',
    'Divider',
)

GRID_SPECS = (
    ('40m', (-40.0, 40.0, 0.4)),
    ('50m', (-50.0, 50.0, 0.5)),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Count positive cells in the six nuScenes map GT channels for '
            '[-40, 40] at 0.4 m and [-50, 50] at 0.5 m.'))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        '--scene',
        help=(
            'Scene name (for example scene-0268), numeric suffix (0268), or '
            'scene token. Defaults to the first scene in the timestamp-sorted '
            'validation dataset.'))
    selection.add_argument(
        '--all',
        dest='all_scenes',
        action='store_true',
        help='Aggregate all scenes and frames in the validation annotation.')
    parser.add_argument(
        '--ann-file',
        default='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
        help='Validation annotation pkl, relative to the repository root.')
    parser.add_argument(
        '--data-root',
        default='data/nuscenes',
        help='nuScenes data root, relative to the repository root.')
    parser.add_argument(
        '--version',
        default=None,
        help=(
            'nuScenes version. Defaults to the version in annotation metadata '
            'or v1.0-trainval.'))
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=250,
        help='Write progress to stderr every N frames; 0 disables it.')
    args = parser.parse_args()
    if args.progress_interval < 0:
        parser.error('--progress-interval must be non-negative')
    return args


def resolve_repo_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_validation_infos(path):
    with path.open('rb') as handle:
        annotation = pickle.load(handle)

    if isinstance(annotation, dict):
        infos = annotation.get('infos')
        metadata = annotation.get('metadata', {})
    else:
        infos = annotation
        metadata = {}

    if not isinstance(infos, (list, tuple)) or not infos:
        raise ValueError(f'No validation infos found in {path}')

    missing = [
        key for key in ('timestamp', 'scene_token')
        if key not in infos[0]
    ]
    if missing:
        raise KeyError(f'Annotation infos are missing required keys: {missing}')

    # Match NuScenesDatasetBEVDet.load_annotations().
    infos = sorted(infos, key=lambda item: item['timestamp'])
    return infos, metadata


def normalize_scene_query(query):
    aliases = {query}
    if query.isdigit():
        aliases.add(f'scene-{int(query):04d}')
    return aliases


def select_infos(infos, scene_query, all_scenes):
    if all_scenes:
        return list(infos)

    if scene_query is None:
        target_token = infos[0]['scene_token']
        return [
            info for info in infos
            if info['scene_token'] == target_token
        ]

    aliases = normalize_scene_query(scene_query)
    selected = [
        info for info in infos
        if info['scene_token'] in aliases
        or info.get('scene_name') in aliases
    ]
    if selected:
        return selected

    examples = []
    seen = set()
    for info in infos:
        name = info.get('scene_name', info['scene_token'])
        if name not in seen:
            seen.add(name)
            examples.append(name)
        if len(examples) == 8:
            break
    raise ValueError(
        f'Scene {scene_query!r} is not in the validation annotation. '
        f'Examples: {", ".join(examples)}')


def build_scene_metadata(data_root, version):
    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    scene2location = {}
    scene2name = {}
    for scene in nusc.scene:
        log = nusc.get('log', scene['log_token'])
        scene2location[scene['token']] = log['location']
        scene2name[scene['token']] = scene['name']
    del nusc
    gc.collect()
    return scene2location, scene2name


def info_to_loader_input(info, location):
    return dict(
        curr=dict(
            lidar2ego_rotation=info['lidar2ego_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            ego2global_translation=info['ego2global_translation'],
        ),
        location=location,
        rotate_bda=0.0,
        scale_bda=1.0,
        flip_dx=False,
        flip_dy=False,
    )


def count_grid_positives(infos, scene2location, data_root, label, bound,
                         progress_interval):
    xbound = list(bound)
    ybound = list(bound)
    cells_x = int(round((xbound[1] - xbound[0]) / xbound[2]))
    cells_y = int(round((ybound[1] - ybound[0]) / ybound[2]))
    if (cells_x, cells_y) != (200, 200):
        raise ValueError(
            f'{label} grid must be 200 x 200, got {cells_x} x {cells_y}')

    # Processing one grid at a time avoids keeping two NuScenesMap instances
    # resident. Grouping by location also prevents repeatedly reloading maps
    # during a full-validation pass.
    records = []
    for info in infos:
        scene_token = info['scene_token']
        if scene_token not in scene2location:
            raise KeyError(f'Unknown nuScenes scene token: {scene_token}')
        records.append((scene2location[scene_token], info))
    records.sort(key=lambda item: item[0])

    loader = LoadBEVSegmentation(
        dataset_root=str(data_root),
        xbound=xbound,
        ybound=ybound,
        classes=MAP_CLASSES,
    )
    counts = np.zeros(len(MAP_CLASSES), dtype=np.int64)
    total = len(records)
    start = time.time()
    print(f'Rasterizing {label}: {total} frames ...', file=sys.stderr,
          flush=True)

    for index, (location, info) in enumerate(records, 1):
        results = info_to_loader_input(info, location)
        masks = loader(results)['gt_masks_bev']
        expected_shape = (len(MAP_CLASSES), cells_x, cells_y)
        if masks.shape != expected_shape:
            raise ValueError(
                f'Unexpected {label} mask shape for {info.get("token")}: '
                f'{masks.shape}, expected {expected_shape}')
        counts += masks.reshape(len(MAP_CLASSES), -1).sum(
            axis=1, dtype=np.int64)

        if progress_interval and (
                index % progress_interval == 0 or index == total):
            elapsed = time.time() - start
            print(
                f'  {label}: {index}/{total} frames, {elapsed:.1f}s',
                file=sys.stderr,
                flush=True,
            )

    del loader
    gc.collect()
    return counts, total * cells_x * cells_y


def format_percentage(value):
    return f'{value:.4f}'.rstrip('0').rstrip('.') or '0'


def print_summary(label, counts, total_cells):
    print(f'{label}:')
    for display_name, count in zip(DISPLAY_NAMES, counts):
        percentage = 100.0 * int(count) / total_cells
        print(
            f'{display_name}: {int(count)}\u683c, '
            f'{format_percentage(percentage)}%')


def main():
    args = parse_args()
    ann_file = resolve_repo_path(args.ann_file)
    data_root = resolve_repo_path(args.data_root)
    if not ann_file.is_file():
        raise FileNotFoundError(f'Validation annotation not found: {ann_file}')
    if not data_root.is_dir():
        raise NotADirectoryError(f'nuScenes data root not found: {data_root}')

    infos, metadata = load_validation_infos(ann_file)
    selected = select_infos(infos, args.scene, args.all_scenes)
    version = args.version or metadata.get('version', 'v1.0-trainval')
    scene2location, scene2name = build_scene_metadata(data_root, version)

    scene_tokens = list(dict.fromkeys(
        info['scene_token'] for info in selected))
    if args.all_scenes:
        selection_label = (
            f'all validation scenes ({len(scene_tokens)} scenes, '
            f'{len(selected)} frames)')
    else:
        token = scene_tokens[0]
        scene_name = selected[0].get('scene_name') or scene2name[token]
        selection_label = f'{scene_name} ({token}, {len(selected)} frames)'

    grid_results = []
    for label, bound in GRID_SPECS:
        counts, total_cells = count_grid_positives(
            selected,
            scene2location,
            data_root,
            label,
            bound,
            args.progress_interval,
        )
        grid_results.append((label, counts, total_cells))

    print(f'Selection: {selection_label}')
    print(
        'Each percentage uses its class channel over '
        f'{len(selected)} x 200 x 200 = {len(selected) * 200 * 200} cells.')
    print('Classes are multi-label channels; percentages need not sum to 100%.')
    print()
    for index, (label, counts, total_cells) in enumerate(grid_results):
        if index:
            print()
        print_summary(label, counts, total_cells)


if __name__ == '__main__':
    main()
