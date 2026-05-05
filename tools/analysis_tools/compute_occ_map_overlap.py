import argparse
import json
import sys
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmdet3d_plugin.datasets.pipelines.loading import (  # noqa: E402
    LoadOccGTFromFile,
)
from projects.mmdet3d_plugin.datasets.pipelines.loading_bev_seg import (  # noqa: E402
    LoadBEVSegmentation,
)


NUSC_OCC_CLASSES = (
    'others',
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
    'free',
)

OCC_CLASS_SETS = {
    'background': (11, 12, 13, 14, 15, 16),
    'non_free': tuple(range(17)),
    'all': tuple(range(18)),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute GT BEV map and low-Z occupancy overlap ratios.')
    parser.add_argument('config', help='Path to a ProtoOcc multitask config.')
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to read.')
    parser.add_argument(
        '--ann-file',
        default=None,
        help='Override annotation pkl. Defaults to cfg.data[split].ann_file.')
    parser.add_argument(
        '--z-layers',
        nargs='+',
        type=int,
        default=[0, 1, 2, 3],
        help='Occupancy Z indices to compare against the BEV map.')
    parser.add_argument(
        '--occ-class-set',
        default='background',
        choices=sorted(OCC_CLASS_SETS.keys()),
        help='Default occupancy classes to print in the report.')
    parser.add_argument(
        '--occ-classes',
        nargs='+',
        default=None,
        help='Optional occupancy class names or indices to print. Overrides '
        '--occ-class-set.')
    parser.add_argument(
        '--topk',
        type=int,
        default=None,
        help='Optional top-K entries to print per row. By default all selected '
        'classes are printed.')
    parser.add_argument(
        '--include-nonvisible',
        action='store_true',
        help='Do not set non-camera-visible occupancy voxels to ignore. The '
        'default matches training LoadOccGTFromFile(ignore_nonvisible=True).')
    parser.add_argument(
        '--sample-count',
        type=int,
        default=None,
        help='Evenly sample N frames across the split for a quick estimate.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Use the first N frames after optional sampling.')
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=1000,
        help='Print progress every N samples.')
    parser.add_argument(
        '--output-json',
        default=None,
        help='Optional JSON output path.')
    parser.add_argument(
        '--output-md',
        default=None,
        help='Optional Markdown output path.')
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def get_split_cfg(cfg, split):
    data_cfg = cfg.data[split]
    if isinstance(data_cfg, list):
        data_cfg = data_cfg[0]
    return data_cfg


def build_scene2location(data_root, version):
    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    scene2location = {}
    for scene in nusc.scene:
        log = nusc.get('log', scene['log_token'])
        scene2location[scene['token']] = log['location']
    return scene2location


def build_map_loader(cfg, data_root):
    return LoadBEVSegmentation(
        dataset_root=str(data_root),
        xbound=cfg.map_xbound,
        ybound=cfg.map_ybound,
        classes=cfg.map_classes)


def make_loader_input(info, scene2location):
    occ_path = Path(info['occ_path'])
    if not occ_path.is_absolute():
        occ_path = REPO_ROOT / occ_path

    return dict(
        curr=dict(
            lidar2ego_rotation=info['lidar2ego_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            ego2global_translation=info['ego2global_translation']),
        location=info.get('location') or scene2location[info['scene_token']],
        occ_gt_path=str(occ_path),
        rotate_bda=0.0,
        scale_bda=1.0,
        flip_dx=False,
        flip_dy=False)


def unwrap_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_occ_indices(args):
    if args.occ_classes is None:
        return list(OCC_CLASS_SETS[args.occ_class_set])

    selected = []
    for item in args.occ_classes:
        if item.isdigit():
            index = int(item)
        else:
            if item not in NUSC_OCC_CLASSES:
                raise ValueError(
                    f'Unknown occupancy class {item}. Expected one of '
                    f'{NUSC_OCC_CLASSES}.')
            index = NUSC_OCC_CLASSES.index(item)
        if index < 0 or index >= len(NUSC_OCC_CLASSES):
            raise ValueError(f'Occupancy class index out of range: {index}')
        selected.append(index)
    return selected


def update_counts(stats, occ_semantics, map_masks, z_layers):
    if occ_semantics.shape[:2] != map_masks.shape[1:]:
        raise ValueError(
            'OCC and map shapes do not align: '
            f'occ={occ_semantics.shape}, map={map_masks.shape}.')

    valid_all = occ_semantics != 255
    map_bool = map_masks.astype(bool)

    for layer_offset, z_index in enumerate(z_layers):
        occ_layer = occ_semantics[:, :, z_index]
        valid = valid_all[:, :, z_index]

        for map_index in range(stats['num_map_classes']):
            map_mask = map_bool[map_index] & valid
            denom = int(map_mask.sum())
            stats['map_valid_denoms'][layer_offset, map_index] += denom
            if denom == 0:
                continue
            occ_values = occ_layer[map_mask]
            counts = np.bincount(
                occ_values.astype(np.int64),
                minlength=stats['num_occ_classes'])
            stats['map_to_occ'][layer_offset, map_index] += counts

        valid_occ = occ_layer[valid]
        if valid_occ.size > 0:
            stats['occ_denoms'][layer_offset] += np.bincount(
                valid_occ.astype(np.int64),
                minlength=stats['num_occ_classes'])

        for occ_index in range(stats['num_occ_classes']):
            occ_mask = (occ_layer == occ_index) & valid
            if not occ_mask.any():
                continue
            for map_index in range(stats['num_map_classes']):
                stats['occ_to_map'][layer_offset, occ_index, map_index] += int(
                    (occ_mask & map_bool[map_index]).sum())


def percentage(value):
    return 100.0 * float(value)


def ranked_pairs(values, names, indices=None, topk=None):
    if indices is None:
        indices = range(len(names))
    pairs = [(index, names[index], float(values[index])) for index in indices]
    pairs.sort(key=lambda item: item[2], reverse=True)
    if topk is not None:
        pairs = pairs[:topk]
    return pairs


def build_layer_summary(stats, map_classes):
    summaries = {}
    layer_keys = [f'z={z}' for z in stats['z_layers']]
    layer_indices = list(range(len(stats['z_layers'])))
    layer_specs = list(zip(layer_keys, layer_indices))
    layer_specs.append((stats['aggregate_layer_key'], None))

    for key, layer_index in layer_specs:
        if layer_index is None:
            map_to_occ_counts = stats['map_to_occ'].sum(axis=0)
            map_denoms = stats['map_valid_denoms'].sum(axis=0)
            occ_to_map_counts = stats['occ_to_map'].sum(axis=0)
            occ_denoms = stats['occ_denoms'].sum(axis=0)
        else:
            map_to_occ_counts = stats['map_to_occ'][layer_index]
            map_denoms = stats['map_valid_denoms'][layer_index]
            occ_to_map_counts = stats['occ_to_map'][layer_index]
            occ_denoms = stats['occ_denoms'][layer_index]

        map_to_occ = {}
        for map_index, map_name in enumerate(map_classes):
            denom = int(map_denoms[map_index])
            if denom == 0:
                ratios = np.zeros(stats['num_occ_classes'], dtype=np.float64)
            else:
                ratios = map_to_occ_counts[map_index] / float(denom)
            map_to_occ[map_name] = dict(
                denom=denom,
                ratios={
                    name: float(ratios[index])
                    for index, name in enumerate(NUSC_OCC_CLASSES)
                })

        occ_to_map = {}
        for occ_index, occ_name in enumerate(NUSC_OCC_CLASSES):
            denom = int(occ_denoms[occ_index])
            if denom == 0:
                ratios = np.zeros(len(map_classes), dtype=np.float64)
            else:
                ratios = occ_to_map_counts[occ_index] / float(denom)
            occ_to_map[occ_name] = dict(
                denom=denom,
                ratios={
                    name: float(ratios[index])
                    for index, name in enumerate(map_classes)
                })

        summaries[key] = dict(map_to_occ=map_to_occ, occ_to_map=occ_to_map)

    return summaries


def format_map_to_occ(summary, map_classes, occ_indices, topk=None):
    lines = ['### Map -> OCC', '']
    lines.append('| Map class | valid map pixels | OCC class ratios |')
    lines.append('| --- | ---: | --- |')
    for map_name in map_classes:
        row = summary['map_to_occ'][map_name]
        ratios = [row['ratios'][name] for name in NUSC_OCC_CLASSES]
        pairs = ranked_pairs(ratios, NUSC_OCC_CLASSES, occ_indices, topk)
        ratio_text = ', '.join(
            f'{name}: {percentage(value):.2f}%' for _, name, value in pairs)
        lines.append(f'| `{map_name}` | {row["denom"]} | {ratio_text} |')
    return lines


def format_occ_to_map(summary, map_classes, occ_indices, topk=None):
    lines = ['### OCC -> Map', '']
    lines.append('| OCC class | valid voxels | Map coverage ratios |')
    lines.append('| --- | ---: | --- |')
    for occ_index in occ_indices:
        occ_name = NUSC_OCC_CLASSES[occ_index]
        row = summary['occ_to_map'][occ_name]
        ratios = [row['ratios'][name] for name in map_classes]
        pairs = ranked_pairs(ratios, map_classes, None, topk)
        ratio_text = ', '.join(
            f'{name}: {percentage(value):.2f}%' for _, name, value in pairs)
        lines.append(f'| `{occ_name}` | {row["denom"]} | {ratio_text} |')
    return lines


def build_markdown(result, occ_indices, topk=None):
    map_classes = result['map_classes']
    lines = [
        '# OCC Map Low-Z Overlap',
        '',
        f'- Config: `{result["config"]}`',
        f'- Ann file: `{result["ann_file"]}`',
        f'- Samples: {result["num_samples"]}',
        f'- Z layers: {result["z_layers"]}',
        f'- Nonvisible OCC ignored: {result["ignore_nonvisible"]}',
        '',
        'Percentages in `Map -> OCC` are conditional on each map mask. '
        'Percentages in `OCC -> Map` are map coverage of each OCC class, '
        'so multi-label map classes may sum above 100%.',
        '',
    ]

    layer_order = [f'z={z}' for z in result['z_layers']]
    layer_order.append(result['aggregate_layer_key'])
    for layer_key in layer_order:
        lines.extend([f'## {layer_key}', ''])
        summary = result['layers'][layer_key]
        lines.extend(format_map_to_occ(summary, map_classes, occ_indices, topk))
        lines.append('')
        lines.extend(format_occ_to_map(summary, map_classes, occ_indices, topk))
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours:d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    split_cfg = get_split_cfg(cfg, args.split)

    data_root = resolve_path(split_cfg.data_root)
    ann_file = resolve_path(args.ann_file or split_cfg.ann_file)
    infos = mmcv.load(str(ann_file))
    infos = infos['infos'] if isinstance(infos, dict) else infos

    if args.sample_count is not None:
        sample_count = min(args.sample_count, len(infos))
        indices = np.linspace(0, len(infos) - 1, sample_count, dtype=np.int64)
        infos = [infos[index] for index in indices]
    if args.max_samples is not None:
        infos = infos[:args.max_samples]

    z_layers = list(dict.fromkeys(args.z_layers))
    if any(z < 0 for z in z_layers):
        raise ValueError(f'Z layers must be non-negative: {z_layers}')

    map_loader = build_map_loader(cfg, data_root)
    occ_loader = LoadOccGTFromFile(
        ignore_nonvisible=not args.include_nonvisible)
    scene2location = build_scene2location(
        data_root, getattr(cfg, 'nusc_version', 'v1.0-trainval'))

    stats = dict(
        z_layers=z_layers,
        aggregate_layer_key=(
            f'z={min(z_layers)}~{max(z_layers)}_weighted'
            if len(z_layers) > 1 else f'z={z_layers[0]}_weighted'),
        num_map_classes=len(cfg.map_classes),
        num_occ_classes=len(NUSC_OCC_CLASSES),
        map_to_occ=np.zeros(
            (len(z_layers), len(cfg.map_classes), len(NUSC_OCC_CLASSES)),
            dtype=np.float64),
        map_valid_denoms=np.zeros(
            (len(z_layers), len(cfg.map_classes)), dtype=np.float64),
        occ_to_map=np.zeros(
            (len(z_layers), len(NUSC_OCC_CLASSES), len(cfg.map_classes)),
            dtype=np.float64),
        occ_denoms=np.zeros(
            (len(z_layers), len(NUSC_OCC_CLASSES)), dtype=np.float64),
    )

    start = time.time()
    for index, info in enumerate(infos, 1):
        results = make_loader_input(info, scene2location)
        results = map_loader(results)
        results = occ_loader(results)

        occ_semantics = unwrap_tensor(results['voxel_semantics'])
        map_masks = unwrap_tensor(results['gt_masks_bev'])

        if max(z_layers) >= occ_semantics.shape[2]:
            raise ValueError(
                f'Max Z layer {max(z_layers)} exceeds OCC shape '
                f'{occ_semantics.shape}.')

        update_counts(stats, occ_semantics, map_masks, z_layers)

        if (args.progress_interval > 0
                and (index % args.progress_interval == 0
                     or index == len(infos))):
            elapsed = time.time() - start
            rate = index / elapsed if elapsed > 0 else 0.0
            print(
                f'progress {index}/{len(infos)} '
                f'elapsed={format_duration(elapsed)} rate={rate:.2f}/s',
                flush=True)

    occ_indices = select_occ_indices(args)
    result = dict(
        config=args.config,
        split=args.split,
        ann_file=str(ann_file),
        num_samples=len(infos),
        z_layers=z_layers,
        aggregate_layer_key=stats['aggregate_layer_key'],
        map_classes=list(cfg.map_classes),
        occ_classes=list(NUSC_OCC_CLASSES),
        printed_occ_indices=occ_indices,
        ignore_nonvisible=not args.include_nonvisible,
        layers=build_layer_summary(stats, list(cfg.map_classes)),
    )

    markdown = build_markdown(result, occ_indices, args.topk)
    print(markdown)

    if args.output_json is not None:
        output_json = resolve_path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2) + '\n')

    if args.output_md is not None:
        output_md = resolve_path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown)


if __name__ == '__main__':
    main()
