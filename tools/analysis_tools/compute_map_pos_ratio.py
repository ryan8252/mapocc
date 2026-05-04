import argparse
import json
import sys
import time
from pathlib import Path

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmdet3d_plugin.datasets.pipelines.loading_bev_seg import (  # noqa: E402
    LoadBEVSegmentation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute BEV map positive pixel ratios from the train '
        'pipeline rasterizer.')
    parser.add_argument('config', help='Path to a ProtoOcc config file.')
    parser.add_argument(
        '--ann-file',
        default=None,
        help='Override annotation pkl. Defaults to cfg.data.train.ann_file.')
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split config to read when --ann-file is omitted.')
    parser.add_argument(
        '--overlay-indices',
        nargs='+',
        type=int,
        default=[1, 3, 5],
        help='Map class indices to print as overlay ref ratios.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional smoke-test limit.')
    parser.add_argument(
        '--sample-count',
        type=int,
        default=None,
        help='Evenly sample N frames across the loaded split. This is useful '
        'for faster bootstrap estimates; omit it for exact full-split stats.')
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=1000,
        help='Print progress every N samples.')
    progress_bar_group = parser.add_mutually_exclusive_group()
    progress_bar_group.add_argument(
        '--progress-bar',
        dest='progress_bar',
        action='store_true',
        default=None,
        help='Force an in-place progress bar on stderr.')
    progress_bar_group.add_argument(
        '--no-progress-bar',
        dest='progress_bar',
        action='store_false',
        help='Disable the in-place progress bar.')
    parser.add_argument(
        '--progress-bar-width',
        type=int,
        default=32,
        help='Width of the in-place progress bar.')
    parser.add_argument(
        '--progress-refresh-seconds',
        type=float,
        default=1.0,
        help='Minimum seconds between progress bar refreshes.')
    parser.add_argument(
        '--output',
        default=None,
        help='Optional JSON output path.')
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


def build_loader(cfg, data_root):
    return LoadBEVSegmentation(
        dataset_root=str(data_root),
        xbound=cfg.map_xbound,
        ybound=cfg.map_ybound,
        classes=cfg.map_classes)


def info_to_loader_input(info, scene2location):
    return dict(
        curr=dict(
            lidar2ego_rotation=info['lidar2ego_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            ego2global_translation=info['ego2global_translation']),
        location=info.get('location') or scene2location[info['scene_token']],
        rotate_bda=0.0,
        scale_bda=1.0,
        flip_dx=False,
        flip_dy=False)


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f'{hours:d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'


def should_show_progress_bar(progress_bar):
    if progress_bar is None:
        return sys.stderr.isatty()
    return bool(progress_bar)


def render_progress_bar(index, total, start_time, width):
    total = max(total, 1)
    width = max(width, 8)
    now = time.time()
    elapsed = now - start_time
    fraction = min(max(index / total, 0.0), 1.0)
    filled = int(round(width * fraction))
    bar = '=' * filled + '-' * (width - filled)
    rate = index / elapsed if elapsed > 0 and index > 0 else 0.0
    remaining = (total - index) / rate if rate > 0 else 0.0
    sys.stderr.write(
        '\r'
        f'[{bar}] {index}/{total} '
        f'{fraction * 100:6.2f}% '
        f'elapsed={format_duration(elapsed)} '
        f'eta={format_duration(remaining)} '
        f'rate={rate:.2f}/s')
    sys.stderr.flush()


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

    loader = build_loader(cfg, data_root)
    scene2location = build_scene2location(
        data_root, getattr(cfg, 'nusc_version', 'v1.0-trainval'))

    num_classes = len(cfg.map_classes)
    pos_pixels = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0
    start = time.time()
    show_progress_bar = should_show_progress_bar(args.progress_bar)
    last_progress_render = 0.0
    last_progress_index = 0
    progress_bar_visible = False

    for index, info in enumerate(infos, 1):
        results = info_to_loader_input(info, scene2location)
        masks = loader(results)['gt_masks_bev']
        pos_pixels += masks.reshape(num_classes, -1).sum(axis=1)
        total_pixels += masks.shape[1] * masks.shape[2]

        now = time.time()
        if (show_progress_bar
                and (index == 1
                     or index == len(infos)
                     or now - last_progress_render
                     >= args.progress_refresh_seconds)):
            render_progress_bar(
                index, len(infos), start, args.progress_bar_width)
            last_progress_render = now
            last_progress_index = index
            progress_bar_visible = True

        if (args.progress_interval > 0
                and (index % args.progress_interval == 0
                     or index == len(infos))):
            if progress_bar_visible:
                sys.stderr.write('\n')
                sys.stderr.flush()
                progress_bar_visible = False
            ratios = pos_pixels / max(total_pixels, 1)
            formatted = ', '.join(f'{ratio:.8f}' for ratio in ratios)
            print(
                f'progress {index}/{len(infos)} '
                f'elapsed={time.time() - start:.1f}s ratios=[{formatted}]',
                flush=True)

    if show_progress_bar:
        if last_progress_index != len(infos):
            render_progress_bar(
                len(infos), len(infos), start, args.progress_bar_width)
            progress_bar_visible = True
        if progress_bar_visible:
            sys.stderr.write('\n')
            sys.stderr.flush()

    ratios = pos_pixels / total_pixels
    overlay_ratios = [float(ratios[index]) for index in args.overlay_indices]
    summary = dict(
        config=args.config,
        ann_file=str(ann_file),
        num_samples=len(infos),
        classes=list(cfg.map_classes),
        pos_pixels=[int(value) for value in pos_pixels],
        total_pixels=int(total_pixels),
        ratios=[float(value) for value in ratios],
        overlay_indices=args.overlay_indices,
        overlay_ratios=overlay_ratios)

    print(json.dumps(summary, indent=2))
    print(
        'dynamic_overlay_ref_pos_ratio='
        + repr([round(value, 10) for value in overlay_ratios]))

    if args.output is not None:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
