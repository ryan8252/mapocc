#!/usr/bin/env python
import argparse
import ast
import copy
import importlib
import os
import os.path as osp
import sys


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
MMDET3D_ROOT = osp.join(REPO_ROOT, 'mmdetection3d')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-protoocc')
for path in (REPO_ROOT, MMDET3D_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2
import numpy as np
import torch
from mmcv import Config

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.pipelines.loading import (
    LoadAnnotationsBEVDepth,
    LoadOccGTFromFile,
)
from projects.mmdet3d_plugin.datasets.pipelines.loading_bev_seg import (
    LoadBEVSegmentation,
)


OCC_CLASS_NAMES = [
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
]

OCC_COLORS = np.array(
    [
        [0, 0, 0],
        [112, 128, 144],
        [220, 20, 60],
        [255, 127, 80],
        [255, 158, 0],
        [233, 150, 70],
        [255, 61, 99],
        [0, 0, 230],
        [47, 79, 79],
        [255, 140, 0],
        [255, 99, 71],
        [0, 207, 191],
        [175, 0, 75],
        [75, 0, 75],
        [112, 180, 60],
        [222, 184, 135],
        [0, 175, 0],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)

GROUND_GROUPS = {
    0: ('empty/free', (0, 0, 0)),
    1: ('driveable/other_flat', (0, 207, 191)),
    2: ('sidewalk', (75, 0, 75)),
    3: ('terrain', (112, 180, 60)),
    4: ('occupied-other', (120, 120, 120)),
}

MAP_CLASS_COLORS = {
    'drivable_area': (245, 183, 0),
    'ped_crossing': (255, 255, 255),
    'walkway': (70, 180, 255),
    'stop_line': (255, 80, 80),
    'carpark_area': (181, 133, 0),
    'divider': (255, 0, 255),
}

MAP_DRAW_ORDER = [
    'drivable_area',
    'carpark_area',
    'walkway',
    'ped_crossing',
    'divider',
    'stop_line',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize occupancy GT and BEV map GT in the same BEV frame.'
    )
    parser.add_argument(
        'config',
        nargs='?',
        default='projects/configs/ProtoOcc/ProtoOcc_bevfusion_mapgt_protoocc_range.py',
        help='Path to the config file.',
    )
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to visualize.',
    )
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='Starting sample index inside the chosen split.',
    )
    parser.add_argument(
        '--token',
        default=None,
        help='Sample token to visualize. Overrides --index if provided.',
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1,
        help='Number of samples to export.',
    )
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help='Step size between exported samples.',
    )
    parser.add_argument(
        '--out-dir',
        default='work_dirs/vis_occ_map_gt',
        help='Directory for output PNGs.',
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=4,
        help='Nearest-neighbor upsample factor for each BEV panel.',
    )
    parser.add_argument(
        '--bda',
        default='none',
        choices=['none', 'train'],
        help='Use no BEV data augmentation or sample train-time BDA.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed used when --bda=train.',
    )
    parser.add_argument(
        '--ignore-nonvisible',
        action='store_true',
        help='Mask occupancy voxels outside camera visibility like the train pipeline.',
    )
    return parser.parse_args()


def resolve_path(path):
    if osp.isabs(path):
        return path
    return osp.abspath(osp.join(REPO_ROOT, path))


def clone_config_value(value):
    if isinstance(value, dict):
        return {key: clone_config_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clone_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_config_value(item) for item in value)
    return value


def merge_config_dicts(base, child):
    merged = {key: clone_config_value(val) for key, val in base.items()}
    for key, value in child.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
            and not value.get('_delete_', False)
        ):
            merged[key] = merge_config_dicts(merged[key], value)
        else:
            if isinstance(value, dict) and value.get('_delete_', False):
                value = {k: v for k, v in value.items() if k != '_delete_'}
            merged[key] = clone_config_value(value)
    return merged


def load_config_namespace(config_path):
    config_path = osp.abspath(config_path)
    with open(config_path, 'r', encoding='utf-8') as handle:
        source = handle.read()

    tree = ast.parse(source, filename=config_path)
    base_files = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '_base_':
                base_value = ast.literal_eval(node.value)
                if isinstance(base_value, str):
                    base_files = [base_value]
                else:
                    base_files = list(base_value)
                break

    base_namespace = {}
    for base_file in base_files:
        base_path = osp.abspath(osp.join(osp.dirname(config_path), base_file))
        base_namespace = merge_config_dicts(base_namespace, load_config_namespace(base_path))

    namespace = dict(base_namespace)
    namespace['__file__'] = config_path
    exec(compile(source, config_path, 'exec'), namespace)
    namespace.pop('_base_', None)
    current_namespace = {
        key: value
        for key, value in namespace.items()
        if not key.startswith('__')
    }
    return merge_config_dicts(base_namespace, current_namespace)


def load_config(config_path):
    if config_path.endswith('.py'):
        # Avoid mmcv.Config.fromfile() here: in this environment/mmcv version it
        # creates a NamedTemporaryFile inside a TemporaryDirectory, which can
        # emit a noisy FileNotFoundError during interpreter cleanup even though
        # the config loads successfully.
        return Config(load_config_namespace(config_path), filename=config_path)
    return Config.fromfile(config_path)


def import_plugin_modules(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return

    if hasattr(cfg, 'plugin_dir'):
        module_dir = osp.dirname(cfg.plugin_dir)
    else:
        module_dir = osp.dirname(config_path)

    module_dir = module_dir.strip('/').split('/')
    module_path = module_dir[0]
    for part in module_dir[1:]:
        module_path = module_path + '.' + part
    importlib.import_module(module_path)


def find_pipeline_step(pipeline, step_type):
    for step in pipeline:
        if step.get('type') == step_type:
            return copy.deepcopy(step)
        for nested_key in ('pipeline', 'transforms'):
            if nested_key in step:
                nested = find_pipeline_step(step[nested_key], step_type)
                if nested is not None:
                    return nested
    return None


def prepare_dataset_cfg(cfg, split):
    dataset_cfg = copy.deepcopy(cfg.data[split])
    if 'data_root' in dataset_cfg:
        dataset_cfg['data_root'] = resolve_path(dataset_cfg['data_root'])
    if 'ann_file' in dataset_cfg:
        dataset_cfg['ann_file'] = resolve_path(dataset_cfg['ann_file'])
    return dataset_cfg


def sample_bda(load_ann_cfg, class_names, mode):
    ann_loader = LoadAnnotationsBEVDepth(
        bda_aug_conf=load_ann_cfg['bda_aug_conf'],
        classes=class_names,
        is_train=True,
    )

    if mode == 'train':
        rotate_bda, scale_bda, flip_dx, flip_dy = ann_loader.sample_bda_augmentation()
    else:
        rotate_bda, scale_bda, flip_dx, flip_dy = 0.0, 1.0, False, False

    _, bda_rot = ann_loader.bev_transform(
        torch.zeros((0, 9), dtype=torch.float32),
        rotate_bda,
        scale_bda,
        flip_dx,
        flip_dy,
    )
    return rotate_bda, scale_bda, flip_dx, flip_dy, bda_rot


def orient_bev(array):
    if array.ndim == 2:
        return array[::-1, ::-1]
    if array.ndim == 3:
        return array[:, ::-1, ::-1]
    raise ValueError(f'Unsupported BEV tensor shape: {array.shape}')


def collapse_top_labels(semantics):
    valid = (semantics != 17) & (semantics != 255)
    has_valid = valid.any(axis=2)
    rev_valid = valid[:, :, ::-1]
    rev_indices = rev_valid.argmax(axis=2)
    z_indices = semantics.shape[2] - 1 - rev_indices

    top = np.full(semantics.shape[:2], 17, dtype=np.uint8)
    xs, ys = np.nonzero(has_valid)
    top[xs, ys] = semantics[xs, ys, z_indices[xs, ys]]
    return top


def collapse_ground_groups(semantics):
    valid = (semantics != 17) & (semantics != 255)
    has_occ = valid.any(axis=2)
    has_drive = np.isin(semantics, [11, 12]).any(axis=2)
    has_sidewalk = (semantics == 13).any(axis=2)
    has_terrain = (semantics == 14).any(axis=2)

    groups = np.zeros(semantics.shape[:2], dtype=np.uint8)
    groups[has_occ] = 4
    groups[has_terrain] = 3
    groups[has_drive] = 1
    groups[has_sidewalk] = 2
    return groups


def render_occ_surface(top_labels):
    return OCC_COLORS[top_labels]


def render_ground_groups(groups):
    image = np.zeros(groups.shape + (3,), dtype=np.uint8)
    for group_id, (_, color) in GROUND_GROUPS.items():
        image[groups == group_id] = color
    return image


def render_map_masks(gt_masks_bev, map_classes):
    h, w = gt_masks_bev.shape[1:]
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for name in MAP_DRAW_ORDER:
        if name not in map_classes:
            continue
        mask = gt_masks_bev[map_classes.index(name)].astype(bool)
        image[mask] = np.array(MAP_CLASS_COLORS[name], dtype=np.uint8)
    return image


def overlay_map_on_ground(ground_image, gt_masks_bev, map_classes):
    overlay = ground_image.astype(np.float32).copy()
    for name in MAP_DRAW_ORDER:
        if name not in map_classes:
            continue
        mask = gt_masks_bev[map_classes.index(name)].astype(np.uint8)
        color = np.array(MAP_CLASS_COLORS[name], dtype=np.float32)
        overlay[mask.astype(bool)] = overlay[mask.astype(bool)] * 0.72 + color * 0.28
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(v) for v in color), 1)
    return overlay.astype(np.uint8)


def darken_nonvisible(image, visible_cols):
    image = image.copy()
    invisible = ~visible_cols
    image[invisible] = (image[invisible].astype(np.float32) * 0.38).astype(np.uint8)
    return image


def add_center_marker(image):
    image = image.copy()
    h, w = image.shape[:2]
    cx = w // 2
    cy = h // 2
    size = max(6, min(h, w) // 24)
    arrow = np.array(
        [
            [cx, cy - size],
            [cx - size // 2, cy + size // 2],
            [cx + size // 2, cy + size // 2],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [arrow], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.line(image, (cx - size, cy), (cx + size, cy), (255, 255, 255), 1)
    cv2.line(image, (cx, cy - size), (cx, cy + size), (255, 255, 255), 1)
    return image


def make_panel(image, title, scale):
    image = add_center_marker(image)
    panel = cv2.resize(
        image,
        (image.shape[1] * scale, image.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    panel = cv2.copyMakeBorder(panel, 44, 12, 12, 12, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(
        panel,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return panel


def draw_legend_row(canvas, start_x, start_y, items, box_size=18, gap=14):
    x = start_x
    for label, color in items:
        cv2.rectangle(
            canvas,
            (x, start_y),
            (x + box_size, start_y + box_size),
            color,
            thickness=-1,
        )
        cv2.rectangle(
            canvas,
            (x, start_y),
            (x + box_size, start_y + box_size),
            (255, 255, 255),
            thickness=1,
        )
        cv2.putText(
            canvas,
            label,
            (x + box_size + 8, start_y + box_size - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        x += box_size + 8 + max(70, len(label) * 8) + gap


def compose_canvas(panels, token, sample_index, split, bda_desc):
    top = np.concatenate(panels[:2], axis=1)
    bottom = np.concatenate(panels[2:], axis=1)
    body = np.concatenate([top, bottom], axis=0)

    header_h = 90
    legend_h = 130
    canvas = np.full(
        (body.shape[0] + header_h + legend_h, body.shape[1], 3),
        18,
        dtype=np.uint8,
    )
    canvas[header_h:header_h + body.shape[0]] = body

    cv2.putText(
        canvas,
        f'split={split}  index={sample_index}  token={token}',
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        bda_desc,
        (16, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (190, 190, 190),
        2,
        cv2.LINE_AA,
    )

    legend_y = header_h + body.shape[0] + 18
    cv2.putText(
        canvas,
        'Ground legend',
        (16, legend_y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    ground_items = [(name, color) for _, (name, color) in GROUND_GROUPS.items()]
    draw_legend_row(canvas, 16, legend_y + 12, ground_items)

    map_y = legend_y + 52
    cv2.putText(
        canvas,
        'Map legend',
        (16, map_y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    map_items = [(name, MAP_CLASS_COLORS[name]) for name in MAP_DRAW_ORDER]
    draw_legend_row(canvas, 16, map_y + 12, map_items)

    cv2.putText(
        canvas,
        'Darkened cells indicate columns outside mask_camera visibility.',
        (16, header_h + body.shape[0] + legend_h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    return canvas


def locate_index(dataset, token, default_index):
    if token is None:
        return default_index
    for index, info in enumerate(dataset.data_infos):
        if info['token'] == token:
            return index
    raise KeyError(f'Could not find token `{token}` in the selected split.')


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)
    import_plugin_modules(cfg, config_path)

    dataset_cfg = prepare_dataset_cfg(cfg, args.split)
    dataset = build_dataset(dataset_cfg)

    train_pipeline = cfg.data.train.pipeline
    load_ann_cfg = find_pipeline_step(train_pipeline, 'LoadAnnotationsBEVDepth')
    load_occ_cfg = find_pipeline_step(train_pipeline, 'LoadOccGTFromFile')
    load_map_cfg = find_pipeline_step(train_pipeline, 'LoadBEVSegmentation')
    if load_ann_cfg is None or load_occ_cfg is None or load_map_cfg is None:
        raise RuntimeError(
            'Could not find LoadAnnotationsBEVDepth / LoadOccGTFromFile / '
            'LoadBEVSegmentation in cfg.data.train.pipeline.'
        )

    load_map_cfg['dataset_root'] = resolve_path(load_map_cfg['dataset_root'])
    load_map_cfg.pop('type', None)
    load_occ_cfg.pop('type', None)

    if args.ignore_nonvisible:
        load_occ_cfg['ignore_nonvisible'] = True
    else:
        load_occ_cfg['ignore_nonvisible'] = False

    map_loader = LoadBEVSegmentation(**load_map_cfg)
    occ_loader = LoadOccGTFromFile(**load_occ_cfg)

    sample_index = locate_index(dataset, args.token, args.index)
    os.makedirs(resolve_path(args.out_dir), exist_ok=True)

    map_classes = list(getattr(dataset, 'map_classes', None) or cfg.data.train.map_classes)
    class_names = getattr(cfg, 'class_names', None)
    if class_names is None:
        raise RuntimeError('Config does not define `class_names`.')

    for offset in range(args.count):
        index = sample_index + offset * args.step
        if index >= len(dataset):
            print(f'Skip index {index}: out of range ({len(dataset)} samples).')
            continue

        sample = copy.deepcopy(dataset.get_data_info(index))
        sample['occ_gt_path'] = resolve_path(sample['occ_gt_path'])

        rotate_bda, scale_bda, flip_dx, flip_dy, bda_rot = sample_bda(
            load_ann_cfg,
            class_names,
            args.bda,
        )
        sample['img_inputs'] = (None, None, None, None, None, None, bda_rot)
        sample['rotate_bda'] = rotate_bda
        sample['scale_bda'] = scale_bda
        sample['flip_dx'] = flip_dx
        sample['flip_dy'] = flip_dy

        sample = map_loader(sample)
        sample = occ_loader(sample)

        # Trial alignment fix: occupancy voxels need XY flip, and the map masks
        # still appear mirrored on one BEV axis afterwards. Mirror the map once
        # more so the thin walkway/divider structures can line up with occupancy.
        
        gt_masks_bev = sample['gt_masks_bev']
        voxel_semantics = sample['voxel_semantics'].cpu().numpy()
        mask_camera = sample['mask_camera'].cpu().numpy().astype(bool)

        visible_cols = mask_camera.any(axis=2)
        occ_top = collapse_top_labels(voxel_semantics)
        occ_ground = collapse_ground_groups(voxel_semantics)

        occ_surface_img = darken_nonvisible(render_occ_surface(occ_top), visible_cols)
        ground_img = darken_nonvisible(render_ground_groups(occ_ground), visible_cols)
        map_img = darken_nonvisible(render_map_masks(gt_masks_bev, map_classes), visible_cols)
        overlay_img = darken_nonvisible(
            overlay_map_on_ground(render_ground_groups(occ_ground), gt_masks_bev, map_classes),
            visible_cols,
        )

        panels = [
            make_panel(occ_surface_img, 'Occupancy Surface (top visible voxel)', args.scale),
            make_panel(ground_img, 'Ground-Relevant Occupancy', args.scale),
            make_panel(map_img, 'Map GT', args.scale),
            make_panel(overlay_img, 'Overlay: Ground Occupancy + Map GT', args.scale),
        ]

        token = sample['sample_idx']
        bda_desc = (
            f'BDA mode={args.bda}  rotate={rotate_bda:.2f}  scale={scale_bda:.2f}  '
            f'flip_dx={flip_dx}  flip_dy={flip_dy}  ignore_nonvisible={args.ignore_nonvisible}'
        )
        canvas = compose_canvas(panels, token, index, args.split, bda_desc)

        save_path = resolve_path(
            osp.join(args.out_dir, f'{args.split}_{index:05d}_{token}_occ_map_gt.png')
        )
        saved = cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if not saved:
            raise IOError(f'Failed to save visualization to {save_path}')
        print(f'Saved {save_path}')


if __name__ == '__main__':
    main()


# python tools/analysis_tools/vis_occ_map_gt.py \
#   --index 0 \
#   --count 10 \
#   --out-dir work_dirs/vis_occ_map_gt

# python tools/analysis_tools/vis_occ_map_gt.py --bda=train --seed=3  --index=300