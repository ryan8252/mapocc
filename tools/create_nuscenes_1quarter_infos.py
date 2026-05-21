#!/usr/bin/env python
import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a deterministic scene-level 1/4 nuScenes train info pkl.')
    parser.add_argument(
        '--src',
        default='data/nuscenes/bevdetv2-nuscenes_infos_train.pkl',
        help='Source full-train info pkl.')
    parser.add_argument(
        '--out',
        default='data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0.pkl',
        help='Output subset info pkl.')
    parser.add_argument(
        '--scene-list-out',
        default='data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0_scenes.txt',
        help='Output selected scene-name list.')
    parser.add_argument(
        '--summary-out',
        default='data/nuscenes/bevdetv2-nuscenes_infos_train_1quarter_seed0_summary.json',
        help='Output split summary json.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fraction', type=float, default=0.25)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the summary without writing files.')
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)
    scene_out = Path(args.scene_list_out)
    summary_out = Path(args.summary_out)

    with src.open('rb') as f:
        data = pickle.load(f)

    infos = data['infos']
    scene_names = {}
    for info in infos:
        token = info['scene_token']
        scene_names[token] = info.get('scene_name', token)

    scene_items = sorted(scene_names.items(), key=lambda item: item[1])
    rng = random.Random(args.seed)
    rng.shuffle(scene_items)

    num_scenes = int(len(scene_items) * args.fraction)
    if num_scenes <= 0:
        raise ValueError('fraction selects zero scenes.')

    selected = set(token for token, _ in scene_items[:num_scenes])
    subset_infos = [info for info in infos if info['scene_token'] in selected]
    counts = Counter(info['scene_token'] for info in subset_infos)

    summary = {
        'source_ann_file': str(src),
        'out_ann_file': str(out),
        'num_infos': len(subset_infos),
        'num_scenes': len(counts),
        'source_num_infos': len(infos),
        'source_num_scenes': len(scene_items),
        'ratio_infos': len(subset_infos) / len(infos),
        'ratio_scenes': len(counts) / len(scene_items),
        'seed': args.seed,
        'fraction': args.fraction,
        'split_unit': 'scene',
        'min_frames_per_scene': min(counts.values()),
        'max_frames_per_scene': max(counts.values()),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.dry_run:
        return

    subset = dict(data)
    subset['infos'] = subset_infos
    subset['metadata'] = dict(data.get('metadata', {}))
    subset['metadata'].update({
        'subset': f'train_{args.fraction:g}_seed{args.seed}_scene_level',
        'source_ann_file': str(src),
        'source_num_infos': len(infos),
        'source_num_scenes': len(scene_items),
        'num_infos': len(subset_infos),
        'num_scenes': len(counts),
        'seed': args.seed,
        'fraction': args.fraction,
        'split_unit': 'scene',
    })

    out.parent.mkdir(parents=True, exist_ok=True)
    scene_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    with out.open('wb') as f:
        pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)

    selected_names = [scene_names[token] for token in selected]
    scene_out.write_text('\n'.join(sorted(selected_names)) + '\n')
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
