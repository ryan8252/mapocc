#!/usr/bin/env python3
"""Background prototype discriminative sanity check.

Loads an already-trained ProtoOcc checkpoint, runs N val samples, and reports
the inter-class cosine similarity of the background Scene-Aware Queries used
by `MapNeckQueryTSFG`.

If the off-diagonal mean cosine is >= 0.85, the 6 BG prototypes are collapsed
and the prototype-wise branch will be near-degenerate. In that case do NOT
launch the 24-epoch main run; first compare `mask_embed_real_bqc` source or
`similarity_mode='dot'`.

Usage (run from the ProtoOcc repo root):

    conda activate mapocc
    python tools/prototype_sanity_check.py \
        --config projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck.py \
        --checkpoint work_dirs/ProtoOcc_multi_cnn_head_map_neck_weight_4_4090/epoch_15_ema.pth \
        --num-samples 50 \
        --also-mask-embed
"""
import argparse
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


BG_NAMES = {
    11: 'driveable_surface',
    12: 'other_flat',
    13: 'sidewalk',
    14: 'terrain',
    15: 'manmade',
    16: 'vegetation',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--num-samples', type=int, default=50)
    p.add_argument('--background-ids', type=int, nargs='+',
                   default=[11, 12, 13, 14, 15, 16])
    p.add_argument('--also-mask-embed', action='store_true',
                   help='Also compute stats for mask_embed_real_bqc source.')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--plugin-dir', default='projects/mmdet3d_plugin',
                   help='Plugin path relative to repo root.')
    return p.parse_args()


def import_plugin(plugin_dir):
    plugin_dir = Path(plugin_dir).resolve()
    repo_root = plugin_dir.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    mod = '.'.join(plugin_dir.relative_to(repo_root).parts)
    importlib.import_module(mod)


def cosine_matrix(x):
    n = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    return n @ n.T


def offdiag_stats(mat):
    n = mat.shape[0]
    m = ~np.eye(n, dtype=bool)
    v = mat[m]
    return float(v.mean()), float(v.max())


def print_block(label, off_means, off_maxes, cos_list, bg_ids):
    print(f'\n=== {label} ===')
    print(f'Across-sample mean offdiag cos: {np.mean(off_means):.4f}')
    print(f'Across-sample mean offdiag-max: {np.mean(off_maxes):.4f}')
    mean_cos = np.mean(np.stack(cos_list), axis=0)
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    labels = [BG_NAMES.get(c, str(c))[:8] for c in bg_ids]
    print('Cosine matrix (averaged across samples):')
    print('           ' + '  '.join(f'{l:>8}' for l in labels))
    for i, l in enumerate(labels):
        row = '  '.join(f'{mean_cos[i, j]:8.3f}' for j in range(len(labels)))
        print(f'  {l:>8} {row}')


def main():
    args = parse_args()
    import_plugin(args.plugin_dir)

    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1,
        dist=False, shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=[args.gpu_id])
    model.eval()
    detector = model.module

    captured = []

    def patched_simple_test(self, points=None, img_metas=None, img=None,
                            rescale=False, **kwargs):
        voxel_feat, depth, pv_feat = self.extract_feat(
            img_inputs=img, img_metas=img_metas)
        encoder_output = self.dual_branch_encoder(voxel_feat)
        cvf, bev_feature, map_bev_feature = self._split_encoder_output(
            encoder_output)
        proto_occ, mask_feat = self.cnn3d_decoder(cvf.permute(0, 1, 4, 2, 3))
        B = cvf.shape[0]
        img_metas_occ = [
            {'pc_range': self.pc_range, 'occ_size': self.grid_size}
            for _ in range(B)
        ]
        _, query_info = self.prototype_query_decoder.simple_test(
            cvf, img_metas_occ, mask_feat, proto_occ,
            return_query_info=True)
        captured.append(query_info)
        return [np.zeros((1,), dtype=np.uint8)]

    detector.simple_test = types.MethodType(patched_simple_test, detector)

    bg_ids = args.background_ids
    qn_means, qn_maxes, qn_cos_list = [], [], []
    me_means, me_maxes, me_cos_list = [], [], []

    print(f'Background class ids: {bg_ids}')
    print(f'Collecting {args.num_samples} val samples ...')

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= args.num_samples:
                break
            captured.clear()
            _ = model(return_loss=False, rescale=True, **data)
            query_info = captured[-1]

            qn = query_info['query_norm_real_bqc'][:, bg_ids, :][0]
            qn = qn.detach().cpu().numpy()
            c = cosine_matrix(qn)
            qn_cos_list.append(c)
            m, mx = offdiag_stats(c)
            qn_means.append(m)
            qn_maxes.append(mx)

            if args.also_mask_embed:
                me = query_info['mask_embed_real_bqc'][:, bg_ids, :][0]
                me = me.detach().cpu().numpy()
                c = cosine_matrix(me)
                me_cos_list.append(c)
                m, mx = offdiag_stats(c)
                me_means.append(m)
                me_maxes.append(mx)

            if (i + 1) % 10 == 0:
                print(f'  [{i+1}/{args.num_samples}] '
                      f'qn offdiag mean so far={np.mean(qn_means):.4f} '
                      f'max={np.mean(qn_maxes):.4f}')

    print_block('query_norm_real_bqc (post-norm Scene-Aware Query)',
                qn_means, qn_maxes, qn_cos_list, bg_ids)
    if args.also_mask_embed:
        print_block('mask_embed_real_bqc (mask projection of post-norm query)',
                    me_means, me_maxes, me_cos_list, bg_ids)

    qn_mean = float(np.mean(qn_means))
    print('\n=== Verdict (based on query_norm_real_bqc) ===')
    if qn_mean < 0.5:
        print(f'GOOD ({qn_mean:.3f} < 0.5): BG prototypes well-separated.')
        print('  -> Safe to send 24-epoch main run with cosine + post-norm query.')
    elif qn_mean < 0.85:
        print(f'OK ({qn_mean:.3f}): some overlap but still discriminative.')
        print('  -> Main run can proceed; add similarity_mode="dot" ablation.')
    else:
        print(f'BAD ({qn_mean:.3f} >= 0.85): BG prototypes COLLAPSED.')
        print('  -> DO NOT send 24-epoch main run.')
        print('  -> Try mask_embed_real_bqc source or similarity_mode="dot" first.')


if __name__ == '__main__':
    main()
