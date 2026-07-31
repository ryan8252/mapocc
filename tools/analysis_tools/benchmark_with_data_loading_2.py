# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets.builder import PIPELINES

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from tools.misc.fuse_conv_bn import fuse_module


PROTOOCC_CAMERA_ONLY_DROP_TYPES = {
    'LoadBEVSegmentation',
    'LoadOccGTFromFile',
    'LoadPointsFromFile',
    'LoadPointsFromMultiSweeps',
    'PointToMultiViewDepth',
    'LoadLidarsegFromFile',
}


@PIPELINES.register_module()
class CameraOnlyBDA(object):
    """Append identity BDA to img_inputs without reading annotation targets."""

    def __call__(self, results):
        if len(results['img_inputs']) == 7:
            return results
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        bda_rot = torch.eye(3)
        results['img_inputs'] = (
            imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans,
            bda_rot)
        results['flip_dx'] = False
        results['flip_dy'] = False
        results['rotate_bda'] = 0
        results['scale_bda'] = 1.0
        return results


def bytes_to_mb(num_bytes):
    return num_bytes / 1024 / 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet benchmark a model with data loading')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=500, type=int,
                        help='samples to benchmark after warmup')
    parser.add_argument('--log-interval', default=50, type=int,
                        help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--no-acceleration',
        action='store_true',
        help='Omit the pre-computation acceleration')
    parser.add_argument(
        '--camera-only',
        action='store_true',
        help='Strip test pipeline to camera inputs only for inference '
        'throughput measurement.')
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        default=None,
        help='Override cfg.data.workers_per_gpu. Useful for debugging with 0.')
    args = parser.parse_args()
    return args


def strip_camera_only_pipeline(pipeline):
    stripped = []
    for transform in pipeline:
        transform = transform.copy()
        transform_type = transform.get('type')
        if transform_type in PROTOOCC_CAMERA_ONLY_DROP_TYPES:
            continue
        if transform_type == 'LoadAnnotationsBEVDepth':
            transform = dict(type='CameraOnlyBDA')
        elif transform_type == 'Collect3D':
            transform['keys'] = ['img_inputs']
        if 'transforms' in transform:
            transform['transforms'] = strip_camera_only_pipeline(
                transform['transforms'])
        stripped.append(transform)
    return stripped


def enable_camera_only_pipeline(cfg):
    cfg.data.test.pipeline = strip_camera_only_pipeline(cfg.data.test.pipeline)
    if 'test_pipeline' in cfg:
        cfg.test_pipeline = strip_camera_only_pipeline(cfg.test_pipeline)
    if 'evaluation' in cfg and cfg.evaluation.get('pipeline') is not None:
        cfg.evaluation.pipeline = strip_camera_only_pipeline(
            cfg.evaluation.pipeline)


def import_plugin(cfg, config_path):
    if not hasattr(cfg, 'plugin') or not cfg.plugin:
        return

    import importlib
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
        module_dir = os.path.dirname(plugin_dir)
    else:
        module_dir = os.path.dirname(config_path)

    module_parts = module_dir.split('/')
    module_path = module_parts[0]
    for module_part in module_parts[1:]:
        module_path = module_path + '.' + module_part
    print(module_path)
    importlib.import_module(module_path)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    import_plugin(cfg, args.config)
    if args.camera_only:
        enable_camera_only_pipeline(cfg)

    dataset = build_dataset(cfg.data.test)
    workers_per_gpu = (
        cfg.data.workers_per_gpu
        if args.workers_per_gpu is None else args.workers_per_gpu)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False)

    if not args.no_acceleration:
        cfg.model.img_view_transformer.accelerate = True
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    num_warmup = 5
    data_iter = iter(data_loader)

    for _ in range(num_warmup):
        try:
            data = next(data_iter)
        except StopIteration:
            print('No samples were benchmarked. Dataset ended during warmup.')
            return
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()

    measured_samples = 0
    data_time = 0
    forward_time = 0
    total_time = 0

    for i in range(args.samples):
        data_start = time.perf_counter()
        try:
            data = next(data_iter)
        except StopIteration:
            break
        torch.cuda.synchronize()
        forward_start = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        data_elapsed = forward_start - data_start
        forward_elapsed = end_time - forward_start
        elapsed = end_time - data_start
        measured_samples += 1
        data_time += data_elapsed
        forward_time += forward_elapsed
        total_time += elapsed

        if (i + 1) % args.log_interval == 0:
            fps = measured_samples / total_time
            data_ms = data_time * 1000 / measured_samples
            forward_ms = forward_time * 1000 / measured_samples
            print(f'Done image [{measured_samples:<3}/ {args.samples}], '
                  f'fps with data loading: {fps:.1f} img / s, '
                  f'data: {data_ms:.2f} ms, forward: {forward_ms:.2f} ms')

    if measured_samples == 0:
        print('No samples were benchmarked. Dataset ended during warmup.')
        return

    fps = measured_samples / total_time
    data_ms = data_time * 1000 / measured_samples
    forward_ms = forward_time * 1000 / measured_samples
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    peak_extra = max(0, peak_allocated - baseline_allocated)
    print(f'Overall \nfps with data loading: {fps:.2f} img / s '
          f'\ntime per sample: {1000 / fps:.2f} ms'
          f'\ndata time per sample: {data_ms:.2f} ms'
          f'\nforward time per sample: {forward_ms:.2f} ms')
    print('Memory: '
          f'\npeak allocated: {bytes_to_mb(peak_allocated):.1f} MB '
          f'\npeak reserved: {bytes_to_mb(peak_reserved):.1f} MB '
          f'\npeak extra over warmup baseline: {bytes_to_mb(peak_extra):.1f} MB')


if __name__ == '__main__':
    main()
