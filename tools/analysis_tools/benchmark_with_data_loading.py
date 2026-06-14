# Benchmark with data loading included in timing
# Usage: python tools/analysis_tools/benchmark_with_data_loading.py <config> <checkpoint> [--samples N]
import argparse
import time
import sys
sys.path.insert(0, '.')

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark with data loading')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=500, type=int, help='samples to benchmark')
    parser.add_argument('--log-interval', default=50, type=int, help='interval of logging')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # warmup
    num_warmup = 5
    print(f'Warming up ({num_warmup} samples)...')
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        if i >= num_warmup - 1:
            break
    torch.cuda.synchronize()

    # benchmark with data loading included
    print(f'Benchmarking {args.samples} samples (with data loading)...')
    torch.cuda.synchronize()
    total_time = 0
    start_time = time.perf_counter()

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()

        if (i + 1) % args.log_interval == 0:
            elapsed = time.perf_counter() - start_time
            fps = (i + 1) / elapsed
            print(f'Done image [{i + 1:<3}/ {args.samples}], '
                  f'fps: {fps:.1f} img / s')

        if (i + 1) == args.samples:
            total_time = time.perf_counter() - start_time
            fps = args.samples / total_time
            print(f'Overall fps (with data loading): {fps:.2f} img / s')
            break


if __name__ == '__main__':
    import projects.mmdet3d_plugin
    main()
