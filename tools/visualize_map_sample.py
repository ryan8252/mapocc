#!/usr/bin/env python
"""Run one ProtoOcc sample and save only its BEV map prediction."""

import argparse
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mmcv  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint, wrap_fp16_model  # noqa: E402
from mmdet.datasets import replace_ImageToTensor  # noqa: E402
from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

try:
    from mmdet.utils import compat_cfg, setup_multi_processes
except ImportError:
    from mmdet3d.utils import compat_cfg, setup_multi_processes


MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "divider": (106, 61, 154),
}
MAP_CLASSES = tuple(MAP_PALETTE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sample-token")
    group.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=Path("viz/map_preview"))
    parser.add_argument(
        "--render-axis",
        choices=("bevfusion", "protoocc"),
        default="bevfusion",
        help="Use BEVFusion image orientation for direct visual comparison.",
    )
    parser.add_argument(
        "--render-classes",
        nargs="+",
        choices=MAP_CLASSES,
        help=(
            "Optional subset of map channels to draw. For GME inspection, "
            "use: ped_crossing stop_line divider. The default draws all "
            "dataset map classes."
        ),
    )
    parser.add_argument("--save-gt", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if args.sample_index is not None and args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    return args


def import_plugins(cfg, config_path):
    if not getattr(cfg, "plugin", False):
        return
    if hasattr(cfg, "plugin_dir"):
        module_dir = cfg.plugin_dir
    else:
        module_dir = os.path.dirname(str(config_path))
    module_dir = module_dir.strip("/.")
    module_path = ".".join(Path(module_dir).parts)
    importlib.import_module(module_path)


def load_cfg(config_path):
    cfg = Config.fromfile(str(config_path))
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    import_plugins(cfg, config_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.gpu_ids = [0]
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.get("test_dataloader", {}).get("samples_per_gpu", 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    return cfg


def build_loader(cfg, dataset, sample_index):
    loader_cfg = dict(
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    loader_cfg.update(cfg.data.get("test_dataloader", {}))
    loader_cfg["samples_per_gpu"] = 1
    loader_cfg["workers_per_gpu"] = 0
    subset = Subset(dataset, [sample_index])
    return build_dataloader(subset, **loader_cfg)


def locate_sample(dataset, sample_token, sample_index):
    if sample_token is None:
        if sample_index >= len(dataset):
            raise IndexError(
                f"sample index {sample_index} is outside dataset size {len(dataset)}"
            )
        return sample_index
    for index, info in enumerate(dataset.data_infos):
        if str(info.get("token")) == sample_token:
            return index
    raise KeyError(f"sample token not found: {sample_token}")


def take_sample(data_loader):
    return next(iter(data_loader))


def build_inference_model(cfg, checkpoint_path, device):
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]

    if not device.startswith("cuda"):
        raise ValueError("This model visualizer currently requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active environment")
    device_index = int(device.split(":", 1)[1]) if ":" in device else 0
    torch.cuda.set_device(device_index)
    model = MMDataParallel(model.cuda(device_index), device_ids=[device_index])
    model.eval()
    return model


def to_bevfusion_axis(array):
    """Convert ProtoOcc (C, x, y) tensors to BEVFusion display axes."""
    return array.transpose(0, 2, 1)[:, ::-1, :]


def masks_to_rgb(masks, class_names, render_classes=None):
    canvas = np.full((*masks.shape[-2:], 3), 240, dtype=np.uint8)
    render_set = set(class_names if render_classes is None else render_classes)
    for class_index, class_name in enumerate(class_names):
        if class_name not in render_set:
            continue
        color = MAP_PALETTE.get(class_name)
        if color is not None:
            canvas[masks[class_index]] = color
    return canvas


def save_map(path, masks, class_names, render_axis, render_classes=None):
    if render_axis == "bevfusion":
        masks = to_bevfusion_axis(masks)
    rgb = masks_to_rgb(masks, class_names, render_classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not mmcv.imwrite(rgb[:, :, ::-1], str(path)):
        raise IOError(f"failed to save map image: {path}")


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    cfg = load_cfg(config_path)
    dataset = build_dataset(cfg.data.test)
    sample_index = locate_sample(dataset, args.sample_token, args.sample_index)
    data_loader = build_loader(cfg, dataset, sample_index)
    data = take_sample(data_loader)
    model = build_inference_model(cfg, checkpoint_path, args.device)

    with torch.no_grad():
        output = model(return_loss=False, rescale=True, **data)
    result = output[0]
    if "masks_bev" not in result:
        raise KeyError("model output does not contain masks_bev")

    probs = np.asarray(result["masks_bev"], dtype=np.float32)
    class_names = tuple(dataset.map_classes)
    if probs.shape[0] != len(class_names):
        raise ValueError(
            f"prediction has {probs.shape[0]} channels for {len(class_names)} classes"
        )
    render_classes = tuple(args.render_classes or class_names)
    missing_render_classes = set(render_classes) - set(class_names)
    if missing_render_classes:
        raise ValueError(
            "requested render classes are absent from the dataset: "
            + ", ".join(sorted(missing_render_classes))
        )

    info = dataset.data_infos[sample_index]
    token = str(info["token"])
    scene_name = str(info.get("scene_name", info.get("scene_token", "unknown-scene")))
    pred_path = args.out_dir / f"{token}_pred.png"
    save_map(
        pred_path,
        probs >= args.threshold,
        class_names,
        args.render_axis,
        render_classes,
    )

    print(f"sample_index={sample_index}")
    print(f"scene={scene_name}")
    print(f"token={token}")
    print(f"prediction_shape={probs.shape}")
    print(f"threshold={args.threshold}")
    print(f"render_classes={','.join(render_classes)}")
    print(f"prediction={pred_path.resolve()}")

    if args.save_gt:
        if "gt_masks_bev" not in result:
            raise KeyError("model output does not contain gt_masks_bev")
        gt = np.asarray(result["gt_masks_bev"], dtype=np.bool_)
        gt_path = args.out_dir / f"{token}_gt.png"
        save_map(gt_path, gt, class_names, args.render_axis, render_classes)
        print(f"ground_truth={gt_path.resolve()}")


if __name__ == "__main__":
    main()
