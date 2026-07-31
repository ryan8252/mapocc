#!/usr/bin/env python
"""Probe whether model predictions follow ProtoOcc or BEVFusion map GT axes.

For one validation sample this script writes:

- current ProtoOcc GT masks from the active dataset pipeline
- BEVFusion-exact GT masks rasterized from the same sample pose
- model prediction sigmoid maps
- aligned map-feature activation norm
- per-class correlation / threshold IoU tables
- PNG visualizations for selected classes
"""

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

try:
    from mmdet.utils import compat_cfg, setup_multi_processes
except ImportError:
    from mmdet3d.utils import compat_cfg, setup_multi_processes

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from debug_compare_bevfusion_map_gt import (  # noqa: E402
    build_scene_location_lookup,
    parse_bound,
    rasterize_map,
    resolve_location,
)


DEFAULT_CLASSES = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_work_dir = (
        repo_root
        / "work_dirs/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only_nano4_h200_2"
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root
        / "projects/configs/ProtoOcc/ProtoOcc_multi_cnn_head_map_neck_bevfusion_aligned_map_only.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_work_dir / "epoch_24_ema.pth",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=repo_root / "data/nuscenes")
    parser.add_argument("--nusc-version", default="v1.0-trainval")
    parser.add_argument("--xbound", nargs=3, type=float, default=(-50.0, 50.0, 0.5))
    parser.add_argument("--ybound", nargs=3, type=float, default=(-50.0, 50.0, 0.5))
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument(
        "--show-classes",
        nargs="+",
        default=["drivable_area", "walkway", "divider"],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "work_dirs/debug_feature_convention_probe",
    )
    return parser.parse_args()


def import_plugins(cfg: Config, config_path: Path) -> None:
    if not getattr(cfg, "plugin", False):
        return
    if hasattr(cfg, "plugin_dir"):
        module_dir = cfg.plugin_dir
    else:
        try:
            module_dir = str(config_path.parent.resolve().relative_to(REPO_ROOT))
        except ValueError:
            module_dir = str(config_path.parent)
    module_dir = module_dir.strip("/.")
    module_path_parts = module_dir.split("/")
    module_path = module_path_parts[0]
    for part in module_path_parts[1:]:
        module_path += "." + part
    importlib.import_module(module_path)


def load_cfg(config_path: Path) -> Config:
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


def build_one_sample_loader(cfg: Config):
    dataset = build_dataset(cfg.data.test)
    loader_cfg = dict(samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    loader_cfg.update(cfg.data.get("test_dataloader", {}))
    loader_cfg["samples_per_gpu"] = 1
    loader_cfg["workers_per_gpu"] = 0
    data_loader = build_dataloader(dataset, **loader_cfg)
    return dataset, data_loader


def take_sample(data_loader, sample_index: int):
    for idx, data in enumerate(data_loader):
        if idx == sample_index:
            return data
    raise IndexError(f"sample index out of range: {sample_index}")


def build_probe_model(cfg: Config, checkpoint: Path, device: str, capture: Dict[str, torch.Tensor]):
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint_data = load_checkpoint(model, str(checkpoint), map_location="cpu")

    def capture_input(module, inputs):
        feature = inputs[0]
        if isinstance(feature, dict):
            feature = feature["bev_feature"]
        capture["map_feature"] = feature.detach().float().cpu()

    def capture_output(module, inputs, output):
        logits = output["bev_seg_logits"] if isinstance(output, dict) else output
        capture["logits"] = logits.detach().float().cpu()

    model.bev_seg_head.register_forward_pre_hook(capture_input)
    model.bev_seg_head.register_forward_hook(capture_output)

    if "CLASSES" in checkpoint_data.get("meta", {}):
        model.CLASSES = checkpoint_data["meta"]["CLASSES"]

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    if device.startswith("cuda"):
        device_id = int(device.split(":", 1)[1]) if ":" in device else 0
        model = MMDataParallel(model.cuda(device_id), device_ids=[device_id])
    else:
        model = MMDataParallel(model, device_ids=[])
    model.eval()
    return model


def sigmoid_prediction(output, capture: Dict[str, torch.Tensor]) -> np.ndarray:
    if "logits" in capture:
        return torch.sigmoid(capture["logits"])[0].numpy()
    sample = output[0]
    return sample["masks_bev"].astype(np.float32)


def feature_norm_map(capture: Dict[str, torch.Tensor]) -> np.ndarray:
    feature = capture["map_feature"][0]
    norm = torch.linalg.vector_norm(feature, ord=2, dim=0).numpy()
    if np.nanmax(norm) > np.nanmin(norm):
        norm = (norm - np.nanmin(norm)) / (np.nanmax(norm) - np.nanmin(norm))
    return norm.astype(np.float32)


def threshold_iou(pred: np.ndarray, label: np.ndarray, threshold: float) -> float:
    pred_mask = pred >= threshold
    label_mask = label.astype(bool)
    union = np.logical_or(pred_mask, label_mask).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred_mask, label_mask).sum() / union)


def best_iou(pred: np.ndarray, label: np.ndarray) -> float:
    values = [
        threshold_iou(pred, label, threshold)
        for threshold in np.arange(0.05, 0.951, 0.05)
    ]
    valid = [value for value in values if not np.isnan(value)]
    return max(valid) if valid else float("nan")


def pearson(pred: np.ndarray, label: np.ndarray) -> float:
    pred_flat = pred.reshape(-1).astype(np.float64)
    label_flat = label.reshape(-1).astype(np.float64)
    if pred_flat.std() == 0 or label_flat.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred_flat, label_flat)[0, 1])


def mean_pos_neg_gap(pred: np.ndarray, label: np.ndarray) -> float:
    label_bool = label.astype(bool)
    if not label_bool.any() or label_bool.all():
        return float("nan")
    return float(pred[label_bool].mean() - pred[~label_bool].mean())


def metric_rows(
    pred: np.ndarray,
    labels: Dict[str, np.ndarray],
    class_names: Sequence[str],
    show_classes: Iterable[str],
) -> Sequence[Sequence[str]]:
    rows = []
    for class_name in show_classes:
        class_index = class_names.index(class_name)
        for label_name, label in labels.items():
            pred_cls = pred[class_index]
            label_cls = label[class_index]
            rows.append(
                [
                    class_name,
                    label_name,
                    f"{pearson(pred_cls, label_cls):.6f}",
                    f"{best_iou(pred_cls, label_cls):.6f}",
                    f"{threshold_iou(pred_cls, label_cls, 0.5):.6f}",
                    f"{mean_pos_neg_gap(pred_cls, label_cls):.6f}",
                    f"{label_cls.mean():.6f}",
                    f"{pred_cls.mean():.6f}",
                ]
            )
    return rows


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def save_class_figure(
    class_name: str,
    class_index: int,
    pred: np.ndarray,
    labels: Dict[str, np.ndarray],
    feature_norm: np.ndarray,
    out_path: Path,
) -> None:
    panels = [
        ("current ProtoOcc GT", labels["current_protoocc"][class_index], "gray", 0.0, 1.0),
        ("BEVFusion-exact GT", labels["bevfusion_exact"][class_index], "gray", 0.0, 1.0),
        (
            "BEVFusion axes + Proto layers GT",
            labels["bevfusion_axes_proto_layers"][class_index],
            "gray",
            0.0,
            1.0,
        ),
        ("model pred sigmoid", pred[class_index], "viridis", 0.0, 1.0),
        ("map feature L2 norm", feature_norm, "magma", 0.0, 1.0),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for axis, (title, image, cmap, vmin, vmax) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(class_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_feature_figure(feature_norm: np.ndarray, out_path: Path) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(5, 5))
    axis.imshow(feature_norm, cmap="magma", vmin=0.0, vmax=1.0, origin="upper")
    axis.set_title("Aligned map feature L2 norm")
    axis.set_xticks([])
    axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed, deterministic=True)

    cfg = load_cfg(args.config)
    dataset, data_loader = build_one_sample_loader(cfg)
    data = take_sample(data_loader, args.sample_index)

    capture: Dict[str, torch.Tensor] = {}
    model = build_probe_model(cfg, args.checkpoint, args.device, capture)

    with torch.no_grad():
        output = model(return_loss=False, rescale=True, **data)

    pred = sigmoid_prediction(output, capture)
    feature_norm = feature_norm_map(capture)
    current_label = output[0]["gt_masks_bev"].astype(np.uint8)

    info = dataset.data_infos[args.sample_index]
    scene_lookup = None
    if not info.get("location"):
        scene_lookup = build_scene_location_lookup(args.data_root, args.nusc_version)
    location = resolve_location(info, scene_lookup)
    from nuscenes.map_expansion.map_api import NuScenesMap

    nusc_map = NuScenesMap(str(args.data_root), location)
    xbound = parse_bound(args.xbound)
    ybound = parse_bound(args.ybound)
    class_names = tuple(args.classes)
    bevfusion_exact = rasterize_map(
        info,
        nusc_map,
        class_names,
        xbound,
        ybound,
        mapping_mode="bevfusion",
        axis_mode="bevfusion",
    )
    bevfusion_axes_proto_layers = rasterize_map(
        info,
        nusc_map,
        class_names,
        xbound,
        ybound,
        mapping_mode="protoocc",
        axis_mode="bevfusion",
    )

    labels = {
        "current_protoocc": current_label,
        "bevfusion_exact": bevfusion_exact,
        "bevfusion_axes_proto_layers": bevfusion_axes_proto_layers,
    }

    image_paths = []
    for class_name in args.show_classes:
        class_index = class_names.index(class_name)
        image_path = args.out_dir / f"{class_name}.png"
        save_class_figure(
            class_name,
            class_index,
            pred,
            labels,
            feature_norm,
            image_path,
        )
        image_paths.append(image_path)
    feature_path = args.out_dir / "feature_norm.png"
    save_feature_figure(feature_norm, feature_path)

    rows = metric_rows(pred, labels, class_names, args.show_classes)
    result_lines = [
        "# Feature Convention Probe",
        "",
        f"- config: `{args.config}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- sample index: `{args.sample_index}`",
        f"- token: `{info.get('token')}`",
        f"- scene: `{info.get('scene_name', info.get('scene_token'))}`",
        f"- location: `{location}`",
        f"- output dir: `{args.out_dir}`",
        "",
        "## Metrics",
        "",
        markdown_table(
            [
                "class",
                "label",
                "pearson",
                "best_iou",
                "iou@0.5",
                "pos_neg_gap",
                "label_pos_ratio",
                "pred_mean",
            ],
            rows,
        ),
        "",
        "## Images",
        "",
    ]
    for image_path in image_paths:
        result_lines.append(f"- [{image_path.name}]({image_path.name})")
    result_lines.append(f"- [{feature_path.name}]({feature_path.name})")
    result_lines.append("")
    result_lines.append("## Interpretation")
    result_lines.append("")
    result_lines.append(
        "- If prediction metrics are consistently better against "
        "`current_protoocc`, the trained model is following the current "
        "ProtoOcc GT convention."
    )
    result_lines.append(
        "- If prediction metrics are consistently better against "
        "`bevfusion_exact`, the model output convention is closer to "
        "BEVFusion's tensor convention."
    )
    result_lines.append(
        "- If only `drivable_area` changes between `bevfusion_exact` and "
        "`bevfusion_axes_proto_layers`, that is the drivable-area layer "
        "definition mismatch rather than an axis issue."
    )

    result_path = args.out_dir / "result.md"
    result_path.write_text("\n".join(result_lines) + "\n")
    print(result_path.read_text())
    print(f"[WROTE] {result_path}")


if __name__ == "__main__":
    main()
