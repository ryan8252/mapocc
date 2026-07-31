#!/usr/bin/env python3
"""Build an evidence-oriented report from ProtoOcc ``work_dirs`` results.

The workspace contains several generations of ``result*.md`` files.  Some are
structured reports, some are pasted evaluator output, and many contain more
than one checkpoint evaluation.  This script keeps those evaluations separate
instead of combining the last OCC number with the last map number in a file.

The generated report deliberately separates protocols that should not be
ranked together (full vs. 1/4-data smoke, joint vs. map-only, and native vs.
BEVFusion-aligned geometry).  A canonical row is selected without looking at
the score: prefer a complete evaluation, then the highest epoch, then EMA, and
finally the later rerun.  All non-canonical evaluations remain in an appendix.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from summarize_work_dirs_results import (
        FLOAT,
        MAP_CLASSES,
        THIN_CLASSES,
        first_float,
        normalize_map_iou,
        parse_field,
        parse_raw_map_metrics,
        parse_summary_metrics,
        parse_table_section,
    )
except ModuleNotFoundError:  # Support ``python -m tools.analysis_tools...``.
    from tools.analysis_tools.summarize_work_dirs_results import (
        FLOAT,
        MAP_CLASSES,
        THIN_CLASSES,
        first_float,
        normalize_map_iou,
        parse_field,
        parse_raw_map_metrics,
        parse_summary_metrics,
        parse_table_section,
    )


AREA_CLASSES = ("drivable_area", "walkway", "carpark_area")
CONFIG_RE = re.compile(
    r"(?:\./)?(projects/configs/[A-Za-z0-9_./+-]+\.py)", re.IGNORECASE
)
CHECKPOINT_RE = re.compile(
    r"(?:[A-Za-z0-9_./+-]*/)?(epoch_[A-Za-z0-9_.+-]*\.pth)", re.IGNORECASE
)
RAW_OCC_RE = re.compile(
    rf"===>\s*mIoU\s+of\s+\d+\s+samples\s*:\s*({FLOAT})", re.IGNORECASE
)
SUMMARY_OCC_RE = re.compile(
    rf"^\s*(?:[-*]\s*)?OCC\s+mIoU\s*:\s*`?\**\s*({FLOAT}|n/a|not available)",
    re.IGNORECASE | re.MULTILINE,
)
OCC_CLASS_RE = re.compile(
    rf"^===>\s+(.+?)\s+-\s+IoU\s*=\s*({FLOAT})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
CHECKPOINT_HEADING_RE = re.compile(r"^#{1,6}\s+.*\.pth\b", re.IGNORECASE)
OUTPUT_HEADING_RE = re.compile(
    r"^#{1,6}\s+(final(?:\s+output)?|coarse(?:\s+output)?)\s*$",
    re.IGNORECASE,
)
RERUN_HEADING_RE = re.compile(
    r"^#{1,6}\s+(?:re-?run|re-?eval|rerun)\b", re.IGNORECASE
)


@dataclass
class TextBlock:
    text: str
    start_line: int
    start_offset: int


@dataclass
class EvalRecord:
    source_path: Path
    rel_source: str
    work_dir_rel: str
    work_dir_name: str
    block_index: int
    source_line: int
    config: str = ""
    config_source: str = "missing"
    checkpoint: str = ""
    epoch: int | None = None
    ema: bool = False
    output_variant: str = "default"
    protocol: str = "full"
    geometry: str = "native"
    task_mode: str = "unknown"
    train_schedule: str = ""
    gpus: int | None = None
    samples_per_gpu: int | None = None
    learning_rate: float | None = None
    map_loss_weight: float | None = None
    occ_miou: float | None = None
    map_mean: float | None = None
    map_metrics: dict[str, float | None] = field(default_factory=dict)
    thin_avg: float | None = None
    area_avg: float | None = None
    occ_classes: dict[str, float] = field(default_factory=dict)
    metric_source: str = "missing"
    structured_summary: bool = False
    method_tags: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)
    canonical: bool = False
    duplicate_of: str = ""

    @property
    def metric_count(self) -> int:
        return int(self.occ_miou is not None) + int(self.map_mean is not None)

    @property
    def complete(self) -> bool:
        if self.task_mode == "map-only":
            return self.map_mean is not None
        if self.task_mode == "occ-only":
            return self.occ_miou is not None
        if self.task_mode == "joint":
            return self.occ_miou is not None and self.map_mean is not None
        return self.metric_count > 0


def line_has_metric(line: str) -> bool:
    lower = line.lower()
    return (
        "map/mean/iou@max" in lower
        or "map mean iou@max" in lower
        or "occ miou:" in lower
        or "miou of " in lower
    )


def split_evaluation_blocks(text: str) -> list[TextBlock]:
    """Split a result file at checkpoint, rerun, command, and output boundaries."""

    lines = text.splitlines(keepends=True)
    if not lines:
        return [TextBlock("", 1, 0)]

    blocks: list[TextBlock] = []
    current: list[str] = []
    current_start_line = 1
    current_start_offset = 0
    current_has_metric = False
    offset = 0

    def flush() -> None:
        nonlocal current, current_has_metric
        if current and current_has_metric:
            blocks.append(
                TextBlock("".join(current), current_start_line, current_start_offset)
            )
        current = []
        current_has_metric = False

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        checkpoint_boundary = bool(CHECKPOINT_HEADING_RE.match(stripped))
        output_boundary = bool(OUTPUT_HEADING_RE.match(stripped))
        rerun_boundary = bool(RERUN_HEADING_RE.match(stripped))
        command_boundary = "dist_test.sh" in line and current_has_metric
        boundary = checkpoint_boundary or output_boundary or rerun_boundary or command_boundary

        if boundary:
            flush()
            current_start_line = line_number
            current_start_offset = offset

        current.append(line)
        current_has_metric = current_has_metric or line_has_metric(line)
        offset += len(line)

    flush()
    if not blocks:
        return [TextBlock(text, 1, 0)]
    return blocks


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    match = None
    for match in pattern.finditer(text):
        pass
    return match


def extract_config(context: str, source_path: Path) -> tuple[str, str]:
    matches = list(CONFIG_RE.finditer(context))
    if matches:
        return matches[-1].group(1), "eval-command"

    field_value = parse_field(context, "Config")
    if field_value:
        match = CONFIG_RE.search(field_value)
        return (match.group(1) if match else field_value.strip("`"), "metadata")

    candidates = sorted(source_path.parent.glob("*.py"))
    if len(candidates) == 1:
        return candidates[0].as_posix(), "saved-config"
    return "", "missing"


def extract_checkpoint(context: str) -> str:
    matches = list(CHECKPOINT_RE.finditer(context))
    if matches:
        return matches[-1].group(1)
    for field_name in ("Eval checkpoint", "Checkpoint"):
        value = parse_field(context, field_name)
        match = CHECKPOINT_RE.search(value)
        if match:
            return match.group(1)
    return ""


def checkpoint_epoch(checkpoint: str) -> int | None:
    match = re.search(r"epoch_(\d+)", checkpoint, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_output_variant(context: str) -> str:
    last_parent = -1
    last_variant: tuple[int, str] | None = None
    offset = 0
    for line in context.splitlines(keepends=True):
        stripped = line.strip()
        if CHECKPOINT_HEADING_RE.match(stripped) or "dist_test.sh" in line:
            last_parent = offset
        variant_match = OUTPUT_HEADING_RE.match(stripped)
        if variant_match:
            name = "final" if variant_match.group(1).lower().startswith("final") else "coarse"
            last_variant = (offset, name)
        offset += len(line)
    if last_variant is not None and last_variant[0] > last_parent:
        return last_variant[1]
    return "default"


def parse_occ(block: str) -> tuple[float | None, dict[str, float], str, list[str]]:
    warnings: list[str] = []
    raw_values = [float(match.group(1)) for match in RAW_OCC_RE.finditer(block)]
    summary_values: list[float] = []
    for match in SUMMARY_OCC_RE.finditer(block):
        raw = match.group(1)
        if raw.lower() not in ("n/a", "not available"):
            value = first_float(raw)
            if value is not None:
                summary_values.append(value)

    table = parse_table_section(block, "OCC IoU")
    table_value = table.get("miou")
    if raw_values:
        value, source = raw_values[-1], "raw-occ"
    elif summary_values:
        value, source = summary_values[-1], "summary-occ"
    elif table_value is not None:
        value, source = table_value, "table-occ"
    else:
        value, source = None, ""

    if raw_values and summary_values and abs(raw_values[-1] - summary_values[-1]) > 0.02:
        warnings.append(
            f"raw/summary OCC mismatch ({raw_values[-1]:.4f} vs {summary_values[-1]:.4f})"
        )

    classes: dict[str, float] = {}
    for match in OCC_CLASS_RE.finditer(block):
        classes[match.group(1).strip().lower().replace(" ", "_")] = float(match.group(2))
    return value, classes, source, warnings


def parse_map(
    block: str,
) -> tuple[dict[str, float | None], str, list[str]]:
    warnings: list[str] = []
    raw = parse_raw_map_metrics(block)
    for key in (*MAP_CLASSES, "mean"):
        raw_key = "map/mean/iou@max" if key == "mean" else f"map/{key}/iou@max"
        inline_pattern = re.compile(
            rf"^\s*(?:[-*]\s*)?(?:Raw\s+)?`?{re.escape(raw_key)}`?\s*:\s*`?({FLOAT})",
            re.IGNORECASE | re.MULTILINE,
        )
        match = _last_match(inline_pattern, block)
        if match is not None:
            raw[key] = normalize_map_iou(float(match.group(1)))
    summary = parse_summary_metrics(block)
    table_values = parse_any_map_table(block)

    metrics: dict[str, float | None] = {}
    for key in (*MAP_CLASSES, "mean"):
        if key in raw:
            metrics[key] = raw[key]
        elif key in summary:
            metrics[key] = summary[key]
        else:
            metrics[key] = table_values.get(key)

    if "mean" in raw:
        source = "raw-map"
    elif "mean" in summary:
        source = "summary-map"
    elif "mean" in table_values:
        source = "table-map"
    else:
        source = ""

    if "mean" in raw and "mean" in summary:
        if abs(raw["mean"] - summary["mean"]) > 0.0005:
            warnings.append(
                f"raw/summary Map mismatch ({raw['mean']:.6f} vs {summary['mean']:.6f})"
            )
    return metrics, source, warnings


def parse_any_map_table(text: str) -> dict[str, float]:
    """Return the last Markdown table that contains map-class rows and a mean."""

    candidates: list[dict[str, float]] = []
    current: list[str] = []

    def consume(lines: list[str]) -> None:
        if not lines:
            return
        rows: dict[str, float] = {}
        for line in lines:
            cells = [cell.strip().strip("`*") for cell in line.strip().strip("|").split("|")]
            if len(cells) < 2:
                continue
            key = cells[0].lower()
            if key not in {*MAP_CLASSES, "mean"}:
                continue
            value = first_float(cells[1])
            if value is not None:
                rows[key] = normalize_map_iou(value)
        class_count = sum(name in rows for name in MAP_CLASSES)
        if "mean" in rows and class_count >= 2:
            candidates.append(rows)

    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        else:
            consume(current)
            current = []
    consume(current)
    return candidates[-1] if candidates else {}


def infer_protocol(rel_source: str, context: str) -> str:
    lower = f"{rel_source} {context}".lower()
    work_dir_name = Path(rel_source).parent.name.lower()
    if (
        "1quarter" in lower
        or re.search(r"(?:^|_)smoke(?:_|$)", work_dir_name)
        or "quick_test" in lower
    ):
        return "1quarter-smoke"
    return "full-data"


def infer_geometry(rel_source: str, config: str) -> str:
    lower = f"{rel_source} {config}".lower()
    if "51m2" in lower or "51.2" in lower:
        return "51.2m"
    if "bevfusion_aligned" in lower:
        return "bevfusion-aligned"
    return "native"


def extract_eval_names(context: str) -> set[str]:
    matches = list(re.finditer(r"--eval\s+([^\n\r]+)", context, re.IGNORECASE))
    if not matches:
        return set()
    tail = matches[-1].group(1)
    names = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", tail):
        if token in {"miou", "map-miou"}:
            names.add(token)
        elif names:
            break
    return names


def infer_task_mode(
    rel_source: str,
    config: str,
    context: str,
    occ_miou: float | None,
    map_mean: float | None,
) -> str:
    lower = f"{rel_source} {config}".lower()
    if "map_only" in lower or "map-only" in lower:
        return "map-only"
    eval_names = extract_eval_names(context)
    if eval_names == {"map-miou"}:
        return "map-only"
    if eval_names == {"miou"}:
        return "occ-only"
    if {"miou", "map-miou"}.issubset(eval_names):
        return "joint"
    if occ_miou is not None and map_mean is not None:
        return "joint"
    if map_mean is not None:
        return "map-only"
    if occ_miou is not None:
        return "occ-only"
    return "unknown"


def extract_training_metadata(text: str) -> tuple[str, int | None, int | None, float | None]:
    schedule = parse_field(text, "Train schedule")
    samples_matches = re.findall(r"samples_per_gpu\s*=\s*(\d+)", text, re.IGNORECASE)
    samples = int(samples_matches[-1]) if samples_matches else None

    gpu_matches = re.findall(
        r"dist_train\.sh\s+\S+\.py\s+(\d+)(?:\s|$)", text, re.IGNORECASE
    )
    gpus = int(gpu_matches[-1]) if gpu_matches else None

    lr_matches = re.findall(
        rf"(?:optimizer\.lr|\blr)\s*=\s*({FLOAT})", text, re.IGNORECASE
    )
    learning_rate = float(lr_matches[-1]) if lr_matches else None
    return schedule, gpus, samples, learning_rate


def infer_map_loss_weight(rel_source: str, checkpoint: str, text: str) -> float | None:
    matches = re.findall(rf"map_loss_weight\s*=\s*({FLOAT})", text, re.IGNORECASE)
    if matches:
        return float(matches[-1])
    progressive = re.search(r"mapw(\d+(?:\.\d+)?)to(\d+(?:\.\d+)?)", rel_source, re.IGNORECASE)
    if progressive:
        return float(progressive.group(2))
    lower = f"{rel_source} {checkpoint}".lower()
    patterns = [
        r"mapw(\d+(?:\.\d+)?)",
        r"(?:^|[_/-])weight[_-](\d+(?:\.\d+)?)(?:[_./-]|$)",
        r"(?:^|[_/-])w(\d+(?:\.\d+)?)(?:[_./-]|$)",
    ]
    if "focal_weight" not in lower:
        patterns.append(r"(?:^|[_/-])weight(\d+(?:\.\d+)?)(?:[_./-]|$)")
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            return float(match.group(1))
    return None


def infer_method_tags(rel_source: str, config: str) -> tuple[str, ...]:
    lower = f"{rel_source} {config}".lower()
    tags: list[str] = []

    def add(tag: str, *needles: str) -> None:
        if any(needle in lower for needle in needles):
            tags.append(tag)

    add("MAESTRO", "maestro")
    add("map-only", "map_only", "map-only")
    add("VGMR/Map-HFM", "map_hfm", "_hfm", "additive_hfm")
    add("adapter", "adapter")
    add("FPN-ASPP", "fpn_lateral_aspp")
    add("weighted-FD", "focal_dice_weighted")
    add("active-gate", "active_enhance_gate", "active_gate")
    add("prototype/TSFG", "proto_map", "query_tsfg", "cfv_proto", "tsfg_supp")
    add("LSS/Z-aware", "lss2d", "zaware", "zsum", "heightattn", "catz", "zlite")
    add("loss/schedule", "mapw", "weight_4", "progressive", "uncertainty_weighting")
    add("thin-aware", "thin_boundary", "thin_roi", "dual_area_line", "residual_gates")
    add("alternative", "vami", "overlay_dynamic", "tgde", "lgmg", "occ2map")
    add("debug", "debug", "probe", "parity")
    if not tags:
        tags.append("baseline/other")
    return tuple(dict.fromkeys(tags))


def parse_result_file(source_path: Path, work_dirs: Path) -> list[EvalRecord]:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    rel_source = source_path.relative_to(work_dirs).as_posix()
    work_dir_rel = source_path.parent.relative_to(work_dirs).as_posix()
    schedule, gpus, samples, learning_rate = extract_training_metadata(text)
    blocks = split_evaluation_blocks(text)
    records: list[EvalRecord] = []

    for index, block in enumerate(blocks, start=1):
        context_end = block.start_offset + len(block.text)
        context = text[:context_end]
        config, config_source = extract_config(context, source_path)
        checkpoint = extract_checkpoint(context)
        occ_miou, occ_classes, occ_source, occ_warnings = parse_occ(block.text)
        map_metrics, map_source, map_warnings = parse_map(block.text)
        map_mean = map_metrics.get("mean")
        thin_values = [map_metrics.get(name) for name in THIN_CLASSES]
        area_values = [map_metrics.get(name) for name in AREA_CLASSES]
        thin_avg = (
            sum(value for value in thin_values if value is not None) / len(THIN_CLASSES)
            if all(value is not None for value in thin_values)
            else None
        )
        area_avg = (
            sum(value for value in area_values if value is not None) / len(AREA_CLASSES)
            if all(value is not None for value in area_values)
            else None
        )
        task_mode = infer_task_mode(rel_source, config, context, occ_miou, map_mean)
        output_variant = extract_output_variant(context)
        warnings = occ_warnings + map_warnings
        if config_source == "missing":
            warnings.append("config path missing")
        if not checkpoint:
            warnings.append("checkpoint missing")
        if occ_miou is None and map_mean is None:
            warnings.append("no parsed evaluation metric")

        metric_sources = "+".join(source for source in (occ_source, map_source) if source)
        record = EvalRecord(
            source_path=source_path,
            rel_source=rel_source,
            work_dir_rel=work_dir_rel,
            work_dir_name=source_path.parent.name,
            block_index=index,
            source_line=block.start_line,
            config=config,
            config_source=config_source,
            checkpoint=checkpoint,
            epoch=checkpoint_epoch(checkpoint),
            ema="_ema" in checkpoint.lower(),
            output_variant=output_variant,
            protocol=infer_protocol(rel_source, text),
            geometry=infer_geometry(rel_source, config),
            task_mode=task_mode,
            train_schedule=schedule,
            gpus=gpus,
            samples_per_gpu=samples,
            learning_rate=learning_rate,
            map_loss_weight=infer_map_loss_weight(rel_source, checkpoint, context),
            occ_miou=occ_miou,
            map_mean=map_mean,
            map_metrics=map_metrics,
            thin_avg=thin_avg,
            area_avg=area_avg,
            occ_classes=occ_classes,
            metric_source=metric_sources or "missing",
            structured_summary="## Summary" in block.text,
            method_tags=infer_method_tags(rel_source, config),
            warnings=warnings,
        )
        records.append(record)
    return records


def canonicalize(records: list[EvalRecord]) -> None:
    by_source_variant: dict[tuple[str, str], list[EvalRecord]] = defaultdict(list)
    for record in records:
        by_source_variant[(record.rel_source, record.output_variant)].append(record)

    for candidates in by_source_variant.values():
        config_counts = Counter(record.config for record in candidates if record.config)
        dominant_config = config_counts.most_common(1)[0][0] if config_counts else ""

        def key(record: EvalRecord) -> tuple[int, int, int, int, int, int, int]:
            return (
                int(not dominant_config or record.config == dominant_config),
                int(record.complete),
                record.epoch if record.epoch is not None else -1,
                int(record.ema),
                record.metric_count,
                int(record.structured_summary),
                record.block_index,
            )

        max(candidates, key=key).canonical = True


def mark_duplicates(records: list[EvalRecord]) -> None:
    seen: dict[tuple[object, ...], EvalRecord] = {}
    for record in sorted(
        records,
        key=lambda item: (item.source_path.name != "result.md", item.rel_source),
    ):
        key = (
            record.work_dir_rel,
            Path(record.config).name,
            record.checkpoint,
            record.output_variant,
            None if record.occ_miou is None else round(record.occ_miou, 6),
            None if record.map_mean is None else round(record.map_mean, 9),
        )
        if record.metric_count == 0:
            continue
        if key in seen:
            record.duplicate_of = seen[key].rel_source
            record.canonical = False
        else:
            seen[key] = record


def scan_results(work_dirs: Path, include_variants: bool) -> list[EvalRecord]:
    if include_variants:
        paths = sorted(path for path in work_dirs.rglob("result*.md") if path.is_file())
    else:
        paths = sorted(path for path in work_dirs.rglob("result.md") if path.is_file())

    records: list[EvalRecord] = []
    for path in paths:
        records.extend(parse_result_file(path, work_dirs))
    canonicalize(records)
    mark_duplicates(records)
    return records


def score_key(record: EvalRecord) -> tuple[float, float, str]:
    return (
        -(record.map_mean if record.map_mean is not None else -1.0),
        -(record.occ_miou if record.occ_miou is not None else -1.0),
        record.work_dir_name,
    )


def report_section(record: EvalRecord) -> str:
    lower = f"{record.rel_source} {' '.join(record.method_tags)}".lower()
    if "debug" in record.method_tags or record.metric_count == 0:
        return "Diagnostics and incomplete artifacts"
    if "maestro" in lower:
        return "MAESTRO reproductions"
    if record.protocol == "1quarter-smoke":
        if record.geometry == "bevfusion-aligned":
            return "1/4-data smoke - BEVFusion-aligned"
        return "1/4-data smoke - native/other"
    if record.geometry == "bevfusion-aligned":
        return "Full-data - BEVFusion-aligned"
    if record.geometry == "51.2m":
        return "Full-data - 51.2m range"
    if record.task_mode == "map-only":
        return "Full-data - native map-only"
    if record.task_mode == "occ-only":
        return "Full-data - native OCC-only"
    return "Full-data - native joint"


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}"


def occ(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def epoch_label(record: EvalRecord) -> str:
    if record.epoch is None:
        return "n/a"
    return f"{record.epoch}{' EMA' if record.ema else ''}"


def protocol_fingerprint(record: EvalRecord) -> str:
    parts = [record.protocol, record.geometry, record.task_mode, epoch_label(record)]
    if record.gpus is not None and record.samples_per_gpu is not None:
        parts.append(f"{record.gpus}g×{record.samples_per_gpu}")
    if record.learning_rate is not None:
        parts.append(f"lr={record.learning_rate:g}")
    if record.map_loss_weight is not None:
        parts.append(f"mapw={record.map_loss_weight:g}")
    return ", ".join(parts)


def md_link(record: EvalRecord) -> str:
    label = record.work_dir_name
    if record.source_path.name != "result.md":
        label += f"/{record.source_path.name}"
    suffix = "" if record.output_variant == "default" else f" [{record.output_variant}]"
    target = record.rel_source.replace(" ", "%20")
    return f"[`{label}`]({target}){suffix}"


def config_label(record: EvalRecord) -> str:
    if not record.config:
        return "n/a"
    return f"`{Path(record.config).name}`"


def markdown_table(records: list[EvalRecord]) -> list[str]:
    lines = [
        "| Rank | Map | OCC | Thin | Area | Protocol | Epoch | Task | Tags | Run |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for rank, record in enumerate(sorted(records, key=score_key), start=1):
        lines.append(
            "| "
            f"{rank} | {percent(record.map_mean)} | {occ(record.occ_miou)} | "
            f"{percent(record.thin_avg)} | {percent(record.area_avg)} | "
            f"{record.protocol}/{record.geometry} | {epoch_label(record)} | {record.task_mode} | "
            f"{', '.join(record.method_tags)} | {md_link(record)} |"
        )
    return lines


def highest_checkpoint_epoch(work_dir: Path) -> int | None:
    epochs = []
    for checkpoint in work_dir.glob("epoch*.pth"):
        epoch = checkpoint_epoch(checkpoint.name)
        if epoch is not None:
            epochs.append(epoch)
    return max(epochs) if epochs else None


def artifact_warnings(records: list[EvalRecord], work_dirs: Path) -> list[str]:
    warnings: list[str] = []
    canonical = [record for record in records if record.canonical and not record.duplicate_of]
    by_work_dir: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in canonical:
        by_work_dir[record.work_dir_rel].append(record)

    for work_dir_rel, candidates in sorted(by_work_dir.items()):
        disk_epoch = highest_checkpoint_epoch(work_dirs / work_dir_rel)
        evaluated_epochs = [record.epoch for record in candidates if record.epoch is not None]
        if disk_epoch is not None and evaluated_epochs and disk_epoch > max(evaluated_epochs):
            warnings.append(
                f"`{work_dir_rel}` has checkpoint epoch {disk_epoch}, but the highest canonical "
                f"evaluation recorded in result files is epoch {max(evaluated_epochs)}."
            )

    if not any(record.task_mode == "occ-only" and record.complete for record in canonical):
        warnings.append(
            "No complete OCC-only baseline evaluation was found under `work_dirs`; an OCC gain "
            "from map supervision cannot be established from this inventory alone."
        )
    return warnings


def write_markdown(records: list[EvalRecord], work_dirs: Path, output: Path) -> None:
    canonical = [
        record for record in records if record.canonical and not record.duplicate_of
    ]
    sections: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in canonical:
        sections[report_section(record)].append(record)

    source_files = {record.rel_source for record in records}
    standard_files = {name for name in source_files if Path(name).name == "result.md"}
    variant_files = source_files - standard_files
    metric_records = [record for record in records if record.metric_count]
    extra_records = [
        record
        for record in records
        if not record.canonical or record.duplicate_of
    ]

    lines = [
        "# ProtoOcc Work-Dirs Experiment Inventory",
        "",
        f"Generated: `{datetime.now().astimezone().isoformat(timespec='minutes')}`",
        "",
        "## Reading rules",
        "",
        "- Source of truth: local `result*.md` evaluator artifacts under this `work_dirs` tree.",
        "- Map, Thin, and Area values are displayed as percentages; OCC keeps the evaluator's percentage scale.",
        "- Canonical selection never uses the best score: complete evaluation -> highest epoch -> EMA -> later rerun.",
        "- Full-data, 1/4-data smoke, map-only, joint, native, 51.2m, and BEVFusion-aligned runs are separated.",
        "- A row in the same section is not automatically a controlled ablation; check its protocol fingerprint and config.",
        "- Raw evaluator metrics take priority over generated summaries and Markdown tables.",
        "",
        "## Coverage",
        "",
        f"- Standard `result.md` files: `{len(standard_files)}`",
        f"- Additional result variants: `{len(variant_files)}`",
        f"- Parsed evaluation/output blocks with metrics: `{len(metric_records)}`",
        f"- Canonical rows: `{len(canonical)}`",
        f"- Non-canonical or duplicate rows retained in history: `{len(extra_records)}`",
        "",
        "## Canonical results by comparable protocol",
        "",
    ]

    section_order = (
        "Full-data - native joint",
        "Full-data - native map-only",
        "Full-data - native OCC-only",
        "Full-data - BEVFusion-aligned",
        "Full-data - 51.2m range",
        "MAESTRO reproductions",
        "1/4-data smoke - native/other",
        "1/4-data smoke - BEVFusion-aligned",
        "Diagnostics and incomplete artifacts",
    )
    for title in section_order:
        section_records = sections.get(title, [])
        if not section_records:
            continue
        lines.extend([f"### {title}", ""])
        lines.extend(markdown_table(section_records))
        lines.append("")

    lines.extend(["## Artifact coverage warnings", ""])
    warnings = artifact_warnings(records, work_dirs)
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None detected.")
    lines.append("")

    parse_warnings = [
        record
        for record in canonical
        if record.warnings or record.config_source in {"missing", "saved-config"}
    ]
    lines.extend(["## Canonical-row provenance and parser warnings", ""])
    if not parse_warnings:
        lines.append("- None.")
    else:
        lines.extend(
            [
                "| Run | Config | Checkpoint | Fingerprint | Metric source | Warning |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for record in sorted(parse_warnings, key=lambda item: item.rel_source):
            warning_text = "; ".join(record.warnings) or f"config source={record.config_source}"
            lines.append(
                f"| {md_link(record)} | {config_label(record)} | "
                f"`{record.checkpoint or 'n/a'}` | {protocol_fingerprint(record)} | "
                f"{record.metric_source} | {warning_text} |"
            )
    lines.append("")

    lines.extend(["## Additional checkpoint/output history", ""])
    if not extra_records:
        lines.append("- None.")
    else:
        lines.extend(
            [
                "| Source | Line | Checkpoint | Output | Map | OCC | Thin | Config | Status |",
                "| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for record in sorted(
            extra_records,
            key=lambda item: (item.rel_source, item.output_variant, item.epoch or -1, item.block_index),
        ):
            status = f"duplicate of `{record.duplicate_of}`" if record.duplicate_of else "non-canonical"
            lines.append(
                f"| [`{record.rel_source}`]({record.rel_source.replace(' ', '%20')}) | "
                f"{record.source_line} | `{record.checkpoint or 'n/a'}` | "
                f"{record.output_variant} | {percent(record.map_mean)} | {occ(record.occ_miou)} | "
                f"{percent(record.thin_avg)} | {config_label(record)} | {status} |"
            )
    lines.append("")

    lines.extend(["## Complete canonical inventory", ""])
    lines.extend(
        [
            "| Run | Map | OCC | Thin | Area | Protocol fingerprint | Config | Source |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for record in sorted(canonical, key=lambda item: (report_section(item), score_key(item))):
        lines.append(
            f"| {md_link(record)} | {percent(record.map_mean)} | {occ(record.occ_miou)} | "
            f"{percent(record.thin_avg)} | {percent(record.area_avg)} | "
            f"{protocol_fingerprint(record)} | {config_label(record)} | "
            f"`{record.rel_source}:{record.source_line}` |"
        )
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def write_csv(records: list[EvalRecord], output: Path) -> None:
    columns = [
        "canonical",
        "duplicate_of",
        "source_file",
        "source_line",
        "work_dir",
        "config",
        "config_source",
        "checkpoint",
        "epoch",
        "ema",
        "output_variant",
        "protocol",
        "geometry",
        "task_mode",
        "gpus",
        "samples_per_gpu",
        "learning_rate",
        "map_loss_weight",
        "occ_miou",
        "map_mean_raw",
        "map_mean_percent",
        "thin_avg_raw",
        "thin_avg_percent",
        "area_avg_raw",
        "area_avg_percent",
        *MAP_CLASSES,
        "method_tags",
        "metric_source",
        "structured_summary",
        "warnings",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {
                "canonical": int(record.canonical),
                "duplicate_of": record.duplicate_of,
                "source_file": record.rel_source,
                "source_line": record.source_line,
                "work_dir": record.work_dir_rel,
                "config": record.config,
                "config_source": record.config_source,
                "checkpoint": record.checkpoint,
                "epoch": record.epoch,
                "ema": int(record.ema),
                "output_variant": record.output_variant,
                "protocol": record.protocol,
                "geometry": record.geometry,
                "task_mode": record.task_mode,
                "gpus": record.gpus,
                "samples_per_gpu": record.samples_per_gpu,
                "learning_rate": record.learning_rate,
                "map_loss_weight": record.map_loss_weight,
                "occ_miou": record.occ_miou,
                "map_mean_raw": record.map_mean,
                "map_mean_percent": None if record.map_mean is None else 100 * record.map_mean,
                "thin_avg_raw": record.thin_avg,
                "thin_avg_percent": None if record.thin_avg is None else 100 * record.thin_avg,
                "area_avg_raw": record.area_avg,
                "area_avg_percent": None if record.area_avg is None else 100 * record.area_avg,
                "method_tags": ";".join(record.method_tags),
                "metric_source": record.metric_source,
                "structured_summary": int(record.structured_summary),
                "warnings": "; ".join(record.warnings),
            }
            row.update({name: record.map_metrics.get(name) for name in MAP_CLASSES})
            writer.writerow(row)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build a protocol-aware inventory from ProtoOcc work_dirs result files."
    )
    parser.add_argument(
        "--work-dirs",
        type=Path,
        default=repo_root / "work_dirs",
        help="ProtoOcc work_dirs root (default: repository work_dirs).",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Markdown output (default: WORK_DIRS/experiment_results_overview.md).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional machine-readable CSV containing every evaluation block.",
    )
    parser.add_argument(
        "--include-variants",
        action="store_true",
        help="Also scan result_2.md, result_4.md, result copy.md, and similar files.",
    )
    args = parser.parse_args()

    work_dirs = args.work_dirs.resolve()
    md_out = args.md_out or work_dirs / "experiment_results_overview.md"
    records = scan_results(work_dirs, include_variants=args.include_variants)
    write_markdown(records, work_dirs, md_out)
    if args.csv_out is not None:
        write_csv(records, args.csv_out)

    canonical = sum(record.canonical and not record.duplicate_of for record in records)
    sources = len({record.rel_source for record in records})
    print(f"Scanned {sources} result files")
    print(f"Parsed {sum(record.metric_count > 0 for record in records)} metric blocks")
    print(f"Selected {canonical} canonical rows without score-based cherry-picking")
    print(f"Wrote Markdown: {md_out}")
    if args.csv_out is not None:
        print(f"Wrote CSV: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
