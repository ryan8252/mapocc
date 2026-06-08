#!/usr/bin/env python3
"""Summarize ProtoOcc work_dirs result.md metrics.

The parser is intentionally tolerant because older result.md files in this
workspace mix summary bullets, Markdown tables, and raw eval dictionaries.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MAP_CLASSES = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)
THIN_CLASSES = ("ped_crossing", "stop_line", "divider")
TABLE_CLEAN_RE = re.compile(r"[`*]")


def clean_cell(text: str) -> str:
    return TABLE_CLEAN_RE.sub("", text).strip()


def first_float(text: str) -> float | None:
    match = re.search(FLOAT, text)
    return float(match.group(0)) if match else None


def normalize_map_iou(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 100.0 if value > 1.0 else value


def last_match_float(text: str, pattern: str, flags: int = re.IGNORECASE) -> float | None:
    value = None
    for match in re.finditer(pattern, text, flags):
        raw = match.group(1)
        if raw.lower() == "n/a":
            continue
        parsed = first_float(raw)
        if parsed is not None:
            value = parsed
    return value


def parse_field(text: str, field: str) -> str:
    pattern = rf"^\s*-\s*{re.escape(field)}\s*:\s*(.+?)\s*$"
    value = ""
    for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
        value = clean_cell(match.group(1))
    return value


def iter_section_lines(lines: list[str], section_name: str):
    in_section = False
    section_re = re.compile(rf"^#+\s+{re.escape(section_name)}\b", re.IGNORECASE)
    heading_re = re.compile(r"^#+\s+")
    for line in lines:
        if section_re.match(line.strip()):
            in_section = True
            continue
        if in_section and heading_re.match(line.strip()):
            break
        if in_section:
            yield line


def parse_table_section(text: str, section_name: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    for line in iter_section_lines(text.splitlines(), section_name):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [clean_cell(cell) for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        key = cells[0].lower()
        if key in ("item", "---", ""):
            continue
        value = first_float(cells[1])
        if value is not None:
            rows[key] = value
    return rows


def parse_raw_map_metrics(text: str) -> dict[str, float]:
    metrics = {}
    keys = list(MAP_CLASSES) + ["mean"]
    for key in keys:
        raw_key = "map/mean/iou@max" if key == "mean" else f"map/{key}/iou@max"
        pattern = rf"['\"]{re.escape(raw_key)}['\"]\s*:\s*({FLOAT})"
        value = last_match_float(text, pattern, flags=0)
        if value is not None:
            metrics[key] = normalize_map_iou(value)
    return metrics


def parse_summary_metrics(text: str) -> dict[str, float]:
    metrics = {}
    map_mean = last_match_float(
        text, rf"Map\s+mean\s+IoU@max\s*:\s*`?\**\s*({FLOAT}|n/a)"
    )
    if map_mean is not None:
        metrics["mean"] = normalize_map_iou(map_mean)

    thin_re = re.compile(
        rf"Thin\s+classes\s*:\s*`?"
        rf".*?ped_crossing\s*=\s*({FLOAT})"
        rf".*?stop_line\s*=\s*({FLOAT})"
        rf".*?divider\s*=\s*({FLOAT})",
        re.IGNORECASE,
    )
    for match in thin_re.finditer(text):
        metrics["ped_crossing"] = normalize_map_iou(float(match.group(1)))
        metrics["stop_line"] = normalize_map_iou(float(match.group(2)))
        metrics["divider"] = normalize_map_iou(float(match.group(3)))
    return metrics


def parse_map_metrics(text: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {key: None for key in MAP_CLASSES}
    metrics["mean"] = None

    for source in (parse_raw_map_metrics(text), parse_summary_metrics(text)):
        for key, value in source.items():
            metrics[key] = value

    table = parse_table_section(text, "Map IoU")
    table_key_map = {
        "drivable_area": "drivable_area",
        "ped_crossing": "ped_crossing",
        "walkway": "walkway",
        "stop_line": "stop_line",
        "carpark_area": "carpark_area",
        "divider": "divider",
        "mean": "mean",
    }
    for table_key, metric_key in table_key_map.items():
        if metrics.get(metric_key) is None and table_key in table:
            metrics[metric_key] = normalize_map_iou(table[table_key])
    return metrics


def parse_occ_miou(text: str) -> float | None:
    value = last_match_float(text, rf"OCC\s+mIoU\s*:\s*`?\**\s*({FLOAT}|n/a)")
    if value is not None:
        return value
    table = parse_table_section(text, "OCC IoU")
    return table.get("miou")


def infer_run_mode(path: Path, text: str) -> str:
    run_mode = parse_field(text, "Run mode")
    if run_mode:
        return run_mode
    lower = str(path).lower()
    if "1quarter" in lower or path.parent.name.lower().startswith("smoke_"):
        return "smoke"
    if "noaux" in lower:
        return "noaux"
    return "unknown"


def classify_result(rel_path: str, config: str) -> tuple[str, bool, bool, bool]:
    lower_path = rel_path.lower()
    lower_cfg = config.lower()
    parent = Path(rel_path).parent.name.lower()
    parent_no_smoke = parent[len("smoke_"):] if parent.startswith("smoke_") else parent
    is_aligned = "bevfusion_aligned" in lower_path or "bevfusion_aligned" in lower_cfg
    is_mainline = (
        parent_no_smoke.startswith("protoocc_multi_cnn_head_map_neck")
        or "protoocc_multi_cnn_head_map_neck" in lower_cfg
    )
    is_noaux = "noaux" in lower_path or "noaux" in lower_cfg
    if is_aligned and is_mainline:
        group = "aligned_mainline"
    elif is_aligned:
        group = "aligned_other"
    elif is_mainline:
        group = "mainline"
    else:
        group = "other"
    return group, is_aligned, is_mainline, is_noaux


@dataclass
class ResultRow:
    group: str
    run_mode: str
    rel_path: str
    work_dir: str
    config: str
    schedule: str
    occ_miou: float | None
    map_mean: float | None
    thin_avg: float | None
    drivable_area: float | None
    ped_crossing: float | None
    walkway: float | None
    stop_line: float | None
    carpark_area: float | None
    divider: float | None
    is_aligned: bool
    is_mainline: bool
    is_noaux: bool


def parse_result(path: Path, work_dirs: Path) -> ResultRow:
    text = path.read_text(encoding="utf-8", errors="replace")
    rel_path = path.relative_to(work_dirs).as_posix()
    config = parse_field(text, "Config")
    schedule = parse_field(text, "Train schedule")
    run_mode = infer_run_mode(path, text)
    map_metrics = parse_map_metrics(text)
    occ_miou = parse_occ_miou(text)
    thin_values = [map_metrics.get(key) for key in THIN_CLASSES]
    thin_avg = (
        sum(value for value in thin_values if value is not None) / len(THIN_CLASSES)
        if all(value is not None for value in thin_values)
        else None
    )
    group, is_aligned, is_mainline, is_noaux = classify_result(rel_path, config)
    return ResultRow(
        group=group,
        run_mode=run_mode,
        rel_path=rel_path,
        work_dir=path.parent.name,
        config=config,
        schedule=schedule,
        occ_miou=occ_miou,
        map_mean=map_metrics.get("mean"),
        thin_avg=thin_avg,
        drivable_area=map_metrics.get("drivable_area"),
        ped_crossing=map_metrics.get("ped_crossing"),
        walkway=map_metrics.get("walkway"),
        stop_line=map_metrics.get("stop_line"),
        carpark_area=map_metrics.get("carpark_area"),
        divider=map_metrics.get("divider"),
        is_aligned=is_aligned,
        is_mainline=is_mainline,
        is_noaux=is_noaux,
    )


def fmt(value: float | None, precision: int = 6) -> str:
    return "" if value is None else f"{value:.{precision}f}"


def sort_rows(rows: list[ResultRow]) -> list[ResultRow]:
    group_priority = {
        "aligned_mainline": 0,
        "aligned_other": 1,
        "mainline": 2,
        "other": 3,
    }
    return sorted(
        rows,
        key=lambda row: (
            group_priority.get(row.group, 9),
            -(row.map_mean if row.map_mean is not None else -1.0),
            row.rel_path,
        ),
    )


def write_csv(rows: list[ResultRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "group",
        "run_mode",
        "map_mean",
        "occ_miou",
        "thin_avg",
        "ped_crossing",
        "stop_line",
        "divider",
        "drivable_area",
        "walkway",
        "carpark_area",
        "is_aligned",
        "is_mainline",
        "is_noaux",
        "work_dir",
        "config",
        "schedule",
        "rel_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "group": row.group,
                    "run_mode": row.run_mode,
                    "map_mean": fmt(row.map_mean),
                    "occ_miou": fmt(row.occ_miou, precision=2),
                    "thin_avg": fmt(row.thin_avg),
                    "ped_crossing": fmt(row.ped_crossing),
                    "stop_line": fmt(row.stop_line),
                    "divider": fmt(row.divider),
                    "drivable_area": fmt(row.drivable_area),
                    "walkway": fmt(row.walkway),
                    "carpark_area": fmt(row.carpark_area),
                    "is_aligned": int(row.is_aligned),
                    "is_mainline": int(row.is_mainline),
                    "is_noaux": int(row.is_noaux),
                    "work_dir": row.work_dir,
                    "config": row.config,
                    "schedule": row.schedule,
                    "rel_path": row.rel_path,
                }
            )


def md_table(rows: list[ResultRow], limit: int) -> list[str]:
    lines = [
        "| Rank | Map mean | OCC mIoU | Thin avg | Ped | Stop | Divider | Group | Run | Work dir |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for rank, row in enumerate(rows[:limit], start=1):
        lines.append(
            "| "
            f"{rank} | {fmt(row.map_mean)} | {fmt(row.occ_miou, 2)} | "
            f"{fmt(row.thin_avg)} | {fmt(row.ped_crossing)} | "
            f"{fmt(row.stop_line)} | {fmt(row.divider)} | {row.group} | "
            f"{row.run_mode} | `{row.work_dir}` |"
        )
    return lines


def write_markdown(rows: list[ResultRow], path: Path, skipped_proto: int, limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)

    lines = [
        "# Work Dirs Result Summary",
        "",
        "Rules:",
        "",
        "- Recursively scan `result.md` under `work_dirs`.",
        "- Skip directories whose path/config contains `proto_map_head` by default.",
        "- Prioritize `bevfusion_aligned`, then `ProtoOcc_multi_cnn_head_map_neck` mainline.",
        "- Sort each table by `map_mean` descending.",
        "",
        "Counts:",
        "",
        f"- Parsed result files: {len(rows)}",
        f"- Skipped proto_map_head result files: {skipped_proto}",
    ]
    for group in ("aligned_mainline", "aligned_other", "mainline", "other"):
        lines.append(f"- {group}: {len(groups.get(group, []))}")
    lines.append("")

    for title, key in (
        ("BEVFusion-Aligned Mainline", "aligned_mainline"),
        ("BEVFusion-Aligned Other", "aligned_other"),
        ("ProtoOcc Map-Neck Mainline", "mainline"),
        ("Other Parsed Results", "other"),
    ):
        group_rows = groups.get(key, [])
        if not group_rows:
            continue
        lines.extend([f"## {title}", ""])
        lines.extend(md_table(group_rows, limit))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def collect_results(work_dirs: Path, include_proto_map_head: bool) -> tuple[list[ResultRow], int]:
    rows = []
    skipped_proto = 0
    for result_path in sorted(work_dirs.rglob("result.md")):
        text_for_filter = result_path.read_text(encoding="utf-8", errors="replace")
        rel_path = result_path.relative_to(work_dirs).as_posix()
        config = parse_field(text_for_filter, "Config")
        if not include_proto_map_head:
            if "proto_map_head" in rel_path.lower() or "proto_map_head" in config.lower():
                skipped_proto += 1
                continue
        rows.append(parse_result(result_path, work_dirs))
    return sort_rows(rows), skipped_proto


def print_summary(rows: list[ResultRow], skipped_proto: int, limit: int) -> None:
    print(f"Parsed result files: {len(rows)}")
    print(f"Skipped proto_map_head result files: {skipped_proto}")
    for group in ("aligned_mainline", "aligned_other", "mainline", "other"):
        group_rows = [row for row in rows if row.group == group]
        print(f"\n[{group}] top {min(limit, len(group_rows))}")
        for rank, row in enumerate(group_rows[:limit], start=1):
            print(
                f"{rank:02d} map={fmt(row.map_mean)} occ={fmt(row.occ_miou, 2)} "
                f"thin={fmt(row.thin_avg)} run={row.run_mode} dir={row.work_dir}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize ProtoOcc work_dirs result.md metrics."
    )
    parser.add_argument(
        "--work-dirs",
        type=Path,
        default=Path("work_dirs"),
        help="Path to ProtoOcc/work_dirs.",
    )
    parser.add_argument(
        "--include-proto-map-head",
        action="store_true",
        help="Include abandoned proto_map_head result dirs.",
    )
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows, skipped_proto = collect_results(args.work_dirs, args.include_proto_map_head)
    print_summary(rows, skipped_proto, args.limit)
    if args.csv_out is not None:
        write_csv(rows, args.csv_out)
        print(f"\nWrote CSV: {args.csv_out}")
    if args.md_out is not None:
        write_markdown(rows, args.md_out, skipped_proto, args.limit)
        print(f"Wrote Markdown: {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
