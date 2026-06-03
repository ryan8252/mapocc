#!/usr/bin/env python3
"""Rank smoke ProtoOcc result.md files by map mean IoU."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SUMMARY_RE = re.compile(
    r"Map\s+mean\s+IoU@max\s*:\s*`?\**\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
RAW_DICT_RE = re.compile(
    r"['\"]map/mean/iou@max['\"]\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
)
HEADING_RE = re.compile(r"^#{1,6}\s+")
TABLE_CELL_RE = re.compile(r"[`*_]")


def clean_cell(cell: str) -> str:
    return TABLE_CELL_RE.sub("", cell).strip()


def first_float(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    return float(match.group(0)) if match else None


def normalize_iou(value: float) -> float:
    """Result files mix decimal IoU and percent-style IoU."""
    return value / 100.0 if value > 1.0 else value


def extract_summary_iou(text: str) -> tuple[float, str] | None:
    match = SUMMARY_RE.search(text)
    if not match:
        return None
    raw = float(match.group(1))
    return normalize_iou(raw), "summary"


def iter_map_sections(lines: list[str]):
    in_section = False
    for line in lines:
        normalized = line.strip().lower()
        if re.match(r"^#{1,6}\s+map iou\b", normalized):
            in_section = True
            continue
        if normalized.startswith("standard map iou:"):
            in_section = True
            continue
        if in_section and HEADING_RE.match(line):
            in_section = False
        if in_section:
            yield line


def extract_table_iou(text: str) -> tuple[float, str] | None:
    for line in iter_map_sections(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [clean_cell(cell) for cell in stripped.strip("|").split("|")]
        if cells and cells[0].lower() == "mean":
            for cell in cells[1:]:
                value = first_float(cell)
                if value is not None:
                    return normalize_iou(value), "map table"
    return None


def extract_raw_dict_iou(text: str) -> tuple[float, str] | None:
    match = RAW_DICT_RE.search(text)
    if not match:
        return None
    raw = float(match.group(1))
    return normalize_iou(raw), "raw dict"


def extract_map_iou(path: Path) -> tuple[float, str] | None:
    text = path.read_text(encoding="utf-8")
    return extract_summary_iou(text) or extract_table_iou(text) or extract_raw_dict_iou(text)


def collect_results(work_dirs: Path, prefix: str, suffix: str):
    rows = []
    missing = []
    for result_path in sorted(work_dirs.glob(f"{prefix}*{suffix}/result.md")):
        extracted = extract_map_iou(result_path)
        if extracted is None:
            missing.append(result_path)
            continue
        iou, source = extracted
        rows.append(
            {
                "dir": result_path.parent.name,
                "path": result_path,
                "iou": iou,
                "source": source,
            }
        )
    rows.sort(key=lambda row: row["iou"], reverse=True)
    return rows, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank smoke ProtoOcc result.md files by map mean IoU."
    )
    parser.add_argument(
        "--work-dirs",
        default="work_dirs",
        type=Path,
        help="Path to the work_dirs directory.",
    )
    parser.add_argument("--prefix", default="smoke_ProtoOcc")
    parser.add_argument("--suffix", default="_1quarter_4090")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print tab-separated rows instead of a Markdown table.",
    )
    args = parser.parse_args()

    rows, missing = collect_results(args.work_dirs, args.prefix, args.suffix)
    if args.plain:
        print("rank\tmap_iou\tmap_iou_pct\tdir\tsource\tpath")
        for rank, row in enumerate(rows, start=1):
            print(
                f"{rank}\t{row['iou']:.6f}\t{row['iou'] * 100:.2f}\t"
                f"{row['dir']}\t{row['source']}\t{row['path']}"
            )
    else:
        print("| Rank | Map mean IoU | Map mean IoU (%) | Work dir | Source |")
        print("| ---: | ---: | ---: | --- | --- |")
        for rank, row in enumerate(rows, start=1):
            print(
                f"| {rank} | {row['iou']:.6f} | {row['iou'] * 100:.2f} | "
                f"`{row['dir']}` | {row['source']} |"
            )

    if missing:
        print("\nCould not parse Map IoU from:")
        for path in missing:
            print(f"- {path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
