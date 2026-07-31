#!/usr/bin/env python3
"""Render the semantic occupancy class-color legend as a standalone image."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from render_occ_qualitative_comparison import CLASS_COLORS, CLASS_NAMES


DISPLAY_NAMES = {
    "construction_vehicle": "const. veh",
    "traffic_cone": "traf. cone",
    "driveable_surface": "driv. sur.",
    "other_flat": "other flat",
    "vegetation": "veg.",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the OCC semantic class-color legend.")
    parser.add_argument(
        "--output",
        default="work_dirs/qualitative_protoocc_vs_a1/class_color_legend.png",
        help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    handles = [
        Patch(
            facecolor=CLASS_COLORS[index],
            edgecolor="#a0a0a0",
            linewidth=0.8,
            label=DISPLAY_NAMES.get(name, name),
        )
        for index, name in enumerate(CLASS_NAMES)
    ]

    fig, ax = plt.subplots(figsize=(2.25, 6.5))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=1,
        fontsize=12,
        handlelength=2.5,
        handleheight=1.15,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0,
    )
    fig.savefig(
        output,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
