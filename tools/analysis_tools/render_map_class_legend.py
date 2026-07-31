#!/usr/bin/env python3
"""Render a compact vertical legend for the six BEV map classes."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


CLASSES = (
    ("drivable area", (166, 206, 227)),
    ("pedestrian crossing", (251, 154, 153)),
    ("walkway", (227, 26, 28)),
    ("stop line", (253, 191, 111)),
    ("carpark area", (255, 127, 0)),
    ("divider", (106, 61, 154)),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("viz/map_class_legend_vertical.png"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    handles = [
        Patch(
            facecolor=tuple(channel / 255.0 for channel in color),
            edgecolor=(0.45, 0.45, 0.45),
            linewidth=0.7,
            label=label,
        )
        for label, color in CLASSES
    ]

    fig, axis = plt.subplots(figsize=(3.15, 3.0))
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.axis("off")
    axis.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=13,
        handlelength=2.35,
        handleheight=1.05,
        handletextpad=0.55,
        borderpad=0.0,
        labelspacing=0.38,
    )
    fig.savefig(
        args.out,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
