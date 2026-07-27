#!/usr/bin/env python3
"""plot_success_rates.py — grouped bar chart of success rate: our method vs SoTA baseline.

Plots per-prompt success rates (6 prompts) plus an overall average on the far right. Each group has two
FLUSH (touching) bars — our method and the SoTA baseline — in different colors. Y axis is success rate
0-100%.

Usage:
    python3 src/CoPlanVLM/scripts/plot_success_rates.py
    python3 src/CoPlanVLM/scripts/plot_success_rates.py --out debug/success_rates.png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — write to file, no display needed
import matplotlib.pyplot as plt


# ── Data ─────────────────────────────────────────────────────────────────────
LABELS = ["1", "2", "3", "4", "5", "6", "Avg"]
OURS     = [100, 100, 100, 80, 80, 80, 90]
BASELINE = [40, 100, 100,  0,  0,  0, 40]

OURS_COLOR     = "#2c7fb8"   # blue
BASELINE_COLOR = "#f03b20"   # red


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="success_rates.png",
                        help="Output image path (default: success_rates.png)")
    args = parser.parse_args()

    x = np.arange(len(LABELS))
    width = 0.4   # two flush bars -> combined width 0.8, leaving a small gap between groups

    fig, ax = plt.subplots(figsize=(10, 4.25))   # taller to give the bottom legend room
    ax.bar(x - width / 2, OURS, width, label="Our method",
           color=OURS_COLOR, edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, BASELINE, width, label="SoTA baseline",
           color=BASELINE_COLOR, edgecolor="black", linewidth=0.6)

    # Separate the "Avg" group from the per-prompt groups with a light divider.
    ax.axvline(len(LABELS) - 1.5, color="0.7", linestyle="--", linewidth=1)

    # Axis + tick fonts at 2x the matplotlib default (10 -> 20).
    ax.set_ylabel("Success rate (%)", fontsize=20)
    ax.set_xlabel("Prompt #", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=100))
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    # Legend below the x axis, centered, in one row.
    ax.legend(frameon=True, fontsize=16, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
