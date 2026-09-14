"""
Generate a grouped bar chart of per-prompt success rates for CoPlanVLM and three baseline methods,
grouped by task category (Nav2Point / Coverage / Maneuver).

Each method's bar sits on a faint full-height "lane" of the same colour, so a 0% result still shows
which slot it belongs to instead of vanishing into the background — with four methods per cluster and
several zeros in the data, an empty slot is otherwise ambiguous.

A "Mean" summary cluster to the right gives each method's average over all 18 prompts, drawn with the
same bar encoding as the data and separated by a solid divider so it does not read as a 19th prompt.

Output: success_rates.png (raster, 200 dpi).

Usage:  python plot_success_rates.py
Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Data: per-prompt success rates (%) for each method.
# Index i corresponds to "New Prompt #" i+1 (prompts 1..18).
# Prompts are ordered by category: 1-6 Nav2Point, 7-12 Coverage, 13-18 Maneuver.
# ---------------------------------------------------------------------------
SUCCESS_RATES = {
    "CoPlanVLM":       [100, 80, 80, 100, 100, 50, 90, 90, 70, 90, 70, 70, 100, 80, 100, 100, 100, 50],
    "CoNVOI Marking":  [ 70,  0, 100, 100,  0,  0, 50, 60, 90, 50, 20,  0,  90, 10, 100,  60, 100,  0],
    "Grid Overlay":    [  0,  0, 100, 100, 40,  0, 100, 0, 30, 20,  0,  0,  30,  0,  80, 100, 100,  0],
    "Pixel Selection": [  0, 30,  60,  0,  0, 70, 100, 40, 0, 10,  0,  0,  90,  0,   0, 100,  70,  0],
}

# Bar colors, in the order the methods should appear within each cluster.
METHOD_COLORS = {
    "CoPlanVLM":       "#2E5A87",  # deep blue (our method)
    "CoNVOI Marking":  "#C1666B",  # muted red
    "Grid Overlay":    "#6B8E4E",  # green
    "Pixel Selection": "#B0752F",  # gold
}

# Category label -> inclusive prompt-number span (1-indexed).
CATEGORY_SPANS = {
    "Nav2Point": (1, 6),
    "Coverage":  (7, 12),
    "Maneuver":  (13, 18),
}

LANE_ALPHA = 0.18   # faint full-height slot behind each bar, so a 0% result still shows its slot
MEAN_GAP   = 1.25   # x-units from the last prompt to the overall-mean summary cluster
MEAN_SCALE = 1.6    # how much wider the summary cluster is than a per-prompt cluster
HEADER_Y   = 104.0  # y of the category headers, just above the axes (see ylim below)

# Figure size in inches. FIG_W is set so the mean cluster's extra x-range does not squeeze the 18
# data clusters. AXES_H is the height of the PLOT BOX itself, set explicitly rather than left to
# auto-layout: shrinking FIG_H alone does not shrink the bars, because the layout just hands the
# reclaimed space back to the axes. FIG_H only needs to be large enough to hold the axes plus the
# headers above and the ticks/xlabel/legend below — bbox_inches="tight" crops the rest away.
FIG_W, FIG_H = 11.2, 2.5
AXES_H = 1.53       # in; ~2/3 of the 2.23 in plot box this figure had before
# The x-axis label and the legend SHARE one row below the tick labels: "Prompt" sits at the left end
# (under the first few tick numbers) and the legend fills the space to its right, which would
# otherwise be empty. That saves a whole row of figure height versus a centred xlabel with the
# legend stacked beneath it. Value is inches below the axes; bbox_to_anchor wants axes fractions, so
# it is divided by AXES_H at use — keeping it in inches means changing the axes height does not
# silently move the row into the tick labels.
LABEL_ROW_IN = 0.34


def main():
    methods = list(SUCCESS_RATES.keys())
    n_prompts = len(next(iter(SUCCESS_RATES.values())))
    prompts = np.arange(1, n_prompts + 1)

    # sanity check: all methods have the same number of prompts
    for name, vals in SUCCESS_RATES.items():
        assert len(vals) == n_prompts, f"{name} has {len(vals)} values, expected {n_prompts}"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 12,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
    })

    # Explicit axes box (figure fractions): left/width leave room for the y label, bottom leaves
    # room for the tick labels, xlabel and legend, and the height is pinned to AXES_H.
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0.055, 0.40, 0.935, AXES_H / FIG_H])

    n_methods = len(methods)
    # Clusters sit one x-unit apart, so cluster width + gap = 1.0. Narrowing the cluster from 0.8
    # to 0.7 widens the gap between adjacent prompts from 0.2 to 0.3 (1.5x) without moving any
    # tick or changing the figure width.
    bar_w = 0.7 / n_methods                       # total cluster width = 0.7
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_w

    for name, off in zip(methods, offsets):
        vals = np.array(SUCCESS_RATES[name], dtype=float)
        color = METHOD_COLORS[name]
        # Faint lane first, solid bar on top.
        ax.bar(prompts + off, 100, width=bar_w, color=color, alpha=LANE_ALPHA,
               edgecolor="none", zorder=2)
        ax.bar(prompts + off, vals, width=bar_w, color=color, edgecolor="none",
               label=name, zorder=3)

    # Overall-mean summary cluster, set off to the right of the per-prompt data by a full empty
    # slot. Drawn with the SAME bar encoding as the data so it is read the same way — no second
    # visual language to learn. The gap plus the solid divider keep it from reading as a 19th
    # prompt, and it is drawn MEAN_SCALE times wider than a data cluster so it reads as a summary.
    mean_x = n_prompts + MEAN_GAP
    mean_half = bar_w * MEAN_SCALE * n_methods / 2      # half-width of the summary cluster
    for name, off in zip(methods, offsets):
        color = METHOD_COLORS[name]
        m = float(np.mean(SUCCESS_RATES[name]))
        x = mean_x + off * MEAN_SCALE
        ax.bar(x, 100, width=bar_w * MEAN_SCALE, color=color, alpha=LANE_ALPHA,
               edgecolor="none", zorder=2)
        ax.bar(x, m, width=bar_w * MEAN_SCALE, color=color, edgecolor="none", zorder=3)

    # category separators + labels
    for lo, hi in CATEGORY_SPANS.values():
        # draw a separator after each category except the last
        if hi < n_prompts:
            ax.axvline(hi + 0.5, color="black", lw=1.0, ls=(0, (4, 3)), zorder=1)
    # Solid (not dashed) divider before the summary: it separates a different KIND of column from
    # the data, not one category of prompts from the next. Centred in the actual gap — midway
    # between prompt 18's right edge and the summary cluster's left edge — so it stays centred if
    # MEAN_GAP or MEAN_SCALE is retuned.
    ax.axvline(((n_prompts + bar_w * n_methods / 2) + (mean_x - mean_half)) / 2,
               color="black", lw=1.0, zorder=1)
    # Headers sit just ABOVE the axes (y > ylim) rather than inside it. With the top spine removed
    # there is no border for them to collide with, and bbox_inches="tight" still includes them — so
    # the data can use the full plot height instead of reserving 18% of it as header room.
    for cat, (lo, hi) in CATEGORY_SPANS.items():
        ax.text((lo + hi) / 2, HEADER_Y, cat, ha="center", va="bottom",
                fontsize=11, color="black", clip_on=False)
    ax.text(mean_x, HEADER_Y, "Mean", ha="center", va="bottom", fontsize=11, color="black",
            clip_on=False)

    ax.set_xticks(list(prompts) + [mean_x])
    ax.set_xticklabels([str(p) for p in prompts] + ["all"], fontsize=10)
    # "Prompt" as a left-aligned text at the shared row rather than set_xlabel, which can only
    # centre. va="top" hangs it from the same line the legend hangs from, so the two align.
    ax.text(0.0, -LABEL_ROW_IN / AXES_H, "Prompt", transform=ax.transAxes,
            ha="left", va="top", fontsize=12)
    ax.set_ylabel("Success rate (%)")
    # Only just above the 100% lanes: the headers live outside the axes now, so no headroom is
    # needed for them.
    ax.set_ylim(0, 103)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    # Right margin past the summary cluster matched to the left margin past prompt 1 (0.35).
    ax.set_xlim(0.3, mean_x + mean_half + 0.35)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)
    # No top border: it carries no information, and dropping it lets the category headers sit
    # tight against the 100% line. Left/bottom/right stay — the right spine closes the mean panel.
    ax.spines["top"].set_visible(False)
    # borderpad=0: with the frame off the padding is invisible, but it still enlarges the legend's
    # bounding box — which is what bbox_inches="tight" crops to, leaving a blank strip at the bottom.
    # loc="upper center" so the anchor pins the legend's TOP to the shared row line, matching
    # "Prompt"'s va="top". Centred on 0.55 rather than 0.5 to keep clear of the label on the left.
    # Proxy handles: the drawn bars come in pairs (lane + value), so an automatic legend would
    # collect both; full-opacity patches give one clean swatch per method.
    ax.legend(handles=[Patch(facecolor=METHOD_COLORS[n], label=n) for n in methods],
              loc="upper center", ncol=n_methods, frameon=False, fontsize=11, borderpad=0,
              bbox_to_anchor=(0.55, -LABEL_ROW_IN / AXES_H))

    # No tight_layout: the legend is a child of the axes, so tight_layout treats it as content
    # to fit and SHRINKS the axes to make room — the shorter the figure, the worse it gets.
    # bbox_inches="tight" at save time already trims the canvas to the drawn artists.
    # pad_inches=0: bbox_inches="tight" crops to the drawn artists but then re-adds a 0.1 in border
    # on every side (20 px at 200 dpi). Zeroing it puts the crop right on the text.
    fig.savefig("success_rates.png", dpi=200, bbox_inches="tight", pad_inches=0)
    print("Saved success_rates.png")

    # print summary means for reference
    print("\nMean success rate by method:")
    for name in methods:
        print(f"  {name:16s} {np.mean(SUCCESS_RATES[name]):5.1f}%")


if __name__ == "__main__":
    main()
