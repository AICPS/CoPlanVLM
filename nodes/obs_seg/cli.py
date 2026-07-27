#!/usr/bin/env python3
"""
Offline CLIPSeg tuning tool (per-prompt argmax + legend) -- DEBUG ONLY
=====================================================================

NOT part of deployment. Thin CLI around `TraversabilitySegmenter` for tuning
prompts/threshold on a saved overhead image. Colors every pixel by its
best-matching prompt (argmax over all prompts); pixels below --threshold are
"unknown" (gray). Writes a legend next to each image so you can see which prompt
fired where. The live system uses node_Costmap, which shares the same segmenter
library.

Run (after building + sourcing the workspace, or from the nodes/ dir):
    python3 -m obs_seg.cli overhead.png
    ros2 run coplan_vlm obs_seg_cli overhead.png --threshold 0.45

    python3 -m obs_seg.cli overhead.png \
        --traversable "the floor" --obstacle "a brick wall" "a box" --threshold 0.45

Outputs (in --out-dir, default 'clipseg_out/'):
    seg_per_prompt.png    clean per-prompt label map + legend
    overlay.png           per-prompt map blended over the input + legend
    traversable_mask.png  binary mask (white = confidently traversable)
"""
import argparse
import colorsys
import os

import cv2
import numpy as np
from PIL import Image

from . import FREE
from .segmenter import TraversabilitySegmenter

UNKNOWN_COLOR = (128, 128, 128)  # BGR gray

# distinguishable base palette (BGR); extended with HSV if more prompts are given
PALETTE = [
    (0, 200, 0), (0, 0, 220), (220, 0, 0), (0, 200, 220), (220, 0, 220),
    (220, 200, 0), (0, 120, 255), (200, 0, 120), (120, 200, 0), (0, 160, 100),
]


def gen_colors(n):
    """Return n distinct BGR colors."""
    cols = list(PALETTE)
    i = 0
    while len(cols) < n:
        h = (i * 0.61803398875) % 1.0          # golden-ratio hue spacing
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
        i += 1
    return cols[:n]


def make_legend(entries, height, width=480, swatch=26, pad=12, font_scale=0.6):
    """entries: list of (text, BGR color). Returns a white legend panel of given height."""
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    y = pad + swatch
    for text, color in entries:
        cv2.rectangle(panel, (pad, y - swatch), (pad + swatch, y), color, -1)
        cv2.rectangle(panel, (pad, y - swatch), (pad + swatch, y), (0, 0, 0), 1)
        cv2.putText(panel, text, (pad + swatch + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        y += swatch + pad
    return panel


def main():
    ap = argparse.ArgumentParser(description="CLIPSeg per-prompt argmax + legend (offline tuning)")
    ap.add_argument("image", help="overhead image of the environment")
    ap.add_argument("--traversable", nargs="+", default=["the floor"],
                    help="prompts describing traversable regions")
    ap.add_argument("--obstacle", nargs="+", default=[],
                    help="prompts describing untraversable regions")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="if the best prompt score < threshold, label the pixel unknown")
    ap.add_argument("--out-dir", default="clipseg_out")
    args = ap.parse_args()

    seg = TraversabilitySegmenter()
    image = Image.open(args.image).convert("RGB")

    labels, info = seg.classify(image, args.traversable, args.obstacle, args.threshold)
    idx, known = info["idx"], info["known"]
    prompts, classes = info["prompts"], info["classes"]
    print(f"[clipseg] image {image.size}  threshold={args.threshold}  prompts={prompts}")

    # one color per prompt; unknown pixels overridden to gray
    colors = gen_colors(len(prompts))
    seg_img = np.array(colors, dtype=np.uint8)[idx]
    seg_img[~known] = UNKNOWN_COLOR

    traversable_mask = (labels == FREE).astype(np.uint8) * 255

    os.makedirs(args.out_dir, exist_ok=True)

    # legend (color key) built once, appended to both images
    legend_entries = [(f"{p}  [{'trav' if c == 0 else 'obst'}]", colors[i])
                      for i, (p, c) in enumerate(zip(prompts, classes))]
    legend_entries.append(("unknown (< threshold)", UNKNOWN_COLOR))
    legend = make_legend(legend_entries, height=seg_img.shape[0])

    cv2.imwrite(os.path.join(args.out_dir, "seg_per_prompt.png"), np.hstack([seg_img, legend]))
    cv2.imwrite(os.path.join(args.out_dir, "traversable_mask.png"), traversable_mask)

    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = (0.5 * bgr + 0.5 * seg_img).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, "overlay.png"), np.hstack([overlay, legend]))

    print("[clipseg] per-prompt coverage (fraction of all pixels):")
    for i, (p, c) in enumerate(zip(prompts, classes)):
        pct = 100.0 * np.mean((idx == i) & known)
        print(f"           {pct:5.1f}%  {p}  [{'trav' if c == 0 else 'obst'}]")
    print(f"           {100.0*np.mean(~known):5.1f}%  unknown")
    print(f"[clipseg] wrote 3 images to {args.out_dir}/  (see overlay.png for the legend)")


if __name__ == "__main__":
    main()
