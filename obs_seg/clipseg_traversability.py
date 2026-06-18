#!/usr/bin/env python3
"""
CLIPSeg per-prompt segmentation (offline / one-shot)
====================================================

Runs CLIPSeg on a single overhead image with a list of prompts (a "traversable"
group and an "obstacle" group), then labels every pixel by the SINGLE
best-matching prompt (argmax over ALL prompts). Pixels whose best score falls
below --threshold are labeled "unknown".

Each prompt gets its own color, and a legend (color key) is drawn next to the
overlay so you can see exactly which prompt fired where.

Decision rule (per pixel):
    1. Compute a sigmoid heatmap for every prompt (both groups).
    2. label = argmax over ALL prompts          (which prompt matched best)
    3. conf  = max score over all prompts
    4. if conf < --threshold  ->  unknown

Usage
-----
    python3 obs_seg/clipseg_traversability.py overhead.png

    python3 obs_seg/clipseg_traversability.py overhead.png \
        --traversable "grass" "open floor" \
        --obstacle "a wooden crate" "a brick wall" "a robot" \
        --threshold 0.3 --out-dir clipseg_out

Outputs (in --out-dir, default 'clipseg_out/'):
    seg_per_prompt.png    clean per-prompt label map (one color per prompt)
    overlay.png           per-prompt map blended over the input + legend panel
    traversable_mask.png  binary mask (white = winning prompt is a traversable one)
"""
import argparse
import colorsys
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

MODEL_ID = "CIDAS/clipseg-rd64-refined"

TRAVERSABLE, OBSTACLE = 0, 1
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


def prompt_heatmaps(model, processor, image, prompts, device):
    """Return (N, H, W) sigmoid heatmaps, one per prompt, at image resolution."""
    W, H = image.size
    inputs = processor(text=prompts, images=[image] * len(prompts),
                       padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits          # (N, 352, 352) or (352, 352) if N==1
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    probs = torch.sigmoid(logits).cpu().numpy()
    return np.stack([cv2.resize(p, (W, H)) for p in probs], axis=0)


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
    ap = argparse.ArgumentParser(description="CLIPSeg per-prompt argmax + legend")
    ap.add_argument("image", help="overhead image of the environment")
    ap.add_argument("--traversable", nargs="+",
                    default=["the floor"],
                    help="prompts describing traversable regions")
    ap.add_argument("--obstacle", nargs="+",
                    # default=["furniture", "chair", "wall", "box", "person", "colored box"],
                    default=[],
                    help="prompts describing untraversable regions")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="if the best prompt score < threshold, label the pixel unknown")
    ap.add_argument("--out-dir", default="clipseg_out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clipseg] device={device}  model={MODEL_ID}")
    processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).to(device).eval()

    image = Image.open(args.image).convert("RGB")

    # flat list of all prompts, tagged by class
    prompts = list(args.traversable) + list(args.obstacle)
    classes = ([TRAVERSABLE] * len(args.traversable)) + ([OBSTACLE] * len(args.obstacle))
    print(f"[clipseg] image {image.size}  threshold={args.threshold}  prompts={prompts}")

    maps = prompt_heatmaps(model, processor, image, prompts, device)  # (N, H, W)
    idx = maps.argmax(axis=0)          # winning prompt index per pixel
    conf = maps.max(axis=0)            # winning score per pixel
    known = conf >= args.threshold

    # one color per prompt; unknown pixels overridden to gray
    colors = gen_colors(len(prompts))
    color_arr = np.array(colors, dtype=np.uint8)        # (N, 3)
    seg = color_arr[idx]                                # (H, W, 3)
    seg[~known] = UNKNOWN_COLOR

    # binary traversable mask: winning prompt is in the traversable group AND known
    class_arr = np.array(classes)
    pixel_class = class_arr[idx]
    traversable_mask = ((pixel_class == TRAVERSABLE) & known).astype(np.uint8) * 255

    os.makedirs(args.out_dir, exist_ok=True)

    # build the legend (color key) once, then append it to both images
    legend_entries = [(f"{p}  [{'trav' if c == TRAVERSABLE else 'obst'}]", colors[i])
                      for i, (p, c) in enumerate(zip(prompts, classes))]
    legend_entries.append(("unknown (< threshold)", UNKNOWN_COLOR))
    legend = make_legend(legend_entries, height=seg.shape[0])

    cv2.imwrite(os.path.join(args.out_dir, "seg_per_prompt.png"), np.hstack([seg, legend]))
    cv2.imwrite(os.path.join(args.out_dir, "traversable_mask.png"), traversable_mask)

    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = (0.5 * bgr + 0.5 * seg).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, "overlay.png"), np.hstack([overlay, legend]))

    # per-prompt coverage report
    print("[clipseg] per-prompt coverage (fraction of all pixels):")
    for i, (p, c) in enumerate(zip(prompts, classes)):
        pct = 100.0 * np.mean((idx == i) & known)
        print(f"           {pct:5.1f}%  {p}  [{'trav' if c == TRAVERSABLE else 'obst'}]")
    print(f"           {100.0*np.mean(~known):5.1f}%  unknown")
    print(f"[clipseg] wrote 3 images to {args.out_dir}/  (see overlay.png for the legend)")


if __name__ == "__main__":
    main()
