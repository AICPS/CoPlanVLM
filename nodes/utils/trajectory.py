#!/usr/bin/env python3
# csv_traj_overlay.py
# Reads a CSV (x,y or t,x,y), plots the trajectory over an image, and saves a PNG.

import os, csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndimage

# ---------- EDIT THESE ----------
CSV_PATH   = "/home/sandip/vlm_ws/results/magnav/trajectory_xy_1.csv"  # your CSV
IMAGE_PATH = "/home/sandip/vlm_ws/results/magnav/map.png" # background image
X_MIN, X_MAX = -2.5, 2.5   # your map extents
Y_MIN, Y_MAX = -4.5, 4.5
OUT_PATH  = "/home/sandip/vlm_ws/results/magnav/mag_plot.png"
# --------------------------------

def _f(x):
    try: return float(x)
    except: return None

def load_xy(csv_path):
    """Supports header with 'x','y' (any order), or numeric rows:
       [x,y] or [t,x,y]. Returns np.arrays xs, ys."""
    xs, ys = [], []
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        first = next(r)
        # header?
        if any(_f(v) is None for v in first):
            cols = [c.strip().lower() for c in first]
            ix = cols.index("x") if "x" in cols else (cols.index("position_x") if "position_x" in cols else None)
            iy = cols.index("y") if "y" in cols else (cols.index("position_y") if "position_y" in cols else None)
            if ix is None or iy is None:  # fallback: last two columns
                ix, iy = max(0, len(first)-2), max(0, len(first)-1)
            for row in r:
                if not row: continue
                vx, vy = _f(row[ix]), _f(row[iy])
                if vx is None or vy is None: continue
                xs.append(vx); ys.append(vy)
        else:
            nums = [ _f(v) for v in first ]
            if len(nums) >= 3:  # assume t,x,y
                xs.append(nums[1]); ys.append(nums[2])
                ix, iy = 1, 2
            else:               # assume x,y
                xs.append(nums[0]); ys.append(nums[1])
                ix, iy = 0, 1
            for row in r:
                if not row: continue
                vx, vy = _f(row[ix]), _f(row[iy])
                if vx is None or vy is None: continue
                xs.append(vx); ys.append(vy)
    return np.array(xs), np.array(ys)

def main():
    xs, ys = load_xy(CSV_PATH)
    if xs.size == 0:
        print("No points found in CSV."); return

    fig, ax = plt.subplots(figsize=(9,7))
    if os.path.exists(IMAGE_PATH):
        img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
        # rotated_img = ndimage.rotate(img, 90, reshape=True)
        ax.imshow(np.flipud(img), extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin="lower")
    else:
        print(f"Warning: image not found at {IMAGE_PATH}. Plotting without background.")

    ax.plot(xs, ys, linewidth=3, label="trajectory", color="black")
    # ax.scatter(xs[0], ys[0], s=200, marker="o", label="start", color="green")
    ax.scatter(xs[0], ys[0], s=200, marker="x", linewidths=5, label="start", color="green")

    # ax.scatter(xs[-1], ys[-1], s=200, marker="o", label="end", color="red")
    ax.scatter(xs[-1], ys[-1], s=200, marker="x", linewidths=5, label="end", color="red")

    ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); 
    # ax.set_title("CSV Trajectory over Map")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Saved: {OUT_PATH}")
    plt.show()

if __name__ == "__main__":
    main()