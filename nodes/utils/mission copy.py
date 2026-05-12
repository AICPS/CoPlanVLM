#!/usr/bin/env python3
# plot_csv_traj_over_image.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.ndimage as ndimage

# ---------- EDIT THESE ----------
CSV_PATH_1 = "/home/sandip/vlm_ws/results/behind_box_v3/trajectory_20250926_123607.csv"

IMAGE_PATH = "/home/sandip/vlm_ws/results/behind_box_v3/map_raw.png"
OUT_PATH = "/home/sandip/vlm_ws/results/behind_box_v3/undistorted.png"

# Map extents in meters
X_MIN, X_MAX = -4.5, 4.5
Y_MIN, Y_MAX = -2.5, 2.5
# --------------------------------

def main():
    df_1 = pd.read_csv(CSV_PATH_1)
    # df_1= df_1[31000:]
    if "position_x" not in df_1 or "position_y" not in df_1:
        raise ValueError("CSV must contain 'position_x' and 'position_y' columns")

    xs_1 = df_1["position_x"].values
    ys_1 = df_1["position_y"].values

    # Create a normalized time array for colormap
    t_1 = np.linspace(0, 1, len(xs_1))

    # Load image first to get size
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")
    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    height_px, width_px = img.shape[:2]

    # Set DPI and figure size to match image pixel dimensions
    dpi = 100
    figsize = (width_px / dpi, height_px / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(np.flipud(img), extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin="lower")

    # Plot trajectory with temporal colormap
    from matplotlib.collections import LineCollection
    points_1 = np.array([ys_1, xs_1]).T.reshape(-1, 1, 2)
    segments_1 = np.concatenate([points_1[:-1], points_1[1:]], axis=1)
    lc_1 = LineCollection(segments_1, cmap='Greens', norm=plt.Normalize(0, 1))
    lc_1.set_array(t_1)
    lc_1.set_linewidth(4)
    ax.add_collection(lc_1)

    # Start and end points
    ax.scatter(ys_1[0], xs_1[0], s=80, marker="o", color="green", label="start")
    ax.scatter(ys_1[-1], xs_1[-1], s=80, marker="x", color="red", label="end")

    # Axis settings
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.legend(loc="upper right")
    ax.set_aspect('equal')

    # Direction annotations
    cx, cy = -4.0, 2.0
    length = 0.15
    directions = {
        'E': (length, 0),
        'N': (0, length),
        'W': (-length, 0),
        'S': (0, -length),
    }

    for label, (dx, dy) in directions.items():
        tip_x, tip_y = cx + dx, cy + dy

        ax.annotate(
            '',
            xy=(tip_x, tip_y),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle='->', lw=2, color='white')
        )

        label_offset = 0.15
        label_x = tip_x + label_offset * (1 if dx > 0 else -1 if dx < 0 else 0)
        label_y = tip_y + label_offset * (1 if dy > 0 else -1 if dy < 0 else 0)

        ax.text(
            label_x, label_y,
            label,
            ha='center', va='center',
            fontsize=16, color='white'
        )

    # Save with tight bounding box to preserve exact size
    pdf_out_path = OUT_PATH.replace('.png', '.pdf')
    fig.savefig(pdf_out_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved overlay plot -> {OUT_PATH}")
    plt.show()



if __name__ == "__main__":
    main()
