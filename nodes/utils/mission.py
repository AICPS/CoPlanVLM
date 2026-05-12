#!/usr/bin/env python3
# plot_csv_traj_over_image.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.ndimage as ndimage

# ---------- EDIT THESE ----------
CSV_PATH_1 = "/home/sandip/vlm_ws/results/spy_v3/mag_bot.csv"
CSV_PATH_2 = "/home/sandip/vlm_ws/results/spy_v3/spy_v3.csv"

IMAGE_PATH = "/home/sandip/vlm_ws/results/spy_v3/map_raw.png"
OUT_PATH = "/home/sandip/vlm_ws/results/spy_v3/traj_overlay_mag.png"

# Map extents in meters
X_MIN, X_MAX = -4.5, 4.5
Y_MIN, Y_MAX = -2.5, 2.5
# --------------------------------


def main():

    df_1 = pd.read_csv(CSV_PATH_1)
    df_2 = pd.read_csv(CSV_PATH_2)

    if "position_x" not in df_1 or "position_y" not in df_1:
        raise ValueError("CSV must contain 'position_x' and 'position_y' columns")
    if "position_x" not in df_2 or "position_y" not in df_2:
        raise ValueError("CSV must contain 'position_x' and 'position_y' columns")

    xs_1 = df_1["position_x"].values
    ys_1 = -df_1["position_y"].values
    t_1 = np.linspace(0, 1, len(xs_1))

    xs_2 = df_2["position_x"].values
    ys_2 = df_2["position_y"].values
    t_2 = np.linspace(0, 5, len(xs_2))

    # Load image first to get size
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")
    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))

    # --- Undistort image using OpenCV ---
    import cv2
    camera_matrix = np.array([[959.539044, 0, 988.037924], [0, 948.731512, 621.296126], [0, 0, 1]])
    dist_coeffs = np.array([-0.120573, 0.033817, -0.002598, 0.005927, 0.00000])
    img = cv2.undistort(img, camera_matrix, dist_coeffs)

    height_px, width_px = img.shape[:2]

    # Set DPI and figure size to match image pixel dimensions
    dpi = 100
    figsize = (width_px / dpi, height_px / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(np.flipud(img), extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], origin="lower")

    # Plot connected trajectory lines with colormaps
    from matplotlib.collections import LineCollection
    # VLM Agent trajectory
    points_1 = np.array([ys_1, xs_1]).T.reshape(-1, 1, 2)
    segments_1 = np.concatenate([points_1[:-1], points_1[1:]], axis=1)
    lc_1 = LineCollection(segments_1, cmap='Greens', norm=plt.Normalize(0, 1))
    lc_1.set_array(t_1)
    lc_1.set_linewidth(4)
    ax.add_collection(lc_1)
    ax.scatter(ys_1[0], xs_1[0], s=80, marker="o", color="green", label="VLM agent start")
    ax.scatter(ys_1[-1], xs_1[-1], s=80, marker="x", color="red", label="VLM agent end")

    # Leader trajectory
    points_2 = np.array([ys_2, xs_2]).T.reshape(-1, 1, 2)
    segments_2 = np.concatenate([points_2[:-1], points_2[1:]], axis=1)
    lc_2 = LineCollection(segments_2, cmap='Reds', norm=plt.Normalize(0, 5))
    lc_2.set_array(t_2)
    lc_2.set_linewidth(4)
    ax.add_collection(lc_2)
    ax.scatter(ys_2[0], xs_2[0], s=80, marker="o", color="blue", label="Leader start")
    ax.scatter(ys_2[-1], xs_2[-1], s=80, marker="x", color="orange", label="Leader end")

    # Axis settings
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")

    # Custom legend handles for colormap colors
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='VLM Agent trajectory', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Leader trajectory', markerfacecolor='red', markersize=10),
        Line2D([], [], marker='o', linestyle='None', color='green', label='VLM agent start', markersize=10),
        Line2D([], [], marker='x', linestyle='None', color='red', label='VLM agent end', markersize=10),
        Line2D([], [], marker='o', linestyle='None', color='blue', label='Leader start', markersize=10),
        Line2D([], [], marker='x', linestyle='None', color='orange', label='Leader end', markersize=10)
    ]
    ax.legend(handles=legend_handles, loc="upper right")
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
    fig.savefig(OUT_PATH, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved overlay plot -> {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
