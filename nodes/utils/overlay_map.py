from PIL import Image
import os

# ---------- CONFIG ----------
# Input paths
background_path = "/home/sandip/vlm_ws/install/talking-turtle/share/talking-turtle/map_raw.png"         # e.g., map_raw.png
grid_path       = "/home/sandip/vlm_ws/install/talking-turtle/share/talking-turtle/config/transparent_grid_3.png"   # e.g., transparent_grid.png

# Output path
output_path     = "/home/sandip/vlm_ws/install/talking-turtle/share/talking-turtle/map_overlay.png"
# ----------------------------


def overlay_grid_on_image(background_path, grid_path, output_path):
    if not os.path.exists(background_path):
        raise FileNotFoundError(f"Background image not found: {background_path}")
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Grid image not found: {grid_path}")

    # Open background and grid as RGBA
    background = Image.open(background_path).convert("RGBA")
    grid = Image.open(grid_path).convert("RGBA")

    # Resize grid to match background
    grid_resized = grid.resize(background.size)

    # Alpha composite
    result = Image.alpha_composite(background, grid_resized)

    # Save result
    result.save(output_path)
    print(f"✅ Overlay saved to: {output_path}")


if __name__ == "__main__":
    overlay_grid_on_image(background_path, grid_path, output_path)
