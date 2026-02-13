import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
from pathlib import Path
import yaml
import sys

# ==========================================
# 定数定義（基本レイアウト）
# ==========================================
DETECTOR_SIZE_XY = 300.0
DETECTOR_POS_Z   = 80.0
LEAD_DENSITY = 11.34
BLOCK_CENTER = (0.0, 0.0, 0.0)

def get_project_root():
    here = Path(__file__).resolve().parent
    for d in [here] + list(here.parents):
        if (d / "build").exists(): return d
    return Path.cwd()

def resolve_out(name: str) -> Path:
    base = get_project_root() / "build" / "outputs" / "evaluation"
    base.mkdir(parents=True, exist_ok=True)
    return base / name

def load_grid_config():
    path = get_project_root() / "configs" / "grid3d.yml"
    with open(path) as f:
        config = yaml.safe_load(f)
    return config

def draw_box(ax, center, size, color, alpha=0.25, edgecolor="k", linewidth=0.3):
    cx, cy, cz = center
    sx, sy, sz = size
    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy - sy/2, cy + sy/2
    z0, z1 = cz - sz/2, cz + sz/2
    P = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]], dtype=float)
    F = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]
    poly = Poly3DCollection([P[f] for f in F], facecolors=color, edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(poly)

def _clip_index(a0, a1, amin, da, n):
    return int(np.clip((a0 - amin) / da, 0, n)), int(np.clip((a1 - amin) / da, 0, n))

def _fill_box(density_map, grid, x0, x1, y0, y1, z0, z1, value):
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    xmin, ymin, zmin = float(grid["x_min"]), float(grid["y_min"]), float(grid["z_min"])
    dx = (float(grid["x_max"]) - xmin) / nx
    dy = (float(grid["y_max"]) - ymin) / ny
    dz = (float(grid["z_max"]) - zmin) / nz
    ix0, ix1 = _clip_index(x0, x1, xmin, dx, nx)
    iy0, iy1 = _clip_index(y0, y1, ymin, dy, ny)
    iz0, iz1 = _clip_index(z0, z1, zmin, dz, nz)
    density_map[iz0:iz1, iy0:iy1, ix0:ix1] = value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_img", default="setup_geometry.png")
    parser.add_argument("--out_npy", default="true_density.npy")
    parser.add_argument("--angle", type=int, nargs=2, default=[15, -60])
    args = parser.parse_args()

    # 1. YAMLロード
    config = load_grid_config()
    PB_COUNT = config.get("pb_count", 1)
    PB_HOLLOW = config.get("pb_hollow", 0)
    OUTER_HALF = config.get("outer_half", 20.0)
    INNER_HALF = config.get("inner_half", 15.0)

    # 2. 3D可視化 (元のスタイルを復元)
    fig = plt.figure(figsize=(10, 8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    # 検出器 (元のRGB指定)
    draw_box(ax, (0, 0,  DETECTOR_POS_Z), (DETECTOR_SIZE_XY, DETECTOR_SIZE_XY, 1), color=(0.1, 1.0, 0.1))
    draw_box(ax, (0, 0, -DETECTOR_POS_Z), (DETECTOR_SIZE_XY, DETECTOR_SIZE_XY, 1), color=(1.0, 0.1, 0.1))

    centers = [BLOCK_CENTER] if PB_COUNT == 1 else [(-40.0, 0.0, 0.0), (40.0, 0.0, 0.0)]
    for (cx, cy, cz) in centers:
        draw_box(ax, (cx, cy, cz), (2*OUTER_HALF, 2*OUTER_HALF, 2*OUTER_HALF), color=(0.4, 0.4, 0.4), alpha=0.4)
        if PB_HOLLOW:
            draw_box(ax, (cx, cy, cz), (2*INNER_HALF, 2*INNER_HALF, 2*OUTER_HALF), color=(0.7, 0.7, 1.0), alpha=0.2)

    x_limit = max(abs(float(config["x_min"])), abs(float(config["x_max"])))
    y_limit = max(abs(float(config["y_min"])), abs(float(config["y_max"])))
    z_limit = max(abs(float(config["z_min"])), abs(float(config["z_max"])))

    ax.set_xlim(-x_limit, x_limit); ax.set_ylim(-y_limit, y_limit); ax.set_zlim(-z_limit, z_limit)
    ax.set_box_aspect((x_limit*2, y_limit*2, z_limit*2))
    ax.view_init(elev=args.angle[0], azim=args.angle[1])
    
    title_str = "Hollowed Block" if PB_HOLLOW else "Solid Block"
    ax.set_title(f"Simulation Setup (Ground Truth) - {title_str}")

    plt.tight_layout()
    plt.savefig(resolve_out(args.out_img), bbox_inches="tight")
    plt.close(fig)

    # 3. Phantom作成
    density_map = np.zeros((config["nz"], config["ny"], config["nx"]), dtype=np.float32)
    for (cx, cy, cz) in centers:
        _fill_box(density_map, config, cx-OUTER_HALF, cx+OUTER_HALF, cy-OUTER_HALF, cy+OUTER_HALF, cz-OUTER_HALF, cz+OUTER_HALF, LEAD_DENSITY)
        if PB_HOLLOW:
            _fill_box(density_map, config, cx-INNER_HALF, cx+INNER_HALF, cy-INNER_HALF, cy+INNER_HALF, cz-OUTER_HALF, cz+OUTER_HALF, 0.0)

    np.save(resolve_out(args.out_npy), density_map.flatten())
    print(f"Generated {title_str} (Count={PB_COUNT}, Outer={OUTER_HALF})")

if __name__ == "__main__":
    main()