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
# 定数定義 (シミュレーション配置)
# ==========================================
DETECTOR_SIZE_XY = 300.0
DETECTOR_POS_Z   = 80.0
BLOCK_SIZE       = 20.0
BLOCK_POS_X      = 40.0  
LEAD_DENSITY     = 11.34 

# ==========================================
# パス・設定関連
# ==========================================
def get_project_root():
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if (d / "build").exists(): return d
  return Path.cwd()

def resolve_out(name: str) -> Path:
  base = get_project_root() / "build" / "outputs" / "evaluation"
  base.mkdir(parents=True, exist_ok=True)
  return base / name

def config_path(name: str) -> Path:
  return get_project_root() / "configs" / name

def load_grid_config():
  path = config_path("grid3d.yml")
  if not path.exists():
    print(f"Error: Config file not found at {path}")
    sys.exit(1)

  with open(path) as f:
    config = yaml.safe_load(f)
  
  # キー名の修正 (nx, ny, nz)
  required_keys = ["nx", "ny", "nz", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
  for key in required_keys:
    if key not in config:
      print(f"Error: Key '{key}' is missing in {path}.")
      sys.exit(1)
  return config

# ==========================================
# 描画・生成ロジック
# ==========================================
def draw_box(ax, center, size, color, alpha=0.25, edgecolor="k", linewidth=0.3):
  cx, cy, cz = center
  sx, sy, sz = size
  x0, x1 = cx - sx/2, cx + sx/2
  y0, y1 = cy - sy/2, cy + sy/2
  z0, z1 = cz - sz/2, cz + sz/2
  P = np.array([
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
  ], dtype=float)
  F = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]
  poly = Poly3DCollection([P[f] for f in F], facecolors=color, edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)
  ax.add_collection3d(poly)

def create_and_save_phantom(grid_config, filename="true_density.npy"):
  """
  Method Dに合わせて (nz, ny, nx) の順でデータを作成し保存する
  """
  # configから取得
  nx = int(grid_config["nx"])
  ny = int(grid_config["ny"])
  nz = int(grid_config["nz"])
  
  x_min, x_max = float(grid_config["x_min"]), float(grid_config["x_max"])
  y_min, y_max = float(grid_config["y_min"]), float(grid_config["y_max"])
  z_min, z_max = float(grid_config["z_min"]), float(grid_config["z_max"])

  dx = (x_max - x_min) / nx
  dy = (y_max - y_min) / ny
  dz = (z_max - z_min) / nz

  # ★重要★ Method Dに合わせて (Z, Y, X) の順で作成
  density_map = np.zeros((nz, ny, nx), dtype=np.float32)

  blk_size = BLOCK_SIZE
  half_size = blk_size / 2.0
  block_centers = [(-BLOCK_POS_X, 0.0, 0.0), (BLOCK_POS_X, 0.0, 0.0)]

  print(f"Generating Phantom NPY (Method D compatible)...")
  print(f"  Shape: ({nz}, {ny}, {nx}) -> (Z, Y, X)")

  for (cx, cy, cz) in block_centers:
    b_x0, b_x1 = cx - half_size, cx + half_size
    b_y0, b_y1 = cy - half_size, cy + half_size
    b_z0, b_z1 = cz - half_size, cz + half_size

    # インデックス計算
    ix_start = int(np.clip((b_x0 - x_min) / dx, 0, nx))
    ix_end   = int(np.clip((b_x1 - x_min) / dx, 0, nx))
    
    iy_start = int(np.clip((b_y0 - y_min) / dy, 0, ny))
    iy_end   = int(np.clip((b_y1 - y_min) / dy, 0, ny))
    
    iz_start = int(np.clip((b_z0 - z_min) / dz, 0, nz))
    iz_end   = int(np.clip((b_z1 - z_min) / dz, 0, nz))

    # ★重要★ [iz, iy, ix] の順でアクセス
    density_map[iz_start:iz_end, iy_start:iy_end, ix_start:ix_end] = LEAD_DENSITY

  # 1次元化して保存
  flat_density = density_map.flatten()
  
  save_path = resolve_out(filename)
  np.save(save_path, flat_density)
  print(f"  Phantom saved to: {save_path}")
  
  return save_path

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_img", default="setup_geometry.png")
  parser.add_argument("--out_npy", default="true_density.npy")
  parser.add_argument("--angle", type=int, nargs=2, default=[22, -60])
  args = parser.parse_args()

  grid_config = load_grid_config()
  
  # 描画用 (可視化は座標系が変わらないのでそのまま)
  x_limit = max(abs(float(grid_config["x_min"])), abs(float(grid_config["x_max"])))
  y_limit = max(abs(float(grid_config["y_min"])), abs(float(grid_config["y_max"])))
  z_limit = max(abs(float(grid_config["z_min"])), abs(float(grid_config["z_max"])))

  # 1. 3D可視化
  fig = plt.figure(figsize=(10, 8), dpi=220)
  ax = fig.add_subplot(111, projection='3d')
  draw_box(ax, (0, 0, DETECTOR_POS_Z), (DETECTOR_SIZE_XY, DETECTOR_SIZE_XY, 1), color=(0.1, 1.0, 0.1))
  draw_box(ax, (0, 0, -DETECTOR_POS_Z), (DETECTOR_SIZE_XY, DETECTOR_SIZE_XY, 1), color=(1.0, 0.1, 0.1))
  draw_box(ax, (-BLOCK_POS_X, 0, 0), (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), color=(0.4, 0.4, 0.4), alpha=0.8)
  draw_box(ax, ( BLOCK_POS_X, 0, 0), (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), color=(0.4, 0.4, 0.4), alpha=0.8)

  ax.set_xlim(-x_limit, x_limit); ax.set_ylim(-y_limit, y_limit); ax.set_zlim(-z_limit, z_limit)
  ax.set_box_aspect((x_limit*2, y_limit*2, z_limit*2))
  ax.view_init(elev=args.angle[0], azim=args.angle[1])
  ax.set_title(f"Simulation Setup (Ground Truth)")

  out_img_path = resolve_out(args.out_img)
  plt.tight_layout()
  plt.savefig(out_img_path, bbox_inches="tight")
  plt.close(fig)

  # 2. 正解データ作成
  create_and_save_phantom(grid_config, filename=args.out_npy)

if __name__ == "__main__":
  main()