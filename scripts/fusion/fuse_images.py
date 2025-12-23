# ファイル名: scripts/fusion/fuse_images.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
import argparse
from pathlib import Path
import yaml
import sys
import csv

# --- パス解決 ---
def get_project_root():
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if (d / "build").exists(): return d
  return Path.cwd()

def resolve_out(name: str) -> Path:
  return get_project_root() / "build" / "outputs" / name

def config_path(name: str) -> Path:
  return get_project_root() / "configs" / name

# --- レンダリング関数 ---
def make_isos(vol, spacing, level_low=None, level_hi=None):
  vv = vol.astype(np.float32)
  pos = vv[vv > 0]
  if pos.size == 0: return ([], []), ([], []), (0.0, 0.0)

  if level_low is None:
    try: level_low = float(threshold_otsu(pos))
    except Exception: level_low = float(np.percentile(pos, 75))
      
  if level_hi is None:
    level_hi = float(np.percentile(pos, 95)) if len(pos) > 0 else 0.9
  
  if level_hi > vol.max(): level_hi = vol.max()
  if level_low >= level_hi: level_low = level_hi * 0.5

  vertsL, facesL = [], []
  try: vertsL, facesL, _, _ = marching_cubes(vv, level=level_low, spacing=spacing)
  except ValueError: pass

  vertsH, facesH = [], []
  try: vertsH, facesH, _, _ = marching_cubes(vv, level=level_hi, spacing=spacing)
  except ValueError: pass
    
  return (vertsL, facesL), (vertsH, facesH), (level_low, level_hi)

def add_mesh(ax, verts, faces, color, alpha, origin):
  if len(verts) == 0: return
  oz, oy, ox = origin 
  V = np.empty_like(verts)
  V[:, 0] = verts[:, 2] + ox; V[:, 1] = verts[:, 1] + oy; V[:, 2] = verts[:, 0] + oz
  mesh = Poly3DCollection(V[faces], linewidths=0.15, facecolor=color, edgecolor="k", alpha=alpha)
  ax.add_collection3d(mesh)

def draw_plate(ax, xmin,xmax,ymin,ymax,z, color, alpha=0.25, thick=2.0):
  z0, z1 = z - thick/2, z + thick/2
  P = np.array([
    [xmin,ymin,z0],[xmax,ymin,z0],[xmax,ymax,z0],[xmin,ymax,z0],
    [xmin,ymin,z1],[xmax,ymin,z1],[xmax,ymax,z1],[xmin,ymax,z1]
  ], dtype=float)
  F = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
  poly = Poly3DCollection([P[f] for f in F], facecolors=color, edgecolors="k", linewidths=0.3, alpha=alpha)
  ax.add_collection3d(poly)

def save_snapshot_unified(vol, ranges, out_file, z_detectors, title="Fused Result"):
  xmin, xmax, ymin, ymax, zmin, zmax = ranges
  nz, ny, nx = vol.shape
  dx = (xmax - xmin) / nx
  dy = (ymax - ymin) / ny
  dz = (zmax - zmin) / nz
  z_top, z_bot = z_detectors

  (vL, fL), (vH, fH), (lvL, lvH) = make_isos(vol, (dz, dy, dx))

  fig = plt.figure(figsize=(8, 6), dpi=220)
  ax = fig.add_subplot(111, projection='3d')

  add_mesh(ax, vL, fL, color=(0.4,0.7,1.0), alpha=0.25, origin=(zmin, ymin, xmin))
  add_mesh(ax, vH, fH, color=(1.0,0.1,0.1), alpha=0.9,  origin=(zmin, ymin, xmin))

  draw_plate(ax, xmin,xmax,ymin,ymax, z_top, color=(0.1,1.0,0.1), alpha=0.25, thick=2.0)
  draw_plate(ax, xmin,xmax,ymin,ymax, z_bot, color=(1.0,0.1,0.1), alpha=0.25, thick=2.0)

  ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
  ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
  ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
  ax.view_init(elev=22, azim=-60)
  
  ax.set_title(f"{title}\n(Iso-levels: {lvL:.2g}, {lvH:.2g})")
  
  plt.tight_layout()
  plt.savefig(out_file, bbox_inches="tight")
  plt.close(fig)
  print(f"  -> Image saved: {out_file.name}")

# --- メイン処理 ---
def main():
  parser = argparse.ArgumentParser(description="手法Aと手法Bを融合(掛け算)してMethod Cの結果を作る")
  parser.add_argument("--scat", required=True, help="手法Aのnpy (例: progressive_cgls/x_iter_0200.npy)")
  parser.add_argument("--prob", required=True, help="手法Bのnpy (例: prob_map.npy)")
  parser.add_argument("--out_npy", default="fused_result.npy", help="出力: 融合データ")
  parser.add_argument("--out_png", default="fused_render.png", help="出力: 融合画像")
  args = parser.parse_args()

  path_scat = resolve_out(args.scat)
  path_prob = resolve_out(args.prob)

  if not path_scat.exists() or not path_prob.exists():
    print(f"[Error] Inputs not found.\n  Scat: {path_scat}\n  Prob: {path_prob}")
    sys.exit(1)

  print("Loading images...")
  vol_scat = np.load(path_scat)
  vol_prob = np.load(path_prob)

  # ★★★ 修正: 形状合わせ (1D -> 3D) ★★★
  # vol_scat が1次元配列の場合、vol_prob の形状に合わせて変形する
  if vol_scat.ndim == 1 and vol_prob.ndim == 3:
    if vol_scat.size == vol_prob.size:
      print(f"  -> Reshaping Scat {vol_scat.shape} to {vol_prob.shape}")
      vol_scat = vol_scat.reshape(vol_prob.shape)
    else:
      print(f"[Error] Size mismatch: Scat {vol_scat.size} != Prob {vol_prob.size}")
      print("  -> 解像度(grid3d.yml)の設定が計算時と異なっている可能性があります。")
      sys.exit(1)
  
  # 念のため、どちらも3Dか確認
  if vol_scat.shape != vol_prob.shape:
    print(f"[Error] Shape mismatch: Scat{vol_scat.shape} != Prob{vol_prob.shape}")
    sys.exit(1)

  # --- 融合処理 (Masking) ---
  print("Fusing images (Method A * Method B)...")
  
  # Method A (散乱) × Method B (物体確率)
  vol_fused = vol_scat * vol_prob
  
  # 保存
  out_npy = resolve_out(args.out_npy)
  np.save(out_npy, vol_fused)
  print(f"[OK] Fused data saved: {out_npy}")

  # レンダリング
  try:
    with open(config_path("grid3d.yml")) as f:
      g = yaml.safe_load(f)
      ranges = (float(g["x_min"]), float(g["x_max"]), float(g["y_min"]), float(g["y_max"]), float(g["z_min"]), float(g["z_max"]))
      z_center = (ranges[4]+ranges[5])/2
    
    pairs_path = resolve_out("pairs.csv")
    z_top_list, z_bot_list = [], []
    if pairs_path.exists():
      with open(pairs_path) as f:
        reader = csv.DictReader(f)
        try:
          for row in reader:
            if "top_z" in row and "bot_z" in row:
              z_top_list.append(float(row["top_z"]))
              z_bot_list.append(float(row["bot_z"]))
        except Exception: pass
    z_top_pos = float(np.median(z_top_list)) if z_top_list else z_center + 50
    z_bot_pos = float(np.median(z_bot_list)) if z_bot_list else z_center - 50

    out_png = resolve_out(args.out_png)
    save_snapshot_unified(vol_fused, ranges, out_png, (z_top_pos, z_bot_pos), title="Method C: Fused Result")

  except Exception as e:
    print(f"[Warn] Rendering failed: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
  main()