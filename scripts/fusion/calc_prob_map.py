# ファイル名: scripts/fusion/calc_prob_map.py
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

# --- レンダリング関数 (Method A と統一) ---
def make_isos(vol, spacing, level_low=None, level_hi=None):
  # 等値面生成
  vv = vol.astype(np.float32)
  pos = vv[vv > 0]
  
  # データが空、または全て0の場合の安全策
  if pos.size == 0: 
    return ([], []), ([], []), (0.0, 0.0)

  # 閾値の決定 (Method Aと同じロジック + 確率マップ用の調整)
  if level_low is None:
    try:
      # Otsu法で自動決定を試みる
      level_low = float(threshold_otsu(pos))
    except Exception:
      # 失敗したら確率50%をデフォルトにする
      level_low = 0.5
      
  if level_hi is None:
    # 高閾値は分布の95%点
    level_hi = float(np.percentile(pos, 95)) if len(pos) > 0 else 0.9
    
  # 確率マップなので 1.0 を超えないようにクリップ
  if level_hi > 0.99: level_hi = 0.99
  if level_low > level_hi: level_low = level_hi * 0.5

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
  # 検出器板を描画
  z0, z1 = z - thick/2, z + thick/2
  P = np.array([
    [xmin,ymin,z0],[xmax,ymin,z0],[xmax,ymax,z0],[xmin,ymax,z0],
    [xmin,ymin,z1],[xmax,ymin,z1],[xmax,ymax,z1],[xmin,ymax,z1]
  ], dtype=float)
  F = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
  poly = Poly3DCollection([P[f] for f in F], facecolors=color, edgecolors="k", linewidths=0.3, alpha=alpha)
  ax.add_collection3d(poly)

def save_snapshot_unified(vol, ranges, out_file, z_detectors, title="Probability Map"):
  xmin, xmax, ymin, ymax, zmin, zmax = ranges
  nz, ny, nx = vol.shape
  dx = (xmax - xmin) / nx
  dy = (ymax - ymin) / ny
  dz = (zmax - zmin) / nz
  z_top, z_bot = z_detectors

  # 等値面生成 (自動閾値)
  (vL, fL), (vH, fH), (lvL, lvH) = make_isos(vol, (dz, dy, dx))

  # プロット作成 (Method Aと同じ設定)
  fig = plt.figure(figsize=(8, 6), dpi=220)
  ax = fig.add_subplot(111, projection='3d')

  # メッシュ描画
  add_mesh(ax, vL, fL, color=(0.4,0.7,1.0), alpha=0.25, origin=(zmin, ymin, xmin))
  add_mesh(ax, vH, fH, color=(1.0,0.1,0.1), alpha=0.9,  origin=(zmin, ymin, xmin))

  # 検出器板描画
  draw_plate(ax, xmin,xmax,ymin,ymax, z_top, color=(0.1,1.0,0.1), alpha=0.25, thick=2.0)
  draw_plate(ax, xmin,xmax,ymin,ymax, z_bot, color=(1.0,0.1,0.1), alpha=0.25, thick=2.0)

  # 軸設定
  ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
  ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
  ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
  ax.view_init(elev=22, azim=-60)

  # タイトル
  ax.set_title(f"{title}\n(Iso-levels: {lvL:.2f}, {lvH:.2f})")

  plt.tight_layout()
  plt.savefig(out_file, bbox_inches="tight")
  plt.close(fig)
  print(f"  -> Image saved: {out_file.name}")

# --- メイン処理 ---
def main():
  parser = argparse.ArgumentParser(description="FluxMapから物体存在確率マップ(Method B)を作成する")
  parser.add_argument("--all", default="flux_all.npy", help="分母: 全Flux")
  parser.add_argument("--straight", default="flux_straight.npy", help="分子: 直進Flux")
  parser.add_argument("--out_npy", default="prob_map.npy", help="出力: 確率マップデータ")
  parser.add_argument("--out_png", default="prob_map_render.png", help="出力: 確率マップ画像")
  args = parser.parse_args()

  # 1. データの読み込み
  path_all = resolve_out(args.all)
  path_str = resolve_out(args.straight)
  
  if not path_all.exists() or not path_str.exists():
    print("[Error] Flux map not found. Run 'make make-maps' first.")
    sys.exit(1)

  print("Loading flux maps...")
  flux_all = np.load(path_all)
  flux_str = np.load(path_str)

  # 2. 確率計算
  print("Calculating Probability Map (P_obj = 1 - P_str)...")
  with np.errstate(divide='ignore', invalid='ignore'):
    ratio_straight = flux_str / flux_all
  
  ratio_straight = np.nan_to_num(ratio_straight, nan=1.0, posinf=1.0, neginf=1.0)
  ratio_straight = np.clip(ratio_straight, 0.0, 1.0)
  prob_obj = 1.0 - ratio_straight
  
  # 3. 保存 (Method C用)
  out_npy_path = resolve_out(args.out_npy)
  np.save(out_npy_path, prob_obj)
  print(f"[OK] Probability Map saved: {out_npy_path}")

  # 4. レンダリング (Method Aと統一したスタイル)
  try:
    # Configから範囲情報を取得
    with open(config_path("grid3d.yml")) as f:
      g = yaml.safe_load(f)
      ranges = (float(g["x_min"]), float(g["x_max"]), float(g["y_min"]), float(g["y_max"]), float(g["z_min"]), float(g["z_max"]))
      z_center = (ranges[4]+ranges[5])/2
    
    # 検出器位置の取得 (pairs.csv から) - Method Aと同じロジック
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
    z_detectors = (z_top_pos, z_bot_pos)

    # 描画実行
    out_png_path = resolve_out(args.out_png)
    save_snapshot_unified(prob_obj, ranges, out_png_path, z_detectors, title="Method B: Probability Map")
    
  except Exception as e:
    print(f"[Warn] Rendering skipped or failed: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
  main()