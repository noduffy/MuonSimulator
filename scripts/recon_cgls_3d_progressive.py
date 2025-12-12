# ファイル名: scripts/recon_cgls_3d_progressive.py
import numpy as np
from scipy.sparse import load_npz
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
import yaml
import csv

# --- ユーティリティ関数 ---
def resolve_build():
  p = Path.cwd()
  for d in [p] + list(p.parents):
    if d.name == "build": return d
    if (d / "build").exists(): return d / "build"
  return Path("build")

def out_path(name: str) -> Path:
  b = resolve_build()
  o = b / "outputs"
  o.mkdir(parents=True, exist_ok=True)
  return o / name

# --- レンダリング関数 (render_recon_x_3d.py から移植) ---
def make_isos(vol, spacing, level_low=None, level_hi=None):
  # 等値面生成
  vv = vol.astype(np.float32)
  pos = vv[vv>0]
  if pos.size == 0: return ([], []), ([], []), (0.0, 0.0)
  if level_low is None:
    try: level_low = float(threshold_otsu(pos))
    except Exception: level_low = float(np.percentile(pos, 75))
  if level_hi is None: level_hi = float(np.percentile(pos, 95))
  if level_hi < level_low: level_hi = level_low + 1e-6
  
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

# --- スナップショット保存関数 ---
def save_snapshot(x_vector, nx, ny, nz, ranges, iteration, out_file, z_detectors, low_thresh=None):
  xmin, xmax, ymin, ymax, zmin, zmax = ranges
  dx, dy, dz = (xmax-xmin)/nx, (ymax-ymin)/ny, (zmax-zmin)/nz
  z_top, z_bot = z_detectors
  
  # x_vector を 3D 配列に変換
  vol = x_vector.reshape((nz, ny, nx))
  
  # 等値面生成
  (vL, fL), (vH, fH), (lvL, lvH) = make_isos(vol, (dz, dy, dx), level_low=low_thresh)
  
  # プロット作成
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
  ax.set_title(f"Iteration {iteration:04d} (low≈{lvL:.3g}, high≈{lvH:.3g})")
  
  plt.tight_layout()
  plt.savefig(out_file, bbox_inches="tight")
  plt.close(fig)

# --- プログレッシブCGLSソルバー ---
def run_cgls_progressive(A, b, nx, ny, nz, ranges, max_iter, interval, low_thresh, sub_dir, pairs_path):
  # 初期化
  x = np.zeros(A.shape[1])
  r = b - A.dot(x) 
  p = r.copy()     
  rsold = np.dot(r, r)
  
  # 出力ディレクトリを作成 (build/outputs/{sub_dir})
  output_dir = out_path(sub_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  # 検出器位置の取得 (pairs.csv から)
  z_top_list, z_bot_list = [], []
  z_center = (ranges[4]+ranges[5])/2 # (zmin+zmax)/2
  
  if pairs_path.exists():
    with open(pairs_path) as f:
      reader = csv.DictReader(f)
      try:
        for row in reader:
          if "top_z" in row and "bot_z" in row:
            z_top_list.append(float(row["top_z"]))
            z_bot_list.append(float(row["bot_z"]))
      except Exception: pass
  
  z_top_pos = float(np.median(z_top_list)) if z_top_list else z_center + (ranges[5]-ranges[4])/4
  z_bot_pos = float(np.median(z_bot_list)) if z_bot_list else z_center - (ranges[5]-ranges[4])/4
  z_detectors = (z_top_pos, z_bot_pos)
  
  print(f"--- CGLS開始 (最大 {max_iter} 回, {interval} 回ごとに保存) ---")
  print(f"画像保存先: {output_dir}")

  for i in range(1, max_iter + 1):
    Ap = A.dot(p)
    
    # ゼロ除算回避
    denom = np.dot(p, Ap)
    if denom == 0:
      print("分母が0になったため、計算を中断します。")
      break
      
    alpha = rsold / denom
    x = x + alpha * p
    r = r - alpha * Ap
    rsnew = np.dot(r, r)
    p = r + (rsnew / rsold) * p
    rsold = rsnew
    
    # 指定間隔または最後の反復で保存
    if i % interval == 0 or i == max_iter:
      # 非負制約 (物理的に負の密度はありえないためクリップ)
      x_save = np.clip(x, 0, None)
      
      # 画像を保存
      img_path = output_dir / f"render_iter_{i:04d}.png"
      save_snapshot(x_save, nx, ny, nz, ranges, i, img_path, z_detectors, low_thresh)
      print(f"Iteration {i:04d}: 画像を保存しました -> {img_path.name}")

  return x

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_iter", type=int, default=300, help="最大反復回数")
  parser.add_argument("--interval", type=int, default=20, help="画像を保存する間隔")
  parser.add_argument("--low", type=float, default=None, help="表示の閾値 (指定なしなら自動)")
  parser.add_argument("--out_dir", type=str, default="progressive_frames", help="outputs下の保存先ディレクトリ名")
  args = parser.parse_args()

  WTW_PATH = out_path("W_transpose_W_sparse.npz")
  WTy_PATH = out_path("W_transpose_y_vector.npy")
  PAIRS_PATH = out_path("pairs.csv")
  
  # 設定読み込み
  try:
    with open("configs/grid3d.yml") as f: g = yaml.safe_load(f)
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    ranges = (float(g["x_min"]), float(g["x_max"]), float(g["y_min"]), float(g["y_max"]), float(g["z_min"]), float(g["z_max"]))
    A = load_npz(WTW_PATH)
    b = np.load(WTy_PATH)
  except Exception as e:
    print(f"エラー: 必要なファイルが見つかりません ({e})。先に build_system_matrix.py を実行してください。")
    return

  # 実行
  run_cgls_progressive(A, b, nx, ny, nz, ranges, args.max_iter, args.interval, args.low, args.out_dir, PAIRS_PATH)

if __name__ == "__main__":
  main()