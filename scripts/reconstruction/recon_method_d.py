import numpy as np
from scipy.sparse import load_npz
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import sys
from pathlib import Path

# --- 共通ライブラリのインポート ---
# scripts/common を見つけるためにパスを通す
sys.path.append(str(Path(__file__).resolve().parent.parent)) 
from common import paths, viz

# --- スナップショット保存関数 (ライブラリ活用版) ---
def save_snapshot_d(x_vector, nx, ny, nz, ranges, iteration, out_file):
  xmin, xmax, ymin, ymax, zmin, zmax = ranges
  dx, dy, dz = (xmax-xmin)/nx, (ymax-ymin)/ny, (zmax-zmin)/nz
  
  # 3D変形 & 等値面計算 (vizライブラリ使用)
  vol = x_vector.reshape((nz, ny, nx))
  (vL, fL), (vH, fH), (lvL, lvH) = viz.make_isos(vol, (dz, dy, dx))

  fig = plt.figure(figsize=(8, 6), dpi=220)
  ax = fig.add_subplot(111, projection='3d')
  
  # 検出器描画 (vizライブラリ使用: 統一スタイル)
  # ※ grid3d.ymlの範囲外にある場合でも、位置関係を示すために描画
  viz.draw_detectors(ax, z_pos=80.0, size=300.0)

  # メッシュ描画 (vizライブラリ使用)
  # Low Level (Blue)
  viz.add_mesh(ax, vL, fL, color=(0.4,0.7,1.0), alpha=0.25, origin=(zmin, ymin, xmin))
  # High Level (Red)
  viz.add_mesh(ax, vH, fH, color=(1.0,0.1,0.1), alpha=0.9,  origin=(zmin, ymin, xmin))
  
  # 軸設定
  ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
  ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
  ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
  
  # 視点
  ax.view_init(elev=22, azim=-60)
  
  ax.set_title(f"Method D: Iteration {iteration:04d}\n(Prior-constrained CGLS)")
  
  plt.tight_layout()
  plt.savefig(out_file, bbox_inches="tight")
  plt.close(fig)

# --- Method D: 確率マップ制約付きCGLS ---
def run_cgls_constrained(A, b, prob_map_1d, nx, ny, nz, ranges, max_iter, interval, out_dir):
  # 初期化
  x = np.zeros(A.shape[1])
  r = b - A.dot(x) 
  p = r.copy()     
  rsold = np.dot(r, r)
  
  # 出力先解決 (pathsライブラリ使用)
  output_dir = paths.resolve_out(out_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  print(f"--- Method D (Constrained CGLS) ---")
  print(f"Prior Map applied. Max Iter: {max_iter}")

  for i in range(1, max_iter + 1):
    Ap = A.dot(p)
    denom = np.dot(p, Ap)
    if denom == 0: break
      
    alpha = rsold / denom
    x = x + alpha * p
    
    # ★ 制約適用 ★
    x = np.clip(x, 0, None)  # 非負制約
    x = x * prob_map_1d      # 空間確率制約 (手法B)

    r = r - alpha * Ap
    rsnew = np.dot(r, r)
    p = r + (rsnew / rsold) * p
    rsold = rsnew
    
    if i % interval == 0 or i == max_iter:
      npy_path = output_dir / f"x_iter_{i:04d}.npy"
      img_path = output_dir / f"render_iter_{i:04d}.png"
      
      np.save(npy_path, x)
      save_snapshot_d(x, nx, ny, nz, ranges, i, img_path)
      print(f"  Iter {i:04d}: Saved.")

  return x

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--prob_map", required=True, help="手法Bの確率マップ(.npy)")
  parser.add_argument("--max_iter", type=int, default=60)
  parser.add_argument("--interval", type=int, default=10)
  parser.add_argument("--out_dir", default="method_d_result")
  args = parser.parse_args()

  # 1. データの読み込み
  try:
    A = load_npz(paths.resolve_out("W_transpose_W_sparse.npz"))
    b = np.load(paths.resolve_out("W_transpose_y_vector.npy"))
  except:
    print("Error: System Matrix not found. Run 'make cgls' first.")
    return

  # 2. 確率マップの読み込み
  try:
    prob_map_3d = np.load(paths.resolve_out(args.prob_map))
    prob_map_1d = prob_map_3d.flatten()
    if prob_map_1d.shape[0] != A.shape[1]:
      print(f"Error: Size mismatch. ProbMap:{prob_map_1d.shape[0]}, Matrix:{A.shape[1]}")
      return
  except Exception as e:
    print(f"Error loading probability map: {e}")
    return

  # 3. 設定読み込み
  with open(paths.config_path("grid3d.yml")) as f: g = yaml.safe_load(f)
  nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
  ranges = (float(g["x_min"]), float(g["x_max"]), float(g["y_min"]), float(g["y_max"]), float(g["z_min"]), float(g["z_max"]))

  run_cgls_constrained(A, b, prob_map_1d, nx, ny, nz, ranges, args.max_iter, args.interval, args.out_dir)

if __name__ == "__main__":
  main()