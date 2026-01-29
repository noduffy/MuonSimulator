import numpy as np
from scipy.sparse import load_npz, diags
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import sys
from pathlib import Path

# --- 共通ライブラリのインポート ---
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import paths, viz

# --- スナップショット保存関数 (元コード踏襲) ---
def save_snapshot_d(x_vector, nx, ny, nz, ranges, iteration, out_file):
  xmin, xmax, ymin, ymax, zmin, zmax = ranges
  dx, dy, dz = (xmax-xmin)/nx, (ymax-ymin)/ny, (zmax-zmin)/nz

  vol = x_vector.reshape((nz, ny, nx))
  (vL, fL), (vH, fH), (lvL, lvH) = viz.make_isos(vol, (dz, dy, dx))

  fig = plt.figure(figsize=(8, 6), dpi=220)
  ax = fig.add_subplot(111, projection="3d")

  viz.draw_detectors(ax, z_pos=80.0, size=300.0)
  viz.add_mesh(ax, vL, fL, color=(0.4, 0.7, 1.0), alpha=0.25, origin=(zmin, ymin, xmin))
  viz.add_mesh(ax, vH, fH, color=(1.0, 0.1, 0.1), alpha=0.9,  origin=(zmin, ymin, xmin))

  ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
  ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
  ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
  ax.view_init(elev=22, azim=-60)

  ax.set_title(f"Method D': Iteration {iteration:04d}\n(Spatially-varying regularization with P)")
  plt.tight_layout()
  plt.savefig(out_file, bbox_inches="tight")
  plt.close(fig)

# --- Method D' : 空間可変正則化つきCG ---
def run_cg_spatial_reg(A, b, P_1d, nx, ny, nz, ranges, lam, max_iter, interval, out_dir,
                      nonneg=False, p_eps=1e-6):
  """
  解く方程式：
    (A + lam * R) x = b
  ここで
    R = diag((1 - P)^2)  （Pが小さいほど罰則が強い）
  """

  # 0,1の端での数値安定化（P=0でもOKだが、極端なときの扱いを安定させる）
  P = np.clip(P_1d, 0.0, 1.0)

  # R の対角成分
  r_diag = (1.0 - P) ** 2
  # 極端に小さい罰則を避けたい場合は下限を入れてもよい（必要なら）
  # r_diag = np.maximum(r_diag, p_eps)

  R = diags(r_diag, 0, shape=A.shape, format="csr")
  Areg = A + (lam * R)

  # 初期化
  x = np.zeros(A.shape[1], dtype=np.float64)
  r = b - Areg.dot(x)
  p = r.copy()
  rsold = float(np.dot(r, r))

  output_dir = paths.resolve_out(out_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  print("--- Method D' (Spatially-varying Regularized CG) ---")
  print(f"Equation: (A + λR)x = b,  R = diag((1-P)^2)")
  print(f"λ = {lam}, Max Iter: {max_iter}, Interval: {interval}")
  print(f"Nonnegativity projection: {nonneg}")

  for i in range(1, max_iter + 1):
    Ap = Areg.dot(p)
    denom = float(np.dot(p, Ap))
    if denom == 0.0:
      print("  denom became 0, stopping.")
      break

    alpha = rsold / denom
    x = x + alpha * p

    # 非負制約（必要なら）
    # ※厳密なCGの性質は崩れるが、再構成としては有効なことが多い
    if nonneg:
      x = np.clip(x, 0.0, None)

    r = r - alpha * Ap
    rsnew = float(np.dot(r, r))

    # 収束判定（任意）
    if rsnew < 1e-30:
      rsold = rsnew
      print(f"  Iter {i:04d}: converged (rs={rsnew:.3e})")
      break

    p = r + (rsnew / rsold) * p
    rsold = rsnew

    if i % interval == 0 or i == max_iter:
      npy_path = output_dir / f"x_iter_{i:04d}.npy"
      img_path = output_dir / f"render_iter_{i:04d}.png"
      np.save(npy_path, x)
      save_snapshot_d(x, nx, ny, nz, ranges, i, img_path)
      print(f"  Iter {i:04d}: Saved. (rs={rsnew:.3e})")

  return x

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--prob_map", required=True, help="確率マップP(.npy) 例: method_b_result/prob_map.npy")
  parser.add_argument("--lam", type=float, default=1.0, help="正則化強度λ（大きいほどPが小さい領域を強く抑える）")
  parser.add_argument("--max_iter", type=int, default=60)
  parser.add_argument("--interval", type=int, default=10)
  parser.add_argument("--out_dir", default="method_dprime_result")
  parser.add_argument("--nonneg", action="store_true", help="非負制約を入れる（必要なら）")
  args = parser.parse_args()

  # 1. データの読み込み
  try:
    A = load_npz(paths.resolve_out("W_transpose_W_sparse.npz"))
    b = np.load(paths.resolve_out("W_transpose_y_vector.npy"))
  except Exception:
    print("Error: System Matrix not found. Run 'make cgls' first.")
    return

  # 2. 確率マップの読み込み
  try:
    P_3d = np.load(paths.resolve_out(args.prob_map))
    P_1d = P_3d.flatten()
    if P_1d.shape[0] != A.shape[1]:
      print(f"Error: Size mismatch. ProbMap:{P_1d.shape[0]}, Matrix:{A.shape[1]}")
      return
  except Exception as e:
    print(f"Error loading probability map: {e}")
    return

  # 3. 設定読み込み
  with open(paths.config_path("grid3d.yml")) as f:
    g = yaml.safe_load(f)
  nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
  ranges = (float(g["x_min"]), float(g["x_max"]),
            float(g["y_min"]), float(g["y_max"]),
            float(g["z_min"]), float(g["z_max"]))

  run_cg_spatial_reg(
    A=A, b=b, P_1d=P_1d,
    nx=nx, ny=ny, nz=nz, ranges=ranges,
    lam=args.lam,
    max_iter=args.max_iter, interval=args.interval,
    out_dir=args.out_dir,
    nonneg=args.nonneg
  )

if __name__ == "__main__":
  main()
