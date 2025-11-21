#!/usr/bin/env python3
import argparse, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from pathlib import Path

def load_vol(path):
  return np.load(path)

def load_grid(gpath):
  import yaml
  with open(gpath) as f:
    return yaml.safe_load(f)

def draw_plate(ax, z, x_min, x_max, y_min, y_max, color, alpha=0.25):
  X = [x_min, x_max, x_max, x_min]
  Y = [y_min, y_min, y_max, y_max]
  Z = [z, z, z, z]
  verts = [list(zip(X, Y, Z))]
  pc = Poly3DCollection([verts[0]], alpha=alpha, facecolor=color, edgecolor='k', linewidths=0.3)
  ax.add_collection3d(pc)

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--vol", default="build/outputs/recon3d_vol.npy")
  ap.add_argument("--grid3d", default="configs/grid3d.yml")
  ap.add_argument("--pairs", default="build/outputs/pairs.csv")
  ap.add_argument("--low", type=float, default=None, help="absolute iso low threshold")
  ap.add_argument("--hi",  type=float, default=None, help="absolute iso high threshold")
  ap.add_argument("--plow", type=float, default=None, help="percentile for low iso (e.g., 70)")
  ap.add_argument("--phi",  type=float, default=None, help="percentile for high iso (e.g., 95)")
  ap.add_argument("--mask-z-mm", type=float, default=0.0, help="zero voxels within this [mm] from z_min/z_max")
  ap.add_argument("--view", choices=["iso","side","front","top"], default="iso")
  ap.add_argument("--elev", type=float, default=None)
  ap.add_argument("--azim", type=float, default=None)
  ap.add_argument("--dpi", type=int, default=160)
  ap.add_argument("--out", default="recon3d_view.png")
  args = ap.parse_args()

  V = load_vol(args.vol)  # shape = (nz, ny, nx)
  g = load_grid(args.grid3d)
  nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
  x_min, x_max = float(g["x_min"]), float(g["x_max"])
  y_min, y_max = float(g["y_min"]), float(g["y_max"])
  z_min, z_max = float(g["z_min"]), float(g["z_max"])

  # Z端の簡易マスク（板近傍アーチファクト抑制）
  mmm = args.mask_z_mm
  if mmm and mmm > 0:
    zc = np.linspace(z_min, z_max, nz, endpoint=False) + (z_max - z_min) / nz * 0.5
    mask = (zc <= z_min + mmm) | (zc >= z_max - mmm)
    V[mask, :, :] = 0.0

  flat = V.ravel()
  flat_pos = flat[flat > 0]
  if flat_pos.size == 0:
    print("[WARN] volume is all zeros after masking; nothing to render", file=sys.stderr)

  # しきい値（percentile優先）
  if args.plow is not None or args.phi is not None:
    pL = 70.0 if args.plow is None else float(args.plow)
    pH = 95.0 if args.phi  is None else float(args.phi)
    low = np.percentile(flat, pL)
    hi  = np.percentile(flat, pH)
  else:
    low = float(args.low) if args.low is not None else (np.percentile(flat, 70.0) if flat_pos.size else 0.0)
    hi  = float(args.hi)  if args.hi  is not None else (np.percentile(flat, 95.0) if flat_pos.size else (low*1.05 if low>0 else 0.0))
  if hi < low:
    hi = low * 1.05

  # --- marching cubes helper ---
  # skimage の頂点は (z, y, x)。spacing=(dz, dy, dx) を与えた上で、
  # (x_min, y_min, z_min) を原点補正し (x, y, z) に並べ替える。
  def mc(level):
    dz = (z_max - z_min) / nz
    dy = (y_max - y_min) / ny
    dx = (x_max - x_min) / nx
    verts_zyx, faces, _, _ = marching_cubes(V.astype(np.float32), level=level, spacing=(dz, dy, dx))
    X = verts_zyx[:, 2] + x_min
    Y = verts_zyx[:, 1] + y_min
    Z = verts_zyx[:, 0] + z_min
    verts_xyz = np.column_stack([X, Y, Z])
    return verts_xyz, faces

  fig = plt.figure(figsize=(7.2, 7.2), dpi=args.dpi)
  ax = fig.add_subplot(111, projection="3d")

  # 等値面（低・高の2段）。ここが“検出器の外に見える”原因だったので mc を修正。
  try:
    v1, f1 = mc(low)
    m1 = Poly3DCollection(v1[f1], alpha=0.30, facecolor='C0', edgecolor='none')
    ax.add_collection3d(m1)
  except Exception:
    pass
  try:
    v2, f2 = mc(hi)
    m2 = Poly3DCollection(v2[f2], alpha=0.80, facecolor='C3', edgecolor='none')
    ax.add_collection3d(m2)
  except Exception:
    pass

  # プレート（pairs の中央値 z を使う。なければ z_min/z_max 近傍）
  zt, zb = None, None
  try:
    import csv, statistics
    tz, bz = [], []
    with open(args.pairs) as f:
      r = csv.DictReader(f)
      for i, row in enumerate(r):
        tz.append(float(row["top_z"])); bz.append(float(row["bot_z"]))
        if i > 10000: break
    zt, zb = statistics.median(tz), statistics.median(bz)
  except Exception:
    pass
  if zt is None: zt = z_max - (z_max - z_min) * 0.2
  if zb is None: zb = z_min + (z_max - z_min) * 0.2
  draw_plate(ax, zb, x_min, x_max, y_min, y_max, color="tab:red",   alpha=0.25)
  draw_plate(ax, zt, x_min, x_max, y_min, y_max, color="tab:green", alpha=0.25)

  ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
  ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")

  if args.view == "side":
    ax.view_init(elev=0 if args.elev is None else args.elev,
                 azim=0 if args.azim is None else args.azim)
  elif args.view == "front":
    ax.view_init(elev=0 if args.elev is None else args.elev,
                 azim=90 if args.azim is None else args.azim)
  elif args.view == "top":
    ax.view_init(elev=90 if args.elev is None else args.elev,
                 azim=0 if args.azim is None else args.azim)
  else:
    ax.view_init(elev=25 if args.elev is None else args.elev,
                 azim=45 if args.azim is None else args.azim)

  title = f"3D Reconstruction  (low≈{low:.3g}, high≈{hi:.3g})"
  if args.view != "iso": title += f"  view={args.view}"
  ax.set_title(title)

  plt.tight_layout()
  out = args.out  # ← 出力先の仕様は変更しない
  Path(out).parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(out, dpi=args.dpi)
  print(f"[OK] wrote PNG: {Path(out).resolve()}")

if __name__ == "__main__":
  sys.exit(main())
