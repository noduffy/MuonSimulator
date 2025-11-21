#!/usr/bin/env python3
import argparse, sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path

# === robust path helpers (★修正) ==========================================
def resolve_build_dir():
  """今いる場所やスクリプトの場所から、一番近い build ディレクトリを返す。"""
  candidates = []
  # 1) CWD とその親を走査
  p = Path.cwd()
  for d in [p] + list(p.parents):
    if d.name == "build":
      return d
    if (d / "build").exists():
      candidates.append(d / "build")
  # 2) スクリプト位置からも走査
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if d.name == "build":
      return d
    if (d / "build").exists():
      candidates.append(d / "build")
  # 3) 最後の手段
  return candidates[0] if candidates else Path("build")

def resolve_in(p: str):
  """入力ファイル: 存在すればそのまま。無ければ build/outputs/ を探す。"""
  P = Path(p)
  if P.is_absolute() and P.exists():
    return P
  if P.exists():
    return P
  build = resolve_build_dir()
  alt = build / "outputs" / P.name
  return alt if alt.exists() else P  # 見つからなければそのまま（エラーは呼び出し側）

def resolve_out(name: str):
  """出力ファイル: どこで実行しても build/outputs/<basename> に1回だけ置く。"""
  build = resolve_build_dir()
  outdir = build / "outputs"
  outdir.mkdir(parents=True, exist_ok=True)
  return outdir / Path(name).name
# ==========================================================================

def load_Wnpz(fname):
  Z = np.load(fname, allow_pickle=False)
  row, col, data, shape = Z["row"], Z["col"], Z["data"], tuple(Z["shape"])
  W = sp.coo_matrix((data,(row,col)), shape=shape).tocsr()
  nx = int(Z["nx"]) if "nx" in Z else int(round(shape[1] ** (1/3)))
  ny = int(Z["ny"]) if "ny" in Z else nx
  nz = int(Z["nz"]) if "nz" in Z else nx
  return W, (nx,ny,nz)

def mlem(W, y, iters=50, eps=1e-12):
  m,n = W.shape
  x = np.ones(n, dtype=np.float64)
  WT = W.T.tocsr()
  denom = WT.dot(np.ones(m)); denom[denom==0] = 1.0
  for _ in range(iters):
    Wx = W.dot(x); Wx[Wx==0] = 1.0
    ratio = y / Wx
    x *= (WT.dot(ratio)) / denom
    x = np.maximum(x, eps)
  return x

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, default=100)
  ap.add_argument("--W", default="W3D_coo.npz")
  ap.add_argument("--y", default="y_theta2_3d.npy")
  args = ap.parse_args()

  W_path = resolve_in(args.W)   # ★修正: 入力は自動探索
  y_path = resolve_in(args.y)

  if not W_path.exists():
    sys.exit(f"[ERR] not found W: {W_path}")
  if not y_path.exists():
    sys.exit(f"[ERR] not found y: {y_path}")

  W, (nx,ny,nz) = load_Wnpz(W_path)
  y = np.load(y_path)
  print("W shape:", W.shape, "grid:", (nx,ny,nz), " m=", len(y))

  x = mlem(W, y, iters=args.iters)
  vol = x.reshape(nz, ny, nx)  # (z,y,x)

  np.save(resolve_out("recon3d_vol.npy"), vol)

  cz, cy, cx = nz//2, ny//2, nx//2
  plt.figure(); plt.imshow(vol[cz,:,:], origin="lower"); plt.colorbar(); plt.title(f"XY z={cz} iters={args.iters}")
  plt.savefig(resolve_out("recon3d_xy.png"), dpi=150); plt.close()
  plt.figure(); plt.imshow(vol[:,cy,:], origin="lower"); plt.colorbar(); plt.title(f"XZ y={cy} iters={args.iters}")
  plt.savefig(resolve_out("recon3d_xz.png"), dpi=150); plt.close()
  plt.figure(); plt.imshow(vol[:,:,cx], origin="lower"); plt.colorbar(); plt.title(f"YZ x={cx} iters={args.iters}")
  plt.savefig(resolve_out("recon3d_yz.png"), dpi=150); plt.close()

  print("[OK] wrote: recon3d_vol.npy / recon3d_{xy,xz,yz}.png ->", resolve_build_dir() / "outputs")

if __name__ == "__main__":
  sys.exit(main())
