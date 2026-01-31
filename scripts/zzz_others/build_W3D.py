#!/usr/bin/env python3
import argparse, csv, sys, yaml, math, pathlib
import numpy as np
import scipy.sparse as sp
from pathlib import Path

def median_top_bottom_z(pairs_csv, nsample=20000):
  import csv, statistics
  tz, bz = [], []
  with open(pairs_csv) as f:
    r = csv.DictReader(f)
    for i,row in enumerate(r):
      tz.append(float(row["top_z"])); bz.append(float(row["bot_z"]))
      if nsample and i >= nsample: break
  if not tz or not bz:
    return None, None
  return statistics.median(tz), statistics.median(bz)

def clip_segment_to_zgap(p0, p1, z_lo, z_hi):
  # p0, p1: (x,y,z); 返り値: クリップ後の2点 or None
  x0,y0,z0 = p0; x1,y1,z1 = p1
  if z0 == z1:
    return None
  t0 = (z_lo - z0) / (z1 - z0)
  t1 = (z_hi - z0) / (z1 - z0)
  t_enter, t_exit = min(t0,t1), max(t0,t1)
  # 元の区間 [0,1] と交差しなければ無効
  if t_exit <= 0.0 or t_enter >= 1.0:
    return None
  t_enter = max(0.0, t_enter)
  t_exit  = min(1.0, t_exit)
  if t_exit - t_enter <= 1e-9:
    return None
  def lerp(a,b,t): return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t)
  q0 = lerp(p0, p1, t_enter)
  q1 = lerp(p0, p1, t_exit)
  return q0, q1


# --- helpers: build/outputs 解決は現状踏襲 ---
def _find_build_dir():
  cwd_build = Path.cwd() / "build"
  if cwd_build.exists():
    return cwd_build
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if (d / "build").exists():
      return d / "build"
  return Path("build")

def resolve_out(p):
  P = Path(str(p))
  if P.is_absolute():
    return P
  build = _find_build_dir()
  outdir = build / "outputs"
  outdir.mkdir(parents=True, exist_ok=True)
  if len(P.parts) >= 1 and P.parts[0] == "build":
    rel = Path(*P.parts[1:]) if len(P.parts) > 1 else Path(P.name)
    return outdir / rel
  return outdir / P.name

def load_yaml(p):
  with open(p) as f:
    return yaml.safe_load(f)

# --- 体積AABBへの線分クリッピング（既存ロジック踏襲） ---
def clip_segment_to_box(p0, p1, xmin, xmax, ymin, ymax, zmin, zmax):
  t0, t1 = 0.0, 1.0
  dx, dy, dz = p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]
  def check(p, q, t0, t1):
    if abs(p) < 1e-12:
      return (t0, t1, q >= 0.0)
    t = q / p
    if p < 0:
      if t > t1: return (t0, t1, False)
      if t > t0: t0 = t
    else:
      if t < t0: return (t0, t1, False)
      if t < t1: t1 = t
    return (t0, t1, True)
  for p,q in [(-dx, p0[0]-xmin), (dx, xmax-p0[0]),
              (-dy, p0[1]-ymin), (dy, ymax-p0[1]),
              (-dz, p0[2]-zmin), (dz, zmax-p0[2])]:
    t0,t1,ok = check(p,q,t0,t1)
    if not ok: return None
  if t0>t1: return None
  c0 = (p0[0]+dx*t0, p0[1]+dy*t0, p0[2]+dz*t0)
  c1 = (p0[0]+dx*t1, p0[1]+dy*t1, p0[2]+dz*t1)
  return c0, c1, t0, t1

# --- 3D グリッド交差（Amanatides-Woo） ---
def traverse_3d(p0, p1, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax):
  sx = (xmax - xmin) / nx
  sy = (ymax - ymin) / ny
  sz = (zmax - zmin) / nz

  clipped = clip_segment_to_box(p0, p1, xmin, xmax, ymin, ymax, zmin, zmax)
  if clipped is None: return []
  c0, c1, _, _ = clipped
  x0,y0,z0 = c0; x1,y1,z1 = c1
  dx,dy,dz = x1-x0, y1-y0, z1-z0
  seg_len = math.sqrt(dx*dx + dy*dy + dz*dz)
  if seg_len < 1e-12: return []

  ix = int((x0 - xmin) / sx); ix = max(0, min(nx-1, ix))
  iy = int((y0 - ymin) / sy); iy = max(0, min(ny-1, iy))
  iz = int((z0 - zmin) / sz); iz = max(0, min(nz-1, iz))

  stepx = 1 if dx>0 else (-1 if dx<0 else 0)
  stepy = 1 if dy>0 else (-1 if dy<0 else 0)
  stepz = 1 if dz>0 else (-1 if dz<0 else 0)

  def next_plane(origin, d, cell, step, cell_size, minb):
    if step>0:
      boundary = minb + (cell+1)*cell_size
    elif step<0:
      boundary = minb + (cell)*cell_size
    else:
      return float('inf')
    return (boundary - origin) / (d + (1e-300 if d==0 else 0))

  tx = next_plane(x0, dx, ix, stepx, sx, xmin)
  ty = next_plane(y0, dy, iy, stepy, sy, ymin)
  tz = next_plane(z0, dz, iz, stepz, sz, zmin)
  t_curr = 0.0

  voxels = []
  while 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
    t_next = min(tx, ty, tz, 1.0)
    length = (t_next - t_curr) * seg_len
    if length > 0:
      col = (iz*ny + iy)*nx + ix
      voxels.append((col, length))
    if t_next >= 1.0: break
    if tx <= ty and tx <= tz:
      ix += stepx; tx = next_plane(x0, dx, ix, stepx, sx, xmin)
    elif ty <= tz:
      iy += stepy; ty = next_plane(y0, dy, iy, stepy, sy, ymin)
    else:
      iz += stepz; tz = next_plane(z0, dz, iz, stepz, sz, zmin)
    t_curr = t_next
  return voxels

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--pairs", default="build/outputs/pairs.csv")
  ap.add_argument("--rays",  default="build/outputs/rays.csv")
  ap.add_argument("--grid3d", default="configs/grid3d.yml")
  ap.add_argument("--outW",   default="W3D_coo.npz")
  ap.add_argument("--outy",   default="y_theta2_3d.npy")
  ap.add_argument("--max_rays", type=int, default=0, help="0=all")
  # 追加オプション
  ap.add_argument("--no-row-norm", action="store_true", help="行正規化を無効化")
  ap.add_argument("--z-mask-mm", type=float, default=0.0, help="Z端からこの距離[mm]以内のボクセル寄与を無視")
  args = ap.parse_args()

  g = load_yaml(args.grid3d)
  # 既存: g = load_yaml(...), nx,ny,nz,... の直後あたりに追加
  z_top_med, z_bot_med = median_top_bottom_z(args.pairs)
  if z_top_med is None:
    # 取得失敗時は体積の20%内側を仮採用
    z_top_med = zmax - (zmax - zmin) * 0.2
    z_bot_med = zmin + (zmax - zmin) * 0.2
  # 端のアーチファクトを避けるマージン Δ
  dz_margin = 8.0  # [mm] お好みで 8〜12
  z_gap_lo = min(z_top_med, z_bot_med) + dz_margin
  z_gap_hi = max(z_top_med, z_bot_med) - dz_margin
  
  nx,ny,nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
  xmin,xmax = float(g["x_min"]), float(g["x_max"])
  ymin,ymax = float(g["y_min"]), float(g["y_max"])
  zmin,zmax = float(g["z_min"]), float(g["z_max"])
  nvox = nx*ny*nz
  nxny = nx*ny
  sz = (zmax - zmin) / nz

  # y(θ²)
  theta2 = []
  with open(args.rays) as fr:
    rr = csv.DictReader(fr)
    for row in rr:
      theta2.append(float(row["theta2"]))

  rows, cols, data = [], [], []
  y_vals = []
  used = 0

  with open(args.pairs) as fp:
    rp = csv.DictReader(fp)
    for i, row in enumerate(rp):
      if args.max_rays and used >= args.max_rays: break
      if i >= len(theta2): break

      tx,ty,tz = float(row["top_x"]), float(row["top_y"]), float(row["top_z"])
      bx,by,bz = float(row["bot_x"]), float(row["bot_y"]), float(row["bot_z"])

      seg = clip_segment_to_box((tx,ty,tz), (bx,by,bz), xmin,xmax,ymin,ymax,zmin,zmax)
      if seg is None:
        continue
      seg_start, seg_end, _, _ = seg

      gap_seg = clip_segment_to_zgap(seg_start, seg_end, z_gap_lo, z_gap_hi)
      if gap_seg is None:
        continue
      g0, g1 = gap_seg
      vox = traverse_3d(g0, g1, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
      if not vox:
        continue

      # Z端マスク（オプション）
      if args.z_mask_mm and args.z_mask_mm > 0.0:
        keep = []
        z_lo = zmin + args.z_mask_mm
        z_hi = zmax - args.z_mask_mm
        for col, length in vox:
          iz = col // nxny
          zc = zmin + (iz + 0.5) * sz
          if z_lo <= zc <= z_hi:
            keep.append((col, length))
        vox = keep
        if not vox:
          continue

      for col, length in vox:
        rows.append(used)
        cols.append(col)
        data.append(length)
      y_vals.append(theta2[i])
      used += 1

  # COO として構築
  W = sp.coo_matrix((data, (rows, cols)), shape=(len(y_vals), nvox))

  # --- 行正規化（デフォルトON）---
  if not args.no_row_norm and W.shape[0] > 0:
    # 行和（各レイの総パス長）
    rowsum = np.bincount(W.row, weights=W.data, minlength=W.shape[0]).astype(np.float64)
    safe = np.maximum(rowsum, 1e-12)
    scale = 1.0 / safe
    # COO の data を直接スケール
    W.data *= scale[W.row]
    # 参考統計
    q50 = float(np.median(rowsum))
    q95 = float(np.percentile(rowsum, 95))
    q99 = float(np.percentile(rowsum, 99))
    print(f"[row-norm] rows={W.shape[0]} rowsum med={q50:.3g} p95={q95:.3g} p99={q99:.3g}")

  outW = resolve_out(args.outW)
  outy = resolve_out(args.outy)
  np.savez_compressed(outW, row=W.row, col=W.col, data=W.data, shape=W.shape,
                      nx=nx, ny=ny, nz=nz, x_min=xmin, x_max=xmax,
                      y_min=ymin, y_max=ymax, z_min=zmin, z_max=zmax)
  np.save(outy, np.asarray(y_vals, dtype=np.float64))

  print(f"[OK] wrote 3D W/y: {outW}, {outy}  (nvox={nvox} nnz={len(W.data)} m={len(y_vals)})")
  print(f"[OK] ... (used_rate={len(y_vals)/max(len(theta2),1):.2f})")

if __name__ == "__main__":
  sys.exit(main())
