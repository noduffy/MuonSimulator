#!/usr/bin/env python3
import numpy as np, scipy.sparse as sp, yaml, argparse, pathlib, sys, csv
from pathlib import Path

# === outputs helper ===（あなたの現行版にあるブロックはそのまま残してOK）
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
# === end helper ===

def load_yaml(p):
  with open(p) as f:
    return yaml.safe_load(f)

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--pairs", default="pairs.csv")
  ap.add_argument("--rays", default="rays.csv")
  ap.add_argument("--grid", default="configs/grid.yml")
  ap.add_argument("--outW", default="W_coo.npz")
  ap.add_argument("--outy", default="y_theta2.npy")
  args = ap.parse_args()

  pairs_csv = pathlib.Path(args.pairs)
  rays_csv = pathlib.Path(args.rays)
  if not pairs_csv.exists() or not rays_csv.exists():
    print("pairs.csv / rays.csv not found", file=sys.stderr)
    sys.exit(2)

  grid = load_yaml(args.grid)
  nx, ny = int(grid["nx"]), int(grid["ny"])
  x_min, x_max = float(grid["x_min"]), float(grid["x_max"])
  y_min, y_max = float(grid["y_min"]), float(grid["y_max"])
  roi_z = float(grid.get("roi_z", 0.0))  # 無ければ0.0を既定

  # ★ 変更ここから：レイを roi_z の平面に投影し、その(x,y)が落ちるボクセルに1を立てる簡易版
  # ボクセル幅
  sx = (x_max - x_min) / nx
  sy = (y_max - y_min) / ny

  def voxel_index(x, y):
    ix = int((x - x_min) // sx)
    iy = int((y - y_min) // sy)
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
      return None
    return iy * nx + ix  # 行メジャー

  # pairs と rays を同じ順で走査（生成順が一致している前提）
  rows, cols, data = [], [], []
  y_vals = []

  # rays の theta2 を配列化
  theta2_list = []
  with open(rays_csv) as fr:
    rr = csv.DictReader(fr)
    for row in rr:
      theta2_list.append(float(row["theta2"]))

  # pairs を読みつつ、対応する theta2 を使う
  used = 0
  with open(pairs_csv) as fp:
    rp = csv.DictReader(fp)
    for i, row in enumerate(rp):
      if i >= len(theta2_list):
        break
      tx, ty, tz = float(row["top_x"]), float(row["top_y"]), float(row["top_z"])
      bx, by, bz = float(row["bot_x"]), float(row["bot_y"]), float(row["bot_z"])

      vz = (bz - tz)
      if abs(vz) < 1e-9:
        # ほぼ平行で roi_z に到達しない
        continue
      t = (roi_z - tz) / vz
      if t < 0.0 or t > 1.0:
        # 線分の範囲外ならこのスライスは通らない
        continue

      x = tx + t * (bx - tx)
      y = ty + t * (by - ty)
      col = voxel_index(x, y)
      if col is None:
        # ROIの外
        continue

      # 測定ベクトル（このレイの theta2）
      y_vals.append(theta2_list[i])

      # この行（測定）に対して、そのボクセルに重み1
      rows.append(used)   # 行インデックスは "採用された測定の連番"
      cols.append(col)
      data.append(1.0)
      used += 1
  # ★ 変更ここまで

  n = nx * ny
  W = sp.coo_matrix((data, (rows, cols)), shape=(len(y_vals), n))
  outW = resolve_out(args.outW)
  outy = resolve_out(args.outy)
  np.savez_compressed(outW, row=W.row, col=W.col, data=W.data, shape=W.shape)
  np.save(outy, np.asarray(y_vals, dtype=np.float64))

  print(f"[OK] wrote W/y: {outW}, {outy}  (nvoxels={n} nnz={len(data)})")

if __name__ == "__main__":
  main()
