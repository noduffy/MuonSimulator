# ファイル名: plot_simple_recon3d.py
import argparse, yaml, numpy as np
from pathlib import Path
import csv

# matplotlib は3D描画を使う
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu

# --- ユーティリティ関数 (元のコードから流用) ---
def resolve_build():
  p = Path.cwd()
  for d in [p] + list(p.parents):
    if d.name == "build": return d
    if (d / "build").exists(): return d / "build"
  return Path("build")
def out_path(name: str) -> Path:
  b = resolve_build(); o = b / "outputs"; o.mkdir(parents=True, exist_ok=True)
  return o / name

def make_isos(vol, spacing, level_low=None, level_hi=None):
  # 等値面生成 (元のコードと同じロジック)
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
  # 検出器板を描画 (元のコードと同じロジック)
  z0, z1 = z - thick/2, z + thick/2
  P = np.array([
    [xmin,ymin,z0],[xmax,ymin,z0],[xmax,ymax,z0],[xmin,ymax,z0],
    [xmin,ymin,z1],[xmax,ymin,z1],[xmax,ymax,z1],[xmin,ymax,z1]
  ], dtype=float)
  F = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
  poly = Poly3DCollection([P[f] for f in F], facecolors=color, edgecolors="k", linewidths=0.3, alpha=alpha)
  ax.add_collection3d(poly)

# --- メイン関数 ---
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--vec", default="build/outputs/W_transpose_y_vector.npy", help="W^T y ベクトルファイル")
  ap.add_argument("--grid3d", default="configs/grid3d.yml")
  ap.add_argument("--pairs", default="build/outputs/pairs.csv", help="元のpairs.csv (検出器位置確認用)")
  ap.add_argument("--low", type=float, default=None, help="青の等値面レベル（指定なければ自動）")
  ap.add_argument("--hi",  type=float, default=None, help="赤の等値面レベル（指定なければ自動）")
  ap.add_argument("--dpi", type=int, default=220)
  ap.add_argument("--out", default="simple_recon3d_render.png", help="出力PNGファイル名")
  args = ap.parse_args()

  # --- 1. グリッド情報の読み込み ---
  try: g = yaml.safe_load(Path(args.grid3d).read_text())
  except FileNotFoundError: print(f"[ERR] グリッド設定ファイルが見つかりません: {args.grid3d}"); return
    
  nx,ny,nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
  xmin,xmax = float(g["x_min"]), float(g["x_max"])
  ymin,ymax = float(g["y_min"]), float(g["y_max"])
  zmin,zmax = float(g["z_min"]), float(g["z_max"])
  sx = (xmax-xmin)/nx; sy=(ymax-ymin)/ny; sz=(zmax-zmin)/nz
  
  # --- 2. W^T y ベクトルの読み込みと3Dボリュームへの変換 ---
  try: wty_vector = np.load(args.vec)
  except FileNotFoundError: print(f"[ERR] W^T y ベクトルファイルが見つかりません: {args.vec}"); return
  
  if wty_vector.size != nx * ny * nz:
      print(f"[ERR] ベクトルサイズがグリッドサイズ ({nx*ny*nz}) と一致しません: {wty_vector.size}"); return
      
  # WTy (N_VOXELS,) を (nz, ny, nx) の3D配列にリシェイプ
  vol = wty_vector.reshape((nz, ny, nx))
  
  # --- 3. 等値面生成 ---
  (vL,fL),(vH,fH),(lvL,lvH) = make_isos(vol, spacing=(sz,sy,sx), level_low=args.low, level_hi=args.hi)

  # --- 4. 検出器板のZ座標取得 ---
  tzs,bzs=[],[]
  pairs_path = Path(args.pairs)
  if pairs_path.exists():
    with open(pairs_path) as f:
      r=csv.DictReader(f)
      try:
          for row in r:
              if "top_z" in row and "bot_z" in row:
                  tzs.append(float(row["top_z"])); bzs.append(float(row["bot_z"]))
      except Exception: pass
          
  z_center = (zmin+zmax)/2
  z_top = float(np.median(tzs)) if tzs else z_center + (zmax-zmin)/4 
  z_bot = float(np.median(bzs)) if bzs else z_center - (zmax-zmin)/4 

  # --- 5. 3D描画 ---
  fig = plt.figure(figsize=(8,6), dpi=args.dpi)
  ax = fig.add_subplot(111, projection="3d")

  # メッシュ（再構成結果）
  add_mesh(ax, vL, fL, color=(0.4,0.7,1.0), alpha=0.25, origin=(zmin, ymin, xmin))
  add_mesh(ax, vH, fH, color=(1.0,0.1,0.1), alpha=0.9,  origin=(zmin, ymin, xmin))

  # 検出器板
  draw_plate(ax, xmin,xmax,ymin,ymax, z_top, color=(0.1,1.0,0.1), alpha=0.25, thick=2.0)
  draw_plate(ax, xmin,xmax,ymin,ymax, z_bot, color=(1.0,0.1,0.1), alpha=0.25, thick=2.0)

  # 軸・タイトル
  ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
  ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
  ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
  ax.view_init(elev=22, azim=-60)
  fig.suptitle(f"3D Simple Back-Projection (W^T y) Recon (low≈{lvL:.3g}, high≈{lvH:.3g})", y=0.94)

  out = out_path(Path(args.out).name)
  plt.tight_layout()
  plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
  plt.close(fig)
  print(f"[OK] 3D PNGを保存しました: {out}")

if __name__ == "__main__":
    main()