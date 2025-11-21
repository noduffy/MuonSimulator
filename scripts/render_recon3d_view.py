#!/usr/bin/env python3
import argparse, yaml, numpy as np
import sys
from pathlib import Path

# matplotlib は3D描画を使う
import matplotlib
matplotlib.use("Agg")  # 画面なし保存
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu

# --- パス解決（どこで実行しても build/outputs に出す） ---
def resolve_build():
  p = Path.cwd()
  for d in [p] + list(p.parents):
    if d.name == "build": return d
    if (d / "build").exists(): return d / "build"
  return Path("build")
def out_path(name: str) -> Path:
  b = resolve_build(); o = b / "outputs"; o.mkdir(parents=True, exist_ok=True)
  return o / name

# --- 体積→2段の等値面（青の全体、赤の高値） ---
def make_isos(vol, spacing, level_low=None, level_hi=None):
  vv = vol.astype(np.float32)
  pos = vv[vv>0]
  if pos.size == 0:
    # マスクによって全てゼロになった可能性があるため、エラーではなく警告にする
    print("[WARN] volume is empty (all zeros) after masking.", file=sys.stderr)
    return (None, None), (None, None), (0.0, 0.0)
    
  if level_low is None:
    try:
      # Otsuの閾値はノイズに敏感なため、警告時に安全なパーセンタイルに戻る
      level_low = float(threshold_otsu(pos))
    except Exception:
      level_low = float(np.percentile(pos, 75))  # 75% タイル
  if level_hi is None:
    level_hi = float(np.percentile(pos, 95))     # 95% タイル（“鉛候補”）
  
  if level_hi < level_low:
    level_hi = level_low * 1.05  # hi を low より少し上にする

  # skimage は (z,y,x) 順・spacing=(dz,dy,dx)
  try:
    vertsL, facesL, _, _ = marching_cubes(vv, level=level_low, spacing=spacing)
  except ValueError:
    vertsL, facesL = np.array([]), np.array([])
  
  try:
    vertsH, facesH, _, _ = marching_cubes(vv, level=level_hi, spacing=spacing)
  except ValueError:
    vertsH, facesH = np.array([]), np.array([])

  return (vertsL, facesL), (vertsH, facesH), (level_low, level_hi)

def add_mesh(ax, verts, faces, color, alpha, origin):
  if verts.size == 0: return
  # verts は (z,y,x)。ROI原点 (zmin, ymin, xmin) を加えてから (x,y,z) へ
  oz, oy, ox = origin  # 注意: originは (zmin, ymin, xmin) の順で受け取る
  V = np.empty_like(verts)
  V[:, 0] = verts[:, 2] + ox  # x = x + xmin
  V[:, 1] = verts[:, 1] + oy  # y = y + ymin
  V[:, 2] = verts[:, 0] + oz  # z = z + zmin
  mesh = Poly3DCollection(V[faces], linewidths=0.15, facecolor=color, edgecolor="k", alpha=alpha)
  ax.add_collection3d(mesh)

def draw_plate(ax, xmin,xmax,ymin,ymax,z, color, alpha=0.25, thick=2.0):
  # 薄い直方体：z±thick/2
  z0, z1 = z - thick/2, z + thick/2
  # 8点
  P = np.array([
    [xmin,ymin,z0],[xmax,ymin,z0],[xmax,ymax,z0],[xmin,ymax,z0],
    [xmin,ymin,z1],[xmax,ymin,z1],[xmax,ymax,z1],[xmin,ymax,z1]
  ], dtype=float)
  # 面
  F = [
    [0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]
  ]
  poly = Poly3DCollection([P[f] for f in F], facecolors=color, edgecolors="k", linewidths=0.3, alpha=alpha)
  ax.add_collection3d(poly)

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--vol", default="build/outputs/recon3d_vol.npy")
  ap.add_argument("--grid3d", default="configs/grid3d.yml")
  ap.add_argument("--pairs", default="build/outputs/pairs.csv")
  ap.add_argument("--low", type=float, default=None, help="青の等値面レベル（指定なければ自動）")
  ap.add_argument("--hi",  type=float, default=None, help="赤の等値面レベル（指定なければ自動）")
  ap.add_argument("--dpi", type=int, default=220)
  ap.add_argument("--out", default="recon3d_view.png") # 出力ファイル名を view.png に変更
  
  # --- _view の機能を追加 ---
  ap.add_argument("--mask-z-mm", type=float, default=0.0, help="z境界からこの[mm]内のボクセルをゼロにする（アーチファクト抑制）")
  ap.add_argument("--view", choices=["iso","side","front","top"], default="iso", help="視点プリセット (iso:斜め, side:側面, front:正面, top:上部)")
  ap.add_argument("--elev", type=float, default=None, help="カメラ仰角 (Elevation)")
  ap.add_argument("--azim", type=float, default=None, help="カメラ方位角 (Azimuth)")
  # --- ここまで ---
  
  args = ap.parse_args()

  vol = np.load(args.vol)
  g = yaml.safe_load(Path(args.grid3d).read_text())
  nx,ny,nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
  xmin,xmax = float(g["x_min"]), float(g["x_max"])
  ymin,ymax = float(g["y_min"]), float(g["y_max"])
  zmin,zmax = float(g["z_min"]), float(g["z_max"])
  sx = (xmax-xmin)/nx; sy=(ymax-ymin)/ny; sz=(zmax-zmin)/nz

  # --- Z端の簡易マスク（板近傍アーチファクト抑制）の適用 ---
  mmm = args.mask_z_mm
  if mmm and mmm > 0.0:
    # zmin から zmax の間に nz 個のボクセル中心座標を生成
    zc = np.linspace(zmin, zmax, nz, endpoint=False) + sz * 0.5
    # マスク条件: z_min + mmm 未満、または z_max - mmm より大きい
    mask = (zc <= zmin + mmm) | (zc >= zmax - mmm)
    # マスクされたZスライスを全てゼロにする
    vol[mask, :, :] = 0.0
  # --- ここまで ---

  # 等値面生成
  (vL,fL),(vH,fH),(lvL,lvH) = make_isos(vol, spacing=(sz,sy,sx), level_low=args.low, level_hi=args.hi)

  # Top/Bottom の z（中央値）
  import csv
  tzs,bzs=[],[]
  if Path(args.pairs).exists():
    with open(args.pairs) as f:
      r=csv.DictReader(f)
      for row in r:
        # csv.DictReaderは文字列を返すのでfloatに変換する
        try:
          tzs.append(float(row["top_z"])); bzs.append(float(row["bot_z"]))
        except ValueError:
          pass # データ欠損があればスキップ
  
  # z_top/z_bot の計算は、データが空の場合はz_min/z_maxの付近に設定する
  z_top = float(np.median(tzs)) if tzs else zmax - 30 # Geant4のデフォルト ±3cm=±30mm を想定
  z_bot = float(np.median(bzs)) if bzs else zmin + 30
  
  # 図
  fig = plt.figure(figsize=(8,6), dpi=args.dpi)
  ax = fig.add_subplot(111, projection="3d")

  # メッシュ
  add_mesh(ax, vL, fL, color=(0.4,0.7,1.0), alpha=0.25, origin=(zmin, ymin, xmin))
  add_mesh(ax, vH, fH, color=(1.0,0.1,0.1), alpha=0.9,  origin=(zmin, ymin, xmin))

  # 検出器板
  draw_plate(ax, xmin,xmax,ymin,ymax, z_top, color=(0.1,1.0,0.1), alpha=0.25, thick=2.0)
  draw_plate(ax, xmin,xmax,ymin,ymax, z_bot, color=(1.0,0.1,0.1), alpha=0.25, thick=2.0)

  # 軸・範囲
  ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
  ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
  ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))

  # --- 角度（View Init）の設定 ---
  elev = args.elev
  azim = args.azim
  
  if args.view == "side":
    elev = 0 if elev is None else elev
    azim = 0 if azim is None else azim
  elif args.view == "front":
    elev = 0 if elev is None else elev
    azim = 90 if azim is None else azim
  elif args.view == "top":
    elev = 90 if elev is None else elev
    azim = 0 if azim is None else azim
  else: # "iso" (デフォルト)
    elev = 22 if elev is None else elev
    azim = -60 if azim is None else azim

  ax.view_init(elev=elev, azim=azim)
  # --- ここまで ---
  
  # タイトル
  title = f"3D Reconstruction  (low≈{lvL:.3g}, high≈{lvH:.3g})"
  if args.view != "iso": title += f"  view={args.view}"
  if args.mask_z_mm > 0.0: title += f" (Z-Mask {args.mask_z_mm}mm)"
  fig.suptitle(title, y=0.94)

  out = out_path(Path(args.out).name)
  plt.tight_layout()
  plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
  plt.close(fig)
  print(f"[OK] wrote PNG: {out}")

if __name__ == "__main__":
  sys.exit(main())