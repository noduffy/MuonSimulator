import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
from pathlib import Path
import yaml

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

# --- 箱（直方体）を描画する関数 ---
# Method A/C の draw_plate とスタイルを統一
def draw_box(ax, center, size, color, alpha=0.25, edgecolor="k", linewidth=0.3):
  cx, cy, cz = center
  sx, sy, sz = size
  
  x0, x1 = cx - sx/2, cx + sx/2
  y0, y1 = cy - sy/2, cy + sy/2
  z0, z1 = cz - sz/2, cz + sz/2
  
  # 頂点定義
  P = np.array([
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], # 下面
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]  # 上面
  ], dtype=float)

  # 面定義 (頂点インデックス)
  F = [
    [0,1,2,3], # 底
    [4,5,6,7], # 天
    [0,1,5,4], # 手前
    [1,2,6,5], # 右
    [2,3,7,6], # 奥
    [3,0,4,7]  # 左
  ]
  
  # Method Aと同じ edgecolors="k", linewidths=0.3 をデフォルトに採用
  poly = Poly3DCollection([P[f] for f in F], facecolors=color, edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)
  ax.add_collection3d(poly)

def main():
  parser = argparse.ArgumentParser(description="シミュレーションの配置構成(Ground Truth)を可視化する")
  parser.add_argument("--out", default="setup_geometry.png", help="出力ファイル名")
  parser.add_argument("--angle", type=int, nargs=2, default=[22, -60], help="視点角度 (elev azim)")
  args = parser.parse_args()

  # 設定読み込み (軸の範囲決定のため)
  try:
    with open(config_path("grid3d.yml")) as f:
      g = yaml.safe_load(f)
      # 範囲を少し余裕を持たせて取得
      x_limit = max(abs(float(g["x_min"])), abs(float(g["x_max"])))
      y_limit = max(abs(float(g["y_min"])), abs(float(g["y_max"])))
      z_limit = max(abs(float(g["z_min"])), abs(float(g["z_max"])))
  except Exception:
    # 読み込めない場合のデフォルト
    x_limit, y_limit, z_limit = 150, 150, 100

  # プロット作成
  fig = plt.figure(figsize=(10, 8), dpi=220) # 解像度も再構成画像(220)に合わせました
  ax = fig.add_subplot(111, projection='3d')

  # ---------------------------------------------------------
  # Setup Geometry (Ground Truth)
  # ---------------------------------------------------------

  # 1. 検出器 (DetectorConstruction.cc の設定)
  # サイズ: 300mm x 300mm, 位置: Z = +/- 80mm
  det_size_xy = 300.0
  det_pos_z   = 80.0
  
  # ★修正: 手法A/Cと色を完全に統一
  # Top: Green (0.1, 1.0, 0.1), alpha=0.25
  draw_box(ax, (0, 0, det_pos_z), (det_size_xy, det_size_xy, 1), 
           color=(0.1, 1.0, 0.1), alpha=0.25, linewidth=0.3)
           
  # Bottom: Red (1.0, 0.1, 0.1), alpha=0.25
  draw_box(ax, (0, 0, -det_pos_z), (det_size_xy, det_size_xy, 1), 
           color=(1.0, 0.1, 0.1), alpha=0.25, linewidth=0.3)

  # 2. 鉛ブロック (物体)
  # サイズ: 20mm立方, 位置: X = +/- 40mm
  blk_size = 20.0
  
  # 物体はグレーのまま (再構成のヒートマップと区別するため)
  # ただし、線を少しシャープに
  draw_box(ax, (-40, 0, 0), (blk_size, blk_size, blk_size), 
           color=(0.4, 0.4, 0.4), alpha=0.8, edgecolor="k", linewidth=0.5)
  
  draw_box(ax, (40, 0, 0), (blk_size, blk_size, blk_size), 
           color=(0.4, 0.4, 0.4), alpha=0.8, edgecolor="k", linewidth=0.5)

  # ---------------------------------------------------------

  # 軸設定
  ax.set_xlim(-x_limit, x_limit)
  ax.set_ylim(-y_limit, y_limit)
  ax.set_zlim(-z_limit, z_limit)
  
  ax.set_xlabel("X [mm]")
  ax.set_ylabel("Y [mm]")
  ax.set_zlabel("Z [mm]")
  
  ax.set_box_aspect((x_limit*2, y_limit*2, z_limit*2))
  
  # 視点
  ax.view_init(elev=args.angle[0], azim=args.angle[1])
  
  # タイトルもシンプルに
  ax.set_title(f"Simulation Setup (Ground Truth)")

  # 保存
  out_path = resolve_out(args.out)
  plt.tight_layout()
  plt.savefig(out_path, bbox_inches="tight")
  plt.close(fig)
  print(f"セットアップ図を保存しました: {out_path}")

if __name__ == "__main__":
  main()