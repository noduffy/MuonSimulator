# ファイル名: scripts/fusion/build_flux_map.py
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import argparse
import sys

# --- パス解決 ---
def get_project_root():
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if (d / "build").exists(): return d
  return Path.cwd()

def resolve_build():
  return get_project_root() / "build"

def out_path(name: str) -> Path:
  b = resolve_build()
  o = b / "outputs"
  o.mkdir(parents=True, exist_ok=True)
  return o / name

def config_path(name: str) -> Path:
  root = get_project_root()
  return root / "configs" / name

# --- Flux 計算ロジック ---
def calculate_flux_map(df, nx, ny, nz, ranges):
  xmin, xmax, ymin, ymax, zmin, zmax = ranges
  flux_map = np.zeros(nx * ny * nz, dtype=np.float32)
  
  P_start = df[['top_x', 'top_y', 'top_z']].values
  P_end   = df[['bot_x', 'bot_y', 'bot_z']].values
  
  dx = (xmax - xmin) / nx
  dy = (ymax - ymin) / ny
  dz = (zmax - zmin) / nz
  step_size = min(dx, dy, dz) * 0.5
  
  n_events = len(df)
  print(f"Flux計算開始: {n_events} イベント...")

  for i in range(n_events):
    p0 = P_start[i]
    p1 = P_end[i]
    total_len = np.linalg.norm(p1 - p0)
    if total_len == 0: continue
    
    direction = (p1 - p0) / total_len
    steps = np.arange(0, total_len, step_size)
    points = p0 + direction * steps[:, np.newaxis]
    
    ix = np.floor((points[:, 0] - xmin) / dx).astype(int)
    iy = np.floor((points[:, 1] - ymin) / dy).astype(int)
    iz = np.floor((points[:, 2] - zmin) / dz).astype(int)
    
    mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    ix, iy, iz = ix[mask], iy[mask], iz[mask]
    
    if len(ix) == 0: continue
    
    voxel_ids = ix + iy * nx + iz * nx * ny
    unique_ids, counts = np.unique(voxel_ids, return_counts=True)
    lengths = counts * step_size
    
    flux_map[unique_ids] += lengths

    if (i+1) % 10000 == 0:
      print(f"処理中: {i+1}/{n_events} ({(i+1)/n_events*100:.1f}%)", end='\r')

  print("\n完了。")
  return flux_map.reshape((nz, ny, nx))

def main():
  parser = argparse.ArgumentParser(description="通過量マップ(Flux Map)を作成する")
  parser.add_argument("--scat", default=None, help="散乱データのCSV (任意)")
  parser.add_argument("--straight", default=None, help="直進データのCSV (任意)")
  parser.add_argument("--ratio", type=float, default=1.0, help="直進データの使用率 (0.0-1.0)")
  parser.add_argument("--out", default="flux_map.npy", help="出力ファイル名")
  args = parser.parse_args()

  # 設定読み込み
  try:
    cfg = config_path('grid3d.yml')
    with open(cfg, 'r') as f: g = yaml.safe_load(f)
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    ranges = (float(g["x_min"]), float(g["x_max"]), float(g["y_min"]), float(g["y_max"]), float(g["z_min"]), float(g["z_max"]))
  except Exception as e:
    print(f"Error: Config load failed: {e}")
    sys.exit(1)

  dfs = []
  
  # 散乱データの読み込み (指定があれば)
  if args.scat:
    p = out_path(args.scat)
    if p.exists():
      print(f"読込: {p.name}")
      dfs.append(pd.read_csv(p))
    else:
      print(f"[Warn] 指定された散乱データが見つかりません: {p}")

  # 直進データの読み込み (指定があれば)
  if args.straight:
    p = out_path(args.straight)
    if p.exists():
      print(f"読込: {p.name}")
      df_str = pd.read_csv(p)
      # 間引き処理
      if args.ratio < 1.0:
        n_use = int(len(df_str) * args.ratio)
        print(f"  -> 直進データを間引きます: {len(df_str)} -> {n_use}")
        df_str = df_str.sample(n=n_use, random_state=42)
      dfs.append(df_str)
    else:
      print(f"[Warn] 指定された直進データが見つかりません: {p}")

  if not dfs:
    print("[Error] 入力データがありません。--scat または --straight を指定してください。")
    sys.exit(1)

  df_total = pd.concat(dfs, ignore_index=True)
  
  # Flux計算
  flux_vol = calculate_flux_map(df_total, nx, ny, nz, ranges)

  # 保存
  OUT_PATH = out_path(args.out)
  np.save(OUT_PATH, flux_vol)
  print(f"[OK] Flux Map saved to: {OUT_PATH}")

if __name__ == "__main__":
  main()