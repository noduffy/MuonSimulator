# ファイル名: scripts/fusion/build_passage_matrix.py
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz 
import sys
import argparse

# --- パス解決ヘルパー (階層が深くなっても対応できるように強化) ---
def get_project_root():
  """このスクリプトがある場所から遡って build ディレクトリがある階層(ルート)を探す"""
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if (d / "build").exists() or (d / "configs").exists():
      return d
  return Path.cwd() # 見つからなければカレント

def resolve_build():
  root = get_project_root()
  return root / "build"

def out_path(name: str) -> Path:
  b = resolve_build()
  o = b / "outputs"
  o.mkdir(parents=True, exist_ok=True)
  return o / name

def config_path(name: str) -> Path:
  root = get_project_root()
  return root / "configs" / name

# --- 直線通過長 (SLP) 計算ロジック ---
def calculate_matrix_slp_binary(df_in, nx, ny, nz, ranges):
  """
  直進ミューオンの経路計算。
  yベクトル（観測値）はすべて 1.0 に設定する。
  """
  xmin, xmax, ymin, ymax, zmin, zmax = ranges
  
  # 座標の取得
  P_start = df_in[['top_x', 'top_y', 'top_z']].values
  P_end   = df_in[['bot_x', 'bot_y', 'bot_z']].values
  
  # ★★★ ポイント: yベクトルをすべて 1.0 にする ★★★
  # これにより「散乱角」ではなく「通過した事実(密度1)」を再構成する
  y_values = np.ones(len(df_in), dtype=np.float32)
  
  rows_list = []; cols_list = []; data_list = []
  y_list = []
  
  dx = (xmax - xmin) / nx
  dy = (ymax - ymin) / ny
  dz = (zmax - zmin) / nz
  step_size = min(dx, dy, dz) * 0.5
  
  n_events = len(df_in)
  print(f"直進パス計算中: {n_events} イベント (y=1.0 固定)...")

  for i in range(n_events):
    p0 = P_start[i]
    p1 = P_end[i]
    val = y_values[i] # 常に 1.0
    
    total_len = np.linalg.norm(p1 - p0)
    if total_len == 0: continue
    
    direction = (p1 - p0) / total_len
    steps = np.arange(0, total_len, step_size)
    points = p0 + direction * steps[:, np.newaxis]
    
    # ボクセルインデックス計算
    ix = np.floor((points[:, 0] - xmin) / dx).astype(int)
    iy = np.floor((points[:, 1] - ymin) / dy).astype(int)
    iz = np.floor((points[:, 2] - zmin) / dz).astype(int)
    
    # 範囲外除外
    mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    ix, iy, iz = ix[mask], iy[mask], iz[mask]
    
    if len(ix) == 0: continue
    
    voxel_ids = ix + iy * nx + iz * nx * ny
    unique_ids, counts = np.unique(voxel_ids, return_counts=True)
    lengths = counts * step_size
    
    # 行列要素に追加
    rows_list.extend([len(y_list)] * len(unique_ids))
    cols_list.extend(unique_ids)
    data_list.extend(lengths)
    y_list.append(val)

    if (i+1) % 5000 == 0:
      print(f"処理中: {i+1}/{n_events}", end='\r')

  print("")
  if len(y_list) == 0: return None, None, None, None
  return np.array(rows_list), np.array(cols_list), np.array(data_list), np.array(y_list)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", default="straight_muons.csv", help="入力CSVファイル名 (build/outputs内)")
  parser.add_argument("--ratio", type=float, default=0.1, help="使用するデータの割合 (0.1なら10%%)")
  args = parser.parse_args()

  # 入力パス
  INPUT_CSV = out_path(args.input)
  if not INPUT_CSV.exists():
    print(f"[Error] {INPUT_CSV} が見つかりません。")
    print("  separate_muons.py を実行して straight_muons.csv を作成してください。")
    sys.exit(1)

  # 設定読み込み (パス解決を強化した config_path を使用)
  try:
    df = pd.read_csv(INPUT_CSV)
    cfg = config_path('grid3d.yml')
    with open(cfg, 'r') as f: g = yaml.safe_load(f)
  except Exception as e:
    print(f"Error: 設定ファイルまたはCSVの読み込みに失敗しました。: {e}")
    sys.exit(1)

  # データの間引き
  n_use = int(len(df) * args.ratio)
  if n_use < 100: n_use = len(df) 
  print(f"データ間引き: {len(df)} 件中 {n_use} 件を使用します (Ratio: {args.ratio})")
  df = df.sample(n=n_use, random_state=42).reset_index(drop=True)

  nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
  ranges = (float(g["x_min"]), float(g["x_max"]), float(g["y_min"]), float(g["y_max"]), float(g["z_min"]), float(g["z_max"]))
  
  # 行列計算
  rows, cols, data, y = calculate_matrix_slp_binary(df, nx, ny, nz, ranges)
    
  if rows is None: print("有効データなし"); sys.exit(1)
    
  # 行列作成
  N_E = len(y)
  N_VOXELS = nx * ny * nz
  W = csr_matrix((data, (rows, cols)), shape=(N_E, N_VOXELS), dtype=np.float32)
  
  # ★保存ファイル名を変更 (passage_*)★
  save_npz(out_path('W_passage_sparse.npz'), W)
  np.save(out_path('y_passage_vector.npy'), y)
  
  # 正規方程式用
  print("正規方程式 (W^T W) を計算中...")
  WTy = W.transpose().dot(y)
  WTW = W.transpose().dot(W)
  
  save_npz(out_path('WTW_passage.npz'), WTW)
  np.save(out_path('WTy_passage.npy'), WTy)
  
  print(f"[OK] 通過マップ用行列を作成しました。")
  print(f"  保存先: {out_path('WTW_passage.npz')}")

if __name__ == '__main__':
  main()