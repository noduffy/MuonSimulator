# ファイル名: scripts/reconstruction/build_system_matrix.py
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz, diags
import sys
import argparse

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

# --- PoCA座標計算ヘルパー ---
def compute_poca_points(df):
  """
  データフレームからPoCA座標を計算して返す
  """
  P_top = df[['top_x', 'top_y', 'top_z']].values
  P_bot = df[['bot_x', 'bot_y', 'bot_z']].values
  v_in  = df[['top_dx', 'top_dy', 'top_dz']].values
  v_out = df[['bot_dx', 'bot_dy', 'bot_dz']].values

  r12 = P_top - P_bot
  d_dot = np.sum(v_in * v_out, axis=1)
  denom = 1 - d_dot**2
  
  # 平行に近い場合は計算不可なのでマスク
  valid = np.abs(denom) > 1e-6
  
  P_poca = np.zeros_like(P_top)
  
  # 計算可能なものだけ処理
  if np.any(valid):
    r12_v = r12[valid]
    vi_v  = v_in[valid]
    vo_v  = v_out[valid]
    dd_v  = d_dot[valid]
    den_v = denom[valid]
    
    # 係数t1, t2の計算
    dot_r_vi = np.sum(r12_v * vi_v, axis=1)
    dot_r_vo = np.sum(r12_v * vo_v, axis=1)
    
    t1 = (dot_r_vi - dd_v * dot_r_vo) / den_v
    t2 = (dd_v * dot_r_vi - dot_r_vo) / den_v
    
    # 中点計算
    P_poca[valid] = ( (P_top[valid] + t1[:,np.newaxis] * vi_v) + 
                      (P_bot[valid] + t2[:,np.newaxis] * vo_v) ) * 0.5
                      
  # 計算不可の場合は中点を仮置き（または使用しない）
  P_poca[~valid] = (P_top[~valid] + P_bot[~valid]) * 0.5
  
  return P_poca

# --- 行列計算ロジック (PoCA Trajectory) ---
def calculate_matrix_poca_traj(df_in, nx, ny, nz, ranges):
  xmin, xmax, ymin, ymax, zmin, zmax = ranges
  
  # ★修正: CSVになければここで計算する★
  if 'poca_x' in df_in.columns:
    P_scat = df_in[['poca_x', 'poca_y', 'poca_z']].values
  else:
    # print("  -> PoCA座標を計算中...")
    P_scat = compute_poca_points(df_in)

  P_top  = df_in[['top_x',  'top_y',  'top_z']].values
  P_bot  = df_in[['bot_x',  'bot_y',  'bot_z']].values
  
  # 散乱角の2乗を使用
  if 'scattering_angle_rad' in df_in.columns:
    scattering_sq = df_in['scattering_angle_rad'].values ** 2
  else:
    # theta_x, theta_y がある場合
    theta_x = df_in['theta_x'].values
    theta_y = df_in['theta_y'].values
    scattering_sq = theta_x**2 + theta_y**2
  
  rows_list = []; cols_list = []; data_list = []
  y_list = []
  
  dx = (xmax - xmin) / nx
  dy = (ymax - ymin) / ny
  dz = (zmax - zmin) / nz
  step_size = min(dx, dy, dz) * 0.5
  
  n_events = len(df_in)
  print(f"行列計算中(PoCA Trajectory): {n_events} events...")

  for i in range(n_events):
    # Top -> PoCA
    len1 = np.linalg.norm(P_scat[i] - P_top[i])
    if len1 > 0:
      dir1 = (P_scat[i] - P_top[i]) / len1
      steps1 = np.arange(0, len1, step_size)
      pts1 = P_top[i] + dir1 * steps1[:, np.newaxis]
    else:
      pts1 = np.zeros((0, 3))

    # PoCA -> Bot
    len2 = np.linalg.norm(P_bot[i] - P_scat[i])
    if len2 > 0:
      dir2 = (P_bot[i] - P_scat[i]) / len2
      steps2 = np.arange(0, len2, step_size)
      pts2 = P_scat[i] + dir2 * steps2[:, np.newaxis]
    else:
      pts2 = np.zeros((0, 3))
      
    # 結合 (空配列対策)
    if len(pts1) > 0 and len(pts2) > 0:
      points = np.vstack([pts1, pts2])
    elif len(pts1) > 0:
      points = pts1
    elif len(pts2) > 0:
      points = pts2
    else:
      continue
    
    # ボクセルID計算
    ix = np.floor((points[:, 0] - xmin) / dx).astype(int)
    iy = np.floor((points[:, 1] - ymin) / dy).astype(int)
    iz = np.floor((points[:, 2] - zmin) / dz).astype(int)
    
    mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    ix, iy, iz = ix[mask], iy[mask], iz[mask]
    if len(ix) == 0: continue
    
    voxel_ids = ix + iy * nx + iz * nx * ny
    unique_ids, counts = np.unique(voxel_ids, return_counts=True)
    lengths = counts * step_size
    
    rows_list.extend([len(y_list)] * len(unique_ids))
    cols_list.extend(unique_ids)
    data_list.extend(lengths)
    y_list.append(scattering_sq[i])

    if (i+1) % 5000 == 0: print(f"Progress: {i+1}/{n_events}", end='\r')
  
  print("")
  return np.array(rows_list), np.array(cols_list), np.array(data_list), np.array(y_list)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", default="scattered_muons.csv", help="入力CSV")
  parser.add_argument("--mode", default="poca_traj", choices=["poca_traj"], help="計算モード")
  parser.add_argument("--flux", default=None, help="(Optional) 正規化に使用するFlux Map (.npy)")
  args = parser.parse_args()

  # 入力チェック
  INPUT_CSV = resolve_out(args.input)
  if not INPUT_CSV.exists():
    print(f"[Error] {INPUT_CSV} が見つかりません。")
    sys.exit(1)

  # 設定読み込み
  try:
    df = pd.read_csv(INPUT_CSV)
    with open(config_path('grid3d.yml'), 'r') as f: g = yaml.safe_load(f)
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    ranges = (float(g["x_min"]), float(g["x_max"]), float(g["y_min"]), float(g["y_max"]), float(g["z_min"]), float(g["z_max"]))
  except Exception as e:
    print(f"Error: {e}"); sys.exit(1)

  # 1. 通常通り行列 W を計算
  rows, cols, data, y = calculate_matrix_poca_traj(df, nx, ny, nz, ranges)
  
  N_E = len(y)
  N_VOXELS = nx * ny * nz
  W = csr_matrix((data, (rows, cols)), shape=(N_E, N_VOXELS), dtype=np.float32)

  # 2. Flux Mapによる正規化 (New!)
  if args.flux:
    flux_path = resolve_out(args.flux)
    if flux_path.exists():
      print(f"--- Flux Mapによる正規化を適用します: {flux_path.name} ---")
      flux_vol = np.load(flux_path)
      flux_flat = flux_vol.flatten()
      
      epsilon = 1e-6
      
      # 重み計算: 1 / (Flux + epsilon)
      # Fluxが大きい(感度が高い)場所ほど、値を小さくしてバランスを取る
      weights = 1.0 / (flux_flat + epsilon)
      
      # 誰も通ってない場所は重み0
      weights[flux_flat < 1.0] = 0.0
      
      # 正規化実行: W_norm = W * diag(weights)
      # これにより、Wの各列(ボクセル)がFluxで割り算される
      D = diags(weights)
      W = W.dot(D)
      
      print("  -> 正規化完了。")
    else:
      print(f"[Warn] 指定されたFlux Mapが見つかりません: {flux_path}")

  # 3. 保存
  print("正規方程式 (WTW, WTy) を計算中...")
  WTy = W.transpose().dot(y)
  WTW = W.transpose().dot(W)
  
  save_npz(resolve_out('W_transpose_W_sparse.npz'), WTW)
  np.save(resolve_out('W_transpose_y_vector.npy'), WTy)
  print("[OK] システム行列を作成しました。")

if __name__ == '__main__':
  main()