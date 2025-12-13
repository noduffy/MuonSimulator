# ファイル名: build_system_matrix.py
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz 
from scipy.spatial import cKDTree
import sys
import argparse

# --- 1. 新しいPOCA法 (ガウス重み付き) ---
def calculate_matrix_poca(df_in, nx, ny, nz, ranges):
  xmin, xmax, ymin, ymax, zmin, zmax = ranges
  dx, dy, dz = (xmax - xmin) / nx, (ymax - ymin) / ny, (zmax - zmin) / nz
  
  # POCA計算
  r1 = df_in[['top_x', 'top_y', 'top_z']].values; v1 = df_in[['top_dx', 'top_dy', 'top_dz']].values
  r2 = df_in[['bot_x', 'bot_y', 'bot_z']].values; v2 = df_in[['bot_dx', 'bot_dy', 'bot_dz']].values
  
  r12 = r1 - r2; d = np.sum(v1 * v2, axis=1)
  r12_v1 = np.sum(r12 * v1, axis=1); r12_v2 = np.sum(r12 * v2, axis=1)
  denom = 1 - d**2; epsilon = 1e-6
  
  valid = np.abs(denom) > epsilon
  if np.sum(valid) == 0: return None, None, None, None
  
  r1, v1, r2, v2 = r1[valid], v1[valid], r2[valid], v2[valid]
  t1 = (r12_v1[valid] - d[valid] * r12_v2[valid]) / denom[valid]
  t2 = (d[valid] * r12_v1[valid] - r12_v2[valid]) / denom[valid]
  p_poca = (r1 + t1[:, np.newaxis] * v1 + r2 + t2[:, np.newaxis] * v2) / 2
  y_vector = df_in.loc[valid, 'scattering_angle_rad'].values ** 2

  roi_mask = (p_poca[:, 2] >= zmin) & (p_poca[:, 2] <= zmax)
  p_poca = p_poca[roi_mask]
  y_vector = y_vector[roi_mask]
  
  if len(y_vector) == 0: return None, None, None, None
  
  print(f"POCA (Gaussian) 計算中: 有効数 {len(y_vector)}")
  sigma = min(dx, dy, dz)
  
  voxel_coords = np.zeros((nx * ny * nz, 3))
  for iz in range(nz):
    z = zmin + (iz + 0.5) * dz
    for iy in range(ny):
      y = ymin + (iy + 0.5) * dy
      for ix in range(nx):
        x = xmin + (ix + 0.5) * dx
        idx = ix + iy * nx + iz * nx * ny
        voxel_coords[idx] = [x, y, z]

  tree = cKDTree(voxel_coords)
  neighbors_list = tree.query_ball_point(p_poca, r=3.0 * sigma)
  
  rows_list, cols_list, data_list = [], [], []
  for i, neighbors in enumerate(neighbors_list):
    if not neighbors: continue
    d2 = np.sum((voxel_coords[neighbors] - p_poca[i])**2, axis=1)
    w = np.exp(-d2 / (2 * sigma**2))
    rows_list.extend([i] * len(neighbors))
    cols_list.extend(neighbors)
    data_list.extend(w)

  return np.array(rows_list), np.array(cols_list), np.array(data_list), y_vector

# --- 2. 直線通過長 (SLP) ---
def calculate_matrix_slp(df_in, nx, ny, nz, ranges):
  return _calculate_matrix_path(df_in, nx, ny, nz, ranges, mode='slp')

# --- 3. PoCA Trajectory (折れ線通過長) ---
def calculate_matrix_poca_traj(df_in, nx, ny, nz, ranges):
  return _calculate_matrix_path(df_in, nx, ny, nz, ranges, mode='poca_traj')

# --- 共通パス計算ロジック ---
def _calculate_matrix_path(df_in, nx, ny, nz, ranges, mode='slp'):
  xmin, xmax, ymin, ymax, zmin, zmax = ranges
  P_start = df_in[['top_x', 'top_y', 'top_z']].values
  P_end   = df_in[['bot_x', 'bot_y', 'bot_z']].values
  y_values = df_in['scattering_angle_rad'].values ** 2
  
  # POCA点の計算 (trajモード用)
  if mode == 'poca_traj':
    v1 = df_in[['top_dx', 'top_dy', 'top_dz']].values
    v2 = df_in[['bot_dx', 'bot_dy', 'bot_dz']].values
    r12 = P_start - P_end
    d = np.sum(v1 * v2, axis=1)
    r12_v1 = np.sum(r12 * v1, axis=1)
    r12_v2 = np.sum(r12 * v2, axis=1)
    denom = 1 - d**2
    valid = np.abs(denom) > 1e-6
    # 平行な場合はSLPにフォールバックするため、ここでは計算可能なものだけPOCA点を出す
    # (ただし簡単のため、全イベントループ内で個別に処理する)
  
  rows, cols, data, y_out = [], [], [], []
  dx, dy, dz = (xmax - xmin) / nx, (ymax - ymin) / ny, (zmax - zmin) / nz
  step = min(dx, dy, dz) * 0.5
  
  print(f"Path計算中 ({mode}): {len(df_in)} イベント...")

  for i in range(len(df_in)):
    p0 = P_start[i]
    p1 = P_end[i]
    val = y_values[i]
    
    segments = []
    
    if mode == 'slp':
      segments.append((p0, p1))
    elif mode == 'poca_traj':
      # POCA点計算
      v_in = df_in.iloc[i][['top_dx', 'top_dy', 'top_dz']].values.astype(float)
      v_out = df_in.iloc[i][['bot_dx', 'bot_dy', 'bot_dz']].values.astype(float)
      denom = 1 - np.dot(v_in, v_out)**2
      
      if abs(denom) < 1e-6: # 平行なら直線
        segments.append((p0, p1))
      else:
        r12 = p0 - p1
        d_dot = np.dot(v_in, v_out)
        t1 = (np.dot(r12, v_in) - d_dot * np.dot(r12, v_out)) / denom
        t2 = (d_dot * np.dot(r12, v_in) - np.dot(r12, v_out)) / denom
        p_poca = (p0 - t1 * v_in + p1 + t2 * v_out) / 2
        
        # POCA点が再構成領域内にある場合のみ折れ線、そうでなければ直線
        if (xmin <= p_poca[0] <= xmax and ymin <= p_poca[1] <= ymax and zmin <= p_poca[2] <= zmax):
          segments.append((p0, p_poca))
          segments.append((p_poca, p1))
        else:
          segments.append((p0, p1))

    # 線分ごとの通過長計算 (サンプリング法)
    voxel_counts = {}
    
    for pa, pb in segments:
      length = np.linalg.norm(pb - pa)
      if length == 0: continue
      direction = (pb - pa) / length
      steps = np.arange(0, length, step)
      points = pa + direction * steps[:, np.newaxis]
      
      ix = np.floor((points[:, 0] - xmin) / dx).astype(int)
      iy = np.floor((points[:, 1] - ymin) / dy).astype(int)
      iz = np.floor((points[:, 2] - zmin) / dz).astype(int)
      
      mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
      ix, iy, iz = ix[mask], iy[mask], iz[mask]
      
      if len(ix) > 0:
        ids = ix + iy * nx + iz * nx * ny
        u_ids, counts = np.unique(ids, return_counts=True)
        for uid, c in zip(u_ids, counts):
          voxel_counts[uid] = voxel_counts.get(uid, 0) + c * step

    if voxel_counts:
      # 結果格納
      u_ids = list(voxel_counts.keys())
      lengths = list(voxel_counts.values())
      rows.extend([len(y_out)] * len(u_ids))
      cols.extend(u_ids)
      data.extend(lengths)
      y_out.append(val)

    if (i+1) % 1000 == 0: print(f"処理完了: {i+1}/{len(df_in)}", end='\r')

  print("")
  if len(y_out) == 0: return None, None, None, None
  return np.array(rows), np.array(cols), np.array(data), np.array(y_out)

def main():
  parser = argparse.ArgumentParser(description="システム行列Wの構築")
  parser.add_argument("--mode", choices=['slp', 'poca', 'poca_traj'], default='poca_traj', 
                      help="W生成: 'poca'(ガウス), 'slp'(直線), 'poca_traj'(折れ線)")
  # ★★★ 追加: 入力ファイルを指定可能にする ★★★
  parser.add_argument("--input", default="scattered_muons.csv", 
                      help="入力CSVファイル名 (build/outputs/以下)。例: mixed_muons.csv")
  args = parser.parse_args()

  OUTPUT_DIR = Path("build/outputs")
  # 指定された入力ファイルを読み込む
  INPUT_CSV = OUTPUT_DIR / args.input
  
  if not INPUT_CSV.exists():
    print(f"[Error] 入力ファイルが見つかりません: {INPUT_CSV}")
    print("  -> mix_muon_data.py を実行して mixed_muons.csv を作成したか確認してください。")
    sys.exit(1)

  try:
    print(f"データを読み込んでいます: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    with open('configs/grid3d.yml', 'r') as f: g = yaml.safe_load(f)
  except Exception as e: print(f"Error: {e}"); sys.exit(1)

  nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
  ranges = (float(g["x_min"]), float(g["x_max"]), float(g["y_min"]), float(g["y_max"]), float(g["z_min"]), float(g["z_max"]))
  N_VOXELS = nx * ny * nz
  
  print(f"--- W構築 (Mode: {args.mode}, Input: {args.input}, Angle^2: YES) ---")
  
  # ここで各計算関数を呼び出す (関数定義は既存のものを使用)
  if args.mode == 'poca':
    rows, cols, data, y = calculate_matrix_poca(df, nx, ny, nz, ranges)
  elif args.mode == 'slp':
    rows, cols, data, y = calculate_matrix_slp(df, nx, ny, nz, ranges)
  elif args.mode == 'poca_traj':
    # 未定義エラーを防ぐため、前回の calculate_matrix_poca_traj 関数が必要です
    rows, cols, data, y = calculate_matrix_poca_traj(df, nx, ny, nz, ranges)
    
  if rows is None: print("有効データなし"); sys.exit(1)
    
  N_E = len(y)
  W = csr_matrix((data, (rows, cols)), shape=(N_E, N_VOXELS), dtype=np.float32)
  
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  
  save_npz(OUTPUT_DIR / 'W_mlem_sparse.npz', W)
  np.save(OUTPUT_DIR / 'y_mlem_vector.npy', y)
  
  WTy = W.transpose().dot(y)
  WTW = W.transpose().dot(W)
  save_npz(OUTPUT_DIR / 'W_transpose_W_sparse.npz', WTW)
  np.save(OUTPUT_DIR / 'W_transpose_y_vector.npy', WTy)
  
  print(f"[OK] 完了. 保存先: {OUTPUT_DIR}")

if __name__ == '__main__':
  main()