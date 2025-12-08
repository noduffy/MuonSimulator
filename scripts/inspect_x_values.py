# ファイル名: inspect_x_values.py
import numpy as np
import yaml
from pathlib import Path

def main():
  # パス設定
  x_path = Path("build/outputs/x_cgls_solution_vector.npy")
  grid_path = Path("configs/grid3d.yml")

  # 1. データの読み込み
  if not x_path.exists():
    print(f"[Error] {x_path} が見つかりません。先に recon_cgls_3d.py を実行してください。")
    return
  x = np.load(x_path)

  if not grid_path.exists():
    print(f"[Error] {grid_path} が見つかりません。")
    return
  with open(grid_path, 'r') as f:
    g = yaml.safe_load(f)
  
  # グリッド情報の取得
  nx, ny, nz = int(g['nx']), int(g['ny']), int(g['nz'])
  xmin, xmax = float(g['x_min']), float(g['x_max'])
  ymin, ymax = float(g['y_min']), float(g['y_max'])
  zmin, zmax = float(g['z_min']), float(g['z_max'])
  
  dx = (xmax - xmin) / nx
  dy = (ymax - ymin) / ny
  dz = (zmax - zmin) / nz

  # 2. 統計情報の表示
  print(f"--- 推定密度ベクトル x の統計情報 ---")
  print(f"要素数: {x.size}")
  print(f"最大値: {x.max():.6e}")
  print(f"最小値: {x.min():.6e}")
  print(f"平均値: {x.mean():.6e}")
  print(f"中央値: {np.median(x):.6e}")
  
  # NaN や Inf のチェック
  if not np.isfinite(x).all():
    print(f"!!! 警告 !!! データに NaN (非数) または Inf (無限大) が含まれています。発散している可能性があります。")
    print(f"NaNの数: {np.count_nonzero(np.isnan(x))}")
    print(f"Infの数: {np.count_nonzero(np.isinf(x))}")
    return

  non_zero_count = np.count_nonzero(x)
  print(f"非ゼロ要素数: {non_zero_count} / {x.size} ({non_zero_count/x.size*100:.2f}%)")

  # 3. 値が大きい上位 10 ボクセルの表示
  print("\n--- 値が大きい上位 10 ボクセル (物体がある可能性が高い場所) ---")
  # インデックスを降順にソート
  sorted_indices = np.argsort(x)[::-1]
  top_indices = sorted_indices[:10]

  print(f"{'Rank':<5} {'Value':<12} {'Index':<8} {'(ix, iy, iz)':<15} {'Center (x, y, z) [mm]':<30}")
  print("-" * 80)

  for rank, idx in enumerate(top_indices):
    val = x[idx]
    
    # 1D index から 3D index (ix, iy, iz) への変換
    # build_system_matrix.py では: j_voxel = ix + iy * nx + iz * nx * ny
    iz = idx // (nx * ny)
    rem = idx % (nx * ny)
    iy = rem // nx
    ix = rem % nx
    
    # 物理座標 (ボクセル中心)
    cx = xmin + (ix + 0.5) * dx
    cy = ymin + (iy + 0.5) * dy
    cz = zmin + (iz + 0.5) * dz
    
    print(f"{rank+1:<5} {val:.4e}   {idx:<8} ({ix}, {iy}, {iz})     ({cx:.1f}, {cy:.1f}, {cz:.1f})")

  # 4. 空間中心付近の値の表示 (鉛ブロックがあると予想される場所)
  print("\n--- 空間中心付近のボクセルの値 ---")
  c_ix, c_iy, c_iz = nx // 2, ny // 2, nz // 2
  
  print(f"{'(ix, iy, iz)':<15} {'Center (x, y, z) [mm]':<30} {'Value':<12}")
  print("-" * 60)
  
  # 中心を中心に 3x3x3 の範囲を表示
  for diz in range(-1, 2):
    for diy in range(-1, 2):
      for dix in range(-1, 2):
        tix, tiy, tiz = c_ix + dix, c_iy + diy, c_iz + diz
        
        # 範囲内チェック
        if 0 <= tix < nx and 0 <= tiy < ny and 0 <= tiz < nz:
          tidx = tix + tiy * nx + tiz * nx * ny
          tval = x[tidx]
          tcx = xmin + (tix + 0.5) * dx
          tcy = ymin + (tiy + 0.5) * dy
          tcz = zmin + (tiz + 0.5) * dz
          
          marker = "<-- CENTER" if (dix==0 and diy==0 and diz==0) else ""
          print(f"({tix}, {tiy}, {tiz})     ({tcx:.1f}, {tcy:.1f}, {tcz:.1f})            {tval:.4e} {marker}")

if __name__ == "__main__":
  main()