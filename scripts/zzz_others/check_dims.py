# ファイル名: check_dims.py
import numpy as np
import yaml
from pathlib import Path

def main():
    # ファイルパス
    x_path = Path("build/outputs/x_cgls_solution_vector.npy")
    grid_path = Path("configs/grid3d.yml")

    # 1. grid3d.yml から期待されるボクセル総数を計算
    if not grid_path.exists():
        print(f"[Error] {grid_path} が見つかりません。")
        return

    with open(grid_path, 'r') as f:
        g = yaml.safe_load(f)
    
    nx, ny, nz = int(g['nx']), int(g['ny']), int(g['nz'])
    expected_dim = nx * ny * nz

    print(f"--- 設定ファイル (configs/grid3d.yml) ---")
    print(f"グリッド分割: {nx} x {ny} x {nz}")
    print(f"期待される次元数 (ボクセル総数): {expected_dim}")

    # 2. 生成された x ベクトルの次元数を確認
    if not x_path.exists():
        print(f"[Error] {x_path} が見つかりません。")
        return

    x_vector = np.load(x_path)
    actual_dim = x_vector.size

    print(f"\n--- 推定結果 (x_cgls_solution_vector.npy) ---")
    print(f"ベクトルの形状: {x_vector.shape}")
    print(f"実際の次元数:   {actual_dim}")

    # 3. 照合
    print(f"\n--- 判定 ---")
    if expected_dim == actual_dim:
        print("✅ 一致します。次元数は正常です。")
    else:
        print(f"❌ 一致しません！ ({expected_dim} vs {actual_dim})")

if __name__ == "__main__":
    main()