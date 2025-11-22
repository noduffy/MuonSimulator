import argparse, yaml, sys
import numpy as np
import pandas as pd
import scipy.sparse as sps
from pathlib import Path
from time import time
from tqdm import tqdm
from numpy.linalg import norm 

# --- POCA重み付け用の定数 ---
POCA_SIGMA = 10.0 
POCA_SIGMA_SQ = POCA_SIGMA**2
WEIGHT_THRESHOLD = 1e-6 

# --- パス解決ヘルパー (変更なし) ---
def resolve_build_dir():
  p = Path.cwd()
  for d in [p] + list(p.parents):
    if d.name == "build": return d
    if (d / "build").exists(): return d / "build"
  return Path("build")

def resolve_in(p: str):
  P = Path(p)
  if P.is_absolute() and P.exists(): return P
  if P.exists(): return P
  build = resolve_build_dir()
  alt = build / "outputs" / P.name
  return alt if alt.exists() else P

def resolve_out(name: str):
  build = resolve_build_dir()
  outdir = build / "outputs"
  outdir.mkdir(parents=True, exist_ok=True)
  return outdir / Path(name).name
# ---------------------------------------------------

# --- W行列構築のメイン関数（メモリ最適化済み） ---
def build_W(pairs_path, rays_path, grid3d_path, outW, outy, batch_size=50000):
    # データのロード
    pairs_df = pd.read_csv(pairs_path)
    rays_df = pd.read_csv(rays_path)

    # グリッド設定のロード
    with open(grid3d_path) as f:
        g = yaml.safe_load(f)
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    
    # ★ 修正箇所: xmin/xmaxだけでなく、ymin/ymax, zmin/zmaxも定義
    xmin, xmax = float(g["x_min"]), float(g["x_max"])
    ymin, ymax = float(g["y_min"]), float(g["y_max"]) # ← 定義を追加
    zmin, zmax = float(g["z_min"]), float(g["z_max"]) # ← 定義を修正

    n_rays = len(pairs_df)
    n_voxels = nx * ny * nz
    
    # ボクセル中心座標の生成
    dx = (xmax - xmin) / nx; dy = (ymax - ymin) / ny; dz = (zmax - zmin) / nz
    x_centers = np.linspace(xmin + dx/2, xmax - dx/2, nx)
    y_centers = np.linspace(ymin + dy/2, ymax - dy/2, ny)
    z_centers = np.linspace(zmin + dz/2, zmax - dz/2, nz)
    
    ZC, YC, XC = np.meshgrid(z_centers, y_centers, x_centers, indexing='ij')
    voxel_centers = np.column_stack((XC.ravel(), YC.ravel(), ZC.ravel()))
    all_cols = np.arange(n_voxels) # ボクセル列インデックス

    # ★ 修正箇所: W行列のチャンクを格納するリスト
    W_chunks = [] 
    
    y = rays_df["theta2"].values
    
    if len(y) != n_rays:
        sys.exit(f"[FATAL] Pairs count ({n_rays}) does not match rays count ({len(y)})")
        
    print(f"Building W matrix with POCA weighting (grid: {nx}x{ny}x{nz}, rays: {n_rays}) using batch size {batch_size}...")

    # 全てのレイをベクトル化された処理で反復
    num_batches = (n_rays + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
        start_ray = batch_idx * batch_size
        end_ray = min((batch_idx + 1) * batch_size, n_rays)
        current_rays = pairs_df.iloc[start_ray:end_ray]
        
        # バッチごとのデータ格納リスト (メモリを抑える)
        W_rows_batch, W_cols_batch, W_data_batch = [], [], []
        
        # --- レイの反復（バッチ内）---
        for i in range(len(current_rays)):
            ray_idx = start_ray + i # グローバルな行インデックス
            ray = current_rays.iloc[i] 
            
            # 始点と終点 (numpy array)
            A = np.array([ray["top_x"], ray["top_y"], ray["top_z"]])
            B = np.array([ray["bot_x"], ray["bot_y"], ray["bot_z"]])
            
            # --- ベクトル化された POCA 重み計算 ---
            V = B - A
            V_sq_norm = np.sum(V**2)
            W_vec = voxel_centers - A
            
            t_POCA = np.clip(np.dot(W_vec, V) / V_sq_norm, 0.0, 1.0)
            P = A + t_POCA[:, None] * V
            poca_distance_sq = np.sum((voxel_centers - P)**2, axis=1)
            poca_weights = np.exp(-poca_distance_sq / (2 * POCA_SIGMA_SQ))

            # フィルタリング（メモリ最適化）
            mask = poca_weights > WEIGHT_THRESHOLD
            filtered_weights = poca_weights[mask]
            filtered_cols = all_cols[mask] 

            # W行列に追加 (グローバルな行インデックスを使用)
            W_rows_batch.extend([ray_idx] * len(filtered_weights))
            W_cols_batch.extend(filtered_cols)
            W_data_batch.extend(filtered_weights)
        
        # --- バッチのCOO行列への変換と格納 ---
        W_chunk = sps.coo_matrix((W_data_batch, (W_rows_batch, W_cols_batch)), shape=(n_rays, n_voxels))
        W_chunks.append(W_chunk)

    # --- 最終的なW行列の構築（チャンクの合計）---
    print("\nSumming sparse matrix chunks...")
    W = sum(W_chunks).tocoo()
    
    # yベクトルの保存
    np.save(resolve_out(Path(outy).name), y)
    
    # W行列の保存（nx, ny, nz情報も保存）
    np.savez_compressed(resolve_out(Path(outW).name),
                       row=W.row, col=W.col, data=W.data, shape=W.shape,
                       nx=nx, ny=ny, nz=nz)


    print(f"[OK] wrote 3D W/y with POCA weighting: {resolve_out(Path(outW).name).resolve()}, {resolve_out(Path(outy).name).resolve()}")
    print(f"     (nvox={n_voxels} nnz={len(W.data)} m={n_rays})")

# --- main関数とその他のロジックは変更なし ---
def main():
    ap = argparse.ArgumentParser(description="3D Muon Tomography W Matrix Builder with POCA Weighting.")
    ap.add_argument("--pairs", default="pairs.csv", help="Input pairs CSV file.")
    ap.add_argument("--rays", default="rays.csv", help="Input rays CSV file.")
    ap.add_argument("--grid3d", default="configs/grid3d.yml", help="Input 3D grid config file.")
    ap.add_argument("--outW", default="W3D_poca.npz", help="Output W matrix file (COO format).")
    ap.add_argument("--outy", default="y_theta2_3d.npy", help="Output y vector file.")
    ap.add_argument("--batch-size", type=int, default=50000, help="Number of rays to process in one batch (affects RAM usage).")
    args = ap.parse_args()

    pairs_path = resolve_in(args.pairs)
    rays_path = resolve_in(args.rays)
    grid3d_path = resolve_in(args.grid3d)
    
    if not rays_path.exists():
        sys.exit(f"[ERR] Rays file not found: {rays_path.resolve()}")
    if not pairs_path.exists():
        sys.exit(f"[ERR] Pairs file not found: {pairs_path.resolve()}")
    
    t_start = time()
    build_W(pairs_path, rays_path, grid3d_path, args.outW, args.outy, args.batch_size)
    t_end = time()
    print(f"Total time for W matrix build: {t_end - t_start:.2f}s")


if __name__ == "__main__":
    main()