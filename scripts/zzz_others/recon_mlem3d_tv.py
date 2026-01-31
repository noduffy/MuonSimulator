import argparse, sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from numpy.linalg import norm
from pathlib import Path
from time import time
import yaml

# === robust path helpers (元のrecon_mlem3d.pyから移植) ====================
def resolve_build_dir():
  """今いる場所やスクリプトの場所から、一番近い build ディレクトリを返す。"""
  candidates = []
  # 1) CWD とその親を走査
  p = Path.cwd()
  for d in [p] + list(p.parents):
    if d.name == "build":
      return d
    if (d / "build").exists():
      candidates.append(d / "build")
  # 2) スクリプト位置からも走査
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if d.name == "build":
      return d
    if (d / "build").exists():
      candidates.append(d / "build")
  # 3) 最後の手段
  return candidates[0] if candidates else Path("build")

def resolve_in(p: str):
  """入力ファイル: 存在すればそのまま。無ければ build/outputs/ を探す。"""
  P = Path(p)
  if P.is_absolute() and P.exists():
    return P
  if P.exists():
    return P
  build = resolve_build_dir()
  alt = build / "outputs" / P.name
  return alt if alt.exists() else P

def resolve_out(name: str):
  """出力ファイル: どこで実行しても build/outputs/<basename> に1回だけ置く。"""
  build = resolve_build_dir()
  outdir = build / "outputs"
  outdir.mkdir(parents=True, exist_ok=True)
  return outdir / Path(name).name
# ==========================================================================

# === W行列のロード (元のrecon_mlem3d.pyから移植) ==========================
def load_Wnpz(fname):
  """W3D_coo.npzをロードし、CSR形式のWとグリッドサイズ(nx,ny,nz)を返す。"""
  # scipy.sparse.load_npzではなく、numpy.loadで中身を展開する
  Z = np.load(fname, allow_pickle=False)
  
  # row, col, data, shape を使ってCOOSparseMatrixを作成し、CSRに変換
  row, col, data, shape = Z["row"], Z["col"], Z["data"], tuple(Z["shape"])
  W = sp.coo_matrix((data,(row,col)), shape=shape).tocsr()
  
  # グリッドサイズをnpzファイルから取得 (nx,ny,nzが保存されているはず)
  nx = int(Z["nx"]) if "nx" in Z else int(round(shape[1] ** (1/3)))
  ny = int(Z["ny"]) if "ny" in Z else nx
  nz = int(Z["nz"]) if "nz" in Z else nx
  return W, (nx,ny,nz)
# ==========================================================================


# --- TV 正則化に必要な関数 (シンプルなラプラシアン平滑化) ---
# 厳密なTotal Variationの勾配計算は複雑なため、ここではシンプルな平滑化を適用
def TV_gradient_approx(vol, nx, ny, nz):
    """隣接ボクセルとの差に基づき、ノイズ抑制の方向を示す簡易勾配"""
    grad = np.zeros_like(vol)
    
    # x方向
    grad[:, :, 1:-1] += 2 * vol[:, :, 1:-1] - vol[:, :, 2:] - vol[:, :, :-2]
    # y方向
    grad[:, 1:-1, :] += 2 * vol[:, 1:-1, :] - vol[:, 2:, :] - vol[:, :-2, :]
    # z方向
    grad[1:-1, :, :] += 2 * vol[1:-1, :, :] - vol[2:, :, :] - vol[:-2, :, :]
    
    return grad

# --- MLEM-TV 再構成関数 ---
def mlem_tv(W, y, nx, ny, nz, iters=50, beta=0.0, eps=1e-12):
    m,n = W.shape
    x = np.ones(n, dtype=np.float64) / n # 初期推定（均一なボリューム）
    WT = W.T.tocsr()
    
    # 分母の計算 (denom = W.T * 1)
    denom = WT.dot(np.ones(m)); denom[denom==0] = 1.0
    
    for k in range(iters):
        t0 = time()
        
        # --- 1. MLEM E/Mステップ (更新) ---
        # 予測: Wx = W * x_k
        Wx = W.dot(x); Wx[Wx==0] = 1.0
        
        # 比率: ratio = y / Wx
        ratio = y / Wx
        
        # MLEM更新: x_k+1_mlem = x_k * (W.T * ratio) / denom
        correction = WT.dot(ratio)
        x_mlem = x * correction / denom

        # --- 2. TV正則化の適用 ---
        if beta > 0.0:
            # ボリューム形状に戻す
            x_vol = x_mlem.reshape(nz, ny, nx)

            # TV勾配の計算
            tv_grad = TV_gradient_approx(x_vol, nx, ny, nz)
            
            # TVペナルティを適用した更新 (近似的な勾配降下)
            x_vol_tv = x_vol - beta * tv_grad
            
            # ボリューム形状をフラットに戻し、負の値をクリップ
            x = x_vol_tv.ravel()
            x = np.maximum(x, eps)
        else:
            x = np.maximum(x_mlem, eps)

        # 3. 正規化
        x /= np.sum(x)
        
        dt = time() - t0
        # MLEM更新前のx_mlemとの差ではなく、収束指標としてnormをそのまま使用
        print(f"Iter {k+1:04d}/{iters}: Time={dt:.2f}s")
        
    return x


def main():
  ap = argparse.ArgumentParser(description="3D Muon Tomography Reconstruction using MLEM with Total Variation (TV) Regularization.")
  ap.add_argument("--iters", type=int, default=100)
  ap.add_argument("--W", default="W3D_coo.npz")
  ap.add_argument("--y", default="y_theta2_3d.npy")
  ap.add_argument("--beta", type=float, default=0.0, help="TV regularization strength (beta). Set > 0 for regularization.") # TVオプションを追加
  args = ap.parse_args()

  W_path = resolve_in(args.W)
  y_path = resolve_in(args.y)

  if not W_path.exists():
    sys.exit(f"[ERR] not found W: {W_path.resolve()}")
  if not y_path.exists():
    sys.exit(f"[ERR] not found y: {y_path.resolve()}")

  # W行列とグリッドサイズのロード (元のスクリプトのロジックを使用)
  W, (nx,ny,nz) = load_Wnpz(W_path)
  y = np.load(y_path)
  
  print("W shape:", W.shape, "grid:", (nx,ny,nz), "m=", len(y))
  
  t_start = time()
  # MLEM-TVの実行
  x = mlem_tv(W, y, nx, ny, nz, iters=args.iters, beta=args.beta)
  t_end = time()
  
  print(f"Reconstruction finished in {t_end - t_start:.2f} seconds.")

  vol = x.reshape(nz, ny, nx)  # (z,y,x)

  np.save(resolve_out("recon3d_vol_tv.npy"), vol) # 出力ファイル名を変更

  # 元のスクリプトと同様に2D投影図を出力
  cz, cy, cx = nz//2, ny//2, nx//2
  title_suffix = f"iters={args.iters} beta={args.beta}"
  
  plt.figure(); plt.imshow(vol[cz,:,:], origin="lower"); plt.colorbar(); plt.title(f"XY z={cz} {title_suffix}")
  plt.savefig(resolve_out("recon3d_xy_tv.png"), dpi=150); plt.close()
  plt.figure(); plt.imshow(vol[:,cy,:], origin="lower"); plt.colorbar(); plt.title(f"XZ y={cy} {title_suffix}")
  plt.savefig(resolve_out("recon3d_xz_tv.png"), dpi=150); plt.close()
  plt.figure(); plt.imshow(vol[:,:,cx], origin="lower"); plt.colorbar(); plt.title(f"YZ x={cx} {title_suffix}")
  plt.savefig(resolve_out("recon3d_yz_tv.png"), dpi=150); plt.close()

  print("[OK] wrote: recon3d_vol_tv.npy / recon3d_{xy,xz,yz}_tv.png ->", resolve_build_dir() / "outputs")

if __name__ == "__main__":
  sys.exit(main())