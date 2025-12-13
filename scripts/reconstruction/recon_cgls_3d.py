# ファイル名: recon_cgls_3d.py
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import cg # 共役勾配法 (Conjugate Gradient)
from pathlib import Path

# --- 設定値 ---
OUTPUT_DIR = Path("build/outputs")
MAX_ITER = 200  # 反復回数 (シミュレーションデータの量に応じて調整してください)
# --------------

def solve_normal_equation(WTW_path, WTy_path, max_iter):
    """
    正規方程式 W^T W x = W^T y を共役勾配法(CG)で解き、推定密度xを返す。
    """
    try:
        # W^T W (疎行列) と W^T y (ベクトル) を読み込み
        A = load_npz(WTW_path)
        b = np.load(WTy_path)
    except FileNotFoundError as e:
        print(f"エラー: 必要なファイルが見つかりません。先に 'build_system_matrix.py' を実行してください: {e.filename}")
        return None

    print(f"--- 1. 正規方程式の解法 (W^T W x = W^T y) ---")
    print(f"W^T Wのサイズ: {A.shape}, W^T yのサイズ: {b.shape}")
    print(f"共役勾配法 (CG) を {max_iter} 回反復して実行中...")

    # CGソルバを使用して x を解く
    x_solution, exit_code = cg(
        A, b, maxiter=max_iter, rtol=1e-6, x0=None
    )

    if exit_code == 0:
        print(f"[OK] 成功: {max_iter} 回の反復で収束しました。")
    elif exit_code > 0:
        print(f"[WARN] 警告: 最大反復回数 ({max_iter} 回) に達しました。")
    else:
        print(f"[ERR] エラー: CG法の計算に失敗しました (Exit Code: {exit_code})。")

    # 推定密度 x は物理的な量であるため、負の値を持たないようにクリップする
    x_solution = np.clip(x_solution, a_min=0, a_max=None)
    
    return x_solution

def main():
    WTW_PATH = OUTPUT_DIR / 'W_transpose_W_sparse.npz'
    WTy_PATH = OUTPUT_DIR / 'W_transpose_y_vector.npy'
    X_SOLUTION_PATH = OUTPUT_DIR / 'x_cgls_solution_vector.npy'

    # W^T W x = W^T y を解く
    x_vector = solve_normal_equation(WTW_PATH, WTy_PATH, MAX_ITER)
    
    if x_vector is not None:
        # 結果を保存
        np.save(X_SOLUTION_PATH, x_vector)
        print(f"\n[OK] 推定密度ベクトル x を {X_SOLUTION_PATH} に保存しました。")

if __name__ == '__main__':
    main()