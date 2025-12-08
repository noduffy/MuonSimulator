# ファイル名: plot_poca_3d_simple.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3Dプロットに必要

def calculate_poca(df_in):
    """入力DataFrameからPOCA点を計算し、(N, 3)のNumpy配列として返す"""
    if df_in.empty:
        print("入力データが空です。")
        return np.array([])
        
    r1 = df_in[['top_x', 'top_y', 'top_z']].values
    v1 = df_in[['top_dx', 'top_dy', 'top_dz']].values
    r2 = df_in[['bot_x', 'bot_y', 'bot_z']].values
    v2 = df_in[['bot_dx', 'bot_dy', 'bot_dz']].values

    # POCA計算のための線形代数演算
    r12 = r1 - r2
    d = np.sum(v1 * v2, axis=1) # v1・v2
    r12_v1 = np.sum(r12 * v1, axis=1) # r12・v1
    r12_v2 = np.sum(r12 * v2, axis=1) # r12・v2
    
    denom = 1 - d**2
    
    # ほぼ平行な光線を除外するためのフィルタリング
    epsilon = 1e-6
    valid_indices = np.abs(denom) > epsilon

    if np.sum(valid_indices) == 0:
        print("有効なPOCA点が得られませんでした。")
        return np.array([])

    # 有効なイベントのみを抽出
    r1_valid, v1_valid = r1[valid_indices], v1[valid_indices]
    r2_valid, v2_valid = r2[valid_indices], v2[valid_indices]
    denom_valid = denom[valid_indices]
    d_valid = d[valid_indices]
    r12_v1_valid = r12_v1[valid_indices]
    r12_v2_valid = r12_v2[valid_indices]
    
    # パラメータ t1, t2 を計算
    t1 = (r12_v1_valid - d_valid * r12_v2_valid) / denom_valid
    t2 = (d_valid * r12_v1_valid - r12_v2_valid) / denom_valid

    # POCA点 (最近接点) を計算: P_poca = (r1(t1) + r2(t2)) / 2
    p1 = r1_valid + t1[:, np.newaxis] * v1_valid
    p2 = r2_valid + t2[:, np.newaxis] * v2_valid
    
    return (p1 + p2) / 2

def main():
    try:
        # 'scattered_muons.csv'ファイルを読み込み
        df_scattered = pd.read_csv('build/outputs/scattered_muons.csv')
    except FileNotFoundError:
        print("エラー: 'scattered_muons.csv' が見つかりません。")
        print("先に 'separate_muons.py' を実行して、散乱データを抽出してください。")
        return

    print("--- 1. POCA点 (曲がり予想点) の計算 ---")
    p_poca = calculate_poca(df_scattered)

    if p_poca.size == 0:
        return

    print(f"計算された有効なPOCA点の数: {len(p_poca)}")

    # --- 2. 3D 散布図の作成 (最小の可視化) ---
    fig = plt.figure(figsize=(10, 8))
    # '3d' プロジェクションで Axes を追加
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = p_poca[:, 0], p_poca[:, 1], p_poca[:, 2]

    # POCA点をプロット (s=1 で小さな点にする)
    ax.scatter(x, y, z, s=1, alpha=0.5, marker='o', c=z, cmap='viridis')
    
    # 軸ラベルとタイトル
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title(f'Minimal 3D POCA Plot ({len(p_poca)} Scattered Muons)')
    
    # 3D表示のアスペクト比を調整 (物理空間に合わせて)
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][0].flatten() + 0.5*(x.max()+x.min())
    Yb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][1].flatten() + 0.5*(y.max()+y.min())
    Zb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][2].flatten() + 0.5*(z.max()+z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w') # 空のプロットで範囲を強制

    # ファイルに保存
    output_filename = 'build/outputs/poca_3d_scatter.png'
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n[OK] 3D散布図を保存しました: {output_filename}")

if __name__ == '__main__':
    main()