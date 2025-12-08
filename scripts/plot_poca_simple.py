# ファイル名: plot_poca_simple.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 設定値 ---
# 2Dヒストグラムのビンの数（解像度に相当）
BIN_COUNT = 50
# ----------------

def calculate_poca_and_plot(df_in, bin_count):
    """散乱ミューオンのデータを使用してPOCA点を計算し、2Dヒストグラムとしてプロットする"""
    if df_in.empty:
        print("エラー: 入力データが空です。散乱ミューオンが見つからなかったか、ファイルが正しくありません。")
        return

    # データの抽出
    r1 = df_in[['top_x', 'top_y', 'top_z']].values
    v1 = df_in[['top_dx', 'top_dy', 'top_dz']].values
    r2 = df_in[['bot_x', 'bot_y', 'bot_z']].values
    v2 = df_in[['bot_dx', 'bot_dy', 'bot_dz']].values

    # --- POCA (Point of Closest Approach) の計算 ---
    # パラメータt1, t2を求める
    r12 = r1 - r2
    d = np.sum(v1 * v2, axis=1) # v1・v2
    r12_v1 = np.sum(r12 * v1, axis=1) # r12・v1
    r12_v2 = np.sum(r12 * v2, axis=1) # r12・v2
    
    denom = 1 - d**2 # 行列式（分母）
    epsilon = 1e-6
    valid_indices = np.abs(denom) > epsilon # ほぼ平行な光線を除外
    
    if np.sum(valid_indices) == 0:
        print("POCA計算が安定して行えるイベントがありませんでした。")
        return

    # 有効なイベントのみを抽出
    r1_valid, v1_valid = r1[valid_indices], v1[valid_indices]
    r2_valid, v2_valid = r2[valid_indices], v2[valid_indices]
    denom_valid = denom[valid_indices]
    d_valid = d[valid_indices]
    r12_v1_valid = r12_v1[valid_indices]
    r12_v2_valid = r12_v2[valid_indices]
    
    # t1, t2を計算
    t1 = (r12_v1_valid - d_valid * r12_v2_valid) / denom_valid
    t2 = (d_valid * r12_v1_valid - r12_v2_valid) / denom_valid

    # POCA点 (最近接点) を計算: P_poca = (r1(t1) + r2(t2)) / 2
    p1 = r1_valid + t1[:, np.newaxis] * v1_valid
    p2 = r2_valid + t2[:, np.newaxis] * v2_valid
    
    p_poca = (p1 + p2) / 2
    
    # --- 2D ヒストグラム (画像再構成) ---
    x_poca = p_poca[:, 0]
    y_poca = p_poca[:, 1]
    
    # プロット範囲の設定（プロットの対象データのみを考慮）
    x_range = (x_poca.min() - 5, x_poca.max() + 5)
    y_range = (y_poca.min() - 5, y_poca.max() + 5)
    
    # 2Dヒストグラムの生成 (これが画像再構成結果となる)
    H, xedges, yedges = np.histogram2d(x_poca, y_poca, bins=bin_count, range=[x_range, y_range])

    # プロットのタイトル設定
    title = f"Primitive Muon Tomography (POCA Plot) - {len(df_in)} Events"

    # プロット
    plt.figure(figsize=(8, 8))
    plt.imshow(H.T, interpolation='nearest', origin='lower',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='viridis') # 密度に応じた色付け
    plt.colorbar(label='POCA Point Count (Density)')
    plt.title(title)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.show()

def main():
    try:
        # 'scattered_muons.csv'ファイルを読み込む
        df_scattered = pd.read_csv('build/outputs/scattered_muons.csv')
    except FileNotFoundError:
        print("エラー: 'scattered_muons.csv' が見つかりませんでした。")
        print("先に 'separate_muons.py' を実行して、このファイルを作成してください。")
        return

    print("--- 原始的なPOCAプロットによる画像再構成の検証 ---")
    calculate_poca_and_plot(df_scattered, BIN_COUNT)
    print("プロットの表示が完了しました。")

if __name__ == '__main__':
    main()