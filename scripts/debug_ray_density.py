# ファイル名: debug_ray_density.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path("build/outputs/scattered_muons.csv")
    if not csv_path.exists():
        print(f"{csv_path} がありません。")
        return

    print("データを読み込んでいます...")
    df = pd.read_csv(csv_path)
    
    # Z=0 面での通過位置 (x, y) を計算 (線形補間)
    # x(z) = top_x + (bot_x - top_x) * (z - top_z) / (bot_z - top_z)
    # z = 0 なので...
    
    top_z = df['top_z'].values
    bot_z = df['bot_z'].values
    top_x = df['top_x'].values
    bot_x = df['bot_x'].values
    top_y = df['top_y'].values
    bot_y = df['bot_y'].values

    # ゼロ除算回避
    dz = bot_z - top_z
    dz[dz == 0] = 1e-6

    # Z=0 での X, Y 座標
    t = (0 - top_z) / dz
    mid_x = top_x + (bot_x - top_x) * t
    mid_y = top_y + (bot_y - top_y) * t

    # ヒストグラムの作成
    plt.figure(figsize=(8, 6))
    plt.hist2d(mid_x, mid_y, bins=50, range=[[-100, 100], [-100, 100]], cmap='inferno')
    plt.colorbar(label='Ray Count Density')
    plt.title(f'Density of "Scattered" Muons at Z=0 (N={len(df)})')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    
    # 中心（鉛ブロック）の場所を四角で表示
    plt.gca().add_patch(plt.Rectangle((-10, -10), 20, 20, linewidth=2, edgecolor='cyan', facecolor='none', label='Pb Block'))
    plt.legend()

    out_file = "build/outputs/debug_ray_density.png"
    plt.savefig(out_file)
    print(f"[OK] 分布図を保存しました: {out_file}")
    print("この画像を見て、中心の四角の中に点が集中しているか確認してください。")

if __name__ == "__main__":
    main()