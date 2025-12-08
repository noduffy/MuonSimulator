# ファイル名: analyze_angle_distribution.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # 全データ（pairs.csv）を読み込む
    csv_path = Path("build/outputs/pairs.csv")
    if not csv_path.exists():
        # build/outputsになければカレントを探す
        csv_path = Path("pairs.csv")
    
    if not csv_path.exists():
        print("[Error] pairs.csv が見つかりません。")
        return

    print("データを読み込んでいます...")
    df = pd.read_csv(csv_path)

    # 散乱角の計算
    v_top = df[['top_dx', 'top_dy', 'top_dz']].values
    v_bot = df[['bot_dx', 'bot_dy', 'bot_dz']].values
    dot_product = np.sum(v_top * v_bot, axis=1)
    norm_top = np.linalg.norm(v_top, axis=1)
    norm_bot = np.linalg.norm(v_bot, axis=1)
    cos_theta = np.clip(dot_product / (norm_top * norm_bot), -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)

    # ヒストグラムのプロット
    plt.figure(figsize=(10, 6))
    
    # ログスケールで表示することで、少ないけど大きな角度（鉛）を見やすくする
    counts, bins, _ = plt.hist(angles_rad, bins=200, range=(0, 0.1), color='blue', alpha=0.7, log=True)
    
    plt.title("Distribution of Scattering Angles (Log Scale)")
    plt.xlabel("Scattering Angle [rad]")
    plt.ylabel("Count (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # 目安となるラインを描画
    # 0.002 (現在の設定)
    plt.axvline(x=0.002, color='red', linestyle='-', label='Current Threshold (0.002)')
    
    # 一般的な鉛の散乱角付近 (0.015 - 0.03)
    plt.axvspan(0.015, 0.030, color='yellow', alpha=0.3, label='Typical Lead Signal (approx)')

    plt.legend()
    
    out_file = "build/outputs/angle_dist_log.png"
    # ディレクトリがない場合の保険
    Path("build/outputs").mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_file)
    print(f"[OK] ヒストグラムを保存しました: {out_file}")
    print("この画像を見て、'急激に数が減った後の、なだらかな裾野（鉛の信号）' が始まる角度を確認してください。")

if __name__ == "__main__":
    main()