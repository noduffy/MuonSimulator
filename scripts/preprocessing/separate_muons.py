# ファイル名: scripts/separate_muons.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # 画面表示がない環境用
import matplotlib.pyplot as plt
from pathlib import Path

# --- 設定値 ---
SCATTERING_ANGLE_THRESHOLD_RAD = 0.01 #現状0.043radが一番綺麗に出力される
# ----------------

def calculate_scattering_angle(df):
    """入射方向と出射方向から散乱角（ラジアン）を計算する"""
    # 方向ベクトルを取得
    v_top = df[['top_dx', 'top_dy', 'top_dz']].values
    v_bot = df[['bot_dx', 'bot_dy', 'bot_dz']].values

    # ベクトルの正規化（念のため）
    norm_top = np.linalg.norm(v_top, axis=1)
    norm_bot = np.linalg.norm(v_bot, axis=1)
    
    # ノルムが0のデータ（エラーデータ）を除外または警告するためのマスク
    valid_mask = (norm_top > 0) & (norm_bot > 0)
    if not valid_mask.all():
        print(f"警告: 方向ベクトルの大きさが0のイベントが {len(df) - np.sum(valid_mask)} 件あります。これらは除外されます。")
        df = df[valid_mask].copy()
        v_top = v_top[valid_mask]
        v_bot = v_bot[valid_mask]
        norm_top = norm_top[valid_mask]
        norm_bot = norm_bot[valid_mask]

    # 内積とCos計算
    dot_product = np.sum(v_top * v_bot, axis=1)
    cos_theta = dot_product / (norm_top * norm_bot)
    
    # 浮動小数点誤差対策 (-1.0 ~ 1.0 に収める)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 散乱角 (ラジアン)
    df['scattering_angle_rad'] = np.arccos(cos_theta)
    
    return df

def main():
    input_path = Path('build/outputs/pairs.csv')
    if not input_path.exists():
        print(f"エラー: {input_path} が見つかりません。")
        return

    print("--- 1. データの読み込みと散乱角計算 ---")
    df = pd.read_csv(input_path)
    df = calculate_scattering_angle(df)
    
    # --- 診断: 統計情報の表示 ---
    angles = df['scattering_angle_rad']
    print(f"\n【散乱角の統計情報 (単位: rad)】")
    print(f"  最小値: {angles.min():.6f}")
    print(f"  最大値: {angles.max():.6f}")
    print(f"  平均値: {angles.mean():.6f}")
    print(f"  中央値: {angles.median():.6f}")
    
    # 逆向きベクトル疑いのチェック (1.5 rad以上 = 90度以上曲がっている)
    backward_count = np.sum(angles > 1.5)
    if backward_count > 0:
        print(f"⚠️ 警告: 90度以上曲がっているイベントが {backward_count} 件あります。")
        print("   -> 入射/出射ベクトルのZ成分の符号が逆転している可能性があります。")

    # --- 2. 散乱/直進ミューオンの分離 ---
    df['is_scattered'] = df['scattering_angle_rad'] > SCATTERING_ANGLE_THRESHOLD_RAD
    df_scattered = df[df['is_scattered']].copy()
    df_straight = df[~df['is_scattered']].copy()

    # ファイル保存
    out_dir = Path('build/outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    df_scattered.to_csv(out_dir / 'scattered_muons.csv', index=False)
    df_straight.to_csv(out_dir / 'straight_muons.csv', index=False)

    # ヒストグラムの作成（分布の可視化）
    plt.figure(figsize=(10, 6))
    plt.hist(angles, bins=100, log=True, range=(0, 0.1), color='skyblue', label='All Events')
    plt.axvline(SCATTERING_ANGLE_THRESHOLD_RAD, color='red', linestyle='dashed', linewidth=2, label=f'Threshold {SCATTERING_ANGLE_THRESHOLD_RAD}')
    plt.xlabel('Scattering Angle [rad]')
    plt.ylabel('Count (Log Scale)')
    plt.title('Distribution of Scattering Angles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / 'scattering_angle_dist.png')
    
    # 結果表示
    scattered_count = len(df_scattered)
    straight_count = len(df_straight)
    total = len(df)

    print(f"\n--- 3. 分離結果 ---")
    print(f"  閾値: {SCATTERING_ANGLE_THRESHOLD_RAD} rad")
    print(f"  散乱 (Scattered): {scattered_count} ({scattered_count/total*100:.2f}%)")
    print(f"  直進 (Straight) : {straight_count} ({straight_count/total*100:.2f}%)")
    print(f"\n[OK] ヒストグラムを保存しました: {out_dir}/scattering_angle_dist.png")
    print("この画像を確認し、閾値が山の『裾野』を適切に切り取れているか判断してください。")

if __name__ == '__main__':
    main()