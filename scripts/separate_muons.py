# ファイル名: separate_muons.py
# mygeom下で実行
import pandas as pd
import numpy as np

# --- 設定値 ---
# 散乱角の分類閾値（ラジアン）。この値を調整することで、散乱ミューオンの定義が変わります。
# 0.05ラジアンは、約2.86度です。
SCATTERING_ANGLE_THRESHOLD_RAD = 0.03
# ----------------

def calculate_scattering_angle(df):
    """入射方向と出射方向から散乱角（ラジアン）を計算する"""
    # 方向ベクトルを取得 (top_dx, top_dy, top_dz) と (bot_dx, bot_dy, bot_dz)
    v_top = df[['top_dx', 'top_dy', 'top_dz']].values
    v_bot = df[['bot_dx', 'bot_dy', 'bot_dz']].values

    # ベクトルの内積 (v_top ・ v_bot)
    dot_product = np.sum(v_top * v_bot, axis=1)

    # ベクトルのノルム (大きさ)
    norm_top = np.linalg.norm(v_top, axis=1)
    norm_bot = np.linalg.norm(v_bot, axis=1)

    # cos(theta) = 内積 / (ノルムの積) を計算
    # 浮動小数点誤差を考慮して[-1, 1]にクリップ
    cos_theta = dot_product / (norm_top * norm_bot)
    cos_theta = np.nan_to_num(cos_theta, nan=1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 散乱角 (ラジアン) を計算
    df['scattering_angle_rad'] = np.arccos(cos_theta)
    
    return df

def main():
    try:
        # 'pairs.csv'ファイルを読み込む
        df = pd.read_csv('build/outputs/pairs.csv')
    except FileNotFoundError:
        print("エラー: 'pairs.csv' が見つかりませんでした。データファイルを用意してください。")
        return

    print("--- 1. 散乱角の計算 ---")
    df = calculate_scattering_angle(df)
    
    # 散乱/直進ミューオンの分離
    df['is_scattered'] = df['scattering_angle_rad'] > SCATTERING_ANGLE_THRESHOLD_RAD
    df_scattered = df[df['is_scattered']].copy()
    df_straight = df[~df['is_scattered']].copy()

    # 結果をCSVファイルとして保存
    df_scattered.to_csv('build/outputs/scattered_muons.csv', index=False)
    df_straight.to_csv('build/outputs/straight_muons.csv', index=False)

    # 統計情報の表示
    scattered_count = len(df_scattered)
    straight_count = len(df_straight)

    print("--- 2. 散乱・直進ミューオンの分離結果 ---")
    print(f"全イベント数: {len(df)}")
    print(f"分類閾値（散乱角）: {SCATTERING_ANGLE_THRESHOLD_RAD:.4f} ラジアン (約 {np.degrees(SCATTERING_ANGLE_THRESHOLD_RAD):.2f} 度)")
    print(f"👉 散乱ミューオンのイベント数: {scattered_count} ({scattered_count / len(df) * 100:.2f}%) -> scattered_muons.csv に保存")
    print(f"👉 直進ミューオンのイベント数: {straight_count} ({straight_count / len(df) * 100:.2f}%) -> straight_muons.csv に保存")
    print("\n分離が完了しました。次に 'plot_poca_simple.py' を実行してください。")

if __name__ == '__main__':
    main()