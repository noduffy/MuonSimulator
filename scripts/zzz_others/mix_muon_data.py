# ファイル名: scripts/mix_muon_data.py
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

def main():
  parser = argparse.ArgumentParser(description="散乱データと直進データを混合する")
  parser.add_argument("--ratio", type=float, default=1.0, help="直進データの混合比率")
  parser.add_argument("--out", default="mixed_muons.csv", help="出力ファイル名")
  args = parser.parse_args()

  OUTPUT_DIR = Path("build/outputs")
  SCATTERED_PATH = OUTPUT_DIR / "scattered_muons.csv"
  STRAIGHT_PATH = OUTPUT_DIR / "straight_muons.csv"
  
  if not SCATTERED_PATH.exists() or not STRAIGHT_PATH.exists():
    print("[Error] 入力ファイルが見つかりません。")
    sys.exit(1)
    
  print("データを読み込んでいます...")
  df_scat = pd.read_csv(SCATTERED_PATH)
  df_str = pd.read_csv(STRAIGHT_PATH)
  
  n_scat = len(df_scat)
  n_str_total = len(df_str)
  
  # 直進データのサンプリング
  n_str_target = int(n_scat * args.ratio)
  if n_str_target > n_str_total: n_str_target = n_str_total
  
  if n_str_target > 0:
    df_str_sampled = df_str.sample(n=n_str_target, random_state=42)
    
    # ★★★ ここが重要！ ★★★
    # 直進データの散乱角を強制的に 0.0 に上書きする
    # これにより、再構成時に「密度ゼロ」の制約として正しく機能させる
    df_str_sampled['scattering_angle_rad'] = 0.0
    
    print(f"  -> 直進データを {n_str_target} 件サンプリングし、散乱角を0.0に設定しました。")
  else:
    df_str_sampled = pd.DataFrame()

  # 結合とシャッフル
  df_mixed = pd.concat([df_scat, df_str_sampled], ignore_index=True)
  df_mixed = df_mixed.sample(frac=1, random_state=42).reset_index(drop=True)
  
  out_path = OUTPUT_DIR / args.out
  df_mixed.to_csv(out_path, index=False)
  print(f"[OK] {out_path} を作成しました。")

if __name__ == "__main__":
  main()