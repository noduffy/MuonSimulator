import os
import glob
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path

# ==========================================
# デフォルト設定 (引数がない場合に使用)
# ==========================================
# ここに比較したいフォルダ名をリストで記述します。
# 3つ記述すれば、グラフの線も自動で3本になります。
DEFAULT_TARGET_DIRS = [
  "proposed_result",
  "progressive_cgls",
  "method_d_result"
]

BASE_OUTPUT_DIR = "build/outputs" # フォルダ名だけで指定された場合の親パス
GROUND_TRUTH_PATH = "build/outputs/evaluation/true_density.npy" 
OUTPUT_DIR = "build/outputs/evaluation"
CONFIG_PATH = "configs/grid3d.yml"

# 評価対象ファイルパターン
TARGET_PATTERNS = ["x_iter_*.npy", "prob_map.npy"]

# ==========================================
# 設定ロード
# ==========================================
def load_grid_config():
  if not os.path.exists(CONFIG_PATH):
    return None
  with open(CONFIG_PATH) as f:
    return yaml.safe_load(f)

# ==========================================
# スケーリング補正 (Optimized Scale)
# ==========================================
def optimize_scale(true_vol, pred_vol):
  """
  再構成データの絶対値のスケールを真値に合わせる補正
  """
  t = true_vol.flatten()
  p = pred_vol.flatten()
  denom = np.dot(p, p)
  if denom == 0:
    return pred_vol, 0.0
  alpha = np.dot(t, p) / denom
  return pred_vol * alpha, alpha

# ==========================================
# 評価指標計算
# ==========================================
def calculate_metrics(true_vol, pred_vol):
  if true_vol.shape != pred_vol.shape:
    return None

  # スケーリング補正
  pred_opt, scale_factor = optimize_scale(true_vol, pred_vol)

  # --- 全体評価 ---
  mse_all = mean_squared_error(true_vol, pred_opt)
  d_range = true_vol.max() - true_vol.min()
  if d_range == 0: d_range = 1.0

  try:
    ssim_all = ssim(true_vol, pred_opt, data_range=d_range, channel_axis=None)
  except:
    ssim_all = ssim(true_vol, pred_opt, data_range=d_range, channel_axis=None, win_size=3)

  # --- ROI評価 ---
  mask = true_vol > 0.1
  if np.sum(mask) < 2:
    return {
      "MSE_All": mse_all, "SSIM_All": ssim_all,
      "MSE_ROI": mse_all, "SSIM_ROI": ssim_all,
      "Scale_Factor": scale_factor
    }

  true_masked = true_vol[mask]
  pred_masked = pred_opt[mask]
  mse_roi = mean_squared_error(true_masked, pred_masked)

  # ROI領域の切り出し(SSIM用)
  coords = np.argwhere(mask)
  z0, y0, x0 = coords.min(axis=0)
  z1, y1, x1 = coords.max(axis=0) + 1
  
  true_crop = true_vol[z0:z1, y0:y1, x0:x1]
  pred_crop = pred_opt[z0:z1, y0:y1, x0:x1]
  
  min_side = min(true_crop.shape)
  win_size = min(7, min_side)
  if win_size % 2 == 0: win_size -= 1
  if win_size < 3: win_size = 3

  try:
    ssim_roi = ssim(true_crop, pred_crop, data_range=d_range, channel_axis=None, win_size=win_size)
  except Exception:
    ssim_roi = 0.0

  return {
    "MSE_All": mse_all,
    "SSIM_All": ssim_all,
    "MSE_ROI": mse_roi,
    "SSIM_ROI": ssim_roi,
    "Scale_Factor": scale_factor
  }

# ==========================================
# グラフ描画 (動的にN本に対応)
# ==========================================
def plot_metrics(df, output_dir, max_iter):
  plt.style.use('default')
  
  # データフレーム内のユニークなディレクトリ（手法）を抽出
  # ここで指定された数だけ手法が見つかり、線が生成されます
  methods = df['Directory'].unique()
  
  # 凡例用にディレクトリ名（末尾）だけを取り出す
  labels = {m: os.path.basename(m.rstrip(os.sep)) for m in methods}
  
  metrics_config = [
    ('MSE_ROI', 'MSE (ROI, Scaled)', 'plot_mse_roi_scaled.png'),
    ('SSIM_ROI', 'SSIM (ROI, Scaled)', 'plot_ssim_roi_scaled.png'),
    ('MSE_All', 'MSE (All, Scaled)', 'plot_mse_all_scaled.png'),
    ('SSIM_All', 'SSIM (All, Scaled)', 'plot_ssim_all_scaled.png')
  ]

  for col, title, fname in metrics_config:
    plt.figure(figsize=(8, 6))
    
    # 手法の数だけループ (Matplotlibが自動で色をサイクルします)
    for method in methods:
      subset = df[df['Directory'] == method].copy()
      
      # Iteration列を数値化してソート
      subset['Iter_Int'] = pd.to_numeric(subset['Iteration'], errors='coerce')
      subset = subset.dropna(subset=['Iter_Int']).sort_values('Iter_Int')
      
      if not subset.empty:
        # label=labels[method] で凡例にフォルダ名が表示されます
        plt.plot(subset['Iter_Int'], subset[col], marker='o', markersize=4, label=labels[method], alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(col)
    
    # X軸の範囲調整
    plt.xlim(0, max_iter + (max_iter * 0.05))
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Graph saved: {out_path}")

# ==========================================
# パス解決ヘルパー
# ==========================================
def resolve_method_dirs(method_names):
  """
  入力された手法名リストから、実在するパスのリストを返す。
  """
  resolved = []
  for m in method_names:
    # 1. そのままパスとして存在するか
    if os.path.isdir(m):
      resolved.append(m)
      continue
    
    # 2. build/outputs/ の下にあるか
    candidate = os.path.join(BASE_OUTPUT_DIR, m)
    if os.path.isdir(candidate):
      resolved.append(candidate)
      continue
      
    print(f"Warning: Directory not found for method '{m}'. Skipped.")
    
  return resolved

# ==========================================
# Main
# ==========================================
def main():
  parser = argparse.ArgumentParser(description="Evaluate reconstruction results comparing multiple methods.")
  parser.add_argument("--max_iter", type=int, default=150, help="Evaluation limit for iteration count")
  parser.add_argument("--interval", type=int, default=10, help="Interval (just for info/validation)")
  
  # 引数で手法を上書き可能
  parser.add_argument("--methods", nargs='+', default=None, 
                      help="List of method directory names or paths")

  args = parser.parse_args()

  os.makedirs(OUTPUT_DIR, exist_ok=True)
  
  # Config読み込み
  config = load_grid_config()
  if config is None:
    print("Error: Config not found.")
    return
  nx, ny, nz = int(config["nx"]), int(config["ny"]), int(config["nz"])
  target_shape = (nz, ny, nx)

  # 真値データの読み込み
  if not os.path.exists(GROUND_TRUTH_PATH):
    print("Error: Ground Truth not found.")
    return
  true_data_flat = np.load(GROUND_TRUTH_PATH)
  try:
    true_data = true_data_flat.reshape(target_shape)
  except Exception as e:
    print(f"Error reshaping GT: {e}")
    return

  # --- 対象ディレクトリの決定 ---
  # 引数がなければ上部の DEFAULT_TARGET_DIRS (3つ設定済み) を使用
  input_methods = args.methods if args.methods else DEFAULT_TARGET_DIRS
  target_dirs = resolve_method_dirs(input_methods)

  if not target_dirs:
    print("Error: No valid method directories found.")
    return

  print(f"Comparison Targets: {[os.path.basename(d) for d in target_dirs]}")
  print(f"Evaluation Max Iter: {args.max_iter}")

  # --- データ処理ループ ---
  all_results = []
  
  for target_dir in target_dirs:
    print(f"\nProcessing: {target_dir}")
    
    files = []
    for p in TARGET_PATTERNS:
      files.extend(glob.glob(os.path.join(target_dir, p)))
    files.sort()
    
    if not files:
      print("  No result files found.")
      continue

    for fpath in files:
      fname = os.path.basename(fpath)
      
      # Iteration番号チェック
      iter_num_str = "Final"
      if "x_iter_" in fname:
        iter_num_str = fname.replace("x_iter_", "").replace(".npy", "")
        if iter_num_str.isdigit():
          iter_val = int(iter_num_str)
          if iter_val > args.max_iter: continue
      elif "prob_map" in fname:
        iter_num_str = "ProbMap"

      try:
        recon_flat = np.load(fpath)
        if recon_flat.size != nx*ny*nz: continue
        recon = recon_flat.reshape(target_shape)
        
        # 指標計算
        res = calculate_metrics(true_data, recon)
        if res is None: continue

        # ログ表示 (間引き)
        if iter_num_str == "ProbMap" or (iter_num_str.isdigit() and int(iter_num_str) % args.interval == 0):
          print(f"  Iter {iter_num_str:>3}: MSE(ROI)={res['MSE_ROI']:.5f}, SSIM(ROI)={res['SSIM_ROI']:.4f}")

        all_results.append({
          "Directory": target_dir,
          "Filename": fname,
          "Iteration": iter_num_str,
          **res
        })
      except Exception as e:
        print(f"Error processing {fname}: {e}")

  # --- 結果の保存と描画 ---
  if all_results:
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_summary_optimized.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    # グラフ描画 (ここが自動で3本の線を引きます)
    plot_metrics(df, OUTPUT_DIR, args.max_iter)
    print("Graphs generated.")
  else:
    print("No valid results to plot.")

if __name__ == "__main__":
  main()