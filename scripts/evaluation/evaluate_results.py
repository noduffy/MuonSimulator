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
# 設定
# ==========================================
TARGET_DIRS = [
  "build/outputs/method_d_result",
  "build/outputs/progressive_cgls"
]
# prob_map.npy は反復回数に関係なく評価対象とする
TARGET_PATTERNS = ["x_iter_*.npy", "prob_map.npy"]
GROUND_TRUTH_PATH = "build/outputs/evaluation/true_density.npy" 
OUTPUT_DIR = "build/outputs/evaluation"
CONFIG_PATH = "configs/grid3d.yml"

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
  再構成データ(pred)に係数を掛けて、正解データ(true)に最も近づける(最小二乗法)。
  """
  t = true_vol.flatten()
  p = pred_vol.flatten()
  
  denom = np.dot(p, p)
  if denom == 0:
    return pred_vol, 0.0
    
  alpha = np.dot(t, p) / denom
  return pred_vol * alpha, alpha

# ==========================================
# 評価関数
# ==========================================
def calculate_metrics(true_vol, pred_vol):
  if true_vol.shape != pred_vol.shape:
    return None

  # 1. スケーリング補正
  pred_opt, scale_factor = optimize_scale(true_vol, pred_vol)

  # --- 全体評価 (Optimized) ---
  mse_all = mean_squared_error(true_vol, pred_opt)
  
  d_range = true_vol.max() - true_vol.min()
  if d_range == 0: d_range = 1.0

  try:
    ssim_all = ssim(true_vol, pred_opt, data_range=d_range, channel_axis=None)
  except:
    ssim_all = ssim(true_vol, pred_opt, data_range=d_range, channel_axis=None, win_size=3)

  # --- ROI評価 (Optimized) ---
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
# グラフ描画
# ==========================================
def plot_metrics(df, output_dir, max_iter):
  plt.style.use('default')
  methods = df['Directory'].unique()
  labels = {m: os.path.basename(m) for m in methods}
  
  metrics_config = [
    ('MSE_ROI', 'MSE (ROI, Scaled)', 'plot_mse_roi_scaled.png'),
    ('SSIM_ROI', 'SSIM (ROI, Scaled)', 'plot_ssim_roi_scaled.png')
  ]

  for col, title, fname in metrics_config:
    plt.figure(figsize=(8, 5))
    for method in methods:
      subset = df[df['Directory'] == method].copy()
      subset['Iter_Int'] = pd.to_numeric(subset['Iteration'], errors='coerce')
      subset = subset.dropna(subset=['Iter_Int']).sort_values('Iter_Int')
      
      # ProbMapなどは Iter_Int が NaN になるので除外される
      # グラフには反復過程(数値)のみをプロットする
      if not subset.empty:
        plt.plot(subset['Iter_Int'], subset[col], marker='o', markersize=4, label=labels[method])
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(col)
    plt.xlim(0, max_iter + (max_iter * 0.05)) # グラフの右端を少し余裕を持たせる
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()

# ==========================================
# Main
# ==========================================
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_iter", type=int, default=150, help="Evaluation limit for iteration count")
  parser.add_argument("--interval", type=int, default=10, help="Interval (just for info/validation)")
  args = parser.parse_args()

  os.makedirs(OUTPUT_DIR, exist_ok=True)
  
  config = load_grid_config()
  if config is None:
    print("Error: Config not found.")
    return
  
  nx, ny, nz = int(config["nx"]), int(config["ny"]), int(config["nz"])
  target_shape = (nz, ny, nx) # Method D order

  print(f"Target Shape: {target_shape}")
  print(f"Evaluation Max Iter: {args.max_iter}")

  if not os.path.exists(GROUND_TRUTH_PATH):
    print("Error: Ground Truth not found.")
    return

  true_data_flat = np.load(GROUND_TRUTH_PATH)
  try:
    true_data = true_data_flat.reshape(target_shape)
  except Exception as e:
    print(f"Error reshaping GT: {e}")
    return

  all_results = []
  
  for target_dir in TARGET_DIRS:
    if not os.path.exists(target_dir): continue
    print(f"\nProcessing: {target_dir}")
    
    files = []
    for p in TARGET_PATTERNS:
      files.extend(glob.glob(os.path.join(target_dir, p)))
    files.sort()
    
    for fpath in files:
      fname = os.path.basename(fpath)
      
      # --- フィルタリング処理 ---
      # ファイル名からiter番号を取得し、max_iterを超える古いファイルは無視する
      iter_num_str = "Final"
      if "x_iter_" in fname:
        iter_num_str = fname.replace("x_iter_", "").replace(".npy", "")
        if iter_num_str.isdigit():
          iter_val = int(iter_num_str)
          if iter_val > args.max_iter:
            continue # Skip old files
      elif "prob_map" in fname:
        iter_num_str = "ProbMap"
      # ------------------------

      try:
        recon_flat = np.load(fpath)
        if recon_flat.size != nx*ny*nz: continue
        
        recon = recon_flat.reshape(target_shape)
        res = calculate_metrics(true_data, recon)
        
        if res is None: continue

        # ログ出力 (適度に間引くか、重要なものだけ)
        if iter_num_str == "ProbMap" or (iter_num_str.isdigit() and int(iter_num_str) % 10 == 0):
          print(f"  {fname} -> MSE(ROI):{res['MSE_ROI']:.4f}, SSIM(ROI):{res['SSIM_ROI']:.4f}")

        all_results.append({
          "Directory": target_dir,
          "Filename": fname,
          "Iteration": iter_num_str,
          **res
        })
      except Exception as e:
        print(f"Error processing {fname}: {e}")

  if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_summary_optimized.csv"), index=False)
    # グラフ描画関数に max_iter を渡してX軸を調整
    plot_metrics(df, OUTPUT_DIR, args.max_iter)
    print("\nEvaluation Completed (Optimized & Filtered).")

if __name__ == "__main__":
  main()