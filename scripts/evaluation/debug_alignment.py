import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

# ==============================
# 設定
# ==============================
GT_PATH = "build/outputs/evaluation/true_density.npy"
# 比較したい再構成ファイル（Method D の最後のほうのファイルを選んでください）
RECON_PATH = "build/outputs/method_d_result/x_iter_0150.npy" 
CONFIG_PATH = "configs/grid3d.yml"
OUTPUT_IMG = "build/outputs/evaluation/debug_alignment.png"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def main():
    # 1. Configから形状を取得
    if not os.path.exists(CONFIG_PATH):
        print("Config not found. Using (40,40,30) as fallback.")
        nx, ny, nz = 40, 40, 30
    else:
        c = load_config()
        nx, ny, nz = int(c['n_x']), int(c['n_y']), int(c['n_z'])

    print(f"Expected Shape: ({nx}, {ny}, {nz})")

    # 2. データの読み込み
    if not os.path.exists(GT_PATH) or not os.path.exists(RECON_PATH):
        print("File not found.")
        return

    gt_data = np.load(GT_PATH)
    recon_data = np.load(RECON_PATH)

    print(f"GT Raw Shape: {gt_data.shape}, Max: {gt_data.max():.4f}")
    print(f"Recon Raw Shape: {recon_data.shape}, Max: {recon_data.max():.4f}")

    # 3. Reshape (1次元なら3次元に戻す)
    # ここでの並び順 (order='C' or 'F') がズレの原因の可能性があります
    try:
        if gt_data.ndim == 1: gt_data = gt_data.reshape((nx, ny, nz), order='C')
        if recon_data.ndim == 1: recon_data = recon_data.reshape((nx, ny, nz), order='C')
    except Exception as e:
        print(f"Reshape error: {e}")
        return

    # 4. 中央スライスの取得
    sz_z = gt_data.shape[2] // 2
    sz_y = gt_data.shape[1] // 2
    sz_x = gt_data.shape[0] // 2

    # Z軸方向の断面（XY平面）
    gt_slice_z = gt_data[:, :, sz_z]
    recon_slice_z = recon_data[:, :, sz_z]

    # Y軸方向の断面（XZ平面）
    gt_slice_y = gt_data[:, sz_y, :]
    recon_slice_y = recon_data[:, sz_y, :]

    # 5. プロット
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # --- XY Plane (Z-slice) ---
    axes[0, 0].imshow(gt_slice_z, cmap='jet', origin='lower')
    axes[0, 0].set_title(f"Ground Truth (XY slice)\nMax={gt_slice_z.max():.2f}")
    
    axes[0, 1].imshow(recon_slice_z, cmap='jet', origin='lower')
    axes[0, 1].set_title(f"Reconstruction (XY slice)\nMax={recon_slice_z.max():.2f}")

    # --- XZ Plane (Y-slice) ---
    axes[1, 0].imshow(gt_slice_y.T, cmap='jet', origin='lower') # 向き調整のため転置など試行が必要かも
    axes[1, 0].set_title("Ground Truth (XZ slice)")
    
    axes[1, 1].imshow(recon_slice_y.T, cmap='jet', origin='lower')
    axes[1, 1].set_title("Reconstruction (XZ slice)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG)
    print(f"Debug image saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()