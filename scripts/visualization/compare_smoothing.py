import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import yaml
import sys
from pathlib import Path

# ==========================================
# 1. 平滑化したいフォルダを配列で指定
# ==========================================
TARGET_DIRS = [
    "build/outputs/method_d_result",
    "build/outputs/progressive_cgls"
]

# ==========================================
# 共通ライブラリのインポート
# scripts/common を見つけるためにパスを通す
# ==========================================
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import paths, viz

def load_config():
    """grid3d.ymlから (nx, ny, nz) と ranges を読み込む"""
    try:
        with open(paths.config_path("grid3d.yml")) as f:
            g = yaml.safe_load(f)

        nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
        ranges = (
            float(g["x_min"]), float(g["x_max"]),
            float(g["y_min"]), float(g["y_max"]),
            float(g["z_min"]), float(g["z_max"]),
        )
        return nx, ny, nz, ranges
    except Exception as e:
        print(f"Error loading config: {e}")
        return None, None, None, None

def voxel_sizes_mm(nx, ny, nz, ranges):
    xmin, xmax, ymin, ymax, zmin, zmax = ranges
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz
    return dx, dy, dz

def save_snapshot_3d(x_vector, nx, ny, nz, ranges, iteration, out_file, sigma_zyx):
    """
    method_d の save_snapshot_d と同じ系（vizライブラリ）で3D鳥瞰図を保存
    sigma_zyx: (sz, sy, sx) を表示用にタイトルへ入れる
    """
    xmin, xmax, ymin, ymax, zmin, zmax = ranges
    dx, dy, dz = voxel_sizes_mm(nx, ny, nz, ranges)

    vol = x_vector.reshape((nz, ny, nx))
    (vL, fL), (vH, fH), (lvL, lvH) = viz.make_isos(vol, (dz, dy, dx))

    fig = plt.figure(figsize=(8, 6), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    viz.draw_detectors(ax, z_pos=80.0, size=300.0)

    viz.add_mesh(ax, vL, fL, color=(0.4, 0.7, 1.0), alpha=0.25, origin=(zmin, ymin, xmin))
    viz.add_mesh(ax, vH, fH, color=(1.0, 0.1, 0.1), alpha=0.9,  origin=(zmin, ymin, xmin))

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
    ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
    ax.view_init(elev=90, azim=0)

    sz, sy, sx = sigma_zyx
    ax.set_title(f"Smoothed Volume: Iteration {iteration:04d}\n(Anisotropic Gaussian sigma_zyx=({sz:.3g},{sy:.3g},{sx:.3g}))")

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

def parse_sigma(args, nx, ny, nz, ranges):
    """
    使い方：
      - 等方（従来）: --sigma 0.5
      - 異方（ボクセル）: --sigma_zyx 0.8 0.2 0.2   # (sz, sy, sx)
      - 異方（mm指定）: --sigma_xyz 2.0 2.0 4.0     # (sx_mm, sy_mm, sz_mm)
        → 内部でボクセル単位 (sz,sy,sx) に変換
    """
    dx, dy, dz = voxel_sizes_mm(nx, ny, nz, ranges)

    if args.sigma_zyx is not None:
        sz, sy, sx = args.sigma_zyx
        return (float(sz), float(sy), float(sx))

    if args.sigma_xyz is not None:
        sx_mm, sy_mm, sz_mm = args.sigma_xyz
        # gaussian_filter はボクセル単位なので変換
        sx = float(sx_mm) / dx
        sy = float(sy_mm) / dy
        sz = float(sz_mm) / dz
        return (sz, sy, sx)

    # fallback: 等方
    s = float(args.sigma)
    return (s, s, s)

def process_directory(target_dir_str, sigma_zyx, nx, ny, nz, ranges):
    """1つのディレクトリ内の全 x_iter_*.npy を平滑化し、3Dレンダも保存"""
    target_path = Path(target_dir_str)

    if not target_path.exists():
        alt = Path.cwd() / target_dir_str
        if alt.exists():
            target_path = alt
        else:
            print(f"Skip: Directory not found -> {target_dir_str}")
            return

    print(f"\nProcessing Directory: {target_path}")

    # 保存先フォルダ名（分かりやすく）
    sz, sy, sx = sigma_zyx
    output_subdir = target_path / f"smoothed_sigma_zyx_{sz:g}_{sy:g}_{sx:g}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder: {output_subdir}")

    files = sorted(list(target_path.glob("x_iter_*.npy")))
    if not files:
        print("  No .npy files found.")
        return

    count = 0
    for fpath in files:
        fname = fpath.name
        iter_str = fname.replace("x_iter_", "").replace(".npy", "")
        try:
            iteration = int(iter_str)
        except:
            print(f"  Skip {fname}: iteration parse failed")
            continue

        # 読み込み
        try:
            vec = np.load(fpath)
            if vec.size != nx * ny * nz:
                print(f"  Skip {fname}: Size mismatch (got {vec.size}, expected {nx*ny*nz})")
                continue
            vol = vec.reshape((nz, ny, nx))
        except Exception as e:
            print(f"  Error reading {fname}: {e}")
            continue

        # 異方ガウシアン（sigmaは (z,y,x)）
        vol_smoothed = gaussian_filter(vol, sigma=sigma_zyx)
        vec_smoothed = vol_smoothed.flatten()

        # 保存（npyは同名で維持）
        save_npy_path = output_subdir / fname
        np.save(save_npy_path, vec_smoothed)

        # 保存（png）
        save_img_path = output_subdir / f"render_iter_{iteration:04d}.png"
        save_snapshot_3d(vec_smoothed, nx, ny, nz, ranges, iteration, save_img_path, sigma_zyx)

        count += 1
        if count % 10 == 0:
            print(f"  Processed {count} files...")

    print(f"  Done! {count} files processed.")

def main():
    parser = argparse.ArgumentParser(description="Batch smoothing + 3D render (anisotropic Gaussian).")

    # 互換：等方 sigma
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Isotropic sigma in voxel units (default: 0.5)")

    # 異方：ボクセル単位（順番に注意）
    parser.add_argument("--sigma_zyx", type=float, nargs=3, default=None,
                        metavar=("SZ", "SY", "SX"),
                        help="Anisotropic sigma in voxel units, order=(z,y,x). e.g. --sigma_zyx 0.8 0.2 0.2")

    # 異方：mm指定（入力は直感的に xyz）
    parser.add_argument("--sigma_xyz", type=float, nargs=3, default=None,
                        metavar=("SX_MM", "SY_MM", "SZ_MM"),
                        help="Anisotropic sigma in mm, order=(x,y,z). e.g. --sigma_xyz 2.0 2.0 4.0")

    args = parser.parse_args()

    nx, ny, nz, ranges = load_config()
    if nx is None:
        return

    sigma_zyx = parse_sigma(args, nx, ny, nz, ranges)

    dx, dy, dz = voxel_sizes_mm(nx, ny, nz, ranges)
    print("Grid voxel size [mm]: dx=%.3f, dy=%.3f, dz=%.3f" % (dx, dy, dz))
    print("Using sigma (voxel units) order=(z,y,x):", sigma_zyx)

    for d in TARGET_DIRS:
        process_directory(d, sigma_zyx, nx, ny, nz, ranges)

if __name__ == "__main__":
    main()
