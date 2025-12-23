# ファイル名: scripts/fusion/fuse_images.py
import numpy as np
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# --- パス解決 ---
def get_project_root():
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if (d / "build").exists(): return d
  return Path.cwd()

def resolve_out(name: str) -> Path:
  root = get_project_root()
  return root / "build" / "outputs" / name

def normalize(vol):
  """0.0〜1.0に正規化する"""
  v_min, v_max = vol.min(), vol.max()
  if v_max - v_min == 0: return vol
  return (vol - v_min) / (v_max - v_min)

def main():
  parser = argparse.ArgumentParser(description="散乱画像と通過マップを融合(引き算)する")
  parser.add_argument("--scat", required=True, help="散乱画像のnpyファイル (例: progressive/x_iter_0200.npy)")
  parser.add_argument("--passmap", default="passage_map.npy", help="通過マップのnpyファイル")
  parser.add_argument("--alpha", type=float, default=0.5, help="融合係数 (大きいほど強く削る)")
  parser.add_argument("--out", default="fused_result.npy", help="出力ファイル名")
  args = parser.parse_args()

  # パス解決
  # --scat はサブディレクトリ(progressive)内にあるかもしれないので柔軟に
  if (resolve_out(args.scat)).exists():
    SCAT_PATH = resolve_out(args.scat)
  elif Path(args.scat).exists():
    SCAT_PATH = Path(args.scat)
  else:
    print(f"[Error] 散乱画像が見つかりません: {args.scat}")
    sys.exit(1)
    
  PASS_PATH = resolve_out(args.passmap)
  if not PASS_PATH.exists():
    print(f"[Error] 通過マップが見つかりません: {args.passmap}")
    sys.exit(1)

  print(f"画像を読み込んでいます...")
  vol_scat = np.load(SCAT_PATH)
  vol_pass = np.load(PASS_PATH)

  if vol_scat.shape != vol_pass.shape:
    print(f"[Error] 画像サイズが一致しません: Scat{vol_scat.shape} != Pass{vol_pass.shape}")
    sys.exit(1)

  # --- 融合処理 (Method C) ---
  print(f"融合処理を実行中 (alpha={args.alpha})...")
  
  # 1. 正規化 (重要: スケールを合わせるため)
  #    それぞれの最大値を1.0にしてから演算するのが安全です
  vol_scat_norm = normalize(vol_scat)
  vol_pass_norm = normalize(vol_pass)

  # 2. 引き算 (Subtraction)
  #    Formula: Final = Scat - alpha * Passage
  vol_fused = vol_scat_norm - (args.alpha * vol_pass_norm)
  
  # 3. 負の値のクリップ
  vol_fused = np.clip(vol_fused, 0, None)

  # 保存
  OUT_PATH = resolve_out(args.out)
  np.save(OUT_PATH, vol_fused)
  print(f"[OK] 融合画像を保存しました: {OUT_PATH}")

  # --- 簡易可視化 (中央断面比較) ---
  nx = int(round(len(vol_fused)**(1/3))) # 立方体と仮定
  center = nx // 2
  
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  
  # 3D配列に変形して断面を取得 (形状が合わなければエラーになるのでtry)
  try:
    s_scat = vol_scat_norm.reshape((nx, nx, nx))[:, center, :]
    s_pass = vol_pass_norm.reshape((nx, nx, nx))[:, center, :]
    s_fuse = vol_fused.reshape((nx, nx, nx))[:, center, :]

    axes[0].imshow(s_scat, cmap='jet', origin='lower')
    axes[0].set_title("Original (Scattered)")
    
    axes[1].imshow(s_pass, cmap='gray_r', origin='lower') # 通過マップは白黒反転で見やすく
    axes[1].set_title("Passage Map (Negative)")
    
    axes[2].imshow(s_fuse, cmap='jet', origin='lower')
    axes[2].set_title(f"Fused (alpha={args.alpha})")
    
    img_out = resolve_out(f"fusion_preview_alpha{args.alpha}.png")
    plt.savefig(img_out)
    print(f"  -> プレビュー画像: {img_out}")
    
  except Exception as e:
    print(f"[Warn] プレビュー生成失敗 (形状不一致など): {e}")

if __name__ == "__main__":
  main()