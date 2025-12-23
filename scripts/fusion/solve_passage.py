# ファイル名: scripts/fusion/solve_passage.py
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import cg
from pathlib import Path
import argparse
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

def main():
  parser = argparse.ArgumentParser(description="通過マップ行列を解いて3D画像にする")
  parser.add_argument("--iter", type=int, default=50, help="CG法の反復回数")
  args = parser.parse_args()

  # ファイルパス
  WTW_PATH = resolve_out("WTW_passage.npz")
  WTy_PATH = resolve_out("WTy_passage.npy")
  OUT_PATH = resolve_out("passage_map.npy")

  if not WTW_PATH.exists() or not WTy_PATH.exists():
    print(f"[Error] 行列ファイルが見つかりません。")
    print("  -> 'make fusion-prep' (build_passage_matrix.py) を先に実行してください。")
    sys.exit(1)

  print("行列を読み込んでいます...")
  A = load_npz(WTW_PATH)
  b = np.load(WTy_PATH)

  print(f"共役勾配法(CG)で解いています... (iter={args.iter})")
  # scipy.sparse.linalg.cg は Ax=b を解く関数です
  x, info = cg(A, b, maxiter=args.iter, rtol=1e-5)
  
  if info == 0:
    print("収束しました。")
  else:
    print(f"収束しませんでした (code={info}) が、結果を出力します。")

  # 負の値は物理的にあり得ないので0にクリップ（念のため）
  x = np.clip(x, 0, None)
  
  # 保存
  np.save(OUT_PATH, x)
  print(f"[OK] 通過マップを保存しました: {OUT_PATH}")

  # 簡易ヒストグラム（値の分布確認用）
  plt.figure()
  plt.hist(x[x > 0], bins=50, log=True)
  plt.title("Passage Map Intensity Distribution")
  plt.savefig(resolve_out("passage_map_hist.png"))
  print("  -> 分布図: passage_map_hist.png")

if __name__ == "__main__":
  main()