import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import sys

def get_project_root():
    # このスクリプトの場所: .../mygeom/scripts/paper_figures/synthesis_pngs.py
    # .parent -> paper_figures
    # .parent -> scripts
    # .parent -> mygeom (Project Root)
    return Path(__file__).resolve().parent.parent.parent

def main():
    root = get_project_root()
    
    # 入力ディレクトリ
    cgls_dir = root / "build" / "outputs" / "progressive_cgls"
    method_d_dir = root / "build" / "outputs" / "method_d_result"
    
    # 出力ディレクトリ (変更: build/outputs/paper_figures に保存)
    out_dir = root / "build" / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 対象のIteration
    iters = [20, 40, 60, 80]
    
    # 図の作成 (4行 x 2列)
    fig, axes = plt.subplots(len(iters), 2, figsize=(8, 12), dpi=200)
    
    # 列のタイトル (手法名)
    cols = ["Conventional (CGLS)", "Proposed (Method D)"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold', pad=10)

    for i, it in enumerate(iters):
        fname = f"render_iter_{it:04d}.png"
        
        # --- 左列: 従来法 (CGLS) ---
        path_cgls = cgls_dir / fname
        ax_cgls = axes[i, 0]
        
        if path_cgls.exists():
            img = mpimg.imread(str(path_cgls))
            ax_cgls.imshow(img)
        else:
            # 画像がない場合のプレースホルダー
            ax_cgls.text(0.5, 0.5, "Not Found", ha='center', va='center')
            # 枠線を消して白い空白にする
            ax_cgls.axis('off')
            print(f"[Warn] Missing: {path_cgls}")

        # --- 右列: 提案法 (Method D) ---
        path_md = method_d_dir / fname
        ax_md = axes[i, 1]
        
        if path_md.exists():
            img = mpimg.imread(str(path_md))
            ax_md.imshow(img)
        else:
            ax_md.text(0.5, 0.5, "Not Found", ha='center', va='center')
            ax_md.axis('off')
            print(f"[Warn] Missing: {path_md}")
            
        # --- 行のラベル (Iteration数) ---
        # 左側のプロットのY軸ラベルとして表示
        # 軸を消してもラベルだけ残すため、textで手動配置する方法もありますが、
        # ここでは軸のTicksだけ消してラベルを残す方法をとります。
        ax_cgls.set_ylabel(f"Iter {it}", fontsize=14, fontweight='bold')
        
        # --- 軸の装飾処理 ---
        for ax in axes[i]:
            # 目盛り(数字とヒゲ)を消す
            ax.set_xticks([])
            ax.set_yticks([])
            # 枠線(Spines)は残るため、画像がくっきり分かれます
            
    plt.tight_layout()
    
    # 保存
    save_path = out_dir / "comparison_grid_20_80.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Combined image saved to: {save_path}")

if __name__ == "__main__":
    main()