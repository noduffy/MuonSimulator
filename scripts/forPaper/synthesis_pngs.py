import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import sys

def get_project_root():
    # このスクリプトは scripts/paper_figures/ にあると仮定
    # .../scripts/paper_figures/synthesis_pngs.py
    # -> root is ../../
    return Path(__file__).resolve().parent.parent.parent

def main():
    root = get_project_root()
    
    # 入力ディレクトリ
    cgls_dir = root / "build" / "outputs" / "progressive_cgls"
    method_d_dir = root / "build" / "outputs" / "method_d_result"
    
    # 出力ディレクトリ
    out_dir = root / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 対象のIteration
    iters = [20, 40, 60, 80]
    
    # 図の作成 (4行 x 2列)
    # figsizeは画像の縦横比に合わせて調整してください
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
            ax_cgls.text(0.5, 0.5, "Not Found", ha='center', va='center')
            print(f"[Warn] Missing: {path_cgls}")

        # --- 右列: 提案法 (Method D) ---
        path_md = method_d_dir / fname
        ax_md = axes[i, 1]
        
        if path_md.exists():
            img = mpimg.imread(str(path_md))
            ax_md.imshow(img)
        else:
            ax_md.text(0.5, 0.5, "Not Found", ha='center', va='center')
            print(f"[Warn] Missing: {path_md}")
            
        # --- 行のラベル (Iteration数) ---
        # 左側のプロットのY軸ラベルとして表示
        ax_cgls.set_ylabel(f"Iter {it}", fontsize=14, fontweight='bold')
        
        # --- 軸の装飾を消す ---
        for ax in axes[i]:
            # 軸目盛りと枠線を消す（画像のみ表示）
            ax.set_xticks([])
            ax.set_yticks([])
            # 枠線を消したい場合は以下を有効化
            # ax.axis('off') 
            # ただし axis('off') すると ylabel (Iter XX) も消えるため、
            # 枠だけ残すか、テキストで配置する調整が必要です。
            # ここでは目盛り(Ticks)だけ消して枠とラベルは残します。
            
    plt.tight_layout()
    
    # 保存
    save_path = out_dir / "comparison_grid_20_80.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Combined image saved to: {save_path}")

if __name__ == "__main__":
    main()