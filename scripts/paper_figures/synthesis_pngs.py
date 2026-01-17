import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import sys

# ★ここを変更: ライブラリを使わず、Mac標準フォントを設定
# import japanize_matplotlib 
plt.rcParams['font.family'] = 'Hiragino Sans'  # Mac用の標準日本語フォント(ヒラギノ角ゴ)

def get_project_root():
    # このスクリプトの場所: .../scripts/paper_figures/synthesis_pngs.py
    return Path(__file__).resolve().parent.parent.parent

def main():
    root = get_project_root()
    
    # 入力ディレクトリ
    cgls_dir = root / "build" / "outputs" / "progressive_cgls"
    method_d_dir = root / "build" / "outputs" / "method_d_result"
    
    # 出力ディレクトリ
    out_dir = root / "build" / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    iters = [20, 50, 80]
    
    # 図の作成
    fig, axes = plt.subplots(len(iters), 2, figsize=(8, 12), dpi=200)
    
    cols = ["従来手法 (CGLS)", "提案手法 (空間制約付きCGLS)"]
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold', pad=10)

    for i, it in enumerate(iters):
        fname = f"render_iter_{it:04d}.png"
        
        # --- 左列: 従来手法 ---
        path_cgls = cgls_dir / fname
        ax_cgls = axes[i, 0]
        if path_cgls.exists():
            img = mpimg.imread(str(path_cgls))
            ax_cgls.imshow(img)
        else:
            ax_cgls.text(0.5, 0.5, "Not Found", ha='center', va='center')
            ax_cgls.axis('off')

        # --- 右列: 提案手法 ---
        path_md = method_d_dir / fname
        ax_md = axes[i, 1]
        if path_md.exists():
            img = mpimg.imread(str(path_md))
            ax_md.imshow(img)
        else:
            ax_md.text(0.5, 0.5, "Not Found", ha='center', va='center')
            ax_md.axis('off')
            
        # --- 行ラベル (反復回数) ---
        ax_cgls.set_ylabel(f"Iter {it}", fontsize=14, fontweight='bold')
        
        # 軸の装飾
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    
    save_path = out_dir / "comparison_grid_20_80.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Combined image saved to: {save_path}")

if __name__ == "__main__":
    main()