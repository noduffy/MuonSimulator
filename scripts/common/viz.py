# ファイル名: scripts/common/viz.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu

def make_isos(vol, spacing, level_low=None, level_hi=None):
    """3Dボリュームから等値面(Isosurface)のメッシュデータを生成する"""
    vv = vol.astype(np.float32)
    pos = vv[vv > 0]
    if pos.size == 0: return ([], []), ([], []), (0.0, 0.0)

    # 閾値の自動決定ロジック
    if level_low is None:
        try: level_low = float(threshold_otsu(pos))
        except Exception: level_low = float(np.percentile(pos, 75))
    
    if level_hi is None:
        level_hi = float(np.percentile(pos, 95)) if len(pos) > 0 else 0.9
    
    if level_hi > vol.max(): level_hi = vol.max()
    if level_low >= level_hi: level_low = level_hi * 0.5

    # Marching Cubes
    vertsL, facesL = [], []
    try: vertsL, facesL, _, _ = marching_cubes(vv, level=level_low, spacing=spacing)
    except ValueError: pass

    vertsH, facesH = [], []
    try: vertsH, facesH, _, _ = marching_cubes(vv, level=level_hi, spacing=spacing)
    except ValueError: pass
        
    return (vertsL, facesL), (vertsH, facesH), (level_low, level_hi)

def add_mesh(ax, verts, faces, color, alpha, origin):
    """生成されたメッシュをMatplotlibのAxesに追加する"""
    if len(verts) == 0: return
    oz, oy, ox = origin 
    # marching_cubesの座標系(z,y,x)をプロット用(x,y,z)に変換
    V = np.empty_like(verts)
    V[:, 0] = verts[:, 2] + ox
    V[:, 1] = verts[:, 1] + oy
    V[:, 2] = verts[:, 0] + oz
    
    mesh = Poly3DCollection(V[faces], linewidths=0.15, facecolor=color, edgecolor="k", alpha=alpha)
    ax.add_collection3d(mesh)

def draw_box(ax, center, size, color, alpha=0.25, edgecolor="k", linewidth=0.3):
    """直方体を描画する汎用関数 (Detectorや物体用)"""
    cx, cy, cz = center
    sx, sy, sz = size
    
    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy - sy/2, cy + sy/2
    z0, z1 = cz - sz/2, cz + sz/2
    
    P = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], # 下面
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]  # 上面
    ], dtype=float)

    F = [
        [0,1,2,3], [4,5,6,7], # 底, 天
        [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7] # 側面
    ]
    
    poly = Poly3DCollection([P[f] for f in F], facecolors=color, edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(poly)

def draw_detectors(ax, z_pos=80.0, size=300.0):
    """統一されたスタイルで上下の検出器を描画するヘルパー関数"""
    # Top: Green
    draw_box(ax, (0, 0, z_pos), (size, size, 1), color=(0.1, 1.0, 0.1), alpha=0.25)
    # Bottom: Red
    draw_box(ax, (0, 0, -z_pos), (size, size, 1), color=(1.0, 0.1, 0.1), alpha=0.25)