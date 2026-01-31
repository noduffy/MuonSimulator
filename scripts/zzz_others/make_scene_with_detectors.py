#!/usr/bin/env python3
# scripts/make_scene_with_detectors.py  （インデント2）
import csv, yaml, numpy as np
from pathlib import Path
import trimesh

# ---- 入力パス ----
BUILD = Path("build")
OUTDIR = BUILD / "outputs"
mesh_in = OUTDIR / "recon3d_mesh.stl"     # 既存のSTL（ボクセル単位の縮尺のままでOK）
grid_yml = Path("configs/grid3d.yml")
pairs_csv = OUTDIR / "pairs.csv"          # ない場合は build/outputs にある最新を指定

# ---- 1) grid3d.yml から実寸情報を取得（mm）----
g = yaml.safe_load(grid_yml.read_text())
nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
xmin, xmax = float(g["x_min"]), float(g["x_max"])
ymin, ymax = float(g["y_min"]), float(g["y_max"])
zmin, zmax = float(g["z_min"]), float(g["z_max"])
sx = (xmax - xmin) / nx  # mm/voxel
sy = (ymax - ymin) / ny
sz = (zmax - zmin) / nz

# ---- 2) pairs.csv から Top/Bottom の z を推定（中央値）----
tzs, bzs = [], []
with pairs_csv.open() as f:
  r = csv.DictReader(f)
  for row in r:
    tzs.append(float(row["top_z"]))
    bzs.append(float(row["bot_z"]))
z_top = float(np.median(tzs)) if tzs else (zmax + zmin)/2
z_bot = float(np.median(bzs)) if bzs else (zmax + zmin)/2

# ---- 3) 既存STLを読み込み → 実寸(mm)スケール＆平行移動 ----
# 既存STLは (voxel単位, 原点=0) 前提。ボクセル実寸と ROI 原点に合わせる
recon = trimesh.load_mesh(mesh_in, process=False)
S = np.diag([sx, sy, sz, 1.0])  # スケーリング（x,y,z）
T = np.eye(4); T[:3, 3] = [xmin, ymin, zmin]  # ROI原点へ平行移動
recon.apply_transform(S @ T)  # 先にスケール→次に平行移動（右から適用）

# マテリアル色（鉛推定を目立たせる）
recon.visual.vertex_colors = [80, 170, 255, 200]  # RGBA

# ---- 4) 検出器プレート（薄板）を作る ----
def make_plate(z_center, thickness=2.0, margin=0.0, color=(255,120,60,120)):
  w = (xmax - xmin) + 2*margin
  h = (ymax - ymin) + 2*margin
  box = trimesh.creation.box(extents=(w, h, thickness))
  # 中心を (x_mid, y_mid, z_center) に
  xmid = 0.5*(xmin + xmax); ymid = 0.5*(ymin + ymax)
  M = np.eye(4); M[:3, 3] = [xmid, ymid, z_center]
  box.apply_transform(M)
  box.visual.vertex_colors = color
  return box

top_plate = make_plate(z_top, thickness=2.0, margin=0.0, color=(20,220,20,90))
bot_plate = make_plate(z_bot, thickness=2.0, margin=0.0, color=(220,20,20,90))

# ---- 5) ROIバウンディングボックス＆座標軸 ----
bbox = trimesh.creation.box(extents=(xmax-xmin, ymax-ymin, zmax-zmin))
Mb = np.eye(4); Mb[:3, 3] = [0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)]
bbox.apply_transform(Mb)
bbox.visual.face_colors = [200,200,200,30]

def axis(length=200.0):
  # X=赤, Y=緑, Z=青 を示す線分。色はGLBで無視されるビューアもある点に注意。
  import numpy as np
  geoms = []
  def add_seg(p, q):
    path = trimesh.load_path(np.array([p, q], dtype=float))
    geoms.append(path)
  add_seg([xmin, ymin, zmin], [xmin + length, ymin, zmin])  # X
  add_seg([xmin, ymin, zmin], [xmin, ymin + length, zmin])  # Y
  add_seg([xmin, ymin, zmin], [xmin, ymin, zmin + length])  # Z
  return geoms


axes = axis(length=200.0)

# ---- 6) すべて合成して glTF（.glb）で書き出し ----
scene = trimesh.Scene()
scene.add_geometry(bbox, node_name="ROI_box")
scene.add_geometry(top_plate, node_name="TopPlate")
scene.add_geometry(bot_plate, node_name="BottomPlate")
scene.add_geometry(recon, node_name="Reconstruction")
for i, ax in enumerate(axes):
  scene.add_geometry(ax, node_name=f"axis_{i}")

OUTDIR.mkdir(parents=True, exist_ok=True)
out_glb = OUTDIR / "scene_with_detectors.glb"
scene.export(out_glb)  # glTF2.0バイナリ。色・座標・スケール保持
print(f"[OK] wrote: {out_glb}  (units: mm)")
