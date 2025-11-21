#!/usr/bin/env python3
import argparse, sys
import numpy as np
from pathlib import Path

def _find_build_dir():
  cwd_build = Path.cwd() / "build"
  return cwd_build if cwd_build.exists() else Path("build")

def resolve_out(name):
  outdir = _find_build_dir() / "outputs"
  outdir.mkdir(parents=True, exist_ok=True)
  return outdir / name

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--vol", default="build/outputs/recon3d_vol.npy",
                  help="np.load-able 3D volume (nz,ny,nx)")
  ap.add_argument("--level", type=float, default=None,
                  help="isosurface level; default=Otsu自動 or 98percentile")
  ap.add_argument("--smooth", type=float, default=1.0,
                  help="Gaussian σ (voxel); 0で無効")
  ap.add_argument("--fmt", choices=["stl","ply","obj"], default="stl")
  ap.add_argument("--out", default="recon3d_mesh.stl",
                  help="出力ファイル名（拡張子は --fmt に合わせて上書き）")
  args = ap.parse_args()

  vol = np.load(args.vol)
  if args.smooth > 0:
    from scipy.ndimage import gaussian_filter
    vol = gaussian_filter(vol, sigma=args.smooth)

  # しきい値（自動）: まずOtsu、失敗したら上位2%の値
  level = args.level
  if level is None:
    try:
      from skimage.filters import threshold_otsu
      level = float(threshold_otsu(vol[vol>0]))  # 0は外れや床ノイズ扱い
    except Exception:
      level = float(np.percentile(vol, 98.0))
  print(f"[INFO] iso-level = {level:g}")

  # marching cubes
  from skimage.measure import marching_cubes
  verts, faces, normals, _ = marching_cubes(vol, level=level, spacing=(1.0,1.0,1.0))
  # skimageの座標は (z,y,x)。必要に応じて(mmスケール等)スケーリング可能
  print(f"[OK] mesh: V={len(verts)} F={len(faces)}")

  # 書き出し
  import trimesh
  mesh = trimesh.Trimesh(vertices=verts[:, ::-1], faces=faces, process=False)  # (x,y,z)順に並べ替え
  outpath = resolve_out(Path(args.out).with_suffix(f".{args.fmt}").name)
  mesh.export(outpath)
  print(f"[OK] wrote mesh: {outpath}")

if __name__ == "__main__":
  sys.exit(main())
