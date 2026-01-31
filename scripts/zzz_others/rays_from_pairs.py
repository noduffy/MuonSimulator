#!/usr/bin/env python3
import csv, math, argparse, sys, pathlib
from pathlib import Path
import numpy as np

# === outputs helper ===
def _find_build_dir():
  cwd_build = Path.cwd() / "build"
  if cwd_build.exists():
    return cwd_build
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if (d / "build").exists():
      return d / "build"
  return Path("build")

def resolve_out(p):
  P = Path(str(p))
  if P.is_absolute():
    return P
  build = _find_build_dir()
  outdir = build / "outputs"
  outdir.mkdir(parents=True, exist_ok=True)
  if len(P.parts) >= 1 and P.parts[0] == "build":
    rel = Path(*P.parts[1:]) if len(P.parts) > 1 else Path(P.name)
    return outdir / rel
  return outdir / P.name
# === end helper ===

def rowfloat(r, k):
  try:
    return float(r[k])
  except:
    return float("nan")

def compute_angles(pairs_row):
  tx, ty, tz = rowfloat(pairs_row, "top_x"), rowfloat(pairs_row, "top_y"), rowfloat(pairs_row, "top_z")
  bx, by, bz = rowfloat(pairs_row, "bot_x"), rowfloat(pairs_row, "bot_y"), rowfloat(pairs_row, "bot_z")
  tdx, tdy, tdz = rowfloat(pairs_row, "top_dx"), rowfloat(pairs_row, "top_dy"), rowfloat(pairs_row, "top_dz")
  bdx, bdy, bdz = rowfloat(pairs_row, "bot_dx"), rowfloat(pairs_row, "bot_dy"), rowfloat(pairs_row, "bot_dz")

  vx, vy, vz = (bx - tx), (by - ty), (bz - tz)
  vnorm = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
  vx, vy, vz = vx / vnorm, vy / vnorm, vz / vnorm

  ux, uy, uz = tdx, tdy, tdz
  un = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
  ux, uy, uz = ux / un, uy / un, uz / un

  dot = max(-1.0, min(1.0, ux * vx + uy * vy + uz * vz))
  theta = math.acos(dot)
  return theta, theta * theta

def main(pairs_csv, out_csv, baseline_frac=0.15, disable_baseline=False, clip_p=99.5):
  # 1) 全件読み込み
  rows = []
  th_list = []
  th2_list = []
  with open(pairs_csv) as f:
    r = csv.DictReader(f)
    for row in r:
      theta, theta2 = compute_angles(row)
      rows.append((row.get("event", ""), theta, theta2))
      th_list.append(theta)
      th2_list.append(theta2)

  th2_arr = np.asarray(th2_list, dtype=float)
  if th2_arr.size == 0:
    baseline = 0.0
    qcap = np.inf
  else:
    # 2) 外れ値クリップ上限（Winsorize）。clip_p<=0 なら無効
    qcap = np.percentile(th2_arr, clip_p) if clip_p and clip_p > 0 else np.inf
    # 3) ベースライン（中央値のα倍）。無効化フラグなら 0
    baseline = 0.0 if disable_baseline else float(baseline_frac) * float(np.median(th2_arr))

  # 4) 書き出し（上限クリップ→ベースライン差し引き→0クリップ）
  out_path = resolve_out(out_csv)
  with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["event", "theta", "theta2"])
    for ev, th, th2 in rows:
      th2w = th2 if th2 <= qcap else qcap
      th2c = th2w - baseline
      if th2c < 0.0:
        th2c = 0.0
      w.writerow([ev, f"{th:.8g}", f"{th2c:.8g}"])

  # 5) ログ
  if th2_arr.size:
    print(f"[rays] N={th2_arr.size}  baseline={baseline:.3g}  "
          f"raw_med={np.median(th2_arr):.3g}  raw_p90={np.percentile(th2_arr,90):.3g}  "
          f"clip_p={clip_p} cap={qcap:.3g}")
  print(f"[OK] wrote rays: {out_path}")

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("pairs_csv")
  ap.add_argument("out_csv", nargs="?", default="rays.csv")
  ap.add_argument("--baseline-frac", type=float, default=0.15, help="baseline = frac * median(theta2)")
  ap.add_argument("--no-baseline", action="store_true", help="disable baseline subtraction")
  ap.add_argument("--clip-p", type=float, default=99.5, help="winsorize upper percentile (e.g., 99 or 99.5); <=0 to disable")
  args = ap.parse_args()
  if not pathlib.Path(args.pairs_csv).exists():
    print("missing pairs.csv", file=sys.stderr)
    sys.exit(2)
  main(args.pairs_csv, args.out_csv, baseline_frac=args.baseline_frac, disable_baseline=args.no_baseline, clip_p=args.clip_p)
