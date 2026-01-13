#!/usr/bin/env python3
import csv, argparse, sys
from pathlib import Path
from collections import defaultdict

# === outputs helper ===
def _find_build_dir():
  cwd_build = Path.cwd() / "build"
  if cwd_build.exists(): return cwd_build
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if (d / "build").exists(): return d / "build"
  return Path("build")

def resolve_out(p):
  P = Path(str(p))
  if P.is_absolute(): return P
  build = _find_build_dir()
  outdir = build / "outputs"
  outdir.mkdir(parents=True, exist_ok=True)
  return outdir / P.name
# === end helper ===

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--hit", required=True, help="hits.csv")
  ap.add_argument("--out", default="pairs.csv", help="output pairs.csv")
  args = ap.parse_args()

  hits = []
  skipped_tracks = 0
  
  # ヘッダーチェック用
  required_cols = ["event", "track", "z", "x", "y", "dx", "dy", "dz"]

  try:
    with open(args.hit, "r", encoding="utf-8") as f:
      # ヘッダーがあるか簡易チェック（1行目を読む）
      sample = f.read(1024)
      f.seek(0)
      has_header = csv.Sniffer().has_header(sample)
      
      if has_header:
        r = csv.DictReader(f)
        # カラム名が正しいかチェック (RunActionの実装に依存するため)
        # もしSteppingActionでヘッダーを書いていない場合、ここでKeyErrorになる可能性があります
        # その場合は RunAction を修正する必要があります
        for row in r:
          if row.get("track") != "1": # 一次粒子のみ
            skipped_tracks += 1
            continue
          hits.append(row)
      else:
        # ヘッダーがない場合、SteppingActionの出力順に合わせて手動割り当て
        # evid, trackID, volName, physName, x, y, z, dx, dy, dz
        fieldnames = ["event", "track", "vol", "name", "x", "y", "z", "dx", "dy", "dz"]
        r = csv.DictReader(f, fieldnames=fieldnames)
        for row in r:
          if row.get("track") != "1":
            skipped_tracks += 1
            continue
          hits.append(row)
          
  except FileNotFoundError:
    print(f"[ERR] Input file not found: {args.hit}", file=sys.stderr)
    sys.exit(1)

  # 3. イベントIDごとにグループ化
  events = defaultdict(lambda: {"top": None, "bot": None})

  for h in hits:
    try:
      eid = h["event"]
      z_val = float(h["z"]) # Z座標で判定
    except ValueError:
      continue # 数値変換エラーはスキップ

    # Z > 0 なら Top, Z < 0 なら Bottom (検出器は ±80mm なのでこれで確実)
    if z_val > 0:
      if events[eid]["top"] is None: events[eid]["top"] = h
    else:
      if events[eid]["bot"] is None: events[eid]["bot"] = h

  # 4. ペアの抽出
  pairs = []
  
  # ソート (文字列ID対策)
  try:
    sorted_keys = sorted(events.keys(), key=lambda x: int(x))
  except:
    sorted_keys = sorted(events.keys())

  for eid in sorted_keys:
    t = events[eid]["top"]
    b = events[eid]["bot"]

    if t and b:
      pair = {
        "event": eid,
        "top_x": t["x"], "top_y": t["y"], "top_z": t["z"],
        "top_dx": t["dx"], "top_dy": t["dy"], "top_dz": t["dz"],
        "bot_x": b["x"], "bot_y": b["y"], "bot_z": b["z"],
        "bot_dx": b["dx"], "bot_dy": b["dy"], "bot_dz": b["dz"],
      }
      pairs.append(pair)

  # 5. 出力
  out_path = resolve_out(args.out)
  with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
      "event",
      "top_x","top_y","top_z","top_dx","top_dy","top_dz",
      "bot_x","bot_y","bot_z","bot_dx","bot_dy","bot_dz",
    ])
    w.writeheader()
    w.writerows(pairs)

  print(f"[OK] Generated pairs: {len(pairs)}")
  print(f"     Output: {out_path}")

if __name__ == "__main__":
  main()