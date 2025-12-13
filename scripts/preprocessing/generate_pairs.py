#!/usr/bin/env python3
import csv, argparse, sys
from pathlib import Path
from collections import defaultdict

# === outputs helper (元のまま) ===
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

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--hit", required=True, help="hits.csv")
  ap.add_argument("--out", default="pairs.csv", help="output pairs.csv")
  # policyオプションは廃止しました（常に厳密なマッチングを行います）
  args = ap.parse_args()

  # 1. データの読み込み & Track ID フィルタリング
  hits = []
  skipped_tracks = 0
  
  try:
    with open(args.hit, "r", encoding="utf-8") as f:
      r = csv.DictReader(f)
      for row in r:
        # ★重要: Track ID が '1' (一次粒子) 以外のものは無視する
        # これにより2次粒子（電子など）による逆方向のヒットやノイズを除去します
        if row.get("track") != "1":
          skipped_tracks += 1
          continue
        hits.append(row)
  except FileNotFoundError:
    print(f"[ERR] Input file not found: {args.hit}", file=sys.stderr)
    sys.exit(1)

  # 2. Top/Bottom 判定
  def is_top(row):
    layer = str(row.get("layer", "")).lower()
    if layer in ("top", "t", "0"): return True
    zname = str(row.get("zname", "")).lower()
    return "top" in zname

  def is_bottom(row):
    layer = str(row.get("layer", "")).lower()
    if layer in ("bottom", "bot", "b", "1"): return True
    zname = str(row.get("zname", "")).lower()
    return "bottom" in zname

  # 3. イベントIDごとにグループ化
  events = defaultdict(lambda: {"top": None, "bot": None})

  for h in hits:
    eid = h.get("event")
    if eid is None: continue

    if is_top(h):
      # 同じイベントの同じトラックで複数ヒットがあっても最初を採用
      if events[eid]["top"] is None:
        events[eid]["top"] = h
    elif is_bottom(h):
      if events[eid]["bot"] is None:
        events[eid]["bot"] = h

  # 4. ペアの抽出 (TopとBottomが揃っているものだけ)
  pairs = []
  
  # イベントID順にソートして処理
  try:
    sorted_keys = sorted(events.keys(), key=lambda x: int(x))
  except ValueError:
    sorted_keys = sorted(events.keys())

  for eid in sorted_keys:
    t = events[eid]["top"]
    b = events[eid]["bot"]

    if t is not None and b is not None:
      pair = {
        "event": eid,
        "top_x": t.get("x"), "top_y": t.get("y"), "top_z": t.get("z"),
        "top_dx": t.get("dx"), "top_dy": t.get("dy"), "top_dz": t.get("dz"),
        "bot_x": b.get("x"), "bot_y": b.get("y"), "bot_z": b.get("z"),
        "bot_dx": b.get("dx"), "bot_dy": b.get("dy"), "bot_dz": b.get("dz"),
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

  # 統計表示
  print(f"[OK] Generated pairs: {len(pairs)}")
  print(f"     Output: {out_path}")
  print(f"     (Ignored {skipped_tracks} secondary particle hits)")
  print(f"     (Discarded {len(events) - len(pairs)} incomplete events)")

if __name__ == "__main__":
  sys.exit(main())