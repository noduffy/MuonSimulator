#!/usr/bin/env python3
import csv, argparse, sys, math, pathlib
from pathlib import Path

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
  """出力専用: 相対パスは build/outputs/<basename> に寄せる。
     'build/xxx' 指定なら 'build/outputs/xxx' に付け替え。絶対パスは変更しない。"""
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

def pair_policy_first(top_rows, bot_rows):
  # 既存挙動: 先頭から順に1:1でペアにする（イベント整合などの高度処理はしない）
  return [(t, b) for t, b in zip(top_rows, bot_rows)]

def pair_policy_nearest_z(top_rows, bot_rows):
  pairs = []
  for t in top_rows:
    tz = float(t["top_z"])
    best = min(bot_rows, key=lambda b: abs(float(b["bot_z"]) - tz))
    pairs.append((t, best))
  return pairs

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--hit", required=True, help="hits.csv")
  ap.add_argument("--out", default="pairs.csv", help="output pairs.csv")
  # 元の機能を変えないため、--policy は“指定されていれば使うが、無ければ従来通り first と同等”
  ap.add_argument("--policy", choices=["first", "nearest_z"], default="first")
  args = ap.parse_args()

  hits = []
  with open(args.hit) as f:
    r = csv.DictReader(f)
    for row in r:
      hits.append(row)

  # ★ 変更ポイント：Top/Bottom の判定に layer が無い場合は zname を使う（大文字小文字無視）
  def is_top(row):
    layer = str(row.get("layer", "")).lower()
    if layer in ("top", "t", "0"):
      return True
    zname = str(row.get("zname", "")).lower()
    return "top" in zname

  def is_bottom(row):
    layer = str(row.get("layer", "")).lower()
    if layer in ("bottom", "bot", "b", "1"):
      return True
    zname = str(row.get("zname", "")).lower()
    return "bottom" in zname

  # 既存カラム名に合わせて top_*/bot_* を作るため、座標/方向のキー名を正規化しておく
  def to_top_fmt(h):
    return {
      "event": h.get("event", ""),
      "top_x": h.get("x"), "top_y": h.get("y"), "top_z": h.get("z"),
      "top_dx": h.get("dx"), "top_dy": h.get("dy"), "top_dz": h.get("dz"),
    }

  def to_bot_fmt(h):
    return {
      "event": h.get("event", ""),
      "bot_x": h.get("x"), "bot_y": h.get("y"), "bot_z": h.get("z"),
      "bot_dx": h.get("dx"), "bot_dy": h.get("dy"), "bot_dz": h.get("dz"),
    }

  tops_raw = [h for h in hits if is_top(h)]
  bots_raw = [h for h in hits if is_bottom(h)]

  # 既存の down-stream 互換のため、top_*/bot_* を持つ辞書に変換
  tops = [to_top_fmt(h) for h in tops_raw]
  bots = [to_bot_fmt(h) for h in bots_raw]

  if not tops or not bots:
    print("[WARN] Top/Bot のどちらかが0件です。zname=TopPlate/BottomPlate が入っているか再確認してください。", file=sys.stderr)

  if args.policy == "first":
    pairs = pair_policy_first(tops, bots)
  else:
    # nearest_z は top_z/bot_z を見る既存の振る舞い
    pairs = pair_policy_nearest_z(tops, bots)

  out_path = resolve_out(args.out)
  with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
      "event",
      "top_x","top_y","top_z","top_dx","top_dy","top_dz",
      "bot_x","bot_y","bot_z","bot_dx","bot_dy","bot_dz",
    ])
    for t, b in pairs:
      ev = t.get("event") or b.get("event", "")
      w.writerow([
        ev,
        t.get("top_x"), t.get("top_y"), t.get("top_z"), t.get("top_dx"), t.get("top_dy"), t.get("top_dz"),
        b.get("bot_x"), b.get("bot_y"), b.get("bot_z"), b.get("bot_dx"), b.get("bot_dy"), b.get("bot_dz"),
      ])

  print(f"[OK] wrote pairs: {out_path}  (pairs={len(pairs)})")

if __name__ == "__main__":
  sys.exit(main())
