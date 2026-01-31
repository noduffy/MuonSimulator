# 先頭の import 群の後に置く（2D/3Dどちらも同じでOK）

from pathlib import Path

def resolve_build_dir():
  candidates = []
  p = Path.cwd()
  for d in [p] + list(p.parents):
    if d.name == "build":
      return d
    if (d / "build").exists():
      candidates.append(d / "build")
  here = Path(__file__).resolve().parent
  for d in [here] + list(here.parents):
    if d.name == "build":
      return d
    if (d / "build").exists():
      candidates.append(d / "build")
  return candidates[0] if candidates else Path("build")

def resolve_in(p: str):
  P = Path(p)
  if P.is_absolute() and P.exists():
    return P
  if P.exists():
    return P
  build = resolve_build_dir()
  for base in [build / "outputs", build]:
    alt = base / P.name
    if alt.exists():
      return alt
  return P

def resolve_out(name: str):
  build = resolve_build_dir()
  outdir = build / "outputs"
  outdir.mkdir(parents=True, exist_ok=True)
  return outdir / Path(name).name
