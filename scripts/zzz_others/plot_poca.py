#!/usr/bin/env python3
import pandas as pd, numpy as np, argparse, pathlib, sys, math, matplotlib.pyplot as plt
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
  p = pathlib.Path("poca.csv")
  if not p.exists():
    print("poca.csv not found; this script mostly visualizes distributions.")
    return

  df = pd.read_csv(p)

  if {"x", "y"}.issubset(df.columns):
    plt.figure()
    plt.scatter(df["x"], df["y"], s=2)
    plt.title("PoCA scatter")
    plt.savefig(resolve_out("recon_poca.png"), dpi=150)
    plt.close()

  print("[OK] plotted PoCA")

if __name__ == "__main__":
  main()
