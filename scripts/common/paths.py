# ファイル名: scripts/common/paths.py
from pathlib import Path

def get_project_root() -> Path:
    """プロジェクトのルートディレクトリ(buildがある場所の親)を探す"""
    here = Path(__file__).resolve().parent
    for d in [here] + list(here.parents):
        if (d / "build").exists(): return d
    return Path.cwd()

def resolve_out(name: str) -> Path:
    """build/outputs/ 以下のパスを返す"""
    return get_project_root() / "build" / "outputs" / name

def config_path(name: str) -> Path:
    """configs/ 以下のパスを返す"""
    return get_project_root() / "configs" / name