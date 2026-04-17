"""数据读写占位模块。"""

from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    """统一路径输入。"""
    return Path(path)
