"""通用常量、路径与基础工具。"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = DATA_DIR / "output"


def ensure_basic_dirs() -> None:
    """确保基础数据目录存在。"""
    for path in (RAW_DIR, INTERIM_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)
