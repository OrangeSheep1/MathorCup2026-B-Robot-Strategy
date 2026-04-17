"""运行 Q1 的脚本入口。"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.q1.pipeline import main


if __name__ == "__main__":
    main()
