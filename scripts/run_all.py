"""顺序运行 Q1 到 Q4 的脚本入口。"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.q1.pipeline import main as run_q1
from src.q2.pipeline import main as run_q2
from src.q3.pipeline import main as run_q3
from src.q4.pipeline import main as run_q4


def main() -> None:
    """顺序执行当前阶段的各题流水线。"""
    run_q1()
    run_q2()
    run_q3()
    run_q4()


if __name__ == "__main__":
    main()
