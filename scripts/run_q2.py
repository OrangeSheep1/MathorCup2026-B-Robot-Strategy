"""运行 Q2 的正式脚本入口。"""

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.q2.pipeline import main


if __name__ == "__main__":
    evaluated_pairs, matchup_table, _ = main()
    print("Q2 已完成，以下展示前 5 个攻击动作的 Top1 防守建议：")
    preview = matchup_table.head(5)
    for _, row in preview.iterrows():
        print(
            f"{row['action_id']} -> {row['defense_id_r1']} | "
            f"block={row['block_prob_r1']:.3f}, "
            f"counter={row['counter_window_r1']:.3f}s, "
            f"score={row['defense_score_r1']:.3f}"
        )
