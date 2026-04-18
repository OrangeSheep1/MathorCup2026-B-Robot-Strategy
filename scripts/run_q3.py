"""运行 Q3 的正式脚本入口。"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.q3.pipeline import main


if __name__ == "__main__":
    policy_table, metrics_table, scenario_summary = main()
    print("Q3 已完成，以下展示三类典型场景下的 MDP 推荐动作：")
    for scenario_name in ["领先局", "平局局", "落后局"]:
        subset = scenario_summary[scenario_summary["scenario"] == scenario_name].head(3)
        if subset.empty:
            continue
        summary_text = " | ".join(
            f"{row['action_id']}({row['share']:.2f})"
            for _, row in subset.iterrows()
        )
        print(f"{scenario_name}: {summary_text}")

    print("Q3 蒙特卡洛胜率摘要：")
    preview = metrics_table[metrics_table["method"] == "mdp"].copy()
    for _, row in preview.iterrows():
        print(
            f"{row['scenario']} -> "
            f"win_rate={row['win_rate']:.3f}, "
            f"score_diff={row['mean_score_diff']:.3f}, "
            f"health_my={row['mean_health_my']:.3f}"
        )
