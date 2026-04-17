"""Q1 正式运行入口。"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.q1.pipeline import main


if __name__ == "__main__":
    ranked, ahp_summary = main()
    print("Q1 方法四综合效用前 5 名动作：")
    display_columns = ["rank", "action_id", "action_name", "utility", "proposed_score"]
    print(ranked.loc[:, display_columns].head(5).to_string(index=False))
    print("\nAHP 一致性检验：")
    print(f"lambda_max = {ahp_summary['lambda_max']:.6f}")
    print(f"CI = {ahp_summary['ci']:.6f}")
    print(f"CR = {ahp_summary['cr']:.6f}")
    print("\nQ1 已输出：")
    print("- data/interim/action_features.csv")
    print("- data/output/q1_utility_bar.png")
    print("- data/output/q1_impact_balance.png")
    print("- data/output/q1_method_comparison.png")
    print("- data/output/q1_penalty_curve.png")
    print("- data/output/q1_decision_atlas.png")
    print("- data/output/q1_sensitivity_heatmap.png")
