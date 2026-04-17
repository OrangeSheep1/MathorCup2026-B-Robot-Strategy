"""Q1 全流程流水线。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import INTERIM_DIR, OUTPUT_DIR, RAW_DIR, ensure_basic_dirs
from src.q1.evaluate import evaluate_all_methods, sensitivity_scan
from src.q1.model_v1 import build_feature_table, load_attack_actions, load_robot_params
from src.q1.plot import (
    plot_decision_atlas,
    plot_impact_balance_scatter,
    plot_method_comparison,
    plot_penalty_curve,
    plot_sensitivity_heatmap,
    plot_utility_bar,
)


ACTION_RAW_FILE = RAW_DIR / "q1_attack_actions.csv"
ROBOT_RAW_FILE = RAW_DIR / "q1_robot_params.csv"
INTERIM_FILE = INTERIM_DIR / "action_features.csv"
UTILITY_FIGURE = OUTPUT_DIR / "q1_utility_bar.png"
TRADEOFF_FIGURE = OUTPUT_DIR / "q1_impact_balance.png"
METHOD_COMPARE_FIGURE = OUTPUT_DIR / "q1_method_comparison.png"
PENALTY_FIGURE = OUTPUT_DIR / "q1_penalty_curve.png"
SENSITIVITY_FIGURE = OUTPUT_DIR / "q1_sensitivity_heatmap.png"
DECISION_ATLAS_FIGURE = OUTPUT_DIR / "q1_decision_atlas.png"


def save_action_features(data: pd.DataFrame, output_path: str | Path = INTERIM_FILE) -> Path:
    """保存 Q1 中间结果表。"""

    stable_columns = [
        "action_id",
        "action_name",
        "category",
        "impact_score",
        "balance_cost",
        "score_prob",
        "energy_cost",
        "utility",
        "rank",
        "method1_score",
        "method1_rank",
        "method2_score",
        "method2_rank",
        "method3_score",
        "method3_rank",
        "proposed_score",
        "u0",
        "fall_penalty_raw",
        "fall_penalty",
        "tau_norm",
        "delta_com_norm",
        "energy_norm",
        "time_norm",
        "stability_score",
        "efficiency_score",
        "exec_time",
        "p_reach",
        "p_target",
        "stable_margin",
        "stability_ratio",
        "eta",
        "joint_count",
        "theta_total_deg",
        "benefit_term",
        "cost_term",
        "omega_out",
        "omega_eff",
        "com_height",
    ]
    output = Path(output_path)
    data.loc[:, stable_columns].to_csv(output, index=False, encoding="utf-8-sig")
    return output


def run_pipeline() -> tuple[pd.DataFrame, dict[str, float]]:
    """执行 Q1 全流程。"""

    ensure_basic_dirs()
    robot = load_robot_params(ROBOT_RAW_FILE)
    raw_actions = load_attack_actions(ACTION_RAW_FILE)
    feature_table = build_feature_table(raw_actions, robot)
    evaluated, ahp_summary = evaluate_all_methods(
        data=feature_table,
        stable_margin=robot.stable_margin_m,
    )
    save_action_features(evaluated)

    sensitivity = sensitivity_scan(
        data=evaluated,
        stable_margin=robot.stable_margin_m,
    )
    plot_utility_bar(evaluated, UTILITY_FIGURE)
    plot_impact_balance_scatter(evaluated, TRADEOFF_FIGURE)
    plot_method_comparison(evaluated, METHOD_COMPARE_FIGURE)
    plot_penalty_curve(evaluated, PENALTY_FIGURE)
    plot_decision_atlas(evaluated, DECISION_ATLAS_FIGURE)
    plot_sensitivity_heatmap(sensitivity, SENSITIVITY_FIGURE)
    return evaluated, ahp_summary


def main() -> tuple[pd.DataFrame, dict[str, float]]:
    """命令行友好的 Q1 入口。"""

    return run_pipeline()


if __name__ == "__main__":
    main()
