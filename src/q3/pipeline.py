"""Q3 单场策略优化流水线。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import INTERIM_DIR, OUTPUT_DIR, ensure_basic_dirs
from src.q3.model_v1 import build_environment
from src.q3.plot import (
    plot_method_comparison,
    plot_policy_heatmap,
    plot_trajectory_comparison,
    plot_value_surface,
)
from src.q3.policy import build_policy_table
from src.q3.simulate import run_monte_carlo


Q1_ACTION_FILE = INTERIM_DIR / "action_features.csv"
Q2_MATCHUP_FILE = INTERIM_DIR / "defense_matchup.csv"
Q2_DEFENSE_FEATURE_FILE = INTERIM_DIR / "defense_features.csv"
Q2_PAIR_FILE = INTERIM_DIR / "defense_pair_scores.csv"

KERNEL_OUTPUT_FILE = INTERIM_DIR / "q3_action_kernels.csv"
POLICY_OUTPUT_FILE = INTERIM_DIR / "q3_policy_table.csv"
STATIC_OUTPUT_FILE = INTERIM_DIR / "q3_static_strategy.csv"
SCENARIO_OUTPUT_FILE = INTERIM_DIR / "q3_scenario_summary.csv"
METRICS_OUTPUT_FILE = INTERIM_DIR / "q3_method_metrics.csv"
TRAJECTORY_OUTPUT_FILE = INTERIM_DIR / "q3_trajectory_sample.csv"

POLICY_FIGURE = OUTPUT_DIR / "q3_policy_heatmap.png"
VALUE_FIGURE = OUTPUT_DIR / "q3_value_surface.png"
METHOD_FIGURE = OUTPUT_DIR / "q3_method_comparison.png"
TRAJECTORY_FIGURE = OUTPUT_DIR / "q3_trajectory.png"


def _save_csv(data: pd.DataFrame, output_path: str | Path) -> Path:
    """统一保存 CSV。"""

    path = Path(output_path)
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """执行 Q3 全流程。"""

    ensure_basic_dirs()
    env = build_environment(
        action_feature_file=Q1_ACTION_FILE,
        defense_matchup_file=Q2_MATCHUP_FILE,
        defense_feature_file=Q2_DEFENSE_FEATURE_FILE,
        defense_pair_file=Q2_PAIR_FILE,
    )
    policy_artifacts = build_policy_table(env)
    metrics_table, trajectory_table = run_monte_carlo(
        env,
        policy_artifacts.policy_table,
        policy_artifacts.static_strategy,
    )

    _save_csv(env.kernel_table, KERNEL_OUTPUT_FILE)
    _save_csv(policy_artifacts.policy_table, POLICY_OUTPUT_FILE)
    _save_csv(policy_artifacts.static_strategy, STATIC_OUTPUT_FILE)
    _save_csv(policy_artifacts.scenario_summary, SCENARIO_OUTPUT_FILE)
    _save_csv(metrics_table, METRICS_OUTPUT_FILE)
    _save_csv(trajectory_table, TRAJECTORY_OUTPUT_FILE)

    plot_policy_heatmap(policy_artifacts.policy_table, POLICY_FIGURE)
    plot_value_surface(policy_artifacts.policy_table, VALUE_FIGURE)
    plot_method_comparison(metrics_table, METHOD_FIGURE)
    plot_trajectory_comparison(trajectory_table, TRAJECTORY_FIGURE)

    return policy_artifacts.policy_table, metrics_table, policy_artifacts.scenario_summary


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """命令行友好的 Q3 入口。"""

    return run_pipeline()


if __name__ == "__main__":
    main()
