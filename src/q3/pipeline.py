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
from src.q3.simulate import (
    build_action_collapse_diagnostics,
    build_action_dominance_diagnostics,
    build_policy_difference_diagnostics,
    run_monte_carlo,
)


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
ACTION_COLLAPSE_OUTPUT_FILE = INTERIM_DIR / "q3_action_collapse.csv"
POLICY_DIFFERENCE_OUTPUT_FILE = INTERIM_DIR / "q3_policy_difference.csv"
RECOVERY_DIAGNOSTICS_OUTPUT_FILE = INTERIM_DIR / "q3_recovery_diagnostics.csv"
SAMPLE_SELECTION_OUTPUT_FILE = INTERIM_DIR / "q3_sample_selection.csv"
ACTION_DOMINANCE_OUTPUT_FILE = INTERIM_DIR / "q3_action_dominance.csv"
CONDITION_METRICS_OUTPUT_FILE = INTERIM_DIR / "q3_condition_metrics.csv"
COUNTER_READY_SENSITIVITY_OUTPUT_FILE = INTERIM_DIR / "q3_counter_ready_sensitivity.csv"

POLICY_FIGURE = OUTPUT_DIR / "q3_policy_heatmap.png"
VALUE_FIGURE = OUTPUT_DIR / "q3_value_surface.png"
METHOD_FIGURE = OUTPUT_DIR / "q3_method_comparison.png"
TRAJECTORY_FIGURE = OUTPUT_DIR / "q3_trajectory.png"


def _save_csv(data: pd.DataFrame, output_path: str | Path) -> Path:
    """统一保存 CSV。"""

    path = Path(output_path)
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _run_counter_ready_sensitivity() -> pd.DataFrame:
    """扫描 counter_ready 关键参数，检查策略结构是否稳健。"""

    settings = [
        {"counter_ready_trigger_prob": 0.15, "counter_ready_bonus_min": 0.08, "counter_ready_bonus_max": 0.16},
        {"counter_ready_trigger_prob": 0.12, "counter_ready_bonus_min": 0.06, "counter_ready_bonus_max": 0.12},
        {"counter_ready_trigger_prob": 0.12, "counter_ready_bonus_min": 0.10, "counter_ready_bonus_max": 0.20},
        {"counter_ready_trigger_prob": 0.18, "counter_ready_bonus_min": 0.06, "counter_ready_bonus_max": 0.12},
        {"counter_ready_trigger_prob": 0.18, "counter_ready_bonus_min": 0.10, "counter_ready_bonus_max": 0.20},
    ]
    records: list[dict[str, object]] = []

    for overrides in settings:
        env = build_environment(
            action_feature_file=Q1_ACTION_FILE,
            defense_matchup_file=Q2_MATCHUP_FILE,
            defense_feature_file=Q2_DEFENSE_FEATURE_FILE,
            defense_pair_file=Q2_PAIR_FILE,
            config_overrides=overrides,
        )
        policy_artifacts = build_policy_table(env)
        collapse = build_action_collapse_diagnostics(policy_artifacts.policy_table)
        difference = build_policy_difference_diagnostics(policy_artifacts.policy_table)
        dominance = build_action_dominance_diagnostics(env, policy_artifacts.policy_table)

        action_scope = collapse[collapse["diagnostic_type"] == "action"].copy()
        overall_diff = difference[difference["dimension"] == "overall"].iloc[0]
        trailing = dominance[dominance["scope"] == "trailing_high_high"].copy()
        ready_scope = dominance[dominance["scope"] == "counter_ready_high_high"].copy()

        def _share(frame: pd.DataFrame, action_id: str, column: str = "mdp_selected_share") -> float:
            local = frame[frame["action_id"] == action_id]
            return float(local.iloc[0][column]) if not local.empty else 0.0

        top_actions = action_scope.sort_values(by=["share", "mdp_action_id"], ascending=[False, True]).reset_index(drop=True)
        records.append(
            {
                "trigger_prob": float(overrides["counter_ready_trigger_prob"]),
                "bonus_min": float(overrides["counter_ready_bonus_min"]),
                "bonus_max": float(overrides["counter_ready_bonus_max"]),
                "top1_action_id": str(top_actions.iloc[0]["mdp_action_id"]),
                "top1_action_share": float(top_actions.iloc[0]["share"]),
                "top2_action_id": str(top_actions.iloc[1]["mdp_action_id"]) if len(top_actions) > 1 else "",
                "top2_action_share": float(top_actions.iloc[1]["share"]) if len(top_actions) > 1 else 0.0,
                "top3_action_id": str(top_actions.iloc[2]["mdp_action_id"]) if len(top_actions) > 2 else "",
                "top3_action_share": float(top_actions.iloc[2]["share"]) if len(top_actions) > 2 else 0.0,
                "mdp_greedy_diff_share": float(overall_diff["different_share"]),
                "trailing_A01_share": _share(trailing, "A01"),
                "trailing_A09_share": _share(trailing, "A09"),
                "trailing_D08_share": _share(trailing, "D08"),
                "counter_ready_A01_share": _share(ready_scope, "A01"),
                "counter_ready_A09_share": _share(ready_scope, "A09"),
                "counter_ready_D08_share": _share(ready_scope, "D08"),
            }
        )

    return pd.DataFrame(records)


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
    (
        metrics_table,
        trajectory_table,
        action_collapse,
        policy_difference,
        recovery_diagnostics,
        sample_selection,
        condition_metrics,
    ) = run_monte_carlo(
        env,
        policy_artifacts.policy_table,
        policy_artifacts.static_strategy,
    )
    action_dominance = build_action_dominance_diagnostics(env, policy_artifacts.policy_table)
    counter_ready_sensitivity = _run_counter_ready_sensitivity()

    _save_csv(env.kernel_table, KERNEL_OUTPUT_FILE)
    _save_csv(policy_artifacts.policy_table, POLICY_OUTPUT_FILE)
    _save_csv(policy_artifacts.static_strategy, STATIC_OUTPUT_FILE)
    _save_csv(policy_artifacts.scenario_summary, SCENARIO_OUTPUT_FILE)
    _save_csv(metrics_table, METRICS_OUTPUT_FILE)
    _save_csv(trajectory_table, TRAJECTORY_OUTPUT_FILE)
    _save_csv(action_collapse, ACTION_COLLAPSE_OUTPUT_FILE)
    _save_csv(policy_difference, POLICY_DIFFERENCE_OUTPUT_FILE)
    _save_csv(recovery_diagnostics, RECOVERY_DIAGNOSTICS_OUTPUT_FILE)
    _save_csv(sample_selection, SAMPLE_SELECTION_OUTPUT_FILE)
    _save_csv(action_dominance, ACTION_DOMINANCE_OUTPUT_FILE)
    _save_csv(condition_metrics, CONDITION_METRICS_OUTPUT_FILE)
    _save_csv(counter_ready_sensitivity, COUNTER_READY_SENSITIVITY_OUTPUT_FILE)

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
