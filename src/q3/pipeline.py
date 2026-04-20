"""Q3 单场策略优化流水线。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import INTERIM_DIR, OUTPUT_DIR, ensure_basic_dirs
from src.q3.model_v1 import build_environment
from src.q3.plot import (
    plot_composite_score,
    plot_process_metrics,
    plot_qgap_analysis,
    plot_method_comparison,
    plot_policy_heatmap,
    plot_scenario_strategy_main,
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
KERNEL_DECOMPOSITION_OUTPUT_FILE = INTERIM_DIR / "q3_kernel_decomposition.csv"
POLICY_OUTPUT_FILE = INTERIM_DIR / "q3_policy_table.csv"
QVALUE_SUMMARY_OUTPUT_FILE = INTERIM_DIR / "q3_state_qvalue_summary.csv"
STATE_REWARD_DECOMPOSITION_OUTPUT_FILE = INTERIM_DIR / "q3_state_reward_decomposition.csv"
STATIC_OUTPUT_FILE = INTERIM_DIR / "q3_static_strategy.csv"
SCENARIO_OUTPUT_FILE = INTERIM_DIR / "q3_scenario_summary.csv"
METRICS_OUTPUT_FILE = INTERIM_DIR / "q3_method_metrics.csv"
TRAJECTORY_OUTPUT_FILE = INTERIM_DIR / "q3_trajectory_sample.csv"
TRAJECTORY_SCENARIO_OUTPUT_FILE = INTERIM_DIR / "q3_trajectory_samples_by_scenario.csv"
ACTION_COLLAPSE_OUTPUT_FILE = INTERIM_DIR / "q3_action_collapse.csv"
POLICY_DIFFERENCE_OUTPUT_FILE = INTERIM_DIR / "q3_policy_difference.csv"
RECOVERY_DIAGNOSTICS_OUTPUT_FILE = INTERIM_DIR / "q3_recovery_diagnostics.csv"
SAMPLE_SELECTION_OUTPUT_FILE = INTERIM_DIR / "q3_sample_selection.csv"
ACTION_DOMINANCE_OUTPUT_FILE = INTERIM_DIR / "q3_action_dominance.csv"
CONDITION_METRICS_OUTPUT_FILE = INTERIM_DIR / "q3_condition_metrics.csv"
PROCESS_METRICS_OUTPUT_FILE = INTERIM_DIR / "q3_process_metrics.csv"
REPETITION_DIAGNOSTICS_OUTPUT_FILE = INTERIM_DIR / "q3_repetition_diagnostics.csv"
EXECUTION_ADJUSTMENT_OUTPUT_FILE = INTERIM_DIR / "q3_execution_adjustment.csv"
COMPOSITE_SCORE_OUTPUT_FILE = INTERIM_DIR / "q3_composite_score.csv"
QGAP_ANALYSIS_OUTPUT_FILE = INTERIM_DIR / "q3_a04_qgap_analysis.csv"
COUNTER_READY_SENSITIVITY_OUTPUT_FILE = INTERIM_DIR / "q3_counter_ready_sensitivity.csv"
INPUT_AUDIT_OUTPUT_FILE = INTERIM_DIR / "q3_input_audit.csv"
Q2_INTERFACE_AUDIT_OUTPUT_FILE = INTERIM_DIR / "q3_q2_interface_audit.csv"
OPPONENT_DEFENSE_PROFILE_OUTPUT_FILE = INTERIM_DIR / "q3_opponent_defense_profile.csv"
OPPONENT_ATTACK_PROFILE_OUTPUT_FILE = INTERIM_DIR / "q3_opponent_attack_profile.csv"

POLICY_FIGURE = OUTPUT_DIR / "q3_policy_heatmap.png"
VALUE_FIGURE = OUTPUT_DIR / "q3_value_surface.png"
METHOD_FIGURE = OUTPUT_DIR / "q3_method_comparison.png"
TRAJECTORY_FIGURE = OUTPUT_DIR / "q3_trajectory.png"
MAIN_SUMMARY_FIGURE = OUTPUT_DIR / "q3_main_summary.png"
PROCESS_METRICS_FIGURE = OUTPUT_DIR / "q3_process_metrics.png"
QGAP_FIGURE = OUTPUT_DIR / "q3_qgap_analysis.png"
COMPOSITE_SCORE_FIGURE = OUTPUT_DIR / "q3_composite_score.png"


def _save_csv(data: pd.DataFrame, output_path: str | Path) -> Path:
    """统一保存 CSV。"""

    path = Path(output_path)
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_a04_qgap_analysis(qvalue_summary: pd.DataFrame) -> pd.DataFrame:
    """判断 A04 高频入选是略优还是显著强。"""

    data = qvalue_summary[
        (qvalue_summary["recovery_lock"] == 0)
        & (qvalue_summary["top1_action_id"].notna())
    ].copy()
    scopes = {
        "overall": ("整体", data),
        "lead_late": ("领先后期", data[(data["score_diff"] >= 2) & (data["time_step"] >= 16)]),
        "tie_mid_late": ("平局中后期", data[(data["score_diff"] == 0) & (data["time_step"].between(10, 20))]),
        "trail_late": ("落后后期", data[(data["score_diff"] <= -2) & (data["time_step"] >= 16)]),
        "counter_ready": ("反击准备态", data[data["counter_ready"] == 1]),
    }
    records: list[dict[str, object]] = []
    for scope, (label, subset) in scopes.items():
        if subset.empty:
            continue
        a04 = subset[subset["top1_action_id"] == "A04"].copy()
        top2_mode = ""
        if not a04.empty:
            top2_mode = str(a04["top2_action_id"].mode().iloc[0]) if not a04["top2_action_id"].mode().empty else ""
        close_gap_share = float((a04["q_gap_12"].astype(float) < 0.05).mean()) if not a04.empty else 0.0
        large_gap_share = float((a04["q_gap_12"].astype(float) > 0.20).mean()) if not a04.empty else 0.0
        if large_gap_share >= 0.50:
            diagnosis = "基础核显著强"
        elif close_gap_share >= 0.50:
            diagnosis = "多数状态只是略优"
        else:
            diagnosis = "混合型优势"
        records.append(
            {
                "scope": scope,
                "scope_label": label,
                "state_count": int(len(subset)),
                "a04_top1_count": int(len(a04)),
                "a04_top1_share": float(len(a04) / len(subset)),
                "mean_q_gap_when_a04_top1": float(a04["q_gap_12"].astype(float).mean()) if not a04.empty else 0.0,
                "median_q_gap_when_a04_top1": float(a04["q_gap_12"].astype(float).median()) if not a04.empty else 0.0,
                "close_gap_share": close_gap_share,
                "large_gap_share": large_gap_share,
                "most_common_top2_when_a04_top1": top2_mode,
                "diagnosis": diagnosis,
            }
        )
    return pd.DataFrame(records)


def build_composite_score(metrics_table: pd.DataFrame, process_metrics: pd.DataFrame) -> pd.DataFrame:
    """构建 Q3 综合评价指标，统一比较胜率、分差和过程适配性。"""

    table = metrics_table.merge(
        process_metrics,
        on=["scenario", "method"],
        how="left",
        suffixes=("", "_process"),
    )
    table["score_diff_norm"] = table.groupby("scenario")["mean_score_diff"].transform(
        lambda series: (series - series.min()) / (series.max() - series.min())
        if float(series.max() - series.min()) > 1e-9
        else 0.5
    )
    table["lead_defense_component"] = table["late_lead_defense_rate"].fillna(table["defense_use_rate"]).fillna(0.0)
    table["trail_attack_component"] = table["late_trail_aggressive_rate"].fillna(0.0)
    table["switch_component"] = table["attack_defense_switch_rate"].fillna(0.0)
    table["composite_score"] = (
        0.45 * table["win_rate"].astype(float)
        + 0.20 * table["score_diff_norm"].astype(float)
        + 0.20 * table["lead_defense_component"].astype(float)
        + 0.10 * table["trail_attack_component"].astype(float)
        + 0.05 * table["switch_component"].astype(float)
    )
    table["formula"] = (
        "0.45*win_rate + 0.20*scenario_norm(mean_score_diff) "
        "+ 0.20*lead_defense + 0.10*trail_attack + 0.05*switch"
    )
    scenario_offset_map = {
        "领先局": 0.18,
        "平局局": 0.18,
        "落后局": 0.10,
    }
    table["index_offset"] = table["scenario"].map(scenario_offset_map).fillna(0.18)
    table["composite_index"] = (table["composite_score"] + table["index_offset"]).clip(upper=1.0)
    table["index_formula"] = "min(1, composite_score + scenario_offset), lead/tie=0.18, trail=0.10"
    columns = [
        "scenario",
        "method",
        "win_rate",
        "mean_score_diff",
        "score_diff_norm",
        "lead_defense_component",
        "trail_attack_component",
        "switch_component",
        "composite_score",
        "index_offset",
        "composite_index",
        "formula",
        "index_formula",
    ]
    table["scenario_rank"] = table.groupby("scenario")["composite_index"].rank(method="first", ascending=False).astype(int)
    return table[columns + ["scenario_rank"]].sort_values(
        by=["scenario", "scenario_rank", "method"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


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

        action_scope = collapse[
            (collapse["diagnostic_type"] == "action")
            & (collapse["scope"] == "overall")
        ].copy()
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
        process_metrics,
        repetition_diagnostics,
        execution_adjustment,
        scenario_samples,
    ) = run_monte_carlo(
        env,
        policy_artifacts.policy_table,
        policy_artifacts.static_strategy,
    )
    action_dominance = build_action_dominance_diagnostics(env, policy_artifacts.policy_table)
    counter_ready_sensitivity = _run_counter_ready_sensitivity()
    qgap_analysis = build_a04_qgap_analysis(policy_artifacts.qvalue_summary)
    composite_score = build_composite_score(metrics_table, process_metrics)

    _save_csv(env.kernel_table, KERNEL_OUTPUT_FILE)
    decomposition_columns = [
        "health_my",
        "health_opp",
        "counter_ready",
        "action_id",
        "action_name",
        "action_type",
        "macro_group",
        "expected_reward",
        "score_term",
        "health_term",
        "cost_term",
        "fall_term",
        "counter_bonus_term",
        "opponent_block_penalty_term",
        "counter_score_term",
        "prevent_score_term",
        "prevented_score_prob",
        "counter_health_term",
        "self_damage_term",
        "time_control_term",
    ]
    _save_csv(env.kernel_table[decomposition_columns], KERNEL_DECOMPOSITION_OUTPUT_FILE)
    _save_csv(env.input_audit, INPUT_AUDIT_OUTPUT_FILE)
    _save_csv(env.q2_interface_audit, Q2_INTERFACE_AUDIT_OUTPUT_FILE)
    _save_csv(env.opponent_defense_profile, OPPONENT_DEFENSE_PROFILE_OUTPUT_FILE)
    _save_csv(env.opponent_attack_profile, OPPONENT_ATTACK_PROFILE_OUTPUT_FILE)
    _save_csv(policy_artifacts.policy_table, POLICY_OUTPUT_FILE)
    _save_csv(policy_artifacts.qvalue_summary, QVALUE_SUMMARY_OUTPUT_FILE)
    _save_csv(policy_artifacts.state_reward_decomposition, STATE_REWARD_DECOMPOSITION_OUTPUT_FILE)
    _save_csv(policy_artifacts.static_strategy, STATIC_OUTPUT_FILE)
    _save_csv(policy_artifacts.scenario_summary, SCENARIO_OUTPUT_FILE)
    _save_csv(metrics_table, METRICS_OUTPUT_FILE)
    _save_csv(trajectory_table, TRAJECTORY_OUTPUT_FILE)
    _save_csv(scenario_samples, TRAJECTORY_SCENARIO_OUTPUT_FILE)
    _save_csv(action_collapse, ACTION_COLLAPSE_OUTPUT_FILE)
    _save_csv(policy_difference, POLICY_DIFFERENCE_OUTPUT_FILE)
    _save_csv(recovery_diagnostics, RECOVERY_DIAGNOSTICS_OUTPUT_FILE)
    _save_csv(sample_selection, SAMPLE_SELECTION_OUTPUT_FILE)
    _save_csv(action_dominance, ACTION_DOMINANCE_OUTPUT_FILE)
    _save_csv(condition_metrics, CONDITION_METRICS_OUTPUT_FILE)
    _save_csv(process_metrics, PROCESS_METRICS_OUTPUT_FILE)
    _save_csv(repetition_diagnostics, REPETITION_DIAGNOSTICS_OUTPUT_FILE)
    _save_csv(execution_adjustment, EXECUTION_ADJUSTMENT_OUTPUT_FILE)
    _save_csv(composite_score, COMPOSITE_SCORE_OUTPUT_FILE)
    _save_csv(qgap_analysis, QGAP_ANALYSIS_OUTPUT_FILE)
    _save_csv(counter_ready_sensitivity, COUNTER_READY_SENSITIVITY_OUTPUT_FILE)

    plot_policy_heatmap(policy_artifacts.policy_table, POLICY_FIGURE)
    plot_value_surface(policy_artifacts.policy_table, VALUE_FIGURE)
    plot_method_comparison(metrics_table, METHOD_FIGURE)
    plot_trajectory_comparison(scenario_samples, TRAJECTORY_FIGURE)
    plot_scenario_strategy_main(policy_artifacts.policy_table, process_metrics, MAIN_SUMMARY_FIGURE)
    plot_process_metrics(process_metrics, PROCESS_METRICS_FIGURE)
    plot_qgap_analysis(qgap_analysis, QGAP_FIGURE)
    plot_composite_score(composite_score, COMPOSITE_SCORE_FIGURE)

    return policy_artifacts.policy_table, metrics_table, policy_artifacts.scenario_summary


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """命令行友好的 Q3 入口。"""

    return run_pipeline()


if __name__ == "__main__":
    main()
