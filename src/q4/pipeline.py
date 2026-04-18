"""Q4 双层动态规划流水线。"""

from __future__ import annotations

from pathlib import Path
import json
import pickle

import pandas as pd

from src.common import INTERIM_DIR, OUTPUT_DIR, RAW_DIR, ensure_basic_dirs
from src.q4.decision import (
    build_exhaustive_plan,
    build_pwin_table,
    save_micro_solutions,
    solve_macro_dp,
)
from src.q4.model_v1 import build_context, ensure_fault_param_file
from src.q4.plot import (
    plot_fault_curve,
    plot_method_boxplot,
    plot_policy_tree,
    plot_pwin_heatmaps,
    plot_scenario_radar,
    plot_tornado,
)
from src.q4.simulate import run_bo3_monte_carlo


Q1_ACTION_FILE = INTERIM_DIR / "action_features.csv"
Q2_PAIR_FILE = INTERIM_DIR / "defense_pair_scores.csv"
Q3_KERNEL_FILE = INTERIM_DIR / "q3_action_kernels.csv"
Q3_METRIC_FILE = INTERIM_DIR / "q3_method_metrics.csv"
Q4_FAULT_PARAM_FILE = RAW_DIR / "q4_fault_params.json"

BASE_WIN_OUTPUT_FILE = INTERIM_DIR / "q4_base_win_prob.json"
PWIN_OUTPUT_FILE = INTERIM_DIR / "q4_pwin_table.csv"
USAGE_DISTRIBUTION_FILE = INTERIM_DIR / "q4_usage_distribution.csv"
MICRO_POLICY_OUTPUT_FILE = INTERIM_DIR / "q4_micro_policy.csv"
MICRO_SOLUTION_OUTPUT_FILE = INTERIM_DIR / "q4_pwin_table.pkl"
MACRO_VALUE_OUTPUT_FILE = INTERIM_DIR / "q4_macro_value.csv"
MACRO_POLICY_OUTPUT_FILE = INTERIM_DIR / "q4_macro_policy.csv"
EXHAUSTIVE_OUTPUT_FILE = INTERIM_DIR / "q4_exhaustive_plan.csv"
METHOD_OUTPUT_FILE = INTERIM_DIR / "q4_method_summary.csv"
BATCH_OUTPUT_FILE = INTERIM_DIR / "q4_batch_distribution.csv"
RESOURCE_OUTPUT_FILE = INTERIM_DIR / "q4_resource_usage.csv"
FAULT_PROFILE_OUTPUT_FILE = INTERIM_DIR / "q4_fault_profile.csv"
SENSITIVITY_OUTPUT_FILE = INTERIM_DIR / "q4_sensitivity.csv"

TREE_FIGURE = OUTPUT_DIR / "q4_policy_tree.png"
PWIN_FIGURE = OUTPUT_DIR / "q4_pwin_heatmap.png"
FAULT_FIGURE = OUTPUT_DIR / "q4_fault_curve.png"
RADAR_FIGURE = OUTPUT_DIR / "q4_scenario_radar.png"
METHOD_FIGURE = OUTPUT_DIR / "q4_method_boxplot.png"
TORNADO_FIGURE = OUTPUT_DIR / "q4_tornado.png"


def _save_csv(data: pd.DataFrame, output_path: str | Path) -> Path:
    """统一保存 CSV。"""

    path = Path(output_path)
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _save_json(payload: dict[str, float], output_path: str | Path) -> Path:
    """统一保存 JSON。"""

    path = Path(output_path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path


def _build_sensitivity_table(base_context, baseline_value: float) -> pd.DataFrame:
    """对关键参数做 ±20% 单因素分析。"""

    parameters = [
        ("lambda_0", "基础故障率"),
        ("k_fault", "故障敏感系数"),
        ("delta_H_pause", "暂停恢复量"),
        ("base_win_scale", "基础单局胜率"),
    ]
    records: list[dict[str, object]] = []

    for parameter_key, parameter_label in parameters:
        base_value = getattr(base_context.config, parameter_key if parameter_key != "delta_H_pause" else "delta_h_pause")
        scenario_values: dict[str, float] = {}
        for suffix, factor in [("low", 0.8), ("high", 1.2)]:
            override_key = parameter_key if parameter_key != "delta_H_pause" else "delta_H_pause"
            context = build_context(
                action_feature_file=Q1_ACTION_FILE,
                defense_pair_file=Q2_PAIR_FILE,
                q3_kernel_file=Q3_KERNEL_FILE,
                q3_metric_file=Q3_METRIC_FILE,
                fault_param_file=Q4_FAULT_PARAM_FILE,
                config_overrides={override_key: float(base_value) * factor},
            )
            pwin_table, usage_distribution, _ = build_pwin_table(context)
            macro_solution = solve_macro_dp(context, usage_distribution)
            scenario_values[suffix] = macro_solution.best_initial_win_prob
        records.append(
            {
                "parameter_key": parameter_key,
                "parameter_label": parameter_label,
                "base_value": float(base_value),
                "win_prob_low": scenario_values["low"],
                "win_prob_high": scenario_values["high"],
                "delta_low": scenario_values["low"] - baseline_value,
                "delta_high": scenario_values["high"] - baseline_value,
                "impact_abs": max(abs(scenario_values["low"] - baseline_value), abs(scenario_values["high"] - baseline_value)),
            }
        )
    return pd.DataFrame(records).sort_values(by="impact_abs", ascending=False).reset_index(drop=True)


def run_pipeline() -> dict[str, pd.DataFrame | float]:
    """执行 Q4 全流程。"""

    ensure_basic_dirs()
    ensure_fault_param_file(Q4_FAULT_PARAM_FILE)

    context = build_context(
        action_feature_file=Q1_ACTION_FILE,
        defense_pair_file=Q2_PAIR_FILE,
        q3_kernel_file=Q3_KERNEL_FILE,
        q3_metric_file=Q3_METRIC_FILE,
        fault_param_file=Q4_FAULT_PARAM_FILE,
    )
    _save_json(context.base_win_prob, BASE_WIN_OUTPUT_FILE)

    pwin_table, usage_distribution, micro_solutions = build_pwin_table(context)
    macro_solution = solve_macro_dp(context, usage_distribution)
    exhaustive_table, exhaustive_plan = build_exhaustive_plan(context, pwin_table)
    method_summary, batch_distribution, resource_usage, fault_profile = run_bo3_monte_carlo(
        context=context,
        micro_solutions=micro_solutions,
        macro_policy_table=macro_solution.policy_table,
        macro_value_table=macro_solution.value_table,
        pwin_table=pwin_table,
        exhaustive_plan=exhaustive_plan,
        num_series=5000,
    )
    sensitivity_table = _build_sensitivity_table(context, macro_solution.best_initial_win_prob)

    _save_csv(pwin_table, PWIN_OUTPUT_FILE)
    _save_csv(usage_distribution, USAGE_DISTRIBUTION_FILE)
    save_micro_solutions(micro_solutions, MICRO_SOLUTION_OUTPUT_FILE)
    if (context.config.max_reset, context.config.max_pause, context.config.max_repair) in micro_solutions:
        _save_csv(
            micro_solutions[(context.config.max_reset, context.config.max_pause, context.config.max_repair)].policy_frame,
            MICRO_POLICY_OUTPUT_FILE,
        )
    _save_csv(macro_solution.value_table, MACRO_VALUE_OUTPUT_FILE)
    _save_csv(macro_solution.policy_table, MACRO_POLICY_OUTPUT_FILE)
    _save_csv(exhaustive_table, EXHAUSTIVE_OUTPUT_FILE)
    _save_csv(method_summary, METHOD_OUTPUT_FILE)
    _save_csv(batch_distribution, BATCH_OUTPUT_FILE)
    _save_csv(resource_usage, RESOURCE_OUTPUT_FILE)
    _save_csv(fault_profile, FAULT_PROFILE_OUTPUT_FILE)
    _save_csv(sensitivity_table, SENSITIVITY_OUTPUT_FILE)

    plot_policy_tree(macro_solution.policy_table, TREE_FIGURE)
    plot_pwin_heatmaps(pwin_table, PWIN_FIGURE)
    plot_fault_curve(fault_profile, FAULT_FIGURE)
    plot_scenario_radar(resource_usage, RADAR_FIGURE)
    plot_method_boxplot(batch_distribution, method_summary, METHOD_FIGURE)
    plot_tornado(sensitivity_table, TORNADO_FIGURE)

    return {
        "pwin_table": pwin_table,
        "macro_value_table": macro_solution.value_table,
        "macro_policy_table": macro_solution.policy_table,
        "method_summary": method_summary,
        "resource_usage": resource_usage,
        "fault_profile": fault_profile,
        "sensitivity_table": sensitivity_table,
        "best_initial_win_prob": macro_solution.best_initial_win_prob,
    }


def main() -> dict[str, pd.DataFrame | float]:
    """命令行友好的 Q4 入口。"""

    return run_pipeline()


if __name__ == "__main__":
    main()
