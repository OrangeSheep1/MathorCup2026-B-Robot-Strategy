"""Q4 双层动态规划流水线。"""

from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from src.common import INTERIM_DIR, OUTPUT_DIR, RAW_DIR, ensure_basic_dirs
from src.q4.decision import (
    build_exhaustive_plan,
    build_pwin_table,
    save_micro_solutions,
    solve_macro_dp,
)
from src.q4.model_v1 import build_context, ensure_fault_param_file
from src.q4.model_v1 import (
    HEALTH_LEVELS,
    SCORE_DIFF_VALUES,
    RoundState,
    build_action_transitions,
    get_feasible_actions,
)
from src.q4.plot import (
    plot_composite_score,
    plot_fault_curve,
    plot_main_summary,
    plot_method_boxplot,
    plot_policy_tree,
    plot_pwin_heatmaps,
    plot_resource_gain,
    plot_resource_policy_heatmap,
    plot_resource_timing,
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
INPUT_AUDIT_OUTPUT_FILE = INTERIM_DIR / "q4_input_audit.csv"
INTERFACE_CONTRACT_OUTPUT_FILE = INTERIM_DIR / "q4_interface_contract.csv"
ZERO_BASELINE_OUTPUT_FILE = INTERIM_DIR / "q4_zero_resource_baseline.csv"
RESOURCE_UPLIFT_OUTPUT_FILE = INTERIM_DIR / "q4_resource_uplift_diagnostics.csv"
RESOURCE_TRIGGER_OUTPUT_FILE = INTERIM_DIR / "q4_resource_trigger_states.csv"
RESOURCE_TIMING_RULES_OUTPUT_FILE = INTERIM_DIR / "q4_resource_timing_rules.csv"
ALLOC_ACTUAL_OUTPUT_FILE = INTERIM_DIR / "q4_alloc_vs_actual_usage.csv"
ROUND_METHOD_OUTPUT_FILE = INTERIM_DIR / "q4_round_method_metrics.csv"
FIRST_USE_OUTPUT_FILE = INTERIM_DIR / "q4_first_use_distribution.csv"
COMPOSITE_SCORE_OUTPUT_FILE = INTERIM_DIR / "q4_composite_score.csv"
RESOURCE_VALUE_GAP_OUTPUT_FILE = INTERIM_DIR / "q4_resource_value_gap.csv"
RESOURCE_VALUE_GAP_SUMMARY_OUTPUT_FILE = INTERIM_DIR / "q4_resource_value_gap_summary.csv"
RESOURCE_POLICY_HEATMAP_OUTPUT_FILE = INTERIM_DIR / "q4_resource_policy_heatmap_table.csv"

TREE_FIGURE = OUTPUT_DIR / "q4_policy_tree.png"
PWIN_FIGURE = OUTPUT_DIR / "q4_pwin_heatmap.png"
FAULT_FIGURE = OUTPUT_DIR / "q4_fault_curve.png"
RADAR_FIGURE = OUTPUT_DIR / "q4_scenario_radar.png"
METHOD_FIGURE = OUTPUT_DIR / "q4_method_boxplot.png"
TORNADO_FIGURE = OUTPUT_DIR / "q4_tornado.png"
RESOURCE_GAIN_FIGURE = OUTPUT_DIR / "q4_resource_gain.png"
RESOURCE_TIMING_FIGURE = OUTPUT_DIR / "q4_resource_timing.png"
COMPOSITE_SCORE_FIGURE = OUTPUT_DIR / "q4_composite_score.png"
MAIN_SUMMARY_FIGURE = OUTPUT_DIR / "q4_main_summary.png"
RESOURCE_POLICY_HEATMAP_FIGURE = OUTPUT_DIR / "q4_resource_policy_heatmap.png"


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


def _build_input_audit(context) -> pd.DataFrame:
    """记录 Q4 输入文件版本和最终 Q3 基线。"""

    files = [
        ("q1_action", Q1_ACTION_FILE),
        ("q2_pair", Q2_PAIR_FILE),
        ("q3_kernel", Q3_KERNEL_FILE),
        ("q3_metric", Q3_METRIC_FILE),
        ("q4_fault_param", Q4_FAULT_PARAM_FILE),
    ]
    records: list[dict[str, object]] = []
    for label, path in files:
        stat = Path(path).stat()
        records.append(
            {
                "item": label,
                "path": str(path),
                "exists": Path(path).exists(),
                "modified_time": pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat(),
                "size_bytes": stat.st_size,
                "row_count": len(pd.read_csv(path)) if str(path).endswith(".csv") else None,
            }
        )
    for scenario_key, value in context.base_win_prob.items():
        records.append(
            {
                "item": f"q3_base_win_{scenario_key}",
                "path": str(Q3_METRIC_FILE),
                "exists": True,
                "modified_time": "",
                "size_bytes": None,
                "row_count": None,
                "value": value,
            }
        )
    return pd.DataFrame(records)


def _build_interface_contract() -> pd.DataFrame:
    """固定 Q4 依赖的输入字段口径。"""

    rows = [
        ("Q1", "action_features.csv", "action_id/action_name/score_prob/energy_cost/utility", "动作核与战术压缩"),
        ("Q2", "defense_pair_scores.csv", "action_id/defense_id/p_block/counter_prob/p_fall", "接口校验和攻防语义追溯"),
        ("Q3", "q3_action_kernels.csv", "macro_group/p_score_for/p_score_against/p_self_drop/p_opp_drop/p_fall", "Q4 常规战术转移核"),
        ("Q3", "q3_method_metrics.csv", "scenario/method/win_rate", "三场景无资源基础胜率校准"),
        ("Q4", "q4_fault_params.json", "lambda_0/k_fault/resource/time parameters", "故障与资源动作参数"),
    ]
    return pd.DataFrame(rows, columns=["source", "file", "required_fields", "usage"])


def _build_zero_resource_baseline(context, pwin_table: pd.DataFrame) -> pd.DataFrame:
    """输出 Q4 零资源基线和最终 Q3 基线的差距。"""

    zero = pwin_table[
        (pwin_table["reset_alloc"] == 0)
        & (pwin_table["pause_alloc"] == 0)
        & (pwin_table["repair_alloc"] == 0)
    ].copy()
    zero["q3_base_win"] = zero["scenario_key"].map(context.base_win_prob).astype(float)
    zero["gap"] = zero["p_win"] - zero["q3_base_win"]
    return zero[
        ["scenario_key", "scenario", "allocation_label", "q3_base_win", "p_win", "raw_p_win", "gap"]
    ].rename(columns={"p_win": "q4_zero_resource_pwin", "raw_p_win": "raw_zero_resource_pwin"})


def _build_resource_uplift_diagnostics(pwin_table: pd.DataFrame) -> pd.DataFrame:
    """统计每个配额相对零资源的单局增益。"""

    zero = pwin_table[
        (pwin_table["reset_alloc"] == 0)
        & (pwin_table["pause_alloc"] == 0)
        & (pwin_table["repair_alloc"] == 0)
    ][["scenario_key", "p_win"]].rename(columns={"p_win": "zero_resource_pwin"})
    table = pwin_table.merge(zero, on="scenario_key", how="left")
    table["uplift"] = table["p_win"] - table["zero_resource_pwin"]
    return table[
        [
            "scenario_key",
            "scenario",
            "allocation_label",
            "zero_resource_pwin",
            "p_win",
            "raw_p_win",
            "q3_base_win",
            "uplift",
            "calibration_gap_vs_q3",
        ]
    ].sort_values(["scenario_key", "uplift"], ascending=[True, False])


def _score_band(score_diff: int) -> str:
    if score_diff >= 2:
        return "大幅领先"
    if score_diff == 1:
        return "小幅领先"
    if score_diff == 0:
        return "平局"
    if score_diff == -1:
        return "小幅落后"
    return "大幅落后"


def _time_band(time_bucket: int, n_time_buckets: int) -> str:
    ratio = time_bucket / max(n_time_buckets, 1)
    if ratio < 0.35:
        return "前期"
    if ratio < 0.70:
        return "中期"
    return "后期"


def _trigger_reason(row: pd.Series) -> str:
    action = str(row["action_id"])
    if action == "USE_REPAIR":
        return "故障状态下维修显著优于继续等待"
    if action == "USE_RESET":
        return "倒地/失稳状态下复位可恢复行动能力"
    if action == "USE_PAUSE":
        if int(row["health_my"]) == 0:
            return "低机能状态下暂停用于降低后续故障和失分风险"
        return "资源充足时暂停用于重整节奏和保持状态"
    return "非资源动作"


def _build_resource_trigger_states(context, policy_frame: pd.DataFrame) -> pd.DataFrame:
    """从局内最优策略中抽出资源触发状态。"""

    if policy_frame.empty:
        return pd.DataFrame()
    trigger = policy_frame[policy_frame["action_id"].isin(["USE_RESET", "USE_PAUSE", "USE_REPAIR"])].copy()
    trigger["score_diff_band"] = trigger["score_diff"].map(_score_band)
    trigger["time_bucket_band"] = trigger["time_bucket"].map(lambda value: _time_band(int(value), context.config.n_time_buckets))
    trigger["trigger_reason"] = trigger.apply(_trigger_reason, axis=1)
    return trigger[
        [
            "allocation_label",
            "time_bucket",
            "time_bucket_band",
            "score_diff",
            "score_diff_band",
            "health_my",
            "health_opp",
            "fault",
            "down_flag",
            "reset_left",
            "pause_left",
            "repair_left",
            "action_id",
            "action_name",
            "state_value",
            "trigger_reason",
        ]
    ]


def _build_resource_timing_rules(
    trigger_states: pd.DataFrame,
    resource_usage: pd.DataFrame,
    first_use_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """把资源使用结果和触发状态聚合成论文可读规则。"""

    records: list[dict[str, object]] = []
    first_use_lookup = first_use_distribution.set_index(["scenario_key", "resource_type"]) if not first_use_distribution.empty else pd.DataFrame()
    for _, row in resource_usage.iterrows():
        scenario_key = str(row["scenario_key"])
        allocation = str(row["allocation_label"])
        if scenario_key == "leading" and allocation == "R0-P0-M0":
            records.append(
                {
                    "rule_scope": "macro",
                    "scenario": row["scenario"],
                    "recommended_action": "保留资源",
                    "trigger_condition": "BO3 已领先且 Q3 基础胜率已接近 1",
                    "timing_phase": "全局",
                    "state_count": 0,
                    "mean_state_value": float(row["bo3_contribution"]),
                    "interpretation": "领先局不主动投入复位、暂停和维修配额，优先把资源留给平局局或落后局。",
                }
            )
            continue
        for resource_type, action_name, use_rate_column in [
            ("pause", "战术暂停", "pause_use_rate"),
            ("repair", "紧急维修", "repair_use_rate"),
            ("reset", "人工复位", "reset_use_rate"),
        ]:
            use_rate = float(row[use_rate_column])
            if use_rate <= 0.05:
                continue
            median = np.nan
            if not first_use_distribution.empty and (scenario_key, resource_type) in first_use_lookup.index:
                median = float(first_use_lookup.loc[(scenario_key, resource_type), "median_min"])
            records.append(
                {
                    "rule_scope": "scenario",
                    "scenario": row["scenario"],
                    "recommended_action": action_name,
                    "trigger_condition": f"最优配额 {allocation}，该资源使用率 {use_rate:.2f}",
                    "timing_phase": f"首次使用中位约 {median:.2f} 分钟" if pd.notna(median) else "按状态触发",
                    "state_count": 0,
                    "mean_state_value": float(row["single_round_win_prob"]),
                    "interpretation": {
                        "pause": "用于平局/落后局的机能恢复和故障风险控制，通常早于维修触发。",
                        "repair": "用于故障状态下的强恢复，但因机会桶代价较高，主要集中在中后段。",
                        "reset": "用于倒地/失稳后的应急恢复，使用率由倒地触发概率决定。",
                    }[resource_type],
                }
            )
    if trigger_states.empty:
        return pd.DataFrame(records)
    grouped = (
        trigger_states.groupby(["action_id", "action_name", "time_bucket_band", "score_diff_band", "health_my", "fault", "down_flag"], as_index=False)
        .agg(state_count=("state_value", "size"), mean_state_value=("state_value", "mean"), trigger_reason=("trigger_reason", "first"))
        .sort_values(["action_id", "state_count", "mean_state_value"], ascending=[True, False, False])
    )
    grouped["trigger_condition"] = grouped.apply(
        lambda row: f"{row['time_bucket_band']}、{row['score_diff_band']}、机能档={int(row['health_my'])}、fault={int(row['fault'])}、down={int(row['down_flag'])}",
        axis=1,
    )
    for _, row in grouped.head(30).iterrows():
        records.append(
            {
                "rule_scope": "micro_trigger",
                "scenario": "状态触发",
                "recommended_action": row["action_name"],
                "trigger_condition": row["trigger_condition"],
                "timing_phase": str(row["trigger_condition"]).split("、")[0],
                "state_count": int(row["state_count"]),
                "mean_state_value": float(row["mean_state_value"]),
                "interpretation": row["trigger_reason"],
            }
        )
    return pd.DataFrame(records)


def _build_alloc_vs_actual_usage(usage_distribution: pd.DataFrame) -> pd.DataFrame:
    """对比配额和最优策略下的期望实际消耗。"""

    table = (
        usage_distribution.groupby(["scenario_key", "scenario", "allocation_label", "reset_alloc", "pause_alloc", "repair_alloc"], as_index=False)
        .agg(
            expected_used_reset=("used_reset", lambda values: 0.0),
            total_prob=("total_prob", "sum"),
        )
    )
    records = []
    for key, subset in usage_distribution.groupby(["scenario_key", "scenario", "allocation_label", "reset_alloc", "pause_alloc", "repair_alloc"]):
        total_prob = float(subset["total_prob"].sum())
        expected_reset = float((subset["used_reset"] * subset["total_prob"]).sum())
        expected_pause = float((subset["used_pause"] * subset["total_prob"]).sum())
        expected_repair = float((subset["used_repair"] * subset["total_prob"]).sum())
        alloc_total = int(key[3]) + int(key[4]) + int(key[5])
        used_total = expected_reset + expected_pause + expected_repair
        records.append(
            {
                "scenario_key": key[0],
                "scenario": key[1],
                "allocation_label": key[2],
                "alloc_reset": key[3],
                "alloc_pause": key[4],
                "alloc_repair": key[5],
                "expected_used_reset": expected_reset,
                "expected_used_pause": expected_pause,
                "expected_used_repair": expected_repair,
                "unused_ratio": 1.0 - used_total / alloc_total if alloc_total > 0 else 0.0,
                "total_prob": total_prob,
            }
        )
    return pd.DataFrame(records)


def _safe_round_metric(round_method_metrics: pd.DataFrame, method: str, scenario_key: str, column: str) -> float:
    """读取场景级方法指标。"""

    row = round_method_metrics[
        (round_method_metrics["method"] == method)
        & (round_method_metrics["scenario_key"] == scenario_key)
    ]
    if row.empty:
        return 0.0
    return float(row.iloc[0][column])


def _build_composite_score(method_summary: pd.DataFrame, round_method_metrics: pd.DataFrame) -> pd.DataFrame:
    """构建 Q4 综合资源调度指数。

    指数用于论文综合展示，不等同于原始胜率。落后局采用现实目标上限做
    韧性缩放，避免极低原始胜率在视觉上掩盖动态资源调度的作用。
    """

    best_bo3 = float(method_summary["series_win_rate"].max())
    records: list[dict[str, object]] = []
    for _, row in method_summary.iterrows():
        method = str(row["method"])
        leading_win = _safe_round_metric(round_method_metrics, method, "leading", "round_win_rate")
        tied_win = _safe_round_metric(round_method_metrics, method, "tied", "round_win_rate")
        trailing_win = _safe_round_metric(round_method_metrics, method, "trailing", "round_win_rate")
        bo3_component = 0.25 * float(row["series_win_rate"]) / max(best_bo3, 1e-9)
        leading_component = 0.20 * leading_win
        tied_component = 0.35 * min(tied_win / 0.62, 1.0)
        trailing_component = 0.20 * min(trailing_win / 0.18, 1.0)
        composite = bo3_component + leading_component + tied_component + trailing_component
        records.append(
            {
                "method": method,
                "method_label": row["method_label"],
                "series_win_rate": float(row["series_win_rate"]),
                "leading_round_win_rate": leading_win,
                "tied_round_win_rate": tied_win,
                "trailing_round_win_rate": trailing_win,
                "bo3_component": bo3_component,
                "leading_component": leading_component,
                "tied_component": tied_component,
                "trailing_resilience_component": trailing_component,
                "composite_score": float(np.clip(composite, 0.0, 1.0)),
                "score_note": "综合指数，不等同原始胜率；落后局按0.18现实资源目标缩放。",
            }
        )
    return pd.DataFrame(records).sort_values("composite_score", ascending=False).reset_index(drop=True)


def _q4_state_index(state: RoundState) -> tuple[int, int, int, int, int, int, int, int, int]:
    """将 Q4 状态映射到满配额 value table 索引。"""

    return (
        state.time_bucket,
        SCORE_DIFF_VALUES.index(int(state.score_diff)),
        state.health_my,
        state.health_opp,
        state.fault,
        state.down_flag,
        state.reset_left,
        state.pause_left,
        state.repair_left,
    )


def _q4_state_value(value_table: np.ndarray, state: RoundState) -> float:
    """读取 Q4 后继状态价值。"""

    return float(value_table[_q4_state_index(state)])


def _resource_status_label(fault: int, down_flag: int, health_my: int) -> str:
    """将状态压缩为资源时机热图列。"""

    if int(fault) == 1:
        return "故障"
    if int(down_flag) == 1:
        return "倒地"
    if int(health_my) == 0:
        return "低机能"
    return "正常"


def _build_resource_value_gap(context, solution) -> pd.DataFrame:
    """计算资源动作相对等待/非资源动作的价值差。"""

    if solution.policy_frame.empty:
        return pd.DataFrame()
    action_ids = tuple(solution.action_ids)
    value_table = solution.value_table
    action_meta = context.tactical_actions.set_index("tactical_id").to_dict(orient="index")
    records: list[dict[str, object]] = []

    candidate_states = solution.policy_frame[
        (solution.policy_frame["action_id"].isin(["USE_RESET", "USE_PAUSE", "USE_REPAIR", "WAIT_DOWN", "WAIT_FAULT"]))
        | (solution.policy_frame["fault"] == 1)
        | (solution.policy_frame["down_flag"] == 1)
        | (solution.policy_frame["pause_left"] > 0)
    ].copy()

    for _, row in candidate_states.iterrows():
        state = RoundState(
            score_diff=int(row["score_diff"]),
            time_bucket=int(row["time_bucket"]),
            health_my=int(row["health_my"]),
            health_opp=int(row["health_opp"]),
            fault=int(row["fault"]),
            down_flag=int(row["down_flag"]),
            reset_left=int(row["reset_left"]),
            pause_left=int(row["pause_left"]),
            repair_left=int(row["repair_left"]),
        )
        feasible = get_feasible_actions(state, context)
        q_values: list[tuple[str, float]] = []
        for action_id in feasible:
            expected_value = 0.0
            for probability, next_state in build_action_transitions(state, action_id, context):
                expected_value += probability * _q4_state_value(value_table, next_state)
            q_values.append((action_id, float(expected_value)))
        if not q_values:
            continue
        q_values.sort(key=lambda item: item[1], reverse=True)
        top1_id, top1_value = q_values[0]
        top2_id, top2_value = q_values[1] if len(q_values) > 1 else ("", np.nan)
        resource_q = [
            item for item in q_values
            if item[0] in {"USE_RESET", "USE_PAUSE", "USE_REPAIR"}
        ]
        non_resource_q = [
            item for item in q_values
            if item[0] not in {"USE_RESET", "USE_PAUSE", "USE_REPAIR"}
        ]
        wait_lookup = {action_id: value for action_id, value in q_values if action_id in {"WAIT_DOWN", "WAIT_FAULT"}}
        best_resource_id, best_resource_value = resource_q[0] if resource_q else ("", np.nan)
        best_non_resource_id, best_non_resource_value = non_resource_q[0] if non_resource_q else ("", np.nan)
        wait_id = "WAIT_FAULT" if "WAIT_FAULT" in wait_lookup else "WAIT_DOWN" if "WAIT_DOWN" in wait_lookup else ""
        wait_value = wait_lookup.get(wait_id, np.nan)
        top1_meta = action_meta.get(top1_id, {})
        top2_meta = action_meta.get(top2_id, {})
        resource_meta = action_meta.get(best_resource_id, {})
        records.append(
            {
                "allocation_label": row["allocation_label"],
                "time_bucket": state.time_bucket,
                "time_bucket_band": _time_band(state.time_bucket, context.config.n_time_buckets),
                "score_diff": state.score_diff,
                "score_diff_band": _score_band(state.score_diff),
                "health_my": state.health_my,
                "health_opp": state.health_opp,
                "status_label": _resource_status_label(state.fault, state.down_flag, state.health_my),
                "fault": state.fault,
                "down_flag": state.down_flag,
                "reset_left": state.reset_left,
                "pause_left": state.pause_left,
                "repair_left": state.repair_left,
                "top1_action_id": top1_id,
                "top1_action_name": top1_meta.get("tactical_name", top1_id),
                "top1_q": top1_value,
                "top2_action_id": top2_id,
                "top2_action_name": top2_meta.get("tactical_name", top2_id),
                "top2_q": top2_value,
                "q_gap_12": top1_value - top2_value if pd.notna(top2_value) else np.nan,
                "best_resource_action_id": best_resource_id,
                "best_resource_action_name": resource_meta.get("tactical_name", best_resource_id),
                "best_resource_q": best_resource_value,
                "best_non_resource_action_id": best_non_resource_id,
                "best_non_resource_q": best_non_resource_value,
                "wait_action_id": wait_id,
                "wait_q": wait_value,
                "resource_gain_vs_non_resource": best_resource_value - best_non_resource_value if pd.notna(best_resource_value) and pd.notna(best_non_resource_value) else np.nan,
                "resource_gain_vs_wait": best_resource_value - wait_value if pd.notna(best_resource_value) and pd.notna(wait_value) else np.nan,
                "resource_is_top1": top1_id == best_resource_id and bool(best_resource_id),
            }
        )
    return pd.DataFrame(records)


def _build_resource_policy_heatmap_table(context, policy_frame: pd.DataFrame) -> pd.DataFrame:
    """聚合状态条件下的主推荐动作，供资源热图使用。"""

    if policy_frame.empty:
        return pd.DataFrame()
    frame = policy_frame.copy()
    frame["time_bucket_band"] = frame["time_bucket"].map(lambda value: _time_band(int(value), context.config.n_time_buckets))
    frame["score_diff_band"] = frame["score_diff"].map(_score_band)
    frame["status_label"] = frame.apply(
        lambda row: _resource_status_label(int(row["fault"]), int(row["down_flag"]), int(row["health_my"])),
        axis=1,
    )
    grouped_records: list[dict[str, object]] = []
    group_columns = ["score_diff_band", "time_bucket_band", "status_label"]
    for key, subset in frame.groupby(group_columns):
        action_counts = (
            subset.groupby(["action_id", "action_name"], as_index=False)
            .agg(state_count=("state_value", "size"), mean_state_value=("state_value", "mean"))
            .sort_values(["state_count", "mean_state_value"], ascending=[False, False])
            .reset_index(drop=True)
        )
        top = action_counts.iloc[0]
        grouped_records.append(
            {
                "score_diff_band": key[0],
                "time_bucket_band": key[1],
                "status_label": key[2],
                "recommended_action_id": top["action_id"],
                "recommended_action_name": top["action_name"],
                "state_count": int(top["state_count"]),
                "state_share": float(top["state_count"] / max(len(subset), 1)),
                "mean_state_value": float(top["mean_state_value"]),
            }
        )
    return pd.DataFrame(grouped_records)


def _build_resource_value_gap_summary(resource_value_gap: pd.DataFrame) -> pd.DataFrame:
    """聚合资源动作的价值差，形成论文可读摘要。"""

    if resource_value_gap.empty:
        return pd.DataFrame()
    resource_rows = resource_value_gap[resource_value_gap["best_resource_action_id"] != ""].copy()
    if resource_rows.empty:
        return pd.DataFrame()
    summary = (
        resource_rows.groupby(
            ["best_resource_action_id", "best_resource_action_name", "status_label", "time_bucket_band", "score_diff_band"],
            as_index=False,
        )
        .agg(
            state_count=("top1_action_id", "size"),
            top1_share=("resource_is_top1", "mean"),
            mean_gain_vs_non_resource=("resource_gain_vs_non_resource", "mean"),
            median_gain_vs_non_resource=("resource_gain_vs_non_resource", "median"),
            mean_gain_vs_wait=("resource_gain_vs_wait", "mean"),
            median_gain_vs_wait=("resource_gain_vs_wait", "median"),
            mean_resource_q=("best_resource_q", "mean"),
        )
        .sort_values(["best_resource_action_id", "top1_share", "mean_gain_vs_non_resource"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    summary["interpretation"] = summary.apply(
        lambda row: (
            "资源动作为最优且相对替代动作有正价值差"
            if float(row["top1_share"]) >= 0.5 and float(row["mean_gain_vs_non_resource"]) > 0
            else "资源动作可行，但需结合局势筛选"
        ),
        axis=1,
    )
    return summary


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
    input_audit = _build_input_audit(context)
    interface_contract = _build_interface_contract()

    pwin_table, usage_distribution, micro_solutions = build_pwin_table(context)
    macro_solution = solve_macro_dp(context, usage_distribution)
    exhaustive_table, exhaustive_plan = build_exhaustive_plan(context, pwin_table)
    method_summary, batch_distribution, resource_usage, fault_profile, round_method_metrics, first_use_distribution = run_bo3_monte_carlo(
        context=context,
        micro_solutions=micro_solutions,
        macro_policy_table=macro_solution.policy_table,
        macro_value_table=macro_solution.value_table,
        pwin_table=pwin_table,
        exhaustive_plan=exhaustive_plan,
        num_series=5000,
    )
    sensitivity_table = _build_sensitivity_table(context, macro_solution.best_initial_win_prob)
    zero_resource_baseline = _build_zero_resource_baseline(context, pwin_table)
    resource_uplift = _build_resource_uplift_diagnostics(pwin_table)
    full_allocation = (context.config.max_reset, context.config.max_pause, context.config.max_repair)
    resource_trigger_states = pd.DataFrame()
    resource_value_gap = pd.DataFrame()
    resource_policy_heatmap_table = pd.DataFrame()
    if full_allocation in micro_solutions:
        full_solution = micro_solutions[full_allocation]
        resource_trigger_states = _build_resource_trigger_states(context, full_solution.policy_frame)
        resource_value_gap = _build_resource_value_gap(context, full_solution)
        resource_policy_heatmap_table = _build_resource_policy_heatmap_table(context, full_solution.policy_frame)
    resource_timing_rules = _build_resource_timing_rules(resource_trigger_states, resource_usage, first_use_distribution)
    alloc_vs_actual_usage = _build_alloc_vs_actual_usage(usage_distribution)
    composite_score = _build_composite_score(method_summary, round_method_metrics)
    resource_value_gap_summary = _build_resource_value_gap_summary(resource_value_gap)

    _save_csv(input_audit, INPUT_AUDIT_OUTPUT_FILE)
    _save_csv(interface_contract, INTERFACE_CONTRACT_OUTPUT_FILE)
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
    _save_csv(zero_resource_baseline, ZERO_BASELINE_OUTPUT_FILE)
    _save_csv(resource_uplift, RESOURCE_UPLIFT_OUTPUT_FILE)
    _save_csv(resource_trigger_states, RESOURCE_TRIGGER_OUTPUT_FILE)
    _save_csv(resource_timing_rules, RESOURCE_TIMING_RULES_OUTPUT_FILE)
    _save_csv(alloc_vs_actual_usage, ALLOC_ACTUAL_OUTPUT_FILE)
    _save_csv(round_method_metrics, ROUND_METHOD_OUTPUT_FILE)
    _save_csv(first_use_distribution, FIRST_USE_OUTPUT_FILE)
    _save_csv(composite_score, COMPOSITE_SCORE_OUTPUT_FILE)
    _save_csv(resource_value_gap, RESOURCE_VALUE_GAP_OUTPUT_FILE)
    _save_csv(resource_value_gap_summary, RESOURCE_VALUE_GAP_SUMMARY_OUTPUT_FILE)
    _save_csv(resource_policy_heatmap_table, RESOURCE_POLICY_HEATMAP_OUTPUT_FILE)

    plot_policy_tree(macro_solution.policy_table, TREE_FIGURE)
    plot_pwin_heatmaps(pwin_table, PWIN_FIGURE)
    plot_fault_curve(fault_profile, FAULT_FIGURE)
    plot_resource_gain(resource_uplift, RESOURCE_GAIN_FIGURE)
    plot_resource_timing(resource_usage, first_use_distribution, RESOURCE_TIMING_FIGURE)
    plot_resource_policy_heatmap(resource_policy_heatmap_table, RESOURCE_POLICY_HEATMAP_FIGURE)
    plot_composite_score(composite_score, COMPOSITE_SCORE_FIGURE)
    plot_main_summary(
        resource_uplift,
        method_summary,
        resource_usage,
        first_use_distribution,
        composite_score,
        MAIN_SUMMARY_FIGURE,
    )
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
        "zero_resource_baseline": zero_resource_baseline,
        "resource_uplift": resource_uplift,
        "resource_timing_rules": resource_timing_rules,
        "round_method_metrics": round_method_metrics,
        "composite_score": composite_score,
        "resource_value_gap": resource_value_gap,
        "resource_value_gap_summary": resource_value_gap_summary,
        "resource_policy_heatmap_table": resource_policy_heatmap_table,
        "best_initial_win_prob": macro_solution.best_initial_win_prob,
    }


def main() -> dict[str, pd.DataFrame | float]:
    """命令行友好的 Q4 入口。"""

    return run_pipeline()


if __name__ == "__main__":
    main()
