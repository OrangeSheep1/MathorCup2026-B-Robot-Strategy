"""Q4 BO3 仿真、基线策略与诊断汇总。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.q4.decision import MicroRoundSolution
from src.q4.model_v1 import (
    Q4Context,
    RoundState,
    SCENARIO_INITIAL_SCORE,
    SCENARIO_LABELS,
    allocation_label,
    build_action_transitions,
    get_feasible_actions,
    scenario_from_series_state,
    terminal_win_value,
)


METHOD_LABELS = {
    "optimal_dp": "最优DP",
    "fixed_rule": "固定规则",
    "exhaustive_static": "穷举静态",
    "all_in_first": "首局全投",
}


@dataclass
class RoundSimulationResult:
    """单局仿真结果。"""

    scenario_key: str
    allocation: tuple[int, int, int]
    method: str
    win_flag: int
    final_score_diff: int
    used_reset: int
    used_pause: int
    used_repair: int
    first_reset_bucket: int | None
    first_pause_bucket: int | None
    first_repair_bucket: int | None
    step_log: list[dict[str, object]]


def _policy_action(solution: MicroRoundSolution, state: RoundState) -> str:
    """按局内 DP 策略读取动作。"""

    action_code = int(
        solution.policy_codes[
            state.time_bucket,
            state.score_diff + 5,
            state.health_my,
            state.health_opp,
            state.fault,
            state.down_flag,
            state.reset_left,
            state.pause_left,
            state.repair_left,
        ]
    )
    return solution.action_ids[action_code]


def _sample_next_state(
    state: RoundState,
    action_id: str,
    context: Q4Context,
    rng: np.random.Generator,
) -> RoundState:
    """按给定动作采样后继状态。"""

    transitions = build_action_transitions(state, action_id, context)
    probabilities = np.array([probability for probability, _ in transitions], dtype=float)
    probabilities = probabilities / probabilities.sum()
    index = int(rng.choice(len(transitions), p=probabilities))
    return transitions[index][1]


def _choose_rule_allocation(
    wins_my: int,
    wins_opp: int,
    reset_left: int,
    pause_left: int,
    repair_left: int,
) -> tuple[int, int, int]:
    """固定规则法的跨局分配。"""

    if wins_my == 1 and wins_opp == 1:
        return reset_left, pause_left, repair_left
    if wins_my < wins_opp:
        return min(1, reset_left), min(1, pause_left), min(1, repair_left)
    if wins_my > wins_opp:
        return 0, min(1, pause_left), 0
    return min(1, reset_left), min(1, pause_left), 0


def _choose_rule_action(state: RoundState, context: Q4Context) -> str:
    """固定规则法的局内动作。"""

    feasible = set(get_feasible_actions(state, context))
    if state.fault == 1:
        if "USE_REPAIR" in feasible and state.score_diff <= 0:
            return "USE_REPAIR"
        return "WAIT_FAULT"

    if state.down_flag == 1:
        if "USE_RESET" in feasible:
            return "USE_RESET"
        return "WAIT_DOWN"

    if "USE_PAUSE" in feasible and (state.health_my == 0 or state.score_diff < 0):
        return "USE_PAUSE"
    if state.score_diff <= -2 and "TACT_AGGRESSIVE" in feasible:
        return "TACT_AGGRESSIVE"
    if state.score_diff >= 2 and "TACT_GUARD" in feasible:
        return "TACT_GUARD"
    if state.health_my == 0 and "TACT_RECOVER" in feasible:
        return "TACT_RECOVER"
    if state.score_diff < 0 and "TACT_COUNTER" in feasible:
        return "TACT_COUNTER"
    return "TACT_PROBE"


def _resolve_tie(final_score_diff: int, rng: np.random.Generator) -> int:
    """将平局按 0.5 概率抽签到胜负。"""

    if final_score_diff > 0:
        return 1
    if final_score_diff < 0:
        return 0
    return int(rng.random() < 0.5)


def simulate_round(
    context: Q4Context,
    scenario_key: str,
    allocation: tuple[int, int, int],
    method: str,
    rng: np.random.Generator,
    micro_solutions: dict[tuple[int, int, int], MicroRoundSolution],
    calibrated_round_pwin: float | None = None,
) -> RoundSimulationResult:
    """仿真一局比赛。"""

    state = RoundState(
        score_diff=SCENARIO_INITIAL_SCORE[scenario_key],
        time_bucket=0,
        health_my=2,
        health_opp=2,
        fault=0,
        down_flag=0,
        reset_left=allocation[0],
        pause_left=allocation[1],
        repair_left=allocation[2],
    )
    solution = micro_solutions.get(allocation)
    step_log: list[dict[str, object]] = []
    used_reset = 0
    used_pause = 0
    used_repair = 0
    first_reset_bucket: int | None = None
    first_pause_bucket: int | None = None
    first_repair_bucket: int | None = None

    while state.time_bucket < context.config.n_time_buckets:
        if method == "fixed_rule":
            action_id = _choose_rule_action(state, context)
        else:
            if solution is None:
                raise KeyError(f"缺少分配方案 {allocation} 的局内解。")
            action_id = _policy_action(solution, state)
        next_state = _sample_next_state(state, action_id, context, rng)
        step_log.append(
            {
                "scenario_key": scenario_key,
                "scenario": SCENARIO_LABELS[scenario_key],
                "method": method,
                "time_bucket": state.time_bucket,
                "score_diff_before": state.score_diff,
                "score_diff_after": next_state.score_diff,
                "health_my_before": state.health_my,
                "health_my_after": next_state.health_my,
                "health_opp_after": next_state.health_opp,
                "fault_before": state.fault,
                "fault_after": next_state.fault,
                "down_before": state.down_flag,
                "down_after": next_state.down_flag,
                "action_id": action_id,
                "allocation_label": allocation_label(*allocation),
            }
        )
        if action_id == "USE_RESET":
            used_reset += 1
            first_reset_bucket = state.time_bucket if first_reset_bucket is None else first_reset_bucket
        elif action_id == "USE_PAUSE":
            used_pause += 1
            first_pause_bucket = state.time_bucket if first_pause_bucket is None else first_pause_bucket
        elif action_id == "USE_REPAIR":
            used_repair += 1
            first_repair_bucket = state.time_bucket if first_repair_bucket is None else first_repair_bucket
        state = next_state

    if calibrated_round_pwin is None:
        win_flag = _resolve_tie(state.score_diff, rng)
    else:
        win_flag = int(rng.random() < float(np.clip(calibrated_round_pwin, 0.0, 1.0)))
    return RoundSimulationResult(
        scenario_key=scenario_key,
        allocation=allocation,
        method=method,
        win_flag=win_flag,
        final_score_diff=state.score_diff,
        used_reset=used_reset,
        used_pause=used_pause,
        used_repair=used_repair,
        first_reset_bucket=first_reset_bucket,
        first_pause_bucket=first_pause_bucket,
        first_repair_bucket=first_repair_bucket,
        step_log=step_log,
    )


def _macro_optimal_allocation(
    macro_policy_table: pd.DataFrame,
    wins_my: int,
    wins_opp: int,
    reset_left: int,
    pause_left: int,
    repair_left: int,
) -> tuple[int, int, int]:
    """读取 BO3 宏观最优分配。"""

    row = macro_policy_table[
        (macro_policy_table["wins_my"] == wins_my)
        & (macro_policy_table["wins_opp"] == wins_opp)
        & (macro_policy_table["reset_left"] == reset_left)
        & (macro_policy_table["pause_left"] == pause_left)
        & (macro_policy_table["repair_left"] == repair_left)
    ]
    if row.empty:
        return 0, 0, 0
    selected = row.iloc[0]
    return int(selected["alloc_reset"]), int(selected["alloc_pause"]), int(selected["alloc_repair"])


def _exhaustive_plan_allocation(best_plan: dict[str, object], round_index: int) -> tuple[int, int, int]:
    """按轮次读取穷举静态最优计划。"""

    return tuple(best_plan[f"alloc_round{round_index + 1}"])  # type: ignore[arg-type]


def _all_in_first_allocation(round_index: int, remaining: tuple[int, int, int]) -> tuple[int, int, int]:
    """首局全投策略。"""

    if round_index == 0:
        return remaining
    return 0, 0, 0


def simulate_series(
    context: Q4Context,
    method: str,
    rng: np.random.Generator,
    micro_solutions: dict[tuple[int, int, int], MicroRoundSolution],
    macro_policy_table: pd.DataFrame,
    exhaustive_plan: dict[str, object],
    pwin_table: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """仿真一场 BO3。"""

    wins_my = 0
    wins_opp = 0
    reset_left = context.config.max_reset
    pause_left = context.config.max_pause
    repair_left = context.config.max_repair
    round_logs: list[dict[str, object]] = []
    step_logs: list[dict[str, object]] = []
    round_index = 0

    while wins_my < 2 and wins_opp < 2 and round_index < 3:
        scenario_key = scenario_from_series_state(wins_my, wins_opp)
        remaining = (reset_left, pause_left, repair_left)
        if method == "optimal_dp":
            allocation = _macro_optimal_allocation(macro_policy_table, wins_my, wins_opp, *remaining)
        elif method == "fixed_rule":
            allocation = _choose_rule_allocation(wins_my, wins_opp, *remaining)
        elif method == "exhaustive_static":
            allocation = _exhaustive_plan_allocation(exhaustive_plan, round_index)
        elif method == "all_in_first":
            allocation = _all_in_first_allocation(round_index, remaining)
        else:
            raise ValueError(f"未知 Q4 方法: {method}")

        allocation = (
            min(allocation[0], reset_left),
            min(allocation[1], pause_left),
            min(allocation[2], repair_left),
        )
        pwin_row = pwin_table[
            (pwin_table["scenario_key"] == scenario_key)
            & (pwin_table["reset_alloc"] == allocation[0])
            & (pwin_table["pause_alloc"] == allocation[1])
            & (pwin_table["repair_alloc"] == allocation[2])
        ]
        calibrated_round_pwin = None
        if method != "fixed_rule" and not pwin_row.empty:
            calibrated_round_pwin = float(pwin_row.iloc[0]["p_win"])
        round_result = simulate_round(
            context,
            scenario_key,
            allocation,
            method,
            rng,
            micro_solutions,
            calibrated_round_pwin=calibrated_round_pwin,
        )
        wins_my += round_result.win_flag
        wins_opp += 1 - round_result.win_flag
        reset_left -= round_result.used_reset
        pause_left -= round_result.used_pause
        repair_left -= round_result.used_repair
        round_logs.append(
            {
                "method": method,
                "round_index": round_index + 1,
                "scenario_key": scenario_key,
                "scenario": SCENARIO_LABELS[scenario_key],
                "allocation_label": allocation_label(*allocation),
                "alloc_reset": allocation[0],
                "alloc_pause": allocation[1],
                "alloc_repair": allocation[2],
                "round_win": round_result.win_flag,
                "final_score_diff": round_result.final_score_diff,
                "used_reset": round_result.used_reset,
                "used_pause": round_result.used_pause,
                "used_repair": round_result.used_repair,
                "remain_reset_after": reset_left,
                "remain_pause_after": pause_left,
                "remain_repair_after": repair_left,
                "first_reset_bucket": round_result.first_reset_bucket,
                "first_pause_bucket": round_result.first_pause_bucket,
                "first_repair_bucket": round_result.first_repair_bucket,
                "wins_my_after": wins_my,
                "wins_opp_after": wins_opp,
            }
        )
        step_logs.extend(round_result.step_log)
        round_index += 1

    summary = {
        "method": method,
        "wins_my": wins_my,
        "wins_opp": wins_opp,
        "series_win": int(wins_my > wins_opp),
        "remaining_reset": reset_left,
        "remaining_pause": pause_left,
        "remaining_repair": repair_left,
        "rounds_played": round_index,
    }
    return summary, round_logs + step_logs


def _prepare_batch_distribution(series_table: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
    """将逐场仿真折算为批次胜率，用于箱线图。"""

    records: list[dict[str, object]] = []
    for method, subset in series_table.groupby("method"):
        ordered = subset.reset_index(drop=True)
        ordered["batch_id"] = ordered.index // batch_size
        for batch_id, batch in ordered.groupby("batch_id"):
            records.append(
                {
                    "method": method,
                    "batch_id": int(batch_id),
                    "batch_win_rate": float(batch["series_win"].mean()),
                }
            )
    return pd.DataFrame(records)


def _build_resource_usage_summary(
    round_table: pd.DataFrame,
    macro_policy_table: pd.DataFrame,
    macro_value_table: pd.DataFrame,
    pwin_table: pd.DataFrame,
    time_bucket_s: int,
) -> pd.DataFrame:
    """汇总三种情景下的资源使用画像。"""

    records: list[dict[str, object]] = []
    optimal_rounds = round_table[round_table["method"] == "optimal_dp"].copy()
    for scenario_key, scenario_label in SCENARIO_LABELS.items():
        subset = optimal_rounds[optimal_rounds["scenario_key"] == scenario_key]
        if subset.empty:
            continue
        macro_state = {
            "leading": (1, 0),
            "tied": (0, 0),
            "trailing": (0, 1),
        }[scenario_key]
        macro_row = macro_policy_table[
            (macro_policy_table["wins_my"] == macro_state[0])
            & (macro_policy_table["wins_opp"] == macro_state[1])
            & (macro_policy_table["reset_left"] == 2)
            & (macro_policy_table["pause_left"] == 2)
            & (macro_policy_table["repair_left"] == 1)
        ]
        value_row = macro_value_table[
            (macro_value_table["wins_my"] == macro_state[0])
            & (macro_value_table["wins_opp"] == macro_state[1])
            & (macro_value_table["reset_left"] == 2)
            & (macro_value_table["pause_left"] == 2)
            & (macro_value_table["repair_left"] == 1)
        ]
        bo3_value = float(value_row.iloc[0]["series_value"]) if not value_row.empty else 0.0
        alloc_label = str(macro_row.iloc[0]["allocation_label"]) if not macro_row.empty else ""
        pwin_row = pwin_table[
            (pwin_table["scenario_key"] == scenario_key)
            & (pwin_table["allocation_label"] == alloc_label)
        ]
        single_round_pwin = float(pwin_row.iloc[0]["p_win"]) if not pwin_row.empty else float(subset["round_win"].mean())
        first_times = []
        for column in ["first_reset_bucket", "first_pause_bucket", "first_repair_bucket"]:
            values = subset[column].dropna()
            if not values.empty:
                first_times.extend(values.tolist())
        avg_first_time_min = float(np.mean(first_times)) * (time_bucket_s / 60.0) if first_times else 0.0
        records.append(
            {
                "scenario_key": scenario_key,
                "scenario": scenario_label,
                "reset_use_rate": float((subset["used_reset"] > 0).mean()),
                "pause_use_rate": float((subset["used_pause"] > 0).mean()),
                "repair_use_rate": float((subset["used_repair"] > 0).mean()),
                "avg_first_use_min": avg_first_time_min,
                "single_round_win_prob": single_round_pwin,
                "bo3_contribution": bo3_value,
                "allocation_label": alloc_label,
            }
        )
    return pd.DataFrame(records)


def _build_fault_profile(step_table: pd.DataFrame, time_bucket_s: int) -> pd.DataFrame:
    """统计最优策略下的平均机能与故障率演化。"""

    optimal_steps = step_table[step_table["method"] == "optimal_dp"].copy()
    if optimal_steps.empty:
        return pd.DataFrame()
    profile = (
        optimal_steps.groupby("time_bucket", as_index=False)
        .agg(
            mean_health_my=("health_my_before", "mean"),
            fault_rate=("fault_before", "mean"),
        )
        .sort_values("time_bucket")
        .reset_index(drop=True)
    )
    profile["time_s"] = (profile["time_bucket"] + 1) * time_bucket_s
    return profile


def _build_round_method_metrics(round_table: pd.DataFrame, time_bucket_s: int) -> pd.DataFrame:
    """按方法和场景统计单局表现。"""

    if round_table.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for (method, scenario_key), subset in round_table.groupby(["method", "scenario_key"]):
        first_times = []
        for column in ["first_reset_bucket", "first_pause_bucket", "first_repair_bucket"]:
            values = subset[column].dropna()
            if not values.empty:
                first_times.extend(values.tolist())
        records.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "scenario_key": scenario_key,
                "scenario": SCENARIO_LABELS[scenario_key],
                "round_count": int(len(subset)),
                "round_win_rate": float(subset["round_win"].mean()),
                "mean_score_diff": float(subset["final_score_diff"].mean()),
                "mean_used_reset": float(subset["used_reset"].mean()),
                "mean_used_pause": float(subset["used_pause"].mean()),
                "mean_used_repair": float(subset["used_repair"].mean()),
                "avg_first_use_min": float(np.mean(first_times)) * (time_bucket_s / 60.0) if first_times else 0.0,
            }
        )
    return pd.DataFrame(records)


def _build_first_use_distribution(round_table: pd.DataFrame, time_bucket_s: int) -> pd.DataFrame:
    """输出资源首次使用时机的分位数。"""

    records: list[dict[str, object]] = []
    optimal_rounds = round_table[round_table["method"] == "optimal_dp"].copy()
    resource_columns = {
        "reset": "first_reset_bucket",
        "pause": "first_pause_bucket",
        "repair": "first_repair_bucket",
    }
    for (scenario_key, scenario_label) in SCENARIO_LABELS.items():
        scenario_rows = optimal_rounds[optimal_rounds["scenario_key"] == scenario_key]
        for resource_type, column in resource_columns.items():
            values = scenario_rows[column].dropna().to_numpy(dtype=float) * (time_bucket_s / 60.0)
            if len(values) == 0:
                records.append(
                    {
                        "scenario_key": scenario_key,
                        "scenario": scenario_label,
                        "resource_type": resource_type,
                        "use_count": 0,
                        "p25_min": np.nan,
                        "median_min": np.nan,
                        "p75_min": np.nan,
                    }
                )
                continue
            records.append(
                {
                    "scenario_key": scenario_key,
                    "scenario": scenario_label,
                    "resource_type": resource_type,
                    "use_count": int(len(values)),
                    "p25_min": float(np.percentile(values, 25)),
                    "median_min": float(np.percentile(values, 50)),
                    "p75_min": float(np.percentile(values, 75)),
                }
            )
    return pd.DataFrame(records)


def run_bo3_monte_carlo(
    context: Q4Context,
    micro_solutions: dict[tuple[int, int, int], MicroRoundSolution],
    macro_policy_table: pd.DataFrame,
    macro_value_table: pd.DataFrame,
    pwin_table: pd.DataFrame,
    exhaustive_plan: dict[str, object],
    num_series: int = 5000,
    seed: int = 20260418,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行 BO3 蒙特卡洛仿真并输出摘要。"""

    methods = ["optimal_dp", "fixed_rule", "exhaustive_static", "all_in_first"]
    series_records: list[dict[str, object]] = []
    round_step_records: list[dict[str, object]] = []
    for match_index in range(num_series):
        base_seed = int(seed + match_index * 97)
        for method_index, method in enumerate(methods):
            rng = np.random.default_rng(base_seed + method_index * 10_000)
            summary, logs = simulate_series(
                context,
                method,
                rng,
                micro_solutions,
                macro_policy_table,
                exhaustive_plan,
                pwin_table,
            )
            summary["match_index"] = match_index
            series_records.append(summary)
            for log in logs:
                log["match_index"] = match_index
                round_step_records.append(log)

    series_table = pd.DataFrame(series_records)
    detail_table = pd.DataFrame(round_step_records)
    method_summary = (
        series_table.groupby("method", as_index=False)
        .agg(
            series_count=("series_win", "size"),
            series_win_rate=("series_win", "mean"),
            mean_rounds_played=("rounds_played", "mean"),
            mean_remaining_reset=("remaining_reset", "mean"),
            mean_remaining_pause=("remaining_pause", "mean"),
            mean_remaining_repair=("remaining_repair", "mean"),
        )
        .sort_values("series_win_rate", ascending=False)
        .reset_index(drop=True)
    )
    method_summary["method_label"] = method_summary["method"].map(METHOD_LABELS)
    method_summary["ci_low"] = (
        method_summary["series_win_rate"]
        - 1.96 * np.sqrt(method_summary["series_win_rate"] * (1.0 - method_summary["series_win_rate"]) / method_summary["series_count"])
    ).clip(lower=0.0)
    method_summary["ci_high"] = (
        method_summary["series_win_rate"]
        + 1.96 * np.sqrt(method_summary["series_win_rate"] * (1.0 - method_summary["series_win_rate"]) / method_summary["series_count"])
    ).clip(upper=1.0)
    batch_distribution = _prepare_batch_distribution(series_table)
    round_table = detail_table[detail_table["round_index"].notna()].copy()
    resource_usage = _build_resource_usage_summary(
        round_table,
        macro_policy_table,
        macro_value_table,
        pwin_table,
        context.config.time_bucket_s,
    )
    fault_profile = _build_fault_profile(
        detail_table[detail_table["time_bucket"].notna()].copy(),
        context.config.time_bucket_s,
    )
    round_method_metrics = _build_round_method_metrics(round_table, context.config.time_bucket_s)
    first_use_distribution = _build_first_use_distribution(round_table, context.config.time_bucket_s)
    return method_summary, batch_distribution, resource_usage, fault_profile, round_method_metrics, first_use_distribution
