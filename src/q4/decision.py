"""Q4 局内有限时域动态规划与 BO3 宏观逆推求解。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.q4.model_v1 import (
    DOWN_FLAGS,
    FAULT_FLAGS,
    HEALTH_LEVELS,
    SCENARIO_INITIAL_SCORE,
    SCENARIO_LABELS,
    SCORE_DIFF_VALUES,
    Q4Context,
    RoundState,
    allocation_label,
    build_action_transitions,
    get_feasible_actions,
    scenario_from_series_state,
    terminal_win_value,
)


@dataclass
class MicroRoundSolution:
    """固定资源分配下的局内求解结果。"""

    allocation: tuple[int, int, int]
    action_ids: tuple[str, ...]
    value_table: np.ndarray
    policy_codes: np.ndarray
    pwin_record: pd.DataFrame
    usage_distribution: pd.DataFrame
    policy_frame: pd.DataFrame


@dataclass
class MacroSolution:
    """BO3 宏观动态规划结果。"""

    value_table: pd.DataFrame
    policy_table: pd.DataFrame
    best_initial_win_prob: float


def enumerate_allocations(max_reset: int, max_pause: int, max_repair: int) -> list[tuple[int, int, int]]:
    """枚举单局资源配额方案。"""

    allocations: list[tuple[int, int, int]] = []
    for reset_alloc in range(max_reset + 1):
        for pause_alloc in range(max_pause + 1):
            for repair_alloc in range(max_repair + 1):
                allocations.append((reset_alloc, pause_alloc, repair_alloc))
    return allocations


def _score_index(score_diff: int) -> int:
    """将分差映射为数组索引。"""

    return SCORE_DIFF_VALUES.index(int(score_diff))


def _state_to_index(state: RoundState) -> tuple[int, int, int, int, int, int, int, int, int]:
    """将状态映射为 value/policy 数组索引。"""

    return (
        state.time_bucket,
        _score_index(state.score_diff),
        state.health_my,
        state.health_opp,
        state.fault,
        state.down_flag,
        state.reset_left,
        state.pause_left,
        state.repair_left,
    )


def _get_state_value(value_table: np.ndarray, next_state: RoundState) -> float:
    """从值表中读取后继状态价值。"""

    return float(value_table[_state_to_index(next_state)])


def _usage_from_state(allocation: tuple[int, int, int], state: RoundState) -> tuple[int, int, int]:
    """由终局剩余资源反推本局实际使用量。"""

    return (
        allocation[0] - state.reset_left,
        allocation[1] - state.pause_left,
        allocation[2] - state.repair_left,
    )


def _initial_round_state(scenario_key: str, allocation: tuple[int, int, int]) -> RoundState:
    """构造某一情景和配额下的初始局内状态。"""

    return RoundState(
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


def _build_usage_distribution(
    context: Q4Context,
    allocation: tuple[int, int, int],
    action_ids: tuple[str, ...],
    policy_codes: np.ndarray,
) -> pd.DataFrame:
    """在固定配额与最优局内策略下，前向传播终局输赢与实际资源消耗联合分布。"""

    terminal_time = context.config.n_time_buckets
    records: list[dict[str, object]] = []

    for scenario_key, scenario_label in SCENARIO_LABELS.items():
        state_mass: dict[RoundState, float] = {_initial_round_state(scenario_key, allocation): 1.0}

        for time_bucket in range(terminal_time):
            current_items = [(state, mass) for state, mass in state_mass.items() if state.time_bucket == time_bucket]
            if not current_items:
                continue
            for state, mass in current_items:
                state_mass.pop(state, None)
                if mass <= 0:
                    continue
                action_code = int(policy_codes[_state_to_index(state)])
                if action_code < 0:
                    state_mass[state] = state_mass.get(state, 0.0) + mass
                    continue
                action_id = action_ids[action_code]
                for probability, next_state in build_action_transitions(state, action_id, context):
                    if probability <= 0:
                        continue
                    state_mass[next_state] = state_mass.get(next_state, 0.0) + mass * probability

        grouped: dict[tuple[int, int, int], dict[str, float]] = {}
        for state, probability in state_mass.items():
            if probability <= 0:
                continue
            used_reset, used_pause, used_repair = _usage_from_state(allocation, state)
            key = (used_reset, used_pause, used_repair)
            if key not in grouped:
                grouped[key] = {"win_prob": 0.0, "lose_prob": 0.0}
            if state.score_diff > 0:
                grouped[key]["win_prob"] += probability
            elif state.score_diff < 0:
                grouped[key]["lose_prob"] += probability
            else:
                grouped[key]["win_prob"] += 0.5 * probability
                grouped[key]["lose_prob"] += 0.5 * probability

        for (used_reset, used_pause, used_repair), outcome in grouped.items():
            total_prob = outcome["win_prob"] + outcome["lose_prob"]
            if total_prob <= 0:
                continue
            records.append(
                {
                    "scenario_key": scenario_key,
                    "scenario": scenario_label,
                    "allocation_label": allocation_label(*allocation),
                    "reset_alloc": allocation[0],
                    "pause_alloc": allocation[1],
                    "repair_alloc": allocation[2],
                    "used_reset": used_reset,
                    "used_pause": used_pause,
                    "used_repair": used_repair,
                    "win_prob": outcome["win_prob"],
                    "lose_prob": outcome["lose_prob"],
                    "total_prob": total_prob,
                }
            )

    return pd.DataFrame(records)


def _build_policy_frame(
    context: Q4Context,
    allocation: tuple[int, int, int],
    policy_codes: np.ndarray,
    value_table: np.ndarray,
    action_ids: tuple[str, ...],
) -> pd.DataFrame:
    """将固定配额方案的最优局内策略展开为表。"""

    records: list[dict[str, object]] = []
    max_time = context.config.n_time_buckets
    for time_bucket in range(max_time):
        for score_diff in SCORE_DIFF_VALUES:
            for health_my in HEALTH_LEVELS:
                for health_opp in HEALTH_LEVELS:
                    for fault in FAULT_FLAGS:
                        for down_flag in DOWN_FLAGS:
                            for reset_left in range(allocation[0] + 1):
                                for pause_left in range(allocation[1] + 1):
                                    for repair_left in range(allocation[2] + 1):
                                        policy_code = int(
                                            policy_codes[
                                                time_bucket,
                                                _score_index(score_diff),
                                                health_my,
                                                health_opp,
                                                fault,
                                                down_flag,
                                                reset_left,
                                                pause_left,
                                                repair_left,
                                            ]
                                        )
                                        action_id = action_ids[policy_code] if policy_code >= 0 else ""
                                        action_name = ""
                                        action_type = ""
                                        macro_group = ""
                                        if action_id:
                                            row = context.tactical_actions[
                                                context.tactical_actions["tactical_id"] == action_id
                                            ].iloc[0]
                                            action_name = str(row["tactical_name"])
                                            action_type = str(row["action_type"])
                                            macro_group = str(row["macro_group"])
                                        records.append(
                                            {
                                                "allocation_label": allocation_label(*allocation),
                                                "time_bucket": time_bucket,
                                                "score_diff": score_diff,
                                                "health_my": health_my,
                                                "health_opp": health_opp,
                                                "fault": fault,
                                                "down_flag": down_flag,
                                                "reset_left": reset_left,
                                                "pause_left": pause_left,
                                                "repair_left": repair_left,
                                                "action_id": action_id,
                                                "action_name": action_name,
                                                "action_type": action_type,
                                                "macro_group": macro_group,
                                                "state_value": float(
                                                    value_table[
                                                        time_bucket,
                                                        _score_index(score_diff),
                                                        health_my,
                                                        health_opp,
                                                        fault,
                                                        down_flag,
                                                        reset_left,
                                                        pause_left,
                                                        repair_left,
                                                    ]
                                                ),
                                            }
                                        )
    return pd.DataFrame(records)


def solve_micro_round(context: Q4Context, allocation: tuple[int, int, int]) -> MicroRoundSolution:
    """对单个资源配额方案做局内有限时域动态规划。"""

    action_ids = tuple(context.all_action_ids)
    action_index = {action_id: index for index, action_id in enumerate(action_ids)}
    shape = (
        context.config.n_time_buckets + 1,
        len(SCORE_DIFF_VALUES),
        len(HEALTH_LEVELS),
        len(HEALTH_LEVELS),
        len(FAULT_FLAGS),
        len(DOWN_FLAGS),
        allocation[0] + 1,
        allocation[1] + 1,
        allocation[2] + 1,
    )
    value_table = np.zeros(shape, dtype=float)
    policy_codes = np.full(shape, -1, dtype=np.int16)

    terminal_time = context.config.n_time_buckets
    for score_diff in SCORE_DIFF_VALUES:
        terminal_value = terminal_win_value(score_diff)
        value_table[terminal_time, _score_index(score_diff), :, :, :, :, :, :, :] = terminal_value

    for time_bucket in range(context.config.n_time_buckets - 1, -1, -1):
        for score_diff in SCORE_DIFF_VALUES:
            score_index = _score_index(score_diff)
            for health_my in HEALTH_LEVELS:
                for health_opp in HEALTH_LEVELS:
                    for fault in FAULT_FLAGS:
                        for down_flag in DOWN_FLAGS:
                            for reset_left in range(allocation[0] + 1):
                                for pause_left in range(allocation[1] + 1):
                                    for repair_left in range(allocation[2] + 1):
                                        state = RoundState(
                                            score_diff=score_diff,
                                            time_bucket=time_bucket,
                                            health_my=health_my,
                                            health_opp=health_opp,
                                            fault=fault,
                                            down_flag=down_flag,
                                            reset_left=reset_left,
                                            pause_left=pause_left,
                                            repair_left=repair_left,
                                        )
                                        feasible_actions = get_feasible_actions(state, context)
                                        if not feasible_actions:
                                            value_table[
                                                time_bucket,
                                                score_index,
                                                health_my,
                                                health_opp,
                                                fault,
                                                down_flag,
                                                reset_left,
                                                pause_left,
                                                repair_left,
                                            ] = terminal_win_value(score_diff)
                                            continue

                                        best_action = feasible_actions[0]
                                        best_value = -1.0
                                        for action_id in feasible_actions:
                                            expected_value = 0.0
                                            for probability, next_state in build_action_transitions(state, action_id, context):
                                                expected_value += probability * _get_state_value(value_table, next_state)
                                            if expected_value > best_value + 1e-12:
                                                best_value = expected_value
                                                best_action = action_id

                                        value_table[
                                            time_bucket,
                                            score_index,
                                            health_my,
                                            health_opp,
                                            fault,
                                            down_flag,
                                            reset_left,
                                            pause_left,
                                            repair_left,
                                        ] = best_value
                                        policy_codes[
                                            time_bucket,
                                            score_index,
                                            health_my,
                                            health_opp,
                                            fault,
                                            down_flag,
                                            reset_left,
                                            pause_left,
                                            repair_left,
                                        ] = action_index[best_action]

    usage_distribution = _build_usage_distribution(context, allocation, action_ids, policy_codes)
    pwin_records: list[dict[str, object]] = []
    for scenario_key, scenario_label in SCENARIO_LABELS.items():
        initial_state = _initial_round_state(scenario_key, allocation)
        best_value = _get_state_value(value_table, initial_state)
        action_code = int(policy_codes[_state_to_index(initial_state)])
        action_id = action_ids[action_code] if action_code >= 0 else ""
        action_meta = context.tactical_actions[context.tactical_actions["tactical_id"] == action_id].iloc[0]
        scenario_usage = usage_distribution[usage_distribution["scenario_key"] == scenario_key]
        pwin_records.append(
            {
                "scenario_key": scenario_key,
                "scenario": scenario_label,
                "allocation_label": allocation_label(*allocation),
                "reset_alloc": allocation[0],
                "pause_alloc": allocation[1],
                "repair_alloc": allocation[2],
                "p_win": best_value,
                "initial_action_id": action_id,
                "initial_action_name": str(action_meta["tactical_name"]),
                "initial_action_type": str(action_meta["action_type"]),
                "initial_macro_group": str(action_meta["macro_group"]),
                "expected_used_reset": float((scenario_usage["used_reset"] * scenario_usage["total_prob"]).sum()),
                "expected_used_pause": float((scenario_usage["used_pause"] * scenario_usage["total_prob"]).sum()),
                "expected_used_repair": float((scenario_usage["used_repair"] * scenario_usage["total_prob"]).sum()),
            }
        )

    policy_frame = pd.DataFrame()
    if allocation == (context.config.max_reset, context.config.max_pause, context.config.max_repair):
        policy_frame = _build_policy_frame(context, allocation, policy_codes, value_table, action_ids)

    return MicroRoundSolution(
        allocation=allocation,
        action_ids=action_ids,
        value_table=value_table,
        policy_codes=policy_codes,
        pwin_record=pd.DataFrame(pwin_records),
        usage_distribution=usage_distribution,
        policy_frame=policy_frame,
    )


def build_pwin_table(
    context: Q4Context,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, int, int], MicroRoundSolution]]:
    """枚举全部资源配额方案，构建单局胜率表与实际使用分布表。"""

    allocations = enumerate_allocations(context.config.max_reset, context.config.max_pause, context.config.max_repair)
    pwin_records: list[pd.DataFrame] = []
    usage_records: list[pd.DataFrame] = []
    solutions: dict[tuple[int, int, int], MicroRoundSolution] = {}
    for allocation in allocations:
        solution = solve_micro_round(context, allocation)
        solutions[allocation] = solution
        pwin_records.append(solution.pwin_record)
        usage_records.append(solution.usage_distribution)
    return (
        pd.concat(pwin_records, ignore_index=True),
        pd.concat(usage_records, ignore_index=True),
        solutions,
    )


def save_micro_solutions(
    solutions: dict[tuple[int, int, int], MicroRoundSolution],
    output_path: str | Path,
) -> Path:
    """将局内求解结果以 pickle 形式保存，便于 Q4 仿真复用。"""

    payload = {}
    for allocation, solution in solutions.items():
        payload[allocation] = {
            "action_ids": solution.action_ids,
            "value_table": solution.value_table,
            "policy_codes": solution.policy_codes,
        }
    path = Path(output_path)
    with path.open("wb") as file:
        pickle.dump(payload, file)
    return path


def solve_macro_dp(context: Q4Context, usage_distribution: pd.DataFrame) -> MacroSolution:
    """对 BO3 宏观资源调度做逆向动态规划，资源按实际使用量扣减。"""

    grouped_distribution: dict[tuple[str, int, int, int], list[dict[str, float]]] = {}
    for key, subset in usage_distribution.groupby(["scenario_key", "reset_alloc", "pause_alloc", "repair_alloc"]):
        grouped_distribution[(str(key[0]), int(key[1]), int(key[2]), int(key[3]))] = [
            {
                "used_reset": float(row["used_reset"]),
                "used_pause": float(row["used_pause"]),
                "used_repair": float(row["used_repair"]),
                "win_prob": float(row["win_prob"]),
                "lose_prob": float(row["lose_prob"]),
                "total_prob": float(row["total_prob"]),
            }
            for _, row in subset.iterrows()
        ]

    records_value: list[dict[str, object]] = []
    records_policy: list[dict[str, object]] = []

    @lru_cache(maxsize=None)
    def solve_state(wins_my: int, wins_opp: int, reset_left: int, pause_left: int, repair_left: int) -> float:
        if wins_my >= 2:
            return 1.0
        if wins_opp >= 2:
            return 0.0

        scenario_key = scenario_from_series_state(wins_my, wins_opp)
        best_value = -1.0
        best_alloc = (0, 0, 0)
        best_expected_used = (0.0, 0.0, 0.0)
        best_round_pwin = 0.0

        for reset_alloc in range(reset_left + 1):
            for pause_alloc in range(pause_left + 1):
                for repair_alloc in range(repair_left + 1):
                    distribution_key = (scenario_key, reset_alloc, pause_alloc, repair_alloc)
                    branches = grouped_distribution.get(distribution_key, [])
                    if not branches:
                        continue
                    branch_value = 0.0
                    round_pwin = 0.0
                    expected_used_reset = 0.0
                    expected_used_pause = 0.0
                    expected_used_repair = 0.0

                    for branch in branches:
                        used_reset = int(branch["used_reset"])
                        used_pause = int(branch["used_pause"])
                        used_repair = int(branch["used_repair"])
                        total_prob = branch["total_prob"]
                        expected_used_reset += used_reset * total_prob
                        expected_used_pause += used_pause * total_prob
                        expected_used_repair += used_repair * total_prob
                        round_pwin += branch["win_prob"]
                        if branch["win_prob"] > 0:
                            branch_value += branch["win_prob"] * solve_state(
                                wins_my + 1,
                                wins_opp,
                                reset_left - used_reset,
                                pause_left - used_pause,
                                repair_left - used_repair,
                            )
                        if branch["lose_prob"] > 0:
                            branch_value += branch["lose_prob"] * solve_state(
                                wins_my,
                                wins_opp + 1,
                                reset_left - used_reset,
                                pause_left - used_pause,
                                repair_left - used_repair,
                            )

                    if branch_value > best_value + 1e-12:
                        best_value = branch_value
                        best_alloc = (reset_alloc, pause_alloc, repair_alloc)
                        best_expected_used = (
                            expected_used_reset,
                            expected_used_pause,
                            expected_used_repair,
                        )
                        best_round_pwin = round_pwin

        records_value.append(
            {
                "wins_my": wins_my,
                "wins_opp": wins_opp,
                "scenario_key": scenario_key,
                "scenario": SCENARIO_LABELS[scenario_key],
                "reset_left": reset_left,
                "pause_left": pause_left,
                "repair_left": repair_left,
                "series_value": best_value,
            }
        )
        records_policy.append(
            {
                "wins_my": wins_my,
                "wins_opp": wins_opp,
                "scenario_key": scenario_key,
                "scenario": SCENARIO_LABELS[scenario_key],
                "reset_left": reset_left,
                "pause_left": pause_left,
                "repair_left": repair_left,
                "alloc_reset": best_alloc[0],
                "alloc_pause": best_alloc[1],
                "alloc_repair": best_alloc[2],
                "allocation_label": allocation_label(*best_alloc),
                "round_pwin": best_round_pwin,
                "expected_used_reset": best_expected_used[0],
                "expected_used_pause": best_expected_used[1],
                "expected_used_repair": best_expected_used[2],
                "next_win_state": f"({wins_my + 1},{wins_opp})",
                "next_lose_state": f"({wins_my},{wins_opp + 1})",
            }
        )
        return best_value

    initial_value = solve_state(0, 0, context.config.max_reset, context.config.max_pause, context.config.max_repair)
    value_table = pd.DataFrame(records_value).drop_duplicates().sort_values(
        by=["wins_my", "wins_opp", "reset_left", "pause_left", "repair_left"]
    )
    policy_table = pd.DataFrame(records_policy).drop_duplicates().sort_values(
        by=["wins_my", "wins_opp", "reset_left", "pause_left", "repair_left"]
    )
    return MacroSolution(
        value_table=value_table.reset_index(drop=True),
        policy_table=policy_table.reset_index(drop=True),
        best_initial_win_prob=float(initial_value),
    )


def build_exhaustive_plan(context: Q4Context, pwin_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """穷举法：预先固定三局资源分配方案，选取期望胜率最大的静态计划。"""

    allocations = enumerate_allocations(context.config.max_reset, context.config.max_pause, context.config.max_repair)
    p_lookup = {
        (str(row["scenario_key"]), int(row["reset_alloc"]), int(row["pause_alloc"]), int(row["repair_alloc"])): float(row["p_win"])
        for _, row in pwin_table.iterrows()
    }
    records: list[dict[str, object]] = []
    best_plan: dict[str, object] | None = None
    best_value = -1.0

    for alloc_r1 in allocations:
        for alloc_r2 in allocations:
            for alloc_r3 in allocations:
                total_reset = alloc_r1[0] + alloc_r2[0] + alloc_r3[0]
                total_pause = alloc_r1[1] + alloc_r2[1] + alloc_r3[1]
                total_repair = alloc_r1[2] + alloc_r2[2] + alloc_r3[2]
                if total_reset > context.config.max_reset or total_pause > context.config.max_pause or total_repair > context.config.max_repair:
                    continue
                p1 = p_lookup[("tied", *alloc_r1)]
                p2_lead = p_lookup[("leading", *alloc_r2)]
                p2_trail = p_lookup[("trailing", *alloc_r2)]
                p3 = p_lookup[("tied", *alloc_r3)]
                bo3_value = p1 * p2_lead + p1 * (1.0 - p2_lead) * p3 + (1.0 - p1) * p2_trail * p3
                records.append(
                    {
                        "alloc_round1": allocation_label(*alloc_r1),
                        "alloc_round2": allocation_label(*alloc_r2),
                        "alloc_round3": allocation_label(*alloc_r3),
                        "expected_bo3_win": bo3_value,
                    }
                )
                if bo3_value > best_value + 1e-12:
                    best_value = bo3_value
                    best_plan = {
                        "alloc_round1": alloc_r1,
                        "alloc_round2": alloc_r2,
                        "alloc_round3": alloc_r3,
                        "expected_bo3_win": bo3_value,
                    }

    if best_plan is None:
        raise RuntimeError("穷举静态计划求解失败。")
    return pd.DataFrame(records).sort_values(by="expected_bo3_win", ascending=False).reset_index(drop=True), best_plan
