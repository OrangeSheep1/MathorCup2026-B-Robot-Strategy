"""Q3 单场仿真、蒙特卡洛评估与样例轨迹模块。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.q3.model_v1 import (
    ActionKernel,
    MatchState,
    Q3Environment,
    available_action_ids,
    encode_state,
    sample_transition_event,
)


SCENARIO_INITIAL_STATES = {
    "领先局": MatchState(score_diff=2, time_step=1, health_my=2, health_opp=2),
    "平局局": MatchState(score_diff=0, time_step=1, health_my=2, health_opp=2),
    "落后局": MatchState(score_diff=-2, time_step=1, health_my=2, health_opp=2),
}


@dataclass
class MatchSimulationResult:
    """单场仿真的摘要结果。"""

    method: str
    scenario: str
    final_score_diff: int
    final_health_my: int
    final_health_opp: int
    total_reward: float
    winner_flag: int
    trajectory: pd.DataFrame


def build_action_collapse_diagnostics(policy_table: pd.DataFrame) -> pd.DataFrame:
    """统计 MDP 策略在正常状态下的动作塌缩情况。"""

    base = policy_table[policy_table["recovery_lock"] == 0].copy()
    action_counts = (
        base.groupby(["mdp_action_id", "mdp_action_name", "mdp_macro_group"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(by=["count", "mdp_action_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total = max(int(action_counts["count"].sum()), 1)
    action_counts["share"] = action_counts["count"] / total
    action_counts["cumulative_share"] = action_counts["share"].cumsum()
    action_counts["rank"] = np.arange(1, len(action_counts) + 1)

    macro_counts = (
        base.groupby(["mdp_macro_group"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(by=["count", "mdp_macro_group"], ascending=[False, True])
        .reset_index(drop=True)
    )
    macro_counts["share"] = macro_counts["count"] / max(int(macro_counts["count"].sum()), 1)
    macro_counts["cumulative_share"] = macro_counts["share"].cumsum()
    macro_counts["rank"] = np.arange(1, len(macro_counts) + 1)
    macro_counts["mdp_action_id"] = ""
    macro_counts["mdp_action_name"] = ""
    macro_counts = macro_counts.rename(columns={"mdp_macro_group": "macro_group"})
    action_counts = action_counts.rename(columns={"mdp_macro_group": "macro_group"})

    action_counts["diagnostic_type"] = "action"
    macro_counts["diagnostic_type"] = "macro"
    columns = [
        "diagnostic_type",
        "rank",
        "mdp_action_id",
        "mdp_action_name",
        "macro_group",
        "count",
        "share",
        "cumulative_share",
    ]
    return pd.concat([action_counts[columns], macro_counts[columns]], ignore_index=True)


def build_policy_difference_diagnostics(policy_table: pd.DataFrame) -> pd.DataFrame:
    """统计 MDP 与贪心策略的状态差异率。"""

    base = policy_table[policy_table["recovery_lock"] == 0].copy()
    base["is_different"] = base["mdp_action_id"] != base["greedy_action_id"]

    records: list[dict[str, object]] = []

    def add_record(dimension: str, group_value: object, subset: pd.DataFrame) -> None:
        if subset.empty:
            return
        different_count = int(subset["is_different"].sum())
        records.append(
            {
                "dimension": dimension,
                "group_value": group_value,
                "state_count": int(len(subset)),
                "different_count": different_count,
                "different_share": float(different_count / len(subset)),
            }
        )

    add_record("overall", "all", base)
    for time_phase, subset in base.groupby("time_phase"):
        add_record("time_phase", time_phase, subset)
    for score_diff, subset in base.groupby("score_diff"):
        add_record("score_diff", int(score_diff), subset)
    for health_my, subset in base.groupby("health_my"):
        add_record("health_my", int(health_my), subset)
    return pd.DataFrame(records)


def build_action_dominance_diagnostics(
    env: Q3Environment,
    policy_table: pd.DataFrame,
) -> pd.DataFrame:
    """构造 A01/D08 及全动作的支配诊断表。"""

    action_meta = (
        env.kernel_table[["action_id", "action_name", "action_type", "macro_group"]]
        .drop_duplicates()
        .sort_values(by=["action_type", "action_id"])
        .reset_index(drop=True)
    )
    scopes = {
        "overall_normal": policy_table[policy_table["recovery_lock"] == 0].copy(),
        "main_high_high": policy_table[
            (policy_table["recovery_lock"] == 0)
            & (policy_table["counter_ready"] == 0)
            & (policy_table["health_my"] == 2)
            & (policy_table["health_opp"] == 2)
        ].copy(),
        "trailing_high_high": policy_table[
            (policy_table["recovery_lock"] == 0)
            & (policy_table["counter_ready"] == 0)
            & (policy_table["health_my"] == 2)
            & (policy_table["health_opp"] == 2)
            & (policy_table["score_diff"] < 0)
        ].copy(),
        "counter_ready_high_high": policy_table[
            (policy_table["recovery_lock"] == 0)
            & (policy_table["counter_ready"] == 1)
            & (policy_table["health_my"] == 2)
            & (policy_table["health_opp"] == 2)
        ].copy(),
    }
    metric_columns = [
        "expected_reward",
        "p_score_for",
        "p_score_against",
        "p_self_drop",
        "p_opp_drop",
        "p_fall",
    ]
    records: list[dict[str, object]] = []

    for scope_name, subset in scopes.items():
        if subset.empty:
            continue
        total_states = int(len(subset))
        mdp_counts = subset["mdp_action_id"].value_counts()
        greedy_counts = subset["greedy_action_id"].value_counts()

        available_counts: dict[str, int] = {action_id: 0 for action_id in action_meta["action_id"]}
        grouped_states = subset.groupby(["health_my", "health_opp", "recovery_lock", "counter_ready"]).size().reset_index(name="count")
        weighted_metrics: dict[str, dict[str, float]] = {}

        for _, state_row in grouped_states.iterrows():
            health_my = int(state_row["health_my"])
            health_opp = int(state_row["health_opp"])
            recovery_lock = int(state_row["recovery_lock"])
            counter_ready = int(state_row["counter_ready"])
            state_count = int(state_row["count"])
            available_ids = available_action_ids(env, health_my, recovery_lock, counter_ready)
            kernel_subset = env.kernel_table[
                (env.kernel_table["health_my"] == health_my)
                & (env.kernel_table["health_opp"] == health_opp)
                & (env.kernel_table["counter_ready"] == counter_ready)
                & (env.kernel_table["action_id"].isin(available_ids))
            ]
            for _, kernel_row in kernel_subset.iterrows():
                action_id = str(kernel_row["action_id"])
                available_counts[action_id] += state_count
                action_metrics = weighted_metrics.setdefault(
                    action_id,
                    {"weight_sum": 0.0, **{column: 0.0 for column in metric_columns}},
                )
                action_metrics["weight_sum"] += state_count
                for column in metric_columns:
                    action_metrics[column] += float(kernel_row[column]) * state_count

        for _, meta_row in action_meta.iterrows():
            action_id = str(meta_row["action_id"])
            mdp_selected_count = int(mdp_counts.get(action_id, 0))
            greedy_selected_count = int(greedy_counts.get(action_id, 0))
            available_state_count = int(available_counts.get(action_id, 0))
            metric_bundle = weighted_metrics.get(action_id, {})
            weight_sum = float(metric_bundle.get("weight_sum", 0.0))

            record = {
                "scope": scope_name,
                "state_count": total_states,
                "action_id": action_id,
                "action_name": str(meta_row["action_name"]),
                "action_type": str(meta_row["action_type"]),
                "macro_group": str(meta_row["macro_group"]),
                "available_state_count": available_state_count,
                "available_state_share": float(available_state_count / total_states) if total_states > 0 else 0.0,
                "mdp_selected_count": mdp_selected_count,
                "mdp_selected_share": float(mdp_selected_count / total_states) if total_states > 0 else 0.0,
                "mdp_selected_share_if_available": float(mdp_selected_count / available_state_count)
                if available_state_count > 0
                else 0.0,
                "greedy_selected_count": greedy_selected_count,
                "greedy_selected_share": float(greedy_selected_count / total_states) if total_states > 0 else 0.0,
                "greedy_selected_share_if_available": float(greedy_selected_count / available_state_count)
                if available_state_count > 0
                else 0.0,
                "selection_share_gap": float((mdp_selected_count - greedy_selected_count) / total_states)
                if total_states > 0
                else 0.0,
                "kernel_weight_share": float(weight_sum / total_states) if total_states > 0 else 0.0,
                "is_focus_action": action_id in {"A01", "D08"},
            }
            for column in metric_columns:
                record[column] = float(metric_bundle[column] / weight_sum) if weight_sum > 0 else np.nan
            record["score_margin"] = (
                float(record["p_score_for"]) - float(record["p_score_against"])
                if not pd.isna(record["p_score_for"]) and not pd.isna(record["p_score_against"])
                else np.nan
            )
            records.append(record)

    diagnostic = pd.DataFrame(records)
    if diagnostic.empty:
        return diagnostic
    diagnostic["scope_rank_mdp"] = (
        diagnostic.groupby("scope")["mdp_selected_share"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    diagnostic["scope_rank_greedy"] = (
        diagnostic.groupby("scope")["greedy_selected_share"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    diagnostic = diagnostic.sort_values(
        by=["scope", "action_type", "scope_rank_mdp", "action_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return diagnostic


def _build_policy_lookup(policy_table: pd.DataFrame, action_column: str) -> dict[int, str]:
    """将状态策略表转成快速查询字典。"""

    return policy_table.set_index("state_index")[action_column].to_dict()


def _build_static_distribution(static_strategy: pd.DataFrame) -> dict[str, float]:
    """构造静态博弈混合策略分布。"""

    return static_strategy.set_index("action_id")["static_prob"].to_dict()


def _sample_static_action(
    env: Q3Environment,
    state: MatchState,
    rng: np.random.Generator,
    static_prob_map: dict[str, float],
) -> str:
    """按混合策略在当前可用动作中采样。"""

    available_ids = available_action_ids(env, state.health_my, state.recovery_lock, state.counter_ready)
    probabilities = np.asarray([float(static_prob_map.get(action_id, 0.0)) for action_id in available_ids], dtype=float)
    if np.isclose(float(probabilities.sum()), 0.0):
        probabilities = np.full(len(available_ids), 1.0 / len(available_ids), dtype=float)
    else:
        probabilities = probabilities / probabilities.sum()
    return str(rng.choice(available_ids, p=probabilities))


def _build_attack_event_kernel(
    env: Q3Environment,
    state: MatchState,
    action_id: str,
    pair_row: pd.Series,
) -> ActionKernel:
    """针对显式攻防对构造攻击事件核。"""

    attack_row = env.attack_table.loc[env.attack_table["action_id"] == action_id].iloc[0]
    p_score = float(attack_row["score_prob"]) * (1.0 - float(pair_row["p_block"]))
    p_counter = float(pair_row["counter_prob_effective"])
    counter_ready_bonus = 0.0
    if state.counter_ready > 0:
        counter_ready_bonus = float(attack_row.get("counter_ready_bonus", env.config.counter_ready_bonus_other))
        p_score = p_score * (1.0 + counter_ready_bonus)
    total_score_prob = p_score + p_counter
    if total_score_prob > 1.0:
        p_score = p_score / total_score_prob
        p_counter = p_counter / total_score_prob

    p_energy_drop = 0.0
    if state.health_my > 0:
        if state.health_my == 2:
            p_energy_drop = float(attack_row["energy_drop_high"])
        else:
            p_energy_drop = float(attack_row["energy_drop_mid"])
    p_opp_drop = 0.0
    if state.health_opp > 0:
        p_opp_drop = min(
            1.0,
            p_score * float(attack_row["tau_norm"]) * env.config.impact_to_health_factor,
        )
    return ActionKernel(
        action_id=action_id,
        action_name=str(attack_row["action_name"]),
        action_type="attack",
        macro_group=str(attack_row["macro_group"]),
        health_my=state.health_my,
        health_opp=state.health_opp,
        expected_reward=0.0,
        p_score_for=float(np.clip(p_score, 0.0, 1.0)),
        p_score_against=float(np.clip(p_counter, 0.0, 1.0)),
        p_self_drop=float(np.clip(p_energy_drop, 0.0, 1.0)),
        p_opp_drop=float(np.clip(p_opp_drop, 0.0, 1.0)),
        p_fall=float(np.clip(float(attack_row["attack_fall_risk"]), 0.0, 1.0)),
        preferred_counter_action="",
        preferred_counter_name="",
        counter_ready_bonus=float(counter_ready_bonus),
    )


def _sample_attack_step(
    env: Q3Environment,
    state: MatchState,
    action_id: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    """执行一次攻击动作采样。"""

    attack_row = env.attack_table.loc[env.attack_table["action_id"] == action_id].iloc[0]
    opponent_candidates: list[str] = []
    opponent_weights: list[float] = []
    for rank in (1, 2, 3):
        defense_id = str(attack_row.get(f"opp_defense_id_r{rank}", "")).strip()
        defense_weight = float(attack_row.get(f"opp_defense_weight_r{rank}", 0.0) or 0.0)
        if defense_id and defense_id.lower() != "nan" and defense_weight > 0.0:
            opponent_candidates.append(defense_id)
            opponent_weights.append(defense_weight)
    if not opponent_candidates:
        opponent_candidates = list(env.defense_ids)
        opponent_weights = [1.0 / len(opponent_candidates)] * len(opponent_candidates)
    else:
        total_weight = float(sum(opponent_weights))
        opponent_weights = [weight / total_weight for weight in opponent_weights]

    opponent_defense_id = str(rng.choice(opponent_candidates, p=np.asarray(opponent_weights, dtype=float)))
    pair_row = env.pair_lookup.loc[(action_id, opponent_defense_id)]
    kernel = _build_attack_event_kernel(env, state, action_id, pair_row)
    sampled = sample_transition_event(state, kernel, env.config, rng)
    reward = (
        env.config.reward_score * int(sampled["delta_score"])
        + env.config.reward_health * int(sampled["opp_drop"])
        - env.config.reward_cost * int(sampled["self_drop"])
        - env.config.reward_fall * int(sampled["fall_event"])
    )

    return {
        "action_id": action_id,
        "action_name": str(attack_row["action_name"]),
        "action_type": "attack",
        "macro_group": str(attack_row["macro_group"]),
        "opponent_action_id": opponent_defense_id,
        "opponent_action_name": str(pair_row["defense_name"]),
        "counter_action_id": str(pair_row["counter_action_id"]),
        "counter_action_name": str(pair_row["counter_action_name"]),
        "delta_score": int(sampled["delta_score"]),
        "self_drop": int(sampled["self_drop"]),
        "opp_drop": int(sampled["opp_drop"]),
        "fall_event": int(sampled["fall_event"]),
        "reward": float(reward),
        "next_state": sampled["next_state"],
    }


def _build_defense_event_kernel(
    env: Q3Environment,
    state: MatchState,
    action_id: str,
    pair_row: pd.Series,
) -> ActionKernel:
    """针对显式攻防对构造防守事件核。"""

    defense_row = env.defense_table.loc[env.defense_table["defense_id"] == action_id].iloc[0]
    p_counter = float(pair_row["counter_prob_effective"])
    p_opp_score = float(pair_row["score_prob"]) * (1.0 - float(pair_row["p_block"]))
    total_score_prob = p_counter + p_opp_score
    if total_score_prob > 1.0:
        p_counter = p_counter / total_score_prob
        p_opp_score = p_opp_score / total_score_prob

    p_self_drop = 0.0 if state.health_my <= 0 else float(pair_row["defense_damage"])
    p_opp_drop = 0.0
    if state.health_opp > 0:
        p_opp_drop = min(
            1.0,
            p_counter * float(pair_row["counter_tau_norm"]) * env.config.impact_to_health_factor,
        )
    return ActionKernel(
        action_id=action_id,
        action_name=str(defense_row["defense_name"]),
        action_type="defense",
        macro_group=str(defense_row["macro_group"]),
        health_my=state.health_my,
        health_opp=state.health_opp,
        expected_reward=0.0,
        p_score_for=float(np.clip(p_counter, 0.0, 1.0)),
        p_score_against=float(np.clip(p_opp_score, 0.0, 1.0)),
        p_self_drop=float(np.clip(p_self_drop, 0.0, 1.0)),
        p_opp_drop=float(np.clip(p_opp_drop, 0.0, 1.0)),
        p_fall=float(np.clip(float(pair_row["p_fall"]), 0.0, 1.0)),
        preferred_counter_action=str(pair_row["counter_action_id"]),
        preferred_counter_name=str(pair_row["counter_action_name"]),
        counter_ready_bonus=0.0,
    )


def _sample_defense_step(
    env: Q3Environment,
    state: MatchState,
    action_id: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    """执行一次防守动作采样。"""

    defense_row = env.defense_table.loc[env.defense_table["defense_id"] == action_id].iloc[0]
    opponent_attack_ids = env.attack_ids_by_health[state.health_opp]
    opponent_attack_id = str(rng.choice(opponent_attack_ids))
    pair_row = env.pair_lookup.loc[(opponent_attack_id, action_id)]
    kernel = _build_defense_event_kernel(env, state, action_id, pair_row)
    sampled = sample_transition_event(state, kernel, env.config, rng)
    reward = (
        env.config.reward_score * int(sampled["delta_score"])
        + env.config.reward_health * int(sampled["opp_drop"])
        - env.config.reward_cost * int(sampled["self_drop"])
        - env.config.reward_fall * int(sampled["fall_event"])
    )
    return {
        "action_id": action_id,
        "action_name": str(defense_row["defense_name"]),
        "action_type": "defense",
        "macro_group": str(defense_row["macro_group"]),
        "opponent_action_id": opponent_attack_id,
        "opponent_action_name": str(pair_row["action_name"]),
        "counter_action_id": str(pair_row["counter_action_id"]),
        "counter_action_name": str(pair_row["counter_action_name"]),
        "delta_score": int(sampled["delta_score"]),
        "self_drop": int(sampled["self_drop"]),
        "opp_drop": int(sampled["opp_drop"]),
        "fall_event": int(sampled["fall_event"]),
        "reward": float(reward),
        "next_state": sampled["next_state"],
    }


def simulate_match(
    env: Q3Environment,
    policy_lookup: dict[int, str],
    method_name: str,
    scenario_name: str,
    initial_state: MatchState,
    rng: np.random.Generator,
    static_prob_map: dict[str, float] | None = None,
) -> MatchSimulationResult:
    """按给定策略执行一场单场仿真。"""

    state = initial_state
    records: list[dict[str, object]] = []
    total_reward = 0.0

    for _ in range(env.config.n_time_steps):
        state_index = encode_state(state)
        if method_name == "static":
            if static_prob_map is None:
                raise ValueError("静态博弈仿真缺少混合策略分布。")
            action_id = _sample_static_action(env, state, rng, static_prob_map)
        else:
            action_id = str(policy_lookup[state_index])
        if action_id.startswith("A"):
            event = _sample_attack_step(env, state, action_id, rng)
        else:
            event = _sample_defense_step(env, state, action_id, rng)

        next_state = event["next_state"]
        total_reward += float(event["reward"])
        records.append(
            {
                "method": method_name,
                "scenario": scenario_name,
                "time_step": state.time_step,
                "state_index": state_index,
                "score_diff_before": state.score_diff,
                "health_my_before": state.health_my,
                "health_opp_before": state.health_opp,
                "recovery_lock_before": state.recovery_lock,
                "counter_ready_before": state.counter_ready,
                "action_id": event["action_id"],
                "action_name": event["action_name"],
                "action_type": event["action_type"],
                "macro_group": event["macro_group"],
                "opponent_action_id": event["opponent_action_id"],
                "opponent_action_name": event["opponent_action_name"],
                "counter_action_id": event["counter_action_id"],
                "counter_action_name": event["counter_action_name"],
                "delta_score": event["delta_score"],
                "self_drop": event["self_drop"],
                "opp_drop": event["opp_drop"],
                "fall_event": event["fall_event"],
                "step_reward": event["reward"],
                "score_diff_after": next_state.score_diff,
                "health_my_after": next_state.health_my,
                "health_opp_after": next_state.health_opp,
                "recovery_lock_after": next_state.recovery_lock,
                "counter_ready_after": next_state.counter_ready,
            }
        )
        state = next_state
        if state.time_step > env.config.n_time_steps:
            break

    total_reward += (
        env.config.terminal_reward_win
        if state.score_diff > 0
        else env.config.terminal_reward_loss
        if state.score_diff < 0
        else 0.0
    )
    return MatchSimulationResult(
        method=method_name,
        scenario=scenario_name,
        final_score_diff=state.score_diff,
        final_health_my=state.health_my,
        final_health_opp=state.health_opp,
        total_reward=float(total_reward),
        winner_flag=1 if state.score_diff > 0 else 0,
        trajectory=pd.DataFrame(records),
    )


def _confidence_interval(successes: int, total: int) -> tuple[float, float]:
    """计算胜率的 95% 置信区间。"""

    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    margin = 1.96 * math.sqrt(proportion * (1.0 - proportion) / total) if 0.0 < proportion < 1.0 else 0.0
    return max(0.0, proportion - margin), min(1.0, proportion + margin)


def _winrate_pvalue(success_base: int, total_base: int, success_ref: int, total_ref: int) -> float:
    """计算两比例 z 检验的近似双侧 p 值。"""

    if total_base <= 0 or total_ref <= 0:
        return 1.0
    p_base = success_base / total_base
    p_ref = success_ref / total_ref
    pooled = (success_base + success_ref) / (total_base + total_ref)
    denominator = math.sqrt(max(pooled * (1.0 - pooled) * (1.0 / total_base + 1.0 / total_ref), 1e-12))
    z_score = (p_ref - p_base) / denominator
    return float(2.0 * (1.0 - norm.cdf(abs(z_score))))


def pvalue_to_stars(p_value: float) -> str:
    """将 p 值映射为显著性标记。"""

    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _compare_method_trajectories(
    scenario_name: str,
    match_index: int,
    match_seed: int,
    greedy_result: MatchSimulationResult,
    mdp_result: MatchSimulationResult,
) -> dict[str, object]:
    """比较同一种子下贪心与 MDP 轨迹的差异。"""

    greedy_traj = greedy_result.trajectory[
        ["time_step", "action_id", "macro_group", "score_diff_after", "fall_event", "step_reward"]
    ].copy()
    mdp_traj = mdp_result.trajectory[
        ["time_step", "action_id", "macro_group", "score_diff_after", "fall_event", "step_reward"]
    ].copy()
    merged = greedy_traj.merge(
        mdp_traj,
        on="time_step",
        how="inner",
        suffixes=("_greedy", "_mdp"),
    )
    action_diff_steps = int((merged["action_id_greedy"] != merged["action_id_mdp"]).sum())
    macro_diff_steps = int((merged["macro_group_greedy"] != merged["macro_group_mdp"]).sum())
    fall_diff_steps = int((merged["fall_event_greedy"] != merged["fall_event_mdp"]).sum())
    score_path_l1 = float(np.abs(merged["score_diff_after_greedy"] - merged["score_diff_after_mdp"]).sum())
    final_score_gap = int(mdp_result.final_score_diff - greedy_result.final_score_diff)
    reward_gap = float(mdp_result.total_reward - greedy_result.total_reward)
    win_gap = int(mdp_result.winner_flag - greedy_result.winner_flag)
    return {
        "scenario": scenario_name,
        "match_index": match_index,
        "match_seed": match_seed,
        "action_diff_steps": action_diff_steps,
        "macro_diff_steps": macro_diff_steps,
        "fall_diff_steps": fall_diff_steps,
        "score_path_l1": score_path_l1,
        "final_score_gap": final_score_gap,
        "reward_gap": reward_gap,
        "win_gap": win_gap,
    }


def _select_representative_samples(sample_selection: pd.DataFrame) -> pd.DataFrame:
    """从候选样例中选出场景代表样例和全局绘图样例。"""

    if sample_selection.empty:
        return sample_selection
    ranked = sample_selection.copy()
    ranked["is_scenario_representative"] = False
    ranked["is_plot_sample"] = False

    order_columns = ["win_gap", "final_score_gap", "reward_gap", "action_diff_steps", "score_path_l1"]
    ascending = [False, False, False, False, False]

    for scenario_name in ranked["scenario"].drop_duplicates().tolist():
        subset = ranked[ranked["scenario"] == scenario_name].copy()
        best_index = subset.sort_values(by=order_columns + ["match_index"], ascending=ascending + [True]).index[0]
        ranked.loc[best_index, "is_scenario_representative"] = True

    representatives = ranked[ranked["is_scenario_representative"]].copy()
    plot_index = representatives.sort_values(by=order_columns + ["match_index"], ascending=ascending + [True]).index[0]
    ranked.loc[plot_index, "is_plot_sample"] = True
    return ranked


def build_recovery_diagnostics(trajectory_table: pd.DataFrame, recovery_defense_ids: tuple[str, ...]) -> pd.DataFrame:
    """统计恢复机制是否真的在仿真中起作用。"""

    if trajectory_table.empty:
        return pd.DataFrame()
    data = trajectory_table.copy()
    data["is_recovery_action"] = data["action_id"].isin(recovery_defense_ids)
    records: list[dict[str, object]] = []
    for (scenario, method), subset in data.groupby(["scenario", "method"]):
        step_count = int(len(subset))
        fall_count = int(subset["fall_event"].sum())
        recovery_entry_count = int(subset["recovery_lock_after"].sum())
        recovery_step_count = int(subset["recovery_lock_before"].sum())
        if recovery_step_count > 0:
            recovery_action_match_rate = float(
                subset.loc[subset["recovery_lock_before"] == 1, "is_recovery_action"].mean()
            )
        else:
            recovery_action_match_rate = np.nan
        fall_to_recovery_rate = float(recovery_entry_count / fall_count) if fall_count > 0 else np.nan
        records.append(
            {
                "scenario": scenario,
                "method": method,
                "step_count": step_count,
                "fall_count": fall_count,
                "fall_rate": float(fall_count / step_count) if step_count > 0 else 0.0,
                "recovery_entry_count": recovery_entry_count,
                "recovery_entry_rate": float(recovery_entry_count / step_count) if step_count > 0 else 0.0,
                "recovery_step_count": recovery_step_count,
                "recovery_action_match_rate": recovery_action_match_rate,
                "fall_to_recovery_rate": fall_to_recovery_rate,
            }
        )
    return pd.DataFrame(records)


def build_condition_performance_metrics(env: Q3Environment, trajectory_table: pd.DataFrame) -> pd.DataFrame:
    """统计困难状态表现与序列合理性指标。"""

    if trajectory_table.empty:
        return pd.DataFrame()

    attack_meta = env.attack_table[["action_id", "counter_attack_eligible", "counter_role_score"]].copy()
    data = trajectory_table.merge(attack_meta, on="action_id", how="left")
    data["counter_attack_eligible"] = data["counter_attack_eligible"].astype("boolean").fillna(False).astype(bool)
    data["counter_role_score"] = data["counter_role_score"].fillna(0.0)

    match_level = (
        data.groupby(["scenario", "method", "match_index", "match_seed"], as_index=False)
        .agg(
            initial_score_diff=("score_diff_before", "first"),
            final_score_diff=("score_diff_after", "last"),
            total_reward=("step_reward", "sum"),
        )
    )
    match_level["score_gain"] = match_level["final_score_diff"] - match_level["initial_score_diff"]
    match_level["tie_or_better"] = match_level["final_score_diff"] >= 0
    match_level["reverse_win"] = match_level["initial_score_diff"] < 0
    match_level["reverse_win"] = match_level["reverse_win"] & (match_level["final_score_diff"] > 0)

    records: list[dict[str, object]] = []
    for (scenario, method), subset in data.groupby(["scenario", "method"]):
        step_count = int(len(subset))
        ready_steps = subset[subset["counter_ready_before"] == 1].copy()
        ready_entry_count = int(subset["counter_ready_after"].sum())
        ready_attack_steps = ready_steps[ready_steps["action_type"] == "attack"].copy()

        match_subset = match_level[
            (match_level["scenario"] == scenario)
            & (match_level["method"] == method)
        ].copy()
        records.append(
            {
                "scenario": scenario,
                "method": method,
                "match_count": int(len(match_subset)),
                "mean_final_score_diff": float(match_subset["final_score_diff"].mean()),
                "mean_score_gain": float(match_subset["score_gain"].mean()),
                "tie_or_better_rate": float(match_subset["tie_or_better"].mean()),
                "reverse_win_rate": float(match_subset["reverse_win"].mean()),
                "counter_ready_entry_rate": float(ready_entry_count / step_count) if step_count > 0 else 0.0,
                "counter_ready_step_rate": float(len(ready_steps) / step_count) if step_count > 0 else 0.0,
                "counter_ready_attack_rate": float((ready_steps["action_type"] == "attack").mean())
                if not ready_steps.empty
                else np.nan,
                "counter_ready_focus_attack_rate": float(ready_attack_steps["counter_attack_eligible"].mean())
                if not ready_attack_steps.empty
                else np.nan,
                "counter_ready_mean_delta_score": float(ready_steps["delta_score"].mean())
                if not ready_steps.empty
                else np.nan,
                "counter_ready_mean_role_score": float(ready_attack_steps["counter_role_score"].mean())
                if not ready_attack_steps.empty
                else np.nan,
            }
        )
    return pd.DataFrame(records)


def run_monte_carlo(
    env: Q3Environment,
    policy_table: pd.DataFrame,
    static_strategy: pd.DataFrame,
    n_matches: int = 1000,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对四种策略进行蒙特卡洛对比，并输出诊断表与代表性样例。"""

    policy_columns = {
        "greedy": "greedy_action_id",
        "static": "static_action_id",
        "rule": "rule_action_id",
        "mdp": "mdp_action_id",
    }
    policy_lookups = {
        method: _build_policy_lookup(policy_table, column)
        for method, column in policy_columns.items()
    }
    static_prob_map = _build_static_distribution(static_strategy)

    summary_records: list[dict[str, object]] = []
    comparison_records: list[dict[str, object]] = []
    full_trajectories: list[pd.DataFrame] = []
    rng_master = np.random.default_rng(seed)

    for scenario_name, initial_state in SCENARIO_INITIAL_STATES.items():
        method_results: dict[str, list[MatchSimulationResult]] = {}
        for method_name in policy_lookups:
            method_results[method_name] = []

        for match_index in range(n_matches):
            match_seed = int(rng_master.integers(0, 2**32 - 1))
            match_result_map: dict[str, MatchSimulationResult] = {}
            for method_name, lookup in policy_lookups.items():
                result = simulate_match(
                    env=env,
                    policy_lookup=lookup,
                    method_name=method_name,
                    scenario_name=scenario_name,
                    initial_state=initial_state,
                    rng=np.random.default_rng(match_seed),
                    static_prob_map=static_prob_map if method_name == "static" else None,
                )
                result.trajectory["match_index"] = match_index
                result.trajectory["match_seed"] = match_seed
                method_results[method_name].append(result)
                match_result_map[method_name] = result
                full_trajectories.append(result.trajectory)

            comparison_records.append(
                _compare_method_trajectories(
                    scenario_name=scenario_name,
                    match_index=match_index,
                    match_seed=match_seed,
                    greedy_result=match_result_map["greedy"],
                    mdp_result=match_result_map["mdp"],
                )
            )

        mdp_success = sum(item.winner_flag for item in method_results["mdp"])
        for method_name, results in method_results.items():
            wins = sum(item.winner_flag for item in results)
            total = len(results)
            mean_score_diff = float(np.mean([item.final_score_diff for item in results]))
            mean_health_my = float(np.mean([item.final_health_my for item in results]))
            mean_health_opp = float(np.mean([item.final_health_opp for item in results]))
            mean_reward = float(np.mean([item.total_reward for item in results]))
            ci_low, ci_high = _confidence_interval(wins, total)
            p_value = 1.0 if method_name == "mdp" else _winrate_pvalue(wins, total, mdp_success, len(method_results["mdp"]))
            summary_records.append(
                {
                    "scenario": scenario_name,
                    "method": method_name,
                    "match_count": total,
                    "win_count": wins,
                    "win_rate": wins / total,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "mean_score_diff": mean_score_diff,
                    "mean_health_my": mean_health_my,
                    "mean_health_opp": mean_health_opp,
                    "mean_reward": mean_reward,
                    "p_value_vs_mdp": p_value,
                    "significance_vs_mdp": pvalue_to_stars(p_value),
                    "test_method": "two_proportion_z",
                }
            )

    metrics_table = pd.DataFrame(summary_records)
    full_trajectory_table = pd.concat(full_trajectories, ignore_index=True) if full_trajectories else pd.DataFrame()
    sample_selection = _select_representative_samples(pd.DataFrame(comparison_records))
    action_collapse = build_action_collapse_diagnostics(policy_table)
    policy_difference = build_policy_difference_diagnostics(policy_table)
    recovery_diagnostics = build_recovery_diagnostics(full_trajectory_table, env.recovery_defense_ids)
    condition_metrics = build_condition_performance_metrics(env, full_trajectory_table)

    if not sample_selection.empty and not full_trajectory_table.empty:
        sample_row = sample_selection[sample_selection["is_plot_sample"]].iloc[0]
        trajectory_sample = full_trajectory_table[
            (full_trajectory_table["scenario"] == sample_row["scenario"])
            & (full_trajectory_table["match_index"] == sample_row["match_index"])
            & (full_trajectory_table["match_seed"] == sample_row["match_seed"])
            & (full_trajectory_table["method"].isin(["greedy", "mdp"]))
        ].copy()
        trajectory_sample["is_plot_sample"] = True
    else:
        trajectory_sample = pd.DataFrame()

    return (
        metrics_table,
        trajectory_sample,
        action_collapse,
        policy_difference,
        recovery_diagnostics,
        sample_selection,
        condition_metrics,
    )
