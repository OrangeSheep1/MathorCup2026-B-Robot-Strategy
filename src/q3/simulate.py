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

    available_ids = available_action_ids(env, state.health_my, state.recovery_lock)
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
    sampled = sample_transition_event(state, kernel, rng)
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
    sampled = sample_transition_event(state, kernel, rng)
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


def run_monte_carlo(
    env: Q3Environment,
    policy_table: pd.DataFrame,
    static_strategy: pd.DataFrame,
    n_matches: int = 1000,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对四种策略进行蒙特卡洛对比。"""

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
    sample_trajectories: list[pd.DataFrame] = []
    rng_master = np.random.default_rng(seed)

    for scenario_name, initial_state in SCENARIO_INITIAL_STATES.items():
        method_results: dict[str, list[MatchSimulationResult]] = {}
        for method_name in policy_lookups:
            method_results[method_name] = []

        for match_index in range(n_matches):
            match_seed = int(rng_master.integers(0, 2**32 - 1))
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
                method_results[method_name].append(result)
                if scenario_name == "平局局" and method_name in {"greedy", "mdp"} and match_index == 0:
                    sample_trajectories.append(result.trajectory)

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
    trajectory_sample = pd.concat(sample_trajectories, ignore_index=True) if sample_trajectories else pd.DataFrame()
    return metrics_table, trajectory_sample
