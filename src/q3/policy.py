"""Q3 四种策略方法与 MDP 求解模块。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.q3.model_v1 import (
    HEALTH_LEVELS,
    MatchState,
    Q3Environment,
    SCORE_DIFF_VALUES,
    available_action_ids,
    clamp_score_diff,
    get_kernel,
    iter_transition_branches,
)


@dataclass
class PolicyArtifacts:
    """Q3 策略求解后的主要输出。"""

    policy_table: pd.DataFrame
    static_strategy: pd.DataFrame
    scenario_summary: pd.DataFrame
    qvalue_summary: pd.DataFrame
    state_reward_decomposition: pd.DataFrame
    value_table: np.ndarray


def state_reward_weights(state: MatchState, config) -> dict[str, float]:
    """根据分差、时间和机能返回状态敏感奖励权重。"""

    weights = {
        "score_for": 1.00,
        "score_against": 1.00,
        "health": 0.20,
        "cost": 0.20,
        "fall": 0.67,
        "prevent": 0.35,
        "counter": 0.20,
        "time_control": 0.0,
        "probe": 0.04,
        "counter_setup": 0.06,
    }
    is_late = state.time_step >= int(config.n_time_steps * 0.75)
    is_mid_late = state.time_step >= int(config.n_time_steps * 0.50)

    if state.score_diff >= 2 and is_late:
        weights.update(
            {
                "score_for": 0.65,
                "score_against": 1.45,
                "cost": 0.35,
                "fall": 1.10,
                "prevent": 0.90,
                "counter": 0.25,
                "time_control": 0.38,
                "probe": 0.02,
                "counter_setup": 0.08,
            }
        )
    elif state.score_diff <= -2 and is_late:
        weights.update(
            {
                "score_for": 1.45,
                "score_against": 1.05,
                "health": 0.35,
                "cost": 0.12,
                "fall": 0.75,
                "prevent": 0.20,
                "counter": 0.45,
                "probe": 0.00,
                "counter_setup": 0.04,
            }
        )
    elif state.score_diff == 0 and is_mid_late:
        weights.update(
            {
                "score_for": 1.45,
                "score_against": 0.98,
                "fall": 0.78,
                "prevent": 0.22,
                "counter": 0.24,
                "probe": 0.05,
                "counter_setup": 0.05,
            }
        )

    if state.health_my <= 0:
        weights["cost"] *= 1.35
        weights["fall"] *= 1.25
        weights["prevent"] *= 1.20
    elif state.health_my == 1:
        weights["cost"] *= 1.12
        weights["fall"] *= 1.10

    if state.counter_ready > 0:
        weights["score_for"] *= 1.08
        weights["counter"] *= 1.35
        weights["probe"] *= 0.80
        weights["counter_setup"] *= 1.25

    return weights


def state_adjusted_reward(env: Q3Environment, state: MatchState, kernel) -> float:
    """计算状态敏感即时奖励，用于贪心和 MDP 的一致比较。"""

    return state_reward_breakdown(env, state, kernel)["total_reward"]


def state_reward_breakdown(env: Q3Environment, state: MatchState, kernel) -> dict[str, float]:
    """输出状态-动作层面的唯一主奖励分解。"""

    weights = state_reward_weights(state, env.config)
    if kernel.action_type == "attack":
        is_probe_window = (
            state.time_step <= int(env.config.n_time_steps * 0.35)
            or (abs(state.score_diff) <= 1 and state.time_step <= int(env.config.n_time_steps * 0.70))
        )
        low_risk_value = (
            max(0.0, 1.0 - float(kernel.p_fall))
            * max(0.0, 1.0 - float(kernel.p_self_drop))
            * max(0.0, 1.0 - float(kernel.p_score_against))
        )
        probe_term = (
            weights["probe"] * low_risk_value
            if kernel.macro_group == "试探攻击" and is_probe_window
            else 0.0
        )
        score_for_term = weights["score_for"] * kernel.p_score_for
        score_against_term = -weights["score_against"] * kernel.p_score_against
        health_term = weights["health"] * kernel.p_opp_drop
        cost_term = -weights["cost"] * kernel.p_self_drop
        fall_term = -weights["fall"] * kernel.p_fall
        counter_bonus_term = weights["counter"] * kernel.counter_bonus_term
        total_reward = (
            score_for_term
            + score_against_term
            + health_term
            + cost_term
            + fall_term
            + counter_bonus_term
            + probe_term
        )
        return {
            "score_for_term": float(score_for_term),
            "score_against_term": float(score_against_term),
            "health_term": float(health_term),
            "cost_term": float(cost_term),
            "fall_term": float(fall_term),
            "counter_bonus_term": float(counter_bonus_term),
            "probe_term": float(probe_term),
            "prevent_term": 0.0,
            "counter_term": 0.0,
            "counter_setup_term": 0.0,
            "time_control_term": 0.0,
            "total_reward": float(total_reward),
        }

    lead_factor = max(state.score_diff, 0) / max(abs(SCORE_DIFF_VALUES[0]), 1)
    time_factor = state.time_step / env.config.n_time_steps
    time_control_term = weights["time_control"] * lead_factor * time_factor
    counter_term = weights["counter"] * kernel.p_score_for
    has_counter_followup = bool(str(kernel.preferred_counter_action).strip())
    counter_setup_base = max(float(kernel.p_score_for), float(kernel.prevented_score_prob))
    counter_setup_term = (
        weights["counter_setup"] * counter_setup_base
        if kernel.macro_group == "反击防守" or has_counter_followup
        else 0.0
    )
    score_against_term = -weights["score_against"] * kernel.p_score_against
    prevent_term = weights["prevent"] * kernel.prevent_score_term
    health_term = weights["health"] * kernel.p_opp_drop
    cost_term = -weights["cost"] * kernel.p_self_drop
    fall_term = -weights["fall"] * kernel.p_fall
    total_reward = (
        counter_term
        + score_against_term
        + prevent_term
        + health_term
        + time_control_term
        + counter_setup_term
        + cost_term
        + fall_term
    )
    return {
        "score_for_term": 0.0,
        "score_against_term": float(score_against_term),
        "health_term": float(health_term),
        "cost_term": float(cost_term),
        "fall_term": float(fall_term),
        "counter_bonus_term": 0.0,
        "probe_term": 0.0,
        "prevent_term": float(prevent_term),
        "counter_term": float(counter_term),
        "counter_setup_term": float(counter_setup_term),
        "time_control_term": float(time_control_term),
        "total_reward": float(total_reward),
    }


def _build_immediate_reward_table(env: Q3Environment) -> pd.DataFrame:
    """提取各机能状态下的动作即时奖励。"""

    return env.kernel_table[
        [
            "health_my",
            "health_opp",
            "action_id",
            "action_name",
            "action_type",
            "macro_group",
            "expected_reward",
            "p_score_for",
            "p_score_against",
            "p_self_drop",
            "p_opp_drop",
            "p_fall",
            "preferred_counter_action",
            "preferred_counter_name",
        ]
    ].copy()


def _choose_best_action(
    env: Q3Environment,
    state: MatchState,
    action_ids: tuple[str, ...],
    score_column: str = "expected_reward",
) -> tuple[str, float]:
    """在可用动作集中选出指定评分最高的动作。"""

    subset = env.kernel_table[
        (env.kernel_table["health_my"] == state.health_my)
        & (env.kernel_table["health_opp"] == state.health_opp)
        & (env.kernel_table["counter_ready"] == state.counter_ready)
        & (env.kernel_table["action_id"].isin(action_ids))
    ].copy()
    if subset.empty:
        return "", 0.0
    if score_column == "state_adjusted_reward":
        subset["state_adjusted_reward"] = [
            state_adjusted_reward(env, state, get_kernel(env, state, str(action_id)))
            for action_id in subset["action_id"]
        ]
    row = subset.sort_values(
        by=[score_column, "p_score_for", "action_id"],
        ascending=[False, False, True],
    ).iloc[0]
    return str(row["action_id"]), float(row[score_column])


def build_greedy_policy(env: Q3Environment) -> pd.DataFrame:
    """方法一：贪心策略。"""

    records: list[dict[str, object]] = []
    for _, row in env.state_table.iterrows():
        state = MatchState(
            score_diff=int(row["score_diff"]),
            time_step=int(row["time_step"]),
            health_my=int(row["health_my"]),
            health_opp=int(row["health_opp"]),
            recovery_lock=int(row["recovery_lock"]),
            counter_ready=int(row["counter_ready"]),
        )
        action_id, expected_reward = _choose_best_action(
            env,
            state,
            available_action_ids(env, state.health_my, state.recovery_lock, state.counter_ready),
            score_column="state_adjusted_reward",
        )
        records.append(
            {
                "state_index": int(row["state_index"]),
                "greedy_action_id": action_id,
                "greedy_expected_reward": expected_reward,
            }
        )
    return pd.DataFrame(records)


def _attack_attack_payoff(row_attack: pd.Series, col_attack: pd.Series, env: Q3Environment) -> float:
    """近似计算攻击对攻击的静态收益。"""

    config = env.config
    row_score = float(row_attack["score_prob"])
    col_score = float(col_attack["score_prob"])
    row_health = row_score * float(row_attack["tau_norm"]) * config.impact_to_health_factor
    col_health = col_score * float(col_attack["tau_norm"]) * config.impact_to_health_factor
    row_cost = float(row_attack["energy_drop_high"])
    col_cost = float(col_attack["energy_drop_high"])
    row_fall = float(row_attack["attack_fall_risk"])
    col_fall = float(col_attack["attack_fall_risk"])
    return (
        config.reward_score * (row_score - col_score)
        + config.reward_health * (row_health - col_health)
        - config.reward_cost * (row_cost - col_cost)
        - config.reward_fall * (row_fall - col_fall)
    )


def _attack_defense_payoff(pair_row: pd.Series, attack_row: pd.Series, env: Q3Environment) -> float:
    """近似计算攻击对防守的静态收益。"""

    config = env.config
    p_score = float(attack_row["score_prob"]) * (1.0 - float(pair_row["p_block"]))
    p_score_against = float(pair_row["counter_prob_effective"])
    p_opp_drop = p_score * float(attack_row["tau_norm"]) * config.impact_to_health_factor
    p_self_drop = float(attack_row["energy_drop_high"])
    p_fall = float(attack_row["attack_fall_risk"])
    return (
        config.reward_score * (p_score - p_score_against)
        + config.reward_health * p_opp_drop
        - config.reward_cost * p_self_drop
        - config.reward_fall * p_fall
    )


def _defense_attack_payoff(pair_row: pd.Series, env: Q3Environment) -> float:
    """近似计算防守对攻击的静态收益。"""

    config = env.config
    p_score_for = float(pair_row["counter_prob_effective"])
    p_score_against = float(pair_row["score_prob"]) * (1.0 - float(pair_row["p_block"]))
    p_opp_drop = p_score_for * float(pair_row["counter_tau_norm"]) * config.impact_to_health_factor
    p_self_drop = float(pair_row["defense_damage"])
    p_fall = float(pair_row["p_fall"])
    return (
        config.reward_score * (p_score_for - p_score_against)
        + config.reward_health * p_opp_drop
        - config.reward_cost * p_self_drop
        - config.reward_fall * p_fall
    )


def build_static_payoff_matrix(env: Q3Environment) -> tuple[np.ndarray, list[str]]:
    """构建静态矩阵博弈收益矩阵。"""

    actions = env.action_table["action_id"].tolist()
    attacks = env.attack_table.set_index("action_id")
    neutral_pairs = env.pair_table.copy().set_index(["action_id", "defense_id"])
    payoff = np.zeros((len(actions), len(actions)), dtype=float)

    for row_index, row_action in enumerate(actions):
        for col_index, col_action in enumerate(actions):
            row_is_attack = row_action.startswith("A")
            col_is_attack = col_action.startswith("A")
            if row_is_attack and col_is_attack:
                payoff[row_index, col_index] = _attack_attack_payoff(attacks.loc[row_action], attacks.loc[col_action], env)
            elif row_is_attack and not col_is_attack:
                payoff[row_index, col_index] = _attack_defense_payoff(
                    neutral_pairs.loc[(row_action, col_action)],
                    attacks.loc[row_action],
                    env,
                )
            elif not row_is_attack and col_is_attack:
                payoff[row_index, col_index] = _defense_attack_payoff(
                    neutral_pairs.loc[(col_action, row_action)],
                    env,
                )
            else:
                payoff[row_index, col_index] = 0.0
    return payoff, actions


def solve_static_game(env: Q3Environment) -> pd.DataFrame:
    """方法二：静态矩阵博弈的混合策略近似。"""

    payoff, action_ids = build_static_payoff_matrix(env)
    n_actions = len(action_ids)
    c = np.zeros(n_actions + 1, dtype=float)
    c[-1] = -1.0
    a_ub = np.hstack([-payoff.T, np.ones((n_actions, 1), dtype=float)])
    b_ub = np.zeros(n_actions, dtype=float)
    a_eq = np.zeros((1, n_actions + 1), dtype=float)
    a_eq[0, :n_actions] = 1.0
    b_eq = np.array([1.0], dtype=float)
    bounds = [(0.0, 1.0) for _ in range(n_actions)] + [(None, None)]
    result = linprog(c=c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        probabilities = np.clip(result.x[:n_actions], 0.0, 1.0)
    else:
        row_means = payoff.mean(axis=1)
        shifted = row_means - row_means.max()
        probabilities = np.exp(shifted)
    if np.isclose(probabilities.sum(), 0.0):
        probabilities = np.full(n_actions, 1.0 / n_actions)
    else:
        probabilities = probabilities / probabilities.sum()

    strategy = env.action_table.copy()
    strategy["static_prob"] = probabilities
    strategy["static_rank"] = strategy["static_prob"].rank(method="first", ascending=False).astype(int)
    strategy["static_mean_payoff"] = payoff.mean(axis=1)
    return strategy.sort_values(by=["static_prob", "action_id"], ascending=[False, True]).reset_index(drop=True)


def build_static_policy(env: Q3Environment, static_strategy: pd.DataFrame) -> pd.DataFrame:
    """提取静态博弈在各状态下的主动作，用于可视化与对照。"""

    strategy_map = static_strategy.set_index("action_id")["static_prob"].to_dict()
    records: list[dict[str, object]] = []
    for _, row in env.state_table.iterrows():
        health_my = int(row["health_my"])
        recovery_lock = int(row["recovery_lock"])
        counter_ready = int(row["counter_ready"])
        available_ids = available_action_ids(env, health_my, recovery_lock, counter_ready)
        available_prob = {
            action_id: float(strategy_map.get(action_id, 0.0))
            for action_id in available_ids
        }
        action_id = max(available_prob, key=lambda item: (available_prob[item], item))
        records.append(
            {
                "state_index": int(row["state_index"]),
                "static_action_id": action_id,
                "static_action_prob": available_prob[action_id],
            }
        )
    return pd.DataFrame(records)


def _choose_defense_by_score(defense_table: pd.DataFrame, score_columns: list[str], ascending: list[bool]) -> str:
    """按多个指标排序选防守动作。"""

    row = defense_table.sort_values(by=score_columns, ascending=ascending).iloc[0]
    return str(row["defense_id"])


def _choose_attack_by_score(attack_table: pd.DataFrame, score_columns: list[str]) -> str:
    """按多个指标排序选攻击动作。"""

    row = attack_table.sort_values(by=score_columns, ascending=[False] * len(score_columns)).iloc[0]
    return str(row["action_id"])


def build_rule_policy(env: Q3Environment) -> pd.DataFrame:
    """方法三：启发式规则树。"""

    safe_defense = _choose_defense_by_score(
        env.defense_table,
        ["avg_score", "avg_block_rate", "avg_fall_risk"],
        [False, False, True],
    )
    counter_defense = _choose_defense_by_score(
        env.defense_table,
        ["avg_counter_score", "avg_score", "avg_block_rate"],
        [False, False, False],
    )
    recovery_defense = _choose_defense_by_score(
        env.defense_table[env.defense_table["defense_id"].isin(env.recovery_defense_ids)].copy(),
        ["avg_score", "avg_block_rate", "avg_fall_risk"],
        [False, False, True],
    )
    balanced_attack = _choose_attack_by_score(
        env.attack_table,
        ["attack_utility", "score_margin_response", "tau_norm"],
    )

    records: list[dict[str, object]] = []
    for _, row in env.state_table.iterrows():
        score_diff = int(row["score_diff"])
        time_step = int(row["time_step"])
        health_my = int(row["health_my"])
        recovery_lock = int(row["recovery_lock"])
        available_attacks = env.attack_table[env.attack_table["action_id"].isin(env.attack_ids_by_health[health_my])].copy()
        aggressive_attack = _choose_attack_by_score(
            available_attacks,
            ["score_margin_response", "tau_norm", "attack_utility"],
        )
        low_cost_attack = str(
            available_attacks.sort_values(
                by=["energy_norm", "score_margin_response", "attack_utility", "action_id"],
                ascending=[True, False, False, True],
            ).iloc[0]["action_id"]
        )

        if recovery_lock > 0:
            action_id = recovery_defense
        elif health_my == 0:
            action_id = low_cost_attack if score_diff < 0 else safe_defense
        elif time_step >= 16 and score_diff <= -2:
            action_id = aggressive_attack
        elif time_step >= 16 and score_diff >= 2:
            action_id = safe_defense
        elif score_diff < 0:
            action_id = aggressive_attack
        elif score_diff > 0:
            action_id = counter_defense if time_step <= 12 else safe_defense
        else:
            action_id = counter_defense if 8 <= time_step <= 15 else balanced_attack

        state = MatchState(
            score_diff,
            time_step,
            health_my,
            int(row["health_opp"]),
            recovery_lock,
            int(row["counter_ready"]),
        )
        kernel = get_kernel(env, state, action_id)
        records.append(
            {
                "state_index": int(row["state_index"]),
                "rule_action_id": action_id,
                "rule_expected_reward": state_adjusted_reward(env, state, kernel),
            }
        )
    return pd.DataFrame(records)


def build_state_reward_decomposition(env: Q3Environment) -> pd.DataFrame:
    """构建所有可用 state-action 的状态敏感奖励分解表。"""

    records: list[dict[str, object]] = []
    for _, row in env.state_table.iterrows():
        state = MatchState(
            score_diff=int(row["score_diff"]),
            time_step=int(row["time_step"]),
            health_my=int(row["health_my"]),
            health_opp=int(row["health_opp"]),
            recovery_lock=int(row["recovery_lock"]),
            counter_ready=int(row["counter_ready"]),
        )
        for action_id in available_action_ids(env, state.health_my, state.recovery_lock, state.counter_ready):
            kernel = get_kernel(env, state, action_id)
            breakdown = state_reward_breakdown(env, state, kernel)
            records.append(
                {
                    "state_index": int(row["state_index"]),
                    "score_diff": state.score_diff,
                    "time_step": state.time_step,
                    "time_phase": str(row["time_phase"]),
                    "health_my": state.health_my,
                    "health_opp": state.health_opp,
                    "recovery_lock": state.recovery_lock,
                    "counter_ready": state.counter_ready,
                    "action_id": kernel.action_id,
                    "action_name": kernel.action_name,
                    "action_type": kernel.action_type,
                    "macro_group": kernel.macro_group,
                    **breakdown,
                }
            )
    return pd.DataFrame(records)


def value_iteration(env: Q3Environment) -> tuple[np.ndarray, pd.DataFrame]:
    """方法四：有限时域 MDP 的 Bellman 逆向递推。"""

    config = env.config
    score_card = len(SCORE_DIFF_VALUES)
    health_card = len(HEALTH_LEVELS)
    recovery_card = 2
    counter_card = 2
    values = np.zeros(
        (config.n_time_steps + 1, score_card, health_card, health_card, recovery_card, counter_card),
        dtype=float,
    )
    policy_records: list[dict[str, object]] = []

    for score_index, score_diff in enumerate(SCORE_DIFF_VALUES):
        for health_my in HEALTH_LEVELS:
            for health_opp in HEALTH_LEVELS:
                for recovery_lock in (0, 1):
                    for counter_ready in (0, 1):
                        values[config.n_time_steps, score_index, health_my, health_opp, recovery_lock, counter_ready] = (
                            config.terminal_reward_win
                            if score_diff > 0
                            else config.terminal_reward_loss
                            if score_diff < 0
                            else 0.0
                        )

    for time_step in range(config.n_time_steps, 0, -1):
        value_index = time_step - 1
        for score_index, score_diff in enumerate(SCORE_DIFF_VALUES):
            for health_my in HEALTH_LEVELS:
                for health_opp in HEALTH_LEVELS:
                    for recovery_lock in (0, 1):
                        for counter_ready in (0, 1):
                            state = MatchState(score_diff, time_step, health_my, health_opp, recovery_lock, counter_ready)
                            best_action = ""
                            best_value = -np.inf
                            best_reward = 0.0
                            q_records: list[tuple[str, float, float]] = []

                            for action_id in available_action_ids(env, health_my, recovery_lock, counter_ready):
                                kernel = get_kernel(env, state, action_id)
                                continuation = 0.0
                                for probability, next_state in iter_transition_branches(state, kernel, config):
                                    next_time_index = min(config.n_time_steps, next_state.time_step - 1)
                                    next_score_index = next_state.score_diff - SCORE_DIFF_VALUES[0]
                                    continuation += probability * values[
                                        next_time_index,
                                        next_score_index,
                                        next_state.health_my,
                                        next_state.health_opp,
                                        next_state.recovery_lock,
                                        next_state.counter_ready,
                                    ]
                                immediate_reward = state_adjusted_reward(env, state, kernel)
                                q_value = immediate_reward + config.gamma * continuation
                                q_records.append((action_id, q_value, immediate_reward))
                                if q_value > best_value:
                                    best_value = q_value
                                    best_action = action_id
                                    best_reward = immediate_reward

                            q_records = sorted(q_records, key=lambda item: (item[1], item[0]), reverse=True)
                            top_items = q_records[:3]
                            while len(top_items) < 3:
                                top_items.append(("", np.nan, np.nan))

                            values[value_index, score_index, health_my, health_opp, recovery_lock, counter_ready] = best_value
                            policy_records.append(
                                {
                                    "state_index": int(env.state_table.loc[
                                        (env.state_table["score_diff"] == score_diff)
                                        & (env.state_table["time_step"] == time_step)
                                        & (env.state_table["health_my"] == health_my)
                                        & (env.state_table["health_opp"] == health_opp)
                                        & (env.state_table["recovery_lock"] == recovery_lock)
                                        & (env.state_table["counter_ready"] == counter_ready),
                                        "state_index",
                                    ].iloc[0]),
                                    "mdp_action_id": best_action,
                                    "mdp_value": best_value,
                                    "mdp_immediate_reward": best_reward,
                                    "top1_action_id": top_items[0][0],
                                    "top1_q": top_items[0][1],
                                    "top2_action_id": top_items[1][0],
                                    "top2_q": top_items[1][1],
                                    "top3_action_id": top_items[2][0],
                                    "top3_q": top_items[2][1],
                                    "q_gap_12": float(top_items[0][1] - top_items[1][1])
                                    if not np.isnan(top_items[1][1])
                                    else np.nan,
                                    "q_gap_23": float(top_items[1][1] - top_items[2][1])
                                    if not np.isnan(top_items[1][1]) and not np.isnan(top_items[2][1])
                                    else np.nan,
                                }
                            )

    policy_table = pd.DataFrame(policy_records)
    return values, policy_table


def build_policy_table(env: Q3Environment) -> PolicyArtifacts:
    """统一生成四种方法的策略表与场景摘要。"""

    greedy = build_greedy_policy(env)
    static_strategy = solve_static_game(env)
    static_policy = build_static_policy(env, static_strategy)
    rule_policy = build_rule_policy(env)
    value_table, mdp_policy = value_iteration(env)
    state_reward_decomposition = build_state_reward_decomposition(env)

    policy_table = env.state_table.copy()
    for table in [greedy, static_policy, rule_policy, mdp_policy]:
        policy_table = policy_table.merge(table, on="state_index", how="left")

    action_meta = env.action_table[["action_id", "action_name", "action_type", "macro_group"]].copy()
    for prefix in ["greedy", "static", "rule", "mdp"]:
        policy_table = policy_table.merge(
            action_meta.rename(
                columns={
                    "action_id": f"{prefix}_action_id",
                    "action_name": f"{prefix}_action_name",
                    "action_type": f"{prefix}_action_type",
                    "macro_group": f"{prefix}_macro_group",
                }
            ),
            on=f"{prefix}_action_id",
            how="left",
        )

    scenario_summary = summarize_scenarios(policy_table)
    qvalue_summary = policy_table[
        [
            "state_index",
            "score_diff",
            "time_step",
            "time_phase",
            "health_my",
            "health_opp",
            "recovery_lock",
            "counter_ready",
            "top1_action_id",
            "top1_q",
            "top2_action_id",
            "top2_q",
            "top3_action_id",
            "top3_q",
            "q_gap_12",
            "q_gap_23",
        ]
    ].copy()
    return PolicyArtifacts(
        policy_table=policy_table,
        static_strategy=static_strategy,
        scenario_summary=scenario_summary,
        qvalue_summary=qvalue_summary,
        state_reward_decomposition=state_reward_decomposition,
        value_table=value_table,
    )


def summarize_scenarios(policy_table: pd.DataFrame) -> pd.DataFrame:
    """提取领先、落后、平局三类典型场景的策略摘要。"""

    base_table = policy_table[
        (policy_table["recovery_lock"] == 0)
        & (policy_table["counter_ready"] == 0)
    ].copy()
    scenario_filters = {
        "领先局": (base_table["score_diff"] >= 2) & (base_table["time_step"] >= 16),
        "落后局": (base_table["score_diff"] <= -2) & (base_table["time_step"] >= 16),
        "平局局": (base_table["score_diff"] == 0) & (base_table["time_step"].between(10, 14)),
    }

    records: list[dict[str, object]] = []
    for scenario_name, mask in scenario_filters.items():
        subset = base_table[mask].copy()
        if subset.empty:
            continue
        action_counts = (
            subset.groupby(["mdp_action_id", "mdp_action_name", "mdp_macro_group"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(by=["count", "mdp_action_id"], ascending=[False, True])
            .head(5)
        )
        total = int(action_counts["count"].sum())
        for _, row in action_counts.iterrows():
            records.append(
                {
                    "scenario": scenario_name,
                    "action_id": row["mdp_action_id"],
                    "action_name": row["mdp_action_name"],
                    "macro_group": row["mdp_macro_group"],
                    "count": int(row["count"]),
                    "share": float(row["count"] / max(len(subset), 1)),
                    "top5_total_count": total,
                }
            )
    return pd.DataFrame(records)
