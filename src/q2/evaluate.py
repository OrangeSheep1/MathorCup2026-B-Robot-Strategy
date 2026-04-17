"""Q2 四种方法的评价、筛选与接口输出模块。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


DEFENSE_WEIGHTS = {
    "block": 0.38,
    "damage": 0.12,
    "counter": 0.25,
    "fall": 0.25,
}

DEFENSE_PENALTY_CONFIG = {
    "lambda": 0.30,
    "k": 3.50,
}

FUZZY_GRADE_VECTOR = np.array([1.0, 0.6, 0.2], dtype=float)


def normalize_series(series: pd.Series) -> pd.Series:
    """对序列做极差标准化。"""

    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(maximum, minimum):
        return pd.Series(np.full(len(series), 0.5), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)


def rank_within_attack(data: pd.DataFrame, score_column: str) -> pd.Series:
    """按攻击动作分组生成防守排名。"""

    return data.groupby("action_id")[score_column].rank(method="first", ascending=False).astype(int)


def _rule_match_grade(row: pd.Series) -> float:
    """方法一只保留离散规则等级。"""

    trajectory = str(row["trajectory_type"])
    defense_category = str(row["defense_category"])
    closure_group = str(row["closure_group"])
    direction_match = float(row["direction_match"])
    height_match = float(row["height_match"])

    if direction_match == 1.0 and height_match == 1.0:
        if trajectory == "linear" and defense_category in {"block", "evade"}:
            return 1.0
        if trajectory in {"arc", "spin"} and defense_category == "evade":
            return 1.0
        if trajectory == "rush" and defense_category in {"balance", "evade"}:
            return 1.0
        if trajectory == "sequence" and defense_category in {"combo", "posture"}:
            return 1.0
        if trajectory == "recovery" and closure_group == "fallback":
            return 1.0

    if direction_match > 0.0 and height_match > 0.0:
        return 0.5
    return 0.0


def compute_method1_scores(pair_matrix: pd.DataFrame) -> pd.DataFrame:
    """方法一：规则匹配法。"""

    evaluated = pair_matrix.copy()
    evaluated["method1_score"] = evaluated.apply(_rule_match_grade, axis=1)
    evaluated["method1_rank"] = rank_within_attack(evaluated, "method1_score")
    return evaluated


def compute_method2_scores(pair_matrix: pd.DataFrame) -> pd.DataFrame:
    """方法二：二元拦截概率矩阵。"""

    evaluated = pair_matrix.copy()
    evaluated["method2_score"] = evaluated["p_block"]
    evaluated["method2_rank"] = rank_within_attack(evaluated, "method2_score")
    return evaluated


def membership_high(value: float) -> float:
    """高等级隶属度。"""

    if value < 0.5:
        return 0.0
    if value <= 0.8:
        return (value - 0.5) / 0.3
    return 1.0


def membership_mid(value: float) -> float:
    """中等级隶属度。"""

    if 0.2 <= value < 0.5:
        return (value - 0.2) / 0.3
    if np.isclose(value, 0.5):
        return 1.0
    if 0.5 < value <= 0.8:
        return (0.8 - value) / 0.3
    return 0.0


def membership_low(value: float) -> float:
    """低等级隶属度。"""

    if value < 0.2:
        return 1.0
    if value <= 0.5:
        return (0.5 - value) / 0.3
    return 0.0


def build_membership_row(value: float) -> np.ndarray:
    """构建单指标隶属向量。"""

    return np.array(
        [membership_high(value), membership_mid(value), membership_low(value)],
        dtype=float,
    )


def compute_method3_scores(
    pair_matrix: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """方法三：模糊防守效用评价。"""

    active_weights = dict(DEFENSE_WEIGHTS if weights is None else weights)
    evaluated = pair_matrix.copy()
    evaluated["damage_safety"] = 1.0 - evaluated["defense_damage"]
    evaluated["stability_safety"] = 1.0 - evaluated["p_fall"]

    indicator_weights = np.array(
        [
            active_weights["block"],
            active_weights["damage"],
            active_weights["counter"],
            active_weights["fall"],
        ],
        dtype=float,
    )

    fuzzy_scores: list[float] = []
    for _, row in evaluated.iterrows():
        indicator_values = [
            float(row["p_block"]),
            float(row["damage_safety"]),
            float(row["counter_window_norm"]),
            float(row["stability_safety"]),
        ]
        membership_matrix = np.vstack([build_membership_row(value) for value in indicator_values])
        fuzzy_vector = indicator_weights @ membership_matrix
        fuzzy_scores.append(float(fuzzy_vector @ FUZZY_GRADE_VECTOR))

    evaluated["method3_score"] = fuzzy_scores
    evaluated["method3_rank"] = rank_within_attack(evaluated, "method3_score")
    return evaluated


def compute_base_defense_utility(
    pair_matrix: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """计算主模型的基础效用项。"""

    active_weights = dict(DEFENSE_WEIGHTS if weights is None else weights)
    evaluated = pair_matrix.copy()
    evaluated["v_def"] = (
        active_weights["block"] * evaluated["p_block"]
        - active_weights["damage"] * evaluated["defense_damage"]
        + active_weights["counter"] * evaluated["counter_window_norm"]
        - active_weights["fall"] * evaluated["p_fall"]
    )
    evaluated["benefit_term"] = (
        active_weights["block"] * evaluated["p_block"]
        + active_weights["counter"] * evaluated["counter_window_norm"]
    )
    evaluated["cost_term"] = (
        active_weights["damage"] * evaluated["defense_damage"]
        + active_weights["fall"] * evaluated["p_fall"]
    )
    return evaluated


def compute_defense_penalty_raw(
    pair_matrix: pd.DataFrame,
    penalty_config: Mapping[str, float] | None = None,
) -> pd.Series:
    """计算非线性倒地惩罚。"""

    config = dict(DEFENSE_PENALTY_CONFIG if penalty_config is None else penalty_config)
    return config["lambda"] * np.exp(config["k"] * pair_matrix["p_fall"]) - config["lambda"]


def compute_penalty_normalized(penalty_raw: pd.Series, base_score: pd.Series) -> pd.Series:
    """把惩罚项缩放到与基础效用同量级。"""

    penalty_max = float(penalty_raw.max())
    score_range = float(base_score.max() - base_score.min())
    if np.isclose(penalty_max, 0.0) or np.isclose(score_range, 0.0):
        return pd.Series(np.zeros(len(penalty_raw)), index=penalty_raw.index, dtype=float)
    return penalty_raw / penalty_max * score_range


def compute_method4_scores(
    pair_matrix: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    penalty_config: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """方法四：攻防博弈矩阵 + 反击机会评估。"""

    evaluated = compute_base_defense_utility(pair_matrix, weights=weights)
    evaluated["fall_penalty_raw"] = compute_defense_penalty_raw(evaluated, penalty_config=penalty_config)
    evaluated["fall_penalty"] = compute_penalty_normalized(evaluated["fall_penalty_raw"], evaluated["v_def"])
    evaluated["invalid_geo"] = ~evaluated["geo_covered"]
    evaluated["high_fall"] = evaluated["p_fall"] > 0.70

    raw_score = evaluated["v_def"] - evaluated["fall_penalty"]
    raw_score = raw_score.where(~evaluated["invalid_geo"], 0.0)
    raw_score = raw_score.where(~evaluated["high_fall"], 0.0)
    evaluated["proposed_score"] = raw_score.clip(lower=0.0)
    evaluated["score_def"] = evaluated["proposed_score"]
    if np.isclose(float(evaluated["proposed_score"].max()), 0.0):
        evaluated["proposed_utility"] = 0.0
    else:
        evaluated["proposed_utility"] = normalize_series(evaluated["proposed_score"])
    evaluated["rank"] = rank_within_attack(evaluated, "proposed_score")
    return evaluated


def _pick_counter_actions(
    action_catalog: pd.DataFrame,
    counter_window: float,
    top_n: int = 3,
) -> pd.DataFrame:
    """在反击窗口内选择可执行的攻击动作。"""

    feasible = action_catalog[action_catalog["exec_time"] <= counter_window].copy()
    if feasible.empty:
        return feasible
    feasible = feasible.sort_values(
        by=["attack_utility", "impact_score", "exec_time"],
        ascending=[False, False, True],
    ).head(top_n)
    return feasible


def select_top_defenses(group: pd.DataFrame, top_k: int = 3) -> list[pd.Series]:
    """在闭环约束下为单个攻击动作选择 Top 1-3 防守。"""

    ordered = group.sort_values(
        by=["proposed_score", "p_block", "counter_window", "method2_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    positive = ordered[ordered["proposed_score"] > 0].reset_index(drop=True)
    source = positive if not positive.empty else ordered

    selected: list[pd.Series] = []
    selected_ids: set[str] = set()

    active_candidates = source[source["closure_group"] == "active"]
    if active_candidates.empty:
        active_candidates = ordered[ordered["closure_group"] == "active"]
    fallback_candidates = source[source["closure_group"] == "fallback"]
    if fallback_candidates.empty:
        fallback_candidates = ordered[ordered["closure_group"] == "fallback"]

    if not active_candidates.empty:
        row = active_candidates.iloc[0]
        selected.append(row)
        selected_ids.add(str(row["defense_id"]))
    if not fallback_candidates.empty:
        row = fallback_candidates.iloc[0]
        if str(row["defense_id"]) not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row["defense_id"]))

    for _, row in source.iterrows():
        if len(selected) >= top_k:
            break
        defense_id = str(row["defense_id"])
        if defense_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(defense_id)

    return selected[:top_k]


def build_matchup_outputs(
    evaluated_pairs: pd.DataFrame,
    action_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """输出 Q3 可复用的防守匹配表、反击链表和方法对比表。"""

    matchup_records: list[dict[str, object]] = []
    counter_chain_records: list[dict[str, object]] = []
    method_summary_records: list[dict[str, object]] = []

    for action_id, group in evaluated_pairs.groupby("action_id", sort=False):
        top_choices = select_top_defenses(group, top_k=3)
        if not top_choices:
            continue

        action_row = group.iloc[0]
        selected_groups = {str(item["closure_group"]) for item in top_choices}
        row_record: dict[str, object] = {
            "action_id": action_id,
            "action_name": action_row["action_name"],
            "attack_category": action_row["attack_category"],
            "closure_complete": "active" in selected_groups and "fallback" in selected_groups,
            "closure_note": "OK"
            if ("active" in selected_groups and "fallback" in selected_groups)
            else f"missing:{'active' if 'active' not in selected_groups else 'fallback'}",
        }

        for index, defense_row in enumerate(top_choices, start=1):
            feasible = _pick_counter_actions(action_catalog, float(defense_row["counter_window"]), top_n=3)
            counter_ids = "|".join(feasible["action_id"].tolist())
            counter_names = "|".join(feasible["action_name"].tolist())
            best_counter_id = "" if feasible.empty else str(feasible.iloc[0]["action_id"])
            best_counter_name = "" if feasible.empty else str(feasible.iloc[0]["action_name"])
            best_counter_utility = 0.0 if feasible.empty else float(feasible.iloc[0]["attack_utility"])

            row_record[f"defense_id_r{index}"] = defense_row["defense_id"]
            row_record[f"defense_name_r{index}"] = defense_row["defense_name"]
            row_record[f"block_prob_r{index}"] = float(defense_row["p_block"])
            row_record[f"counter_window_r{index}"] = float(defense_row["counter_window"])
            row_record[f"counter_prob_r{index}"] = float(defense_row["counter_prob"])
            row_record[f"fall_risk_r{index}"] = float(defense_row["p_fall"])
            row_record[f"defense_score_r{index}"] = float(defense_row["proposed_score"])
            row_record[f"counter_action_id_r{index}"] = best_counter_id
            row_record[f"counter_action_name_r{index}"] = best_counter_name
            row_record[f"counter_action_utility_r{index}"] = best_counter_utility

            counter_chain_records.append(
                {
                    "action_id": action_id,
                    "action_name": action_row["action_name"],
                    "priority_rank": index,
                    "defense_id": defense_row["defense_id"],
                    "defense_name": defense_row["defense_name"],
                    "defense_category": defense_row["defense_category"],
                    "defense_score": float(defense_row["proposed_score"]),
                    "block_prob": float(defense_row["p_block"]),
                    "counter_window": float(defense_row["counter_window"]),
                    "fall_risk": float(defense_row["p_fall"]),
                    "counter_action_ids": counter_ids,
                    "counter_action_names": counter_names,
                }
            )

        method_summary_records.append(
            {
                "action_id": action_id,
                "action_name": action_row["action_name"],
                "method1_top1_defense": group.sort_values("method1_score", ascending=False).iloc[0]["defense_id"],
                "method2_top1_defense": group.sort_values("method2_score", ascending=False).iloc[0]["defense_id"],
                "method3_top1_defense": group.sort_values("method3_score", ascending=False).iloc[0]["defense_id"],
                "method4_top1_defense": group.sort_values("proposed_score", ascending=False).iloc[0]["defense_id"],
            }
        )

        matchup_records.append(row_record)

    matchup_table = pd.DataFrame(matchup_records)
    counter_chain_table = pd.DataFrame(counter_chain_records)
    method_summary_table = pd.DataFrame(method_summary_records)
    return matchup_table, counter_chain_table, method_summary_table


def evaluate_all_methods(
    pair_matrix: pd.DataFrame,
    action_catalog: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    penalty_config: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """串联四种方法并生成三个输出表。"""

    evaluated = compute_method1_scores(pair_matrix)
    evaluated = compute_method2_scores(evaluated)
    evaluated = compute_method3_scores(evaluated, weights=weights)
    evaluated = compute_method4_scores(evaluated, weights=weights, penalty_config=penalty_config)
    matchup_table, counter_chain_table, method_summary_table = build_matchup_outputs(evaluated, action_catalog)
    return evaluated, matchup_table, counter_chain_table, method_summary_table
