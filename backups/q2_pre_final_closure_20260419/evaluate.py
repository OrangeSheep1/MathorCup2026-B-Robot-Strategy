"""Q2 evaluation, layered selection, and compatibility outputs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


PRIMARY_WEIGHTS = {
    "success": 0.48,
    "damage": 0.24,
    "counter": 0.18,
    "fall": 0.10,
}

FALLBACK_WEIGHTS = {
    "success": 0.15,
    "damage_safe": 0.50,
    "stability": 0.35,
}

FUZZY_GRADE_VECTOR = np.array([1.0, 0.6, 0.2], dtype=float)

GENERIC_SCOPE_PENALTY = {
    "D14": 0.06,
    "D16": 0.02,
    "D15": 0.00,
}


def rank_within_attack(data: pd.DataFrame, score_column: str) -> pd.Series:
    return data.groupby("action_id")[score_column].rank(method="first", ascending=False).astype(int)


def _score_primary_only(data: pd.DataFrame, raw_score: pd.Series) -> pd.Series:
    mask = data["primary_role_feasible"] & data["state_feasible"]
    return raw_score.where(mask, 0.0).clip(lower=0.0)


def _safe_norm(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(minimum, maximum):
        return pd.Series(np.full(len(series), 0.5), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)


def _generic_scope_penalty(defense_id: pd.Series) -> pd.Series:
    return defense_id.astype(str).map(GENERIC_SCOPE_PENALTY).fillna(0.0).astype(float)


def _rule_match_grade(row: pd.Series) -> float:
    if not bool(row["primary_role_feasible"]):
        return 0.0
    trajectory = str(row["trajectory_type"])
    category = str(row["defense_category"])
    route_type = str(row["route_type"])
    if float(row["p_geo"]) >= 0.85 and float(row["p_react"]) >= 0.5:
        if trajectory == "linear" and category in {"block", "combo"}:
            return 1.0
        if trajectory in {"arc", "spin"} and route_type in {"evade_lateral", "evade_back", "evade_orbit", "combo_lateral_parry"}:
            return 1.0
        if trajectory == "rush" and route_type in {"yield_soft", "evade_lateral", "balance_step"}:
            return 1.0
        if trajectory == "sequence" and category == "combo":
            return 1.0
    if float(row["p_geo"]) > 0.0 and float(row["p_react"]) > 0.0:
        return 0.5
    return 0.0


def compute_method1_scores(pair_matrix: pd.DataFrame) -> pd.DataFrame:
    evaluated = pair_matrix.copy()
    evaluated["method1_score"] = evaluated.apply(_rule_match_grade, axis=1)
    evaluated["method1_rank"] = rank_within_attack(evaluated, "method1_score")
    return evaluated


def compute_method2_scores(pair_matrix: pd.DataFrame) -> pd.DataFrame:
    evaluated = pair_matrix.copy()
    evaluated["method2_score"] = _score_primary_only(evaluated, evaluated["p_success"])
    evaluated["method2_rank"] = rank_within_attack(evaluated, "method2_score")
    return evaluated


def membership_high(value: float) -> float:
    if value < 0.5:
        return 0.0
    if value <= 0.8:
        return (value - 0.5) / 0.3
    return 1.0


def membership_mid(value: float) -> float:
    if 0.2 <= value < 0.5:
        return (value - 0.2) / 0.3
    if np.isclose(value, 0.5):
        return 1.0
    if 0.5 < value <= 0.8:
        return (0.8 - value) / 0.3
    return 0.0


def membership_low(value: float) -> float:
    if value < 0.2:
        return 1.0
    if value <= 0.5:
        return (0.5 - value) / 0.3
    return 0.0


def _membership_row(value: float) -> np.ndarray:
    return np.array([membership_high(value), membership_mid(value), membership_low(value)], dtype=float)


def compute_method3_scores(pair_matrix: pd.DataFrame) -> pd.DataFrame:
    evaluated = pair_matrix.copy()
    indicator_weights = np.array([0.40, 0.25, 0.15, 0.20], dtype=float)
    fuzzy_scores: list[float] = []
    for _, row in evaluated.iterrows():
        if not bool(row["primary_role_feasible"]):
            fuzzy_scores.append(0.0)
            continue
        indicators = [
            float(row["p_success"]),
            float(1.0 - row["residual_damage"]),
            float(row["counter_window_norm"]),
            float(1.0 - row["p_fall"]),
        ]
        membership_matrix = np.vstack([_membership_row(value) for value in indicators])
        fuzzy_vector = indicator_weights @ membership_matrix
        fuzzy_scores.append(float(fuzzy_vector @ FUZZY_GRADE_VECTOR))
    evaluated["method3_score"] = fuzzy_scores
    evaluated["method3_rank"] = rank_within_attack(evaluated, "method3_score")
    return evaluated


def compute_method4_scores(
    pair_matrix: pd.DataFrame,
    primary_weights: Mapping[str, float] | None = None,
    fallback_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    primary = dict(PRIMARY_WEIGHTS if primary_weights is None else primary_weights)
    fallback = dict(FALLBACK_WEIGHTS if fallback_weights is None else fallback_weights)
    evaluated = pair_matrix.copy()
    raw_primary = (
        primary["success"] * evaluated["p_success"]
        - primary["damage"] * evaluated["residual_damage"]
        + primary["counter"] * evaluated["counter_window_norm"]
        - primary["fall"] * evaluated["p_fall"]
    )
    evaluated["proposed_score"] = _score_primary_only(evaluated, raw_primary)
    evaluated["primary_score"] = evaluated["proposed_score"]
    evaluated["score_def"] = evaluated["proposed_score"]
    if np.isclose(float(evaluated["proposed_score"].max()), 0.0):
        evaluated["proposed_utility"] = 0.0
    else:
        evaluated["proposed_utility"] = _safe_norm(evaluated["proposed_score"])

    fallback_mask = evaluated["defense_role"].isin(["fallback_mitigation", "emergency_transition"]) & evaluated["state_feasible"]
    evaluated["fallback_score"] = (
        fallback["success"] * evaluated["p_success"]
        + fallback["damage_safe"] * (1.0 - evaluated["residual_damage"])
        + fallback["stability"] * (1.0 - evaluated["p_fall"])
        - _generic_scope_penalty(evaluated["defense_id"])
    ).where(fallback_mask, 0.0).clip(lower=0.0)
    evaluated["rank"] = rank_within_attack(evaluated, "proposed_score")
    evaluated["fallback_rank"] = rank_within_attack(evaluated, "fallback_score")
    return evaluated


def _pick_counter_rows(row: pd.Series) -> tuple[str, str, str]:
    return (
        str(row.get("counter_action_id", "")),
        str(row.get("counter_action_ids", "")),
        str(row.get("counter_action_names", "")),
    )


def _pick_recovery_action(group: pd.DataFrame, active_row: pd.Series | None, fallback_row: pd.Series | None) -> pd.Series | None:
    if active_row is not None and str(active_row["exit_state_after_defense"]) == "fallen":
        pass
    elif fallback_row is not None and str(fallback_row["exit_state_after_defense"]) == "fallen":
        pass
    else:
        return None

    recovery_pool = group[group["defense_role"].isin(["recovery_only", "ground_only"])].copy()
    if recovery_pool.empty:
        return None
    attack_low = str(group.iloc[0]["height_tag"]) == "low" or str(group.iloc[0]["target_zone"]) == "lower_limbs"
    if attack_low:
        preferred = recovery_pool[recovery_pool["defense_role"] == "ground_only"]
        if not preferred.empty:
            return preferred.iloc[0]
    preferred = recovery_pool[recovery_pool["defense_role"] == "recovery_only"]
    if not preferred.empty:
        return preferred.iloc[0]
    return recovery_pool.iloc[0]


def build_matchup_outputs(
    evaluated_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matchup_records: list[dict[str, object]] = []
    counter_chain_records: list[dict[str, object]] = []
    method_summary_records: list[dict[str, object]] = []

    for action_id, group in evaluated_pairs.groupby("action_id", sort=False):
        action_row = group.iloc[0]
        attack_entry_state = str(action_row["attack_entry_state"])
        ground_row = None
        if attack_entry_state == "fallen":
            active_row = None
            fallback_row = None
            ground_pool = group[group["ground_feasible"]].sort_values(
                by=["fallback_score", "p_success", "stability_safe"],
                ascending=[False, False, False],
            )
            recovery_pool = group[group["recovery_feasible"]].sort_values(
                by=["exec_time_def", "stability_safe"],
                ascending=[True, False],
            )
            ground_row = None if ground_pool.empty else ground_pool.iloc[0]
            recovery_row = None if recovery_pool.empty else recovery_pool.iloc[0]
        else:
            active_pool = group[group["primary_role_feasible"]].sort_values(
                by=["proposed_score", "p_success", "counter_window"],
                ascending=[False, False, False],
            )
            fallback_pool = group[group["fallback_feasible"]].sort_values(
                by=["fallback_score", "p_success", "stability_safe"],
                ascending=[False, False, False],
            )
            active_row = None if active_pool.empty else active_pool.iloc[0]
            fallback_row = None if fallback_pool.empty else fallback_pool.iloc[0]
            recovery_row = _pick_recovery_action(group, active_row, fallback_row)

        selected_groups = set()
        if attack_entry_state == "fallen":
            if ground_row is not None:
                selected_groups.add("ground")
            if recovery_row is not None:
                selected_groups.add("recovery")
        else:
            if active_row is not None:
                selected_groups.add("active")
            if fallback_row is not None:
                selected_groups.add("fallback")

        row_record: dict[str, object] = {
            "action_id": action_id,
            "action_name": action_row["action_name"],
            "attack_category": action_row["attack_category"],
            "closure_complete": (
                ("ground" in selected_groups and "recovery" in selected_groups)
                if attack_entry_state == "fallen"
                else ("active" in selected_groups and "fallback" in selected_groups)
            ),
            "closure_note": (
                "OK"
                if (
                    ("ground" in selected_groups and "recovery" in selected_groups)
                    if attack_entry_state == "fallen"
                    else ("active" in selected_groups and "fallback" in selected_groups)
                )
                else ("missing:ground_or_recovery" if attack_entry_state == "fallen" else "missing:active_or_fallback")
            ),
        }

        ranked_rows = [ground_row, recovery_row, None] if attack_entry_state == "fallen" else [active_row, fallback_row, recovery_row]
        for index, defense_row in enumerate(ranked_rows, start=1):
            if defense_row is None:
                row_record[f"defense_id_r{index}"] = ""
                row_record[f"defense_name_r{index}"] = ""
                row_record[f"block_prob_r{index}"] = 0.0
                row_record[f"counter_window_r{index}"] = 0.0
                row_record[f"counter_prob_r{index}"] = 0.0
                row_record[f"fall_risk_r{index}"] = 0.0
                row_record[f"defense_score_r{index}"] = 0.0
                row_record[f"counter_action_id_r{index}"] = ""
                row_record[f"counter_action_name_r{index}"] = ""
                row_record[f"counter_action_utility_r{index}"] = 0.0
                continue

            counter_action_id, counter_action_ids, counter_action_names = _pick_counter_rows(defense_row)
            row_record[f"defense_id_r{index}"] = defense_row["defense_id"]
            row_record[f"defense_name_r{index}"] = defense_row["defense_name"]
            row_record[f"block_prob_r{index}"] = float(defense_row["p_success"])
            row_record[f"counter_window_r{index}"] = float(defense_row["counter_window"])
            row_record[f"counter_prob_r{index}"] = float(defense_row["counter_prob_effective"])
            row_record[f"fall_risk_r{index}"] = float(defense_row["p_fall"])
            row_record[f"defense_score_r{index}"] = float(
                defense_row["fallback_score"]
                if attack_entry_state == "fallen" and index == 1
                else 1.0 - float(defense_row["exec_time_def"]) / max(float(group["exec_time_def"].max()), 1e-6)
                if attack_entry_state == "fallen" and index == 2
                else defense_row["proposed_score"]
                if index == 1
                else defense_row["fallback_score"] if index == 2
                else 1.0 - float(defense_row["exec_time_def"]) / max(float(group["exec_time_def"].max()), 1e-6)
            )
            row_record[f"counter_action_id_r{index}"] = counter_action_id
            row_record[f"counter_action_name_r{index}"] = counter_action_names.split("|")[0] if counter_action_names else ""
            row_record[f"counter_action_utility_r{index}"] = float(defense_row.get("counter_tau_norm", 0.0))

            counter_chain_records.append(
                {
                    "action_id": action_id,
                    "action_name": action_row["action_name"],
                    "priority_rank": index,
                    "defense_id": defense_row["defense_id"],
                    "defense_name": defense_row["defense_name"],
                    "defense_category": defense_row["defense_category"],
                    "defense_role": defense_row["defense_role"],
                    "defense_score": float(
                        defense_row["fallback_score"]
                        if attack_entry_state == "fallen" and index == 1
                        else 1.0 - float(defense_row["exec_time_def"]) / max(float(group["exec_time_def"].max()), 1e-6)
                        if attack_entry_state == "fallen" and index == 2
                        else defense_row["proposed_score"]
                        if index == 1
                        else defense_row["fallback_score"] if index == 2
                        else 1.0 - float(defense_row["exec_time_def"]) / max(float(group["exec_time_def"].max()), 1e-6)
                    ),
                    "block_prob": float(defense_row["p_success"]),
                    "counter_window": float(defense_row["counter_window"]),
                    "fall_risk": float(defense_row["p_fall"]),
                    "counter_action_ids": counter_action_ids,
                    "counter_action_names": counter_action_names,
                }
            )

        method1_top = group.sort_values("method1_score", ascending=False).iloc[0]
        method2_top = group.sort_values("method2_score", ascending=False).iloc[0]
        method3_top = group.sort_values("method3_score", ascending=False).iloc[0]
        method_summary_records.append(
            {
                "action_id": action_id,
                "action_name": action_row["action_name"],
                "method1_top1_defense": method1_top["defense_id"],
                "method2_top1_defense": method2_top["defense_id"],
                "method3_top1_defense": method3_top["defense_id"],
                "method4_top1_defense": (
                    active_row["defense_id"]
                    if active_row is not None
                    else ground_row["defense_id"] if ground_row is not None else ""
                ),
                "method4_active_top1": "" if active_row is None else active_row["defense_id"],
                "method4_ground_top1": "" if ground_row is None else ground_row["defense_id"],
                "method4_fallback_top1": "" if fallback_row is None else fallback_row["defense_id"],
                "method4_recovery_if_needed": "" if recovery_row is None else recovery_row["defense_id"],
                "method4_active_role": "" if active_row is None else active_row["defense_role"],
                "method4_fallback_role": "" if fallback_row is None else fallback_row["defense_role"],
            }
        )
        matchup_records.append(row_record)

    matchup_table = pd.DataFrame(matchup_records)
    counter_chain_table = pd.DataFrame(counter_chain_records)
    method_summary_table = pd.DataFrame(method_summary_records)
    return matchup_table, counter_chain_table, method_summary_table


def evaluate_all_methods(
    pair_matrix: pd.DataFrame,
    primary_weights: Mapping[str, float] | None = None,
    fallback_weights: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluated = compute_method1_scores(pair_matrix)
    evaluated = compute_method2_scores(evaluated)
    evaluated = compute_method3_scores(evaluated)
    evaluated = compute_method4_scores(evaluated, primary_weights=primary_weights, fallback_weights=fallback_weights)
    matchup_table, counter_chain_table, method_summary_table = build_matchup_outputs(evaluated)
    return evaluated, matchup_table, counter_chain_table, method_summary_table
