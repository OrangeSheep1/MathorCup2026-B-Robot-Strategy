"""Q2 pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import INTERIM_DIR, OUTPUT_DIR, RAW_DIR, ensure_basic_dirs
from src.q1.model_v1 import (
    load_action_phase_templates,
    load_action_templates,
    load_robot_params,
    load_segment_params,
    load_support_mode_config,
)
from src.q2.evaluate import EPSILON, evaluate_all_methods
from src.q2.model_v1 import (
    build_attack_catalog,
    build_defense_feature_table,
    build_pair_matrix,
    load_action_features,
    load_attack_response_policy,
    load_attack_semantics,
    load_defense_actions,
    load_family_compatibility,
    load_route_advantage,
)
from src.q2.plot import (
    plot_decision_waterfall,
    plot_defense_surface,
    plot_parallel_metrics,
    plot_primary_utility_matrix,
    plot_layered_response_overview,
    plot_method_comparison,
)


Q1_ACTION_FILE = INTERIM_DIR / "action_features.csv"
Q1_ROBOT_FILE = RAW_DIR / "q1_robot_params.csv"
Q1_SEGMENT_FILE = RAW_DIR / "q1_segment_params.csv"
Q1_SUPPORT_FILE = RAW_DIR / "q1_support_mode_config.csv"
Q1_ACTION_TEMPLATE_FILE = RAW_DIR / "q1_action_templates.csv"
Q1_PHASE_TEMPLATE_FILE = RAW_DIR / "q1_action_phase_templates.csv"
Q2_ATTACK_SEMANTIC_FILE = RAW_DIR / "q2_attack_semantics.csv"
Q2_ATTACK_RESPONSE_FILE = RAW_DIR / "q2_attack_response_policy.csv"
Q2_DEFENSE_FILE = RAW_DIR / "q2_defense_actions.csv"
Q2_ROUTE_FILE = RAW_DIR / "q2_route_advantage.csv"
Q2_FAMILY_FILE = RAW_DIR / "q2_family_compatibility.csv"

DEFENSE_FEATURE_FILE = INTERIM_DIR / "defense_features.csv"
PAIR_SCORE_FILE = INTERIM_DIR / "defense_pair_scores.csv"
MATCHUP_FILE = INTERIM_DIR / "defense_matchup.csv"
COUNTER_CHAIN_FILE = INTERIM_DIR / "counter_chain.csv"
METHOD_SUMMARY_FILE = INTERIM_DIR / "q2_method_summary.csv"
ATTACK_CONTACT_FEATURE_FILE = INTERIM_DIR / "q2_attack_contact_features.csv"
ATTACK_CONTACT_DEBUG_FILE = INTERIM_DIR / "q2_attack_contact_debug.csv"
DEFENSE_CLEARANCE_DEBUG_FILE = INTERIM_DIR / "q2_defense_clearance_debug.csv"
PAIR_TOP5_DEBUG_FILE = INTERIM_DIR / "q2_pair_top5_debug.csv"
ZERO_TIE_AUDIT_FILE = INTERIM_DIR / "q2_zero_tie_audit.csv"
FAMILY_AUDIT_FILE = INTERIM_DIR / "q2_family_audit.csv"
ACTIVE_DISTRIBUTION_FILE = INTERIM_DIR / "q2_active_distribution.csv"
ACCEPTANCE_CHECK_FILE = INTERIM_DIR / "q2_acceptance_checks.csv"
RULE_COVERAGE_AUDIT_FILE = INTERIM_DIR / "q2_rule_coverage_audit.csv"
RULE_AUDIT_SUMMARY_FILE = INTERIM_DIR / "q2_rule_audit_summary.csv"
TOP1_AUDIT_SUMMARY_FILE = INTERIM_DIR / "q2_top1_audit_summary.csv"

UTILITY_MATRIX_FIGURE = OUTPUT_DIR / "q2_utility_matrix.png"
SURFACE_FIGURE = OUTPUT_DIR / "q2_surface.png"
WATERFALL_FIGURE = OUTPUT_DIR / "q2_waterfall.png"
PARALLEL_FIGURE = OUTPUT_DIR / "q2_parallel.png"
METHOD_COMPARE_FIGURE = OUTPUT_DIR / "q2_method_comparison.png"
LAYERED_OVERVIEW_FIGURE = OUTPUT_DIR / "q2_layered_response_overview.png"


def _save_csv(data: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    data.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def _build_pair_top5_debug(evaluated_pairs: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for _, group in evaluated_pairs.groupby("action_id", sort=False):
        active_pool = group[group["primary_role_feasible"]].sort_values(
            by=["primary_score", "p_success", "residual_damage", "p_fall", "counter_window"],
            ascending=[False, False, True, True, False],
        )
        records.append(
            active_pool.head(5)[
                [
                    "action_id",
                    "defense_id",
                    "defense_role",
                    "p_geo",
                    "p_react",
                    "p_route",
                    "route_rule_hit",
                    "route_rule_specificity",
                    "p_clear",
                    "p_load",
                    "p_family",
                    "family_rule_hit",
                    "geo_cap_applied",
                    "geo_cap_reason",
                    "family_damage_factor",
                    "p_success",
                    "residual_damage",
                    "p_fall",
                    "primary_score",
                    "fallback_score",
                ]
            ]
        )
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _build_zero_tie_audit(evaluated_pairs: pd.DataFrame, method_summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    summary_map = method_summary.set_index("action_id").to_dict("index")
    method_specs = [
        ("method1", "method1_score", "method1_top1_defense"),
        ("method2", "method2_score", "method2_top1_defense"),
        ("method3", "method3_score", "method3_top1_defense"),
        ("method4", "primary_score", "method4_active_top1"),
    ]
    for action_id, group in evaluated_pairs.groupby("action_id", sort=False):
        active_pool = group[group["primary_role_feasible"]].copy()
        for method_key, score_column, summary_column in method_specs:
            if active_pool.empty:
                records.append(
                    {
                        "action_id": action_id,
                        "method_key": method_key,
                        "top_score": 0.0,
                        "tie_count": 0,
                        "emitted_blank": True,
                        "reason": "no_active_primary_feasible",
                    }
                )
                continue
            top_score = float(active_pool[score_column].max())
            tie_count = int((active_pool[score_column] >= top_score - EPSILON).sum()) if top_score > EPSILON else 0
            emitted_blank = str(summary_map[action_id][summary_column]).strip() == ""
            reason = "ok"
            if top_score <= EPSILON:
                reason = "top_score_below_epsilon"
            elif tie_count > 1 and emitted_blank:
                reason = "tie_then_blank"
            elif tie_count > 1:
                reason = "non_unique_top"
            elif emitted_blank:
                reason = "blank_output"
            records.append(
                {
                    "action_id": action_id,
                    "method_key": method_key,
                    "top_score": top_score,
                    "tie_count": tie_count,
                    "emitted_blank": emitted_blank,
                    "reason": reason,
                }
            )
    return pd.DataFrame(records)


def _build_family_audit(evaluated_pairs: pd.DataFrame) -> pd.DataFrame:
    return evaluated_pairs[
        [
            "action_id",
            "defense_id",
            "contact_load_family",
            "mechanism_tag",
            "family_rule_hit",
            "p_family",
            "family_damage_factor",
            "route_rule_hit",
            "route_rule_specificity",
            "route_rule_specificity_note",
            "geo_cap_applied",
            "geo_cap_reason",
        ]
    ].copy()


def _route_relevant_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["route_type"].isin(
        [
            "evade_lateral",
            "evade_duck",
            "evade_back",
            "evade_spin",
            "evade_orbit",
            "combo_lateral_parry",
            "combo_duck_slip",
            "combo_block_retreat",
            "yield_soft",
            "balance_step",
            "ground_hold",
            "ground_transfer",
        ]
    )


def _build_rule_audit_summary(evaluated_pairs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def _append_record(scope_name: str, action_id: str, frame: pd.DataFrame) -> None:
        total_pairs = int(len(frame))
        route_relevant = frame[_route_relevant_mask(frame)] if total_pairs else frame
        route_hit_pairs = int(route_relevant["route_rule_hit"].sum()) if not route_relevant.empty else 0
        records.append(
            {
                "scope_name": scope_name,
                "action_id": action_id,
                "total_pairs": total_pairs,
                "family_rule_hit_pairs": int(frame["family_rule_hit"].sum()) if total_pairs else 0,
                "route_relevant_pairs": int(len(route_relevant)),
                "route_rule_hit_pairs": route_hit_pairs,
                "geo_cap_applied_pairs": int(frame["geo_cap_applied"].sum()) if total_pairs else 0,
                "missing_family_rule_pairs": int((~frame["family_rule_hit"]).sum()) if total_pairs else 0,
                "missing_route_rule_pairs": int((~route_relevant["route_rule_hit"]).sum()) if not route_relevant.empty else 0,
                "non_normal_audit_pairs": int((frame["pair_audit_flag"] != "normal").sum()) if total_pairs else 0,
                "family_rule_hit_rate": float(frame["family_rule_hit"].mean()) if total_pairs else 0.0,
                "route_rule_hit_rate": float(route_relevant["route_rule_hit"].mean()) if not route_relevant.empty else 1.0,
                "geo_cap_applied_rate": float(frame["geo_cap_applied"].mean()) if total_pairs else 0.0,
            }
        )

    _append_record("overall", "", evaluated_pairs)
    for action_id, group in evaluated_pairs.groupby("action_id", sort=False):
        _append_record("by_action", str(action_id), group)
    return pd.DataFrame(records)


def _lookup_top1_pairs(matchup_table: pd.DataFrame, evaluated_pairs: pd.DataFrame, layer_column: str) -> pd.DataFrame:
    lookup = matchup_table[["action_id", layer_column]].copy()
    lookup = lookup[lookup[layer_column].astype(str).ne("")]
    if lookup.empty:
        return evaluated_pairs.iloc[0:0].copy()
    return evaluated_pairs.merge(
        lookup,
        left_on=["action_id", "defense_id"],
        right_on=["action_id", layer_column],
        how="inner",
    )


def _build_top1_audit_summary(matchup_table: pd.DataFrame, evaluated_pairs: pd.DataFrame) -> pd.DataFrame:
    layer_specs = [
        ("active_top1", "active_top1_defense_id"),
        ("fallback_top1", "fallback_top1_defense_id"),
        ("ground_top1", "ground_top1_defense_id"),
    ]
    records: list[dict[str, object]] = []
    for layer_name, layer_column in layer_specs:
        layer_pairs = _lookup_top1_pairs(matchup_table, evaluated_pairs, layer_column)
        pair_count = int(len(layer_pairs))
        route_relevant = layer_pairs[_route_relevant_mask(layer_pairs)] if pair_count else layer_pairs
        records.append(
            {
                "layer_name": layer_name,
                "pair_count": pair_count,
                "family_rule_hit_pairs": int(layer_pairs["family_rule_hit"].sum()) if pair_count else 0,
                "route_relevant_pairs": int(len(route_relevant)),
                "route_rule_hit_pairs": int(route_relevant["route_rule_hit"].sum()) if not route_relevant.empty else 0,
                "geo_cap_applied_pairs": int(layer_pairs["geo_cap_applied"].sum()) if pair_count else 0,
                "non_normal_pair_count": int((layer_pairs["pair_audit_flag"] != "normal").sum()) if pair_count else 0,
                "family_rule_hit_rate": float(layer_pairs["family_rule_hit"].mean()) if pair_count else 0.0,
                "route_rule_hit_rate": float(route_relevant["route_rule_hit"].mean()) if not route_relevant.empty else 1.0,
                "geo_cap_applied_rate": float(layer_pairs["geo_cap_applied"].mean()) if pair_count else 0.0,
                "non_normal_pair_rate": float((layer_pairs["pair_audit_flag"] != "normal").mean()) if pair_count else 0.0,
            }
        )
    return pd.DataFrame(records)


def _build_rule_coverage_audit(evaluated_pairs: pd.DataFrame, layer_table: pd.DataFrame) -> pd.DataFrame:
    scopes: list[tuple[str, pd.DataFrame]] = [
        ("all_pairs", evaluated_pairs),
        ("active_pool", evaluated_pairs[evaluated_pairs["primary_role_feasible"]]),
        ("fallback_pool", evaluated_pairs[evaluated_pairs["fallback_feasible"]]),
    ]
    active_column = "active_top1_defense_id" if "active_top1_defense_id" in layer_table.columns else "method4_active_top1"
    active_lookup = layer_table[["action_id", active_column]].copy()
    active_lookup = active_lookup[active_lookup[active_column].astype(str).ne("")]
    if not active_lookup.empty:
        active_top1 = evaluated_pairs.merge(
            active_lookup,
            left_on=["action_id", "defense_id"],
            right_on=["action_id", active_column],
            how="inner",
        )
        scopes.append(("final_active_top1", active_top1))

    records: list[dict[str, object]] = []
    for scope_name, scope_frame in scopes:
        total_pairs = int(len(scope_frame))
        if total_pairs == 0:
            records.append(
                {
                    "scope_name": scope_name,
                    "pair_count": 0,
                    "family_rule_hit_rate": 0.0,
                    "route_rule_hit_rate": 0.0,
                    "geo_cap_rate": 0.0,
                    "missing_family_pairs": 0,
                    "missing_route_pairs": 0,
                    "capped_pairs": 0,
                }
            )
            continue
        route_relevant = scope_frame[_route_relevant_mask(scope_frame)]
        route_hit_rate = float(route_relevant["route_rule_hit"].mean()) if not route_relevant.empty else 1.0
        records.append(
            {
                "scope_name": scope_name,
                "pair_count": total_pairs,
                "family_rule_hit_rate": float(scope_frame["family_rule_hit"].mean()),
                "route_rule_hit_rate": route_hit_rate,
                "geo_cap_rate": float(scope_frame["geo_cap_applied"].mean()),
                "missing_family_pairs": int((~scope_frame["family_rule_hit"]).sum()),
                "missing_route_pairs": int((~route_relevant["route_rule_hit"]).sum()) if not route_relevant.empty else 0,
                "capped_pairs": int(scope_frame["geo_cap_applied"].sum()),
            }
        )
    return pd.DataFrame(records)


def _build_active_distribution(method_summary: pd.DataFrame, defense_features: pd.DataFrame) -> pd.DataFrame:
    active_ids = method_summary["method4_active_top1"].astype(str)
    distribution = (
        active_ids[active_ids.ne("")]
        .value_counts()
        .rename_axis("defense_id")
        .reset_index(name="count_as_active_top1")
    )
    category_map = defense_features.set_index("defense_id")["defense_category"].to_dict()
    distribution["category"] = distribution["defense_id"].map(category_map).fillna("NA")
    return distribution


def _build_acceptance_checks(
    evaluated_pairs: pd.DataFrame,
    method_summary: pd.DataFrame,
    defense_features: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    summary_map = method_summary.set_index("action_id").to_dict("index")
    active_distribution = _build_active_distribution(method_summary, defense_features)
    active_categories = set(active_distribution["category"].tolist())
    defense_role_map = defense_features.set_index("defense_id")["defense_role"].to_dict()

    checks.append(
        {
            "check_id": "A13_has_standing_response",
            "passed": str(summary_map["A13"]["method4_active_top1"]).strip() != "",
            "detail": f"active={summary_map['A13']['method4_active_top1']}",
        }
    )
    checks.append(
        {
            "check_id": "A13_not_forced_to_ground_layer",
            "passed": str(summary_map["A13"]["method4_ground_top1"]).strip() == "",
            "detail": (
                f"ground={summary_map['A13']['method4_ground_top1']}, "
                f"recovery={summary_map['A13']['method4_recovery_if_needed']}"
            ),
        }
    )
    checks.append(
        {
            "check_id": "A05_not_D02",
            "passed": str(summary_map["A05"]["method4_active_top1"]) != "D02",
            "detail": f"active={summary_map['A05']['method4_active_top1']}",
        }
    )
    checks.append(
        {
            "check_id": "A12_not_D01",
            "passed": str(summary_map["A12"]["method4_active_top1"]) != "D01",
            "detail": f"active={summary_map['A12']['method4_active_top1']}",
        }
    )
    checks.append(
        {
            "check_id": "A11_active_combo_not_blank",
            "passed": str(summary_map["A11"]["method4_active_top1"]).strip() != "",
            "detail": f"active={summary_map['A11']['method4_active_top1']}",
        }
    )
    max_active_count = int(active_distribution["count_as_active_top1"].max()) if not active_distribution.empty else 0
    checks.append(
        {
            "check_id": "active_not_monopolized",
            "passed": max_active_count <= 4,
            "detail": f"max_active_count={max_active_count}",
        }
    )
    category_ok = {"block", "evade", "combo"}.issubset(active_categories)
    checks.append(
        {
            "check_id": "active_category_diversity",
            "passed": category_ok,
            "detail": f"active_categories={sorted(active_categories)}",
        }
    )
    posture_balance_in_active = active_distribution[
        active_distribution["category"].isin(["posture", "balance", "ground"])
    ]
    checks.append(
        {
            "check_id": "posture_balance_outside_active",
            "passed": posture_balance_in_active.empty,
            "detail": f"unexpected_active={posture_balance_in_active.to_dict(orient='records')}",
        }
    )
    layer_isolation_passed = True
    for _, row in method_summary.iterrows():
        active_id = str(row["method4_active_top1"]).strip()
        fallback_id = str(row["method4_fallback_top1"]).strip()
        ground_id = str(row["method4_ground_top1"]).strip()
        recovery_id = str(row["method4_recovery_if_needed"]).strip()
        if active_id and defense_role_map.get(active_id) in {"ground_only", "recovery_only"}:
            layer_isolation_passed = False
        if fallback_id and defense_role_map.get(fallback_id) in {"ground_only", "recovery_only"}:
            layer_isolation_passed = False
        if ground_id and defense_role_map.get(ground_id) != "ground_only":
            layer_isolation_passed = False
        if recovery_id and defense_role_map.get(recovery_id) != "recovery_only":
            layer_isolation_passed = False
    checks.append(
        {
            "check_id": "ground_recovery_layer_isolated",
            "passed": layer_isolation_passed,
            "detail": "ground/recovery roles only appear in their corresponding layers",
        }
    )
    rule_coverage_audit = _build_rule_coverage_audit(evaluated_pairs, method_summary)
    final_active_row = rule_coverage_audit.loc[rule_coverage_audit["scope_name"] == "final_active_top1"]
    final_active_rule_ok = False
    detail = "no_final_active_scope"
    if not final_active_row.empty:
        family_hit_rate = float(final_active_row.iloc[0]["family_rule_hit_rate"])
        route_hit_rate = float(final_active_row.iloc[0]["route_rule_hit_rate"])
        final_active_rule_ok = family_hit_rate >= 0.90 and route_hit_rate >= 0.90
        detail = f"family_hit_rate={family_hit_rate:.3f}, route_hit_rate={route_hit_rate:.3f}"
    checks.append(
        {
            "check_id": "final_active_rule_coverage",
            "passed": final_active_rule_ok,
            "detail": detail,
        }
    )
    no_feasible_methods_emit_na = all(
        str(summary_map["A13"][column]).strip() != ""
        for column in ["method1_top1_defense", "method2_top1_defense", "method3_top1_defense"]
    )
    checks.append(
        {
            "check_id": "A13_baseline_methods_not_blank",
            "passed": no_feasible_methods_emit_na,
            "detail": (
                f"A13_method1={summary_map['A13']['method1_top1_defense']}, "
                f"A13_method2={summary_map['A13']['method2_top1_defense']}, "
                f"A13_method3={summary_map['A13']['method3_top1_defense']}"
            ),
        }
    )
    return pd.DataFrame(checks)


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_basic_dirs()
    robot = load_robot_params(Q1_ROBOT_FILE)
    segments = load_segment_params(Q1_SEGMENT_FILE)
    support_modes = load_support_mode_config(Q1_SUPPORT_FILE)
    action_features = load_action_features(Q1_ACTION_FILE)
    action_templates = load_action_templates(Q1_ACTION_TEMPLATE_FILE)
    phase_templates = load_action_phase_templates(Q1_PHASE_TEMPLATE_FILE)
    attack_semantics = load_attack_semantics(Q2_ATTACK_SEMANTIC_FILE)
    attack_response_policy = load_attack_response_policy(Q2_ATTACK_RESPONSE_FILE)
    defense_actions = load_defense_actions(Q2_DEFENSE_FILE)
    route_advantage = load_route_advantage(Q2_ROUTE_FILE)
    family_compatibility = load_family_compatibility(Q2_FAMILY_FILE)

    attack_catalog = build_attack_catalog(
        action_features=action_features,
        robot=robot,
        action_templates=action_templates,
        phase_templates=phase_templates,
        attack_semantics=attack_semantics,
        attack_response_policy=attack_response_policy,
    )
    defense_features = build_defense_feature_table(
        defense_actions=defense_actions,
        robot=robot,
        segments=segments,
        support_modes=support_modes,
        attack_catalog=attack_catalog,
    )
    pair_matrix = build_pair_matrix(
        attack_catalog=attack_catalog,
        defense_features=defense_features,
        route_advantage=route_advantage,
        family_compatibility=family_compatibility,
        robot=robot,
    )
    evaluated_pairs, matchup_table, counter_chain_table, method_summary_table = evaluate_all_methods(pair_matrix=pair_matrix)

    attack_contact_features = attack_catalog[
        [
            "action_id",
            "contact_phase_no",
            "contact_phase_name",
            "contact_phase_time_ratio",
            "contact_fraction_in_phase",
            "t_contact",
            "t_end",
            "t_recover",
            "opp_recover_time",
            "contact_phase_theta_deg",
            "contact_translation_share",
            "contact_support_mode",
            "contact_strike_segments_json",
            "contact_active_segments_json",
            "contact_reach_m",
            "contact_load_family",
            "contact_plane",
            "response_policy_source",
        ]
    ].copy()
    attack_contact_debug = attack_catalog[
        [
            "action_id",
            "contact_phase_no",
            "contact_phase_name",
            "t_contact",
            "t_recover",
            "opp_recover_time",
            "contact_reach_m",
            "contact_support_mode",
        ]
    ].copy()
    defense_clearance_debug = defense_features[
        [
            "defense_id",
            "route_type",
            "exec_time_def",
            "clear_back_m",
            "clear_lateral_m",
            "clear_orbit_m",
            "clear_drop_m",
            "J_cap",
            "E_cap",
            "absorb_base",
            "mobility_cost",
        ]
    ].copy()
    pair_top5_debug = _build_pair_top5_debug(evaluated_pairs)
    zero_tie_audit = _build_zero_tie_audit(evaluated_pairs, method_summary_table)
    family_audit = _build_family_audit(evaluated_pairs)
    active_distribution = _build_active_distribution(method_summary_table, defense_features)
    acceptance_checks = _build_acceptance_checks(evaluated_pairs, method_summary_table, defense_features)
    rule_coverage_audit = _build_rule_coverage_audit(evaluated_pairs, matchup_table)
    rule_audit_summary = _build_rule_audit_summary(evaluated_pairs)
    top1_audit_summary = _build_top1_audit_summary(matchup_table, evaluated_pairs)

    _save_csv(defense_features, DEFENSE_FEATURE_FILE)
    _save_csv(evaluated_pairs, PAIR_SCORE_FILE)
    _save_csv(matchup_table, MATCHUP_FILE)
    _save_csv(counter_chain_table, COUNTER_CHAIN_FILE)
    _save_csv(method_summary_table, METHOD_SUMMARY_FILE)
    _save_csv(attack_contact_features, ATTACK_CONTACT_FEATURE_FILE)
    _save_csv(attack_contact_debug, ATTACK_CONTACT_DEBUG_FILE)
    _save_csv(defense_clearance_debug, DEFENSE_CLEARANCE_DEBUG_FILE)
    _save_csv(pair_top5_debug, PAIR_TOP5_DEBUG_FILE)
    _save_csv(zero_tie_audit, ZERO_TIE_AUDIT_FILE)
    _save_csv(family_audit, FAMILY_AUDIT_FILE)
    _save_csv(active_distribution, ACTIVE_DISTRIBUTION_FILE)
    _save_csv(acceptance_checks, ACCEPTANCE_CHECK_FILE)
    _save_csv(rule_coverage_audit, RULE_COVERAGE_AUDIT_FILE)
    _save_csv(rule_audit_summary, RULE_AUDIT_SUMMARY_FILE)
    _save_csv(top1_audit_summary, TOP1_AUDIT_SUMMARY_FILE)

    plot_primary_utility_matrix(evaluated_pairs, UTILITY_MATRIX_FIGURE)
    plot_defense_surface(matchup_table, evaluated_pairs, SURFACE_FIGURE)
    plot_parallel_metrics(matchup_table, evaluated_pairs, PARALLEL_FIGURE)
    plot_decision_waterfall(matchup_table, WATERFALL_FIGURE)
    plot_method_comparison(method_summary_table, METHOD_COMPARE_FIGURE)
    plot_layered_response_overview(method_summary_table, LAYERED_OVERVIEW_FIGURE)
    return evaluated_pairs, matchup_table, counter_chain_table


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_pipeline()


if __name__ == "__main__":
    main()
