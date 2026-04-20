"""Q2 全流程流水线。"""

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
from src.q2.evaluate import evaluate_all_methods
from src.q2.model_v1 import (
    build_attack_catalog,
    build_defense_feature_table,
    build_pair_matrix,
    load_action_features,
    load_attack_response_policy,
    load_attack_semantics,
    load_defense_actions,
    load_route_advantage,
)
from src.q2.plot import (
    plot_decision_waterfall,
    plot_defense_surface,
    plot_hierarchical_utility_matrix,
    plot_method_comparison,
    plot_parallel_bands,
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

DEFENSE_FEATURE_FILE = INTERIM_DIR / "defense_features.csv"
PAIR_SCORE_FILE = INTERIM_DIR / "defense_pair_scores.csv"
MATCHUP_FILE = INTERIM_DIR / "defense_matchup.csv"
COUNTER_CHAIN_FILE = INTERIM_DIR / "counter_chain.csv"
METHOD_SUMMARY_FILE = INTERIM_DIR / "q2_method_summary.csv"
ATTACK_CONTACT_FEATURE_FILE = INTERIM_DIR / "q2_attack_contact_features.csv"
ATTACK_CONTACT_DEBUG_FILE = INTERIM_DIR / "q2_attack_contact_debug.csv"
DEFENSE_CLEARANCE_DEBUG_FILE = INTERIM_DIR / "q2_defense_clearance_debug.csv"
PAIR_TOP5_DEBUG_FILE = INTERIM_DIR / "q2_pair_top5_debug.csv"
ACCEPTANCE_CHECK_FILE = INTERIM_DIR / "q2_acceptance_checks.csv"

UTILITY_MATRIX_FIGURE = OUTPUT_DIR / "q2_utility_matrix.png"
SURFACE_FIGURE = OUTPUT_DIR / "q2_surface.png"
WATERFALL_FIGURE = OUTPUT_DIR / "q2_waterfall.png"
PARALLEL_FIGURE = OUTPUT_DIR / "q2_parallel.png"
METHOD_COMPARE_FIGURE = OUTPUT_DIR / "q2_method_comparison.png"


def _save_csv(data: pd.DataFrame, output_path: str | Path) -> Path:
    """统一保存 CSV。"""

    output = Path(output_path)
    data.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def _build_pair_top5_debug(evaluated_pairs: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for action_id, group in evaluated_pairs.groupby("action_id", sort=False):
        active_pool = group[group["primary_role_feasible"]].sort_values(
            by=["proposed_score", "p_success", "counter_window"],
            ascending=[False, False, False],
        )
        if active_pool.empty:
            active_pool = group[group["state_feasible"]].sort_values(
                by=["fallback_score", "p_success", "stability_safe"],
                ascending=[False, False, False],
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
                    "p_clear",
                    "p_load",
                    "p_success",
                    "residual_damage",
                    "p_fall",
                    "proposed_score",
                    "fallback_score",
                ]
            ].rename(columns={"proposed_score": "primary_score"})
        )
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _pair_row(evaluated_pairs: pd.DataFrame, action_id: str, defense_id: str) -> pd.Series | None:
    subset = evaluated_pairs[
        (evaluated_pairs["action_id"] == action_id)
        & (evaluated_pairs["defense_id"] == defense_id)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def _build_acceptance_checks(
    evaluated_pairs: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    summary_map = method_summary.set_index("action_id").to_dict("index")
    a13 = summary_map["A13"]
    checks.append(
        {
            "check_id": "A13_ground_recovery",
            "passed": (
                str(a13["method4_active_top1"]) == ""
                and str(a13["method4_ground_top1"]) == "D19"
                and str(a13["method4_recovery_if_needed"]) == "D18"
            ),
            "detail": (
                f"active={a13['method4_active_top1']}, "
                f"ground={a13['method4_ground_top1']}, "
                f"recovery={a13['method4_recovery_if_needed']}"
            ),
        }
    )

    a12_d06 = _pair_row(evaluated_pairs, "A12", "D06")
    a12_d08 = _pair_row(evaluated_pairs, "A12", "D08")
    a12_d15 = _pair_row(evaluated_pairs, "A12", "D15")
    checks.append(
        {
            "check_id": "A12_not_D08_dominated",
            "passed": (
                a12_d06 is not None
                and a12_d08 is not None
                and float(a12_d06["primary_score"]) >= float(a12_d08["primary_score"])
                and a12_d15 is not None
                and float(a12_d15["p_success"]) >= 0.5
            ),
            "detail": (
                f"D06_primary={0.0 if a12_d06 is None else float(a12_d06['primary_score']):.4f}, "
                f"D08_primary={0.0 if a12_d08 is None else float(a12_d08['primary_score']):.4f}, "
                f"D15_p_success={0.0 if a12_d15 is None else float(a12_d15['p_success']):.4f}"
            ),
        }
    )

    a07_d10 = _pair_row(evaluated_pairs, "A07", "D10")
    a07_d08 = _pair_row(evaluated_pairs, "A07", "D08")
    checks.append(
        {
            "check_id": "A07_D10_competes_or_wins",
            "passed": (
                a07_d10 is not None
                and a07_d08 is not None
                and (
                    float(a07_d10["p_success"]) >= float(a07_d08["p_success"])
                    or float(a07_d10["primary_score"]) >= float(a07_d08["primary_score"])
                )
            ),
            "detail": (
                f"D10_p_success={0.0 if a07_d10 is None else float(a07_d10['p_success']):.4f}, "
                f"D08_p_success={0.0 if a07_d08 is None else float(a07_d08['p_success']):.4f}, "
                f"D10_primary={0.0 if a07_d10 is None else float(a07_d10['primary_score']):.4f}, "
                f"D08_primary={0.0 if a07_d08 is None else float(a07_d08['primary_score']):.4f}"
            ),
        }
    )

    a08 = summary_map["A08"]
    checks.append(
        {
            "check_id": "A08_D04_D16_front",
            "passed": (
                str(a08["method4_active_top1"]) == "D04"
                and str(a08["method4_fallback_top1"]) == "D16"
            ),
            "detail": (
                f"active={a08['method4_active_top1']}, "
                f"fallback={a08['method4_fallback_top1']}"
            ),
        }
    )

    d14_freq = int((method_summary["method4_fallback_top1"].astype(str) == "D14").sum())
    checks.append(
        {
            "check_id": "D14_fallback_frequency",
            "passed": d14_freq <= 5,
            "detail": f"D14_fallback_count={d14_freq}",
        }
    )

    d08_freq = int((method_summary["method4_active_top1"].astype(str) == "D08").sum())
    checks.append(
        {
            "check_id": "D08_active_frequency",
            "passed": d08_freq <= 4,
            "detail": f"D08_active_count={d08_freq}",
        }
    )

    return pd.DataFrame(checks)


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """执行 Q2 全流程。"""

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
        robot=robot,
    )
    evaluated_pairs, matchup_table, counter_chain_table, method_summary_table = evaluate_all_methods(
        pair_matrix=pair_matrix,
    )
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
    acceptance_checks = _build_acceptance_checks(evaluated_pairs, method_summary_table)

    _save_csv(defense_features, DEFENSE_FEATURE_FILE)
    _save_csv(evaluated_pairs, PAIR_SCORE_FILE)
    _save_csv(matchup_table, MATCHUP_FILE)
    _save_csv(counter_chain_table, COUNTER_CHAIN_FILE)
    _save_csv(method_summary_table, METHOD_SUMMARY_FILE)
    _save_csv(attack_contact_features, ATTACK_CONTACT_FEATURE_FILE)
    _save_csv(attack_contact_debug, ATTACK_CONTACT_DEBUG_FILE)
    _save_csv(defense_clearance_debug, DEFENSE_CLEARANCE_DEBUG_FILE)
    _save_csv(pair_top5_debug, PAIR_TOP5_DEBUG_FILE)
    _save_csv(acceptance_checks, ACCEPTANCE_CHECK_FILE)

    plot_hierarchical_utility_matrix(evaluated_pairs, counter_chain_table, UTILITY_MATRIX_FIGURE)
    plot_defense_surface(evaluated_pairs, SURFACE_FIGURE)
    plot_decision_waterfall(matchup_table, evaluated_pairs, WATERFALL_FIGURE)
    plot_parallel_bands(evaluated_pairs, PARALLEL_FIGURE)
    plot_method_comparison(evaluated_pairs, METHOD_COMPARE_FIGURE)
    return evaluated_pairs, matchup_table, counter_chain_table


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """命令行友好的 Q2 入口。"""

    return run_pipeline()


if __name__ == "__main__":
    main()
