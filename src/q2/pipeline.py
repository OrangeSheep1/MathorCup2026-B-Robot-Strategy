"""Q2 全流程流水线。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import INTERIM_DIR, OUTPUT_DIR, RAW_DIR, ensure_basic_dirs
from src.q1.model_v1 import load_robot_params
from src.q2.evaluate import evaluate_all_methods
from src.q2.model_v1 import (
    build_attack_catalog,
    build_defense_feature_table,
    build_pair_matrix,
    load_action_features,
    load_attack_semantics,
    load_defense_actions,
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
Q2_ATTACK_SEMANTIC_FILE = RAW_DIR / "q2_attack_semantics.csv"
Q2_DEFENSE_FILE = RAW_DIR / "q2_defense_actions.csv"

DEFENSE_FEATURE_FILE = INTERIM_DIR / "defense_features.csv"
PAIR_SCORE_FILE = INTERIM_DIR / "defense_pair_scores.csv"
MATCHUP_FILE = INTERIM_DIR / "defense_matchup.csv"
COUNTER_CHAIN_FILE = INTERIM_DIR / "counter_chain.csv"
METHOD_SUMMARY_FILE = INTERIM_DIR / "q2_method_summary.csv"

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


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """执行 Q2 全流程。"""

    ensure_basic_dirs()
    robot = load_robot_params(Q1_ROBOT_FILE)
    action_features = load_action_features(Q1_ACTION_FILE)
    attack_semantics = load_attack_semantics(Q2_ATTACK_SEMANTIC_FILE)
    defense_actions = load_defense_actions(Q2_DEFENSE_FILE)

    attack_catalog = build_attack_catalog(action_features, attack_semantics)
    defense_features = build_defense_feature_table(defense_actions, robot)
    pair_matrix = build_pair_matrix(attack_catalog, defense_features, robot)
    evaluated_pairs, matchup_table, counter_chain_table, method_summary_table = evaluate_all_methods(
        pair_matrix=pair_matrix,
        action_catalog=attack_catalog,
    )

    _save_csv(defense_features, DEFENSE_FEATURE_FILE)
    _save_csv(evaluated_pairs, PAIR_SCORE_FILE)
    _save_csv(matchup_table, MATCHUP_FILE)
    _save_csv(counter_chain_table, COUNTER_CHAIN_FILE)
    _save_csv(method_summary_table, METHOD_SUMMARY_FILE)

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
