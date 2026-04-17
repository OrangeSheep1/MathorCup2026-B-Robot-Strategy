"""Q1 四种方法的评价与排序模块。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


AHP_MATRIX = np.array(
    [
        [1.0, 1.0 / 2.0, 2.0, 4.0],
        [2.0, 1.0, 3.0, 5.0],
        [1.0 / 2.0, 1.0 / 3.0, 1.0, 3.0],
        [1.0 / 4.0, 1.0 / 5.0, 1.0 / 3.0, 1.0],
    ],
    dtype=float,
)

FUZZY_GRADE_VECTOR = np.array([1.0, 0.6, 0.2], dtype=float)

PROPOSED_WEIGHTS = {
    "impact": 0.35,
    "hit": 0.25,
    "energy": 0.20,
    "time": 0.20,
}

FALL_PENALTY_CONFIG = {
    "lambda": 0.30,
    "k": 3.50,
}


def normalize_series(series: pd.Series, reverse: bool = False) -> pd.Series:
    """对序列做极差标准化。"""

    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(maximum, minimum):
        normalized = pd.Series(np.full(len(series), 0.5), index=series.index, dtype=float)
    else:
        normalized = (series - minimum) / (maximum - minimum)
    return 1.0 - normalized if reverse else normalized


def rank_descending(series: pd.Series) -> pd.Series:
    """按降序生成并列保持的排名。"""

    return series.rank(method="min", ascending=False).astype(int)


def compute_method1_scores(data: pd.DataFrame) -> pd.DataFrame:
    """方法一：单维度物理力学评分。"""

    evaluated = data.copy()
    evaluated["method1_score"] = evaluated["impact_score"]
    evaluated["method1_utility"] = normalize_series(evaluated["method1_score"])
    evaluated["method1_rank"] = rank_descending(evaluated["method1_score"])
    return evaluated


def compute_ahp_weights(matrix: np.ndarray = AHP_MATRIX) -> dict[str, float]:
    """计算 AHP 权重及一致性指标。"""

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = int(np.argmax(eigenvalues.real))
    weight_vector = eigenvectors[:, max_index].real
    weight_vector = weight_vector / weight_vector.sum()
    lambda_max = float(eigenvalues[max_index].real)
    ci = (lambda_max - matrix.shape[0]) / (matrix.shape[0] - 1)
    cr = ci / 0.90
    return {
        "w_tau": float(weight_vector[0]),
        "w_stability": float(weight_vector[1]),
        "w_hit": float(weight_vector[2]),
        "w_efficiency": float(weight_vector[3]),
        "lambda_max": lambda_max,
        "ci": float(ci),
        "cr": float(cr),
    }


def compute_method2_scores(data: pd.DataFrame, ahp_weights: Mapping[str, float]) -> pd.DataFrame:
    """方法二：AHP 多准则评分。"""

    evaluated = data.copy()
    evaluated["tau_norm"] = normalize_series(evaluated["impact_score"])
    evaluated["delta_com_norm"] = normalize_series(evaluated["balance_cost"])
    evaluated["energy_norm"] = normalize_series(evaluated["energy_cost"])
    evaluated["time_norm"] = normalize_series(evaluated["exec_time"])
    evaluated["stability_score"] = 1.0 - evaluated["delta_com_norm"]
    evaluated["efficiency_score"] = 1.0 - evaluated["energy_norm"]
    evaluated["method2_score"] = (
        ahp_weights["w_tau"] * evaluated["tau_norm"]
        + ahp_weights["w_stability"] * evaluated["stability_score"]
        + ahp_weights["w_hit"] * evaluated["score_prob"]
        + ahp_weights["w_efficiency"] * evaluated["efficiency_score"]
    )
    evaluated["method2_utility"] = normalize_series(evaluated["method2_score"])
    evaluated["method2_rank"] = rank_descending(evaluated["method2_score"])
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

    if 0.2 <= value <= 0.5:
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
    """构建单个指标的模糊隶属度向量。"""

    return np.array(
        [
            membership_high(value),
            membership_mid(value),
            membership_low(value),
        ],
        dtype=float,
    )


def compute_method3_scores(data: pd.DataFrame, ahp_weights: Mapping[str, float]) -> pd.DataFrame:
    """方法三：模糊综合评价。"""

    evaluated = data.copy()
    fuzzy_scores = []
    indicator_weights = np.array(
        [
            ahp_weights["w_tau"],
            ahp_weights["w_stability"],
            ahp_weights["w_hit"],
            ahp_weights["w_efficiency"],
        ],
        dtype=float,
    )

    for _, row in evaluated.iterrows():
        indicator_values = [
            row["tau_norm"],
            row["stability_score"],
            row["score_prob"],
            row["efficiency_score"],
        ]
        membership_matrix = np.vstack([build_membership_row(value) for value in indicator_values])
        fuzzy_vector = indicator_weights @ membership_matrix
        fuzzy_scores.append(float(fuzzy_vector @ FUZZY_GRADE_VECTOR))

    evaluated["method3_score"] = fuzzy_scores
    evaluated["method3_utility"] = normalize_series(evaluated["method3_score"])
    evaluated["method3_rank"] = rank_descending(evaluated["method3_score"])
    return evaluated


def compute_base_utility(
    data: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """计算方法四的基础效用项。"""

    active_weights = dict(PROPOSED_WEIGHTS if weights is None else weights)
    evaluated = data.copy()
    evaluated["u0"] = (
        active_weights["impact"] * evaluated["tau_norm"]
        + active_weights["hit"] * evaluated["score_prob"]
        - active_weights["energy"] * evaluated["energy_norm"]
        - active_weights["time"] * evaluated["time_norm"]
    )
    return evaluated


def compute_fall_penalty_raw(
    balance_cost: pd.Series,
    stable_margin: float,
    penalty_config: Mapping[str, float] | None = None,
) -> pd.Series:
    """计算指数型原始倒地惩罚。"""

    config = dict(FALL_PENALTY_CONFIG if penalty_config is None else penalty_config)
    ratio = balance_cost / stable_margin
    return config["lambda"] * np.exp(config["k"] * ratio) - config["lambda"]


def compute_penalty_normalized(penalty_raw: pd.Series, u0: pd.Series) -> pd.Series:
    """将原始惩罚量纲对齐到基础效用。"""

    penalty_max = float(penalty_raw.max())
    utility_max = float(u0.max())
    if np.isclose(penalty_max, 0.0):
        return pd.Series(np.zeros(len(penalty_raw)), index=penalty_raw.index, dtype=float)
    return penalty_raw / penalty_max * utility_max


def compute_final_utility(
    data: pd.DataFrame,
    stable_margin: float,
    weights: Mapping[str, float] | None = None,
    penalty_config: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """方法四：双目标效用函数 + 非线性平衡惩罚。"""

    active_weights = dict(PROPOSED_WEIGHTS if weights is None else weights)
    evaluated = compute_base_utility(data, weights=active_weights)
    evaluated["fall_penalty_raw"] = compute_fall_penalty_raw(
        balance_cost=evaluated["balance_cost"],
        stable_margin=stable_margin,
        penalty_config=penalty_config,
    )
    evaluated["fall_penalty"] = compute_penalty_normalized(
        penalty_raw=evaluated["fall_penalty_raw"],
        u0=evaluated["u0"],
    )
    evaluated["benefit_term"] = (
        active_weights["impact"] * evaluated["tau_norm"]
        + active_weights["hit"] * evaluated["score_prob"]
    )
    evaluated["cost_term"] = (
        active_weights["energy"] * evaluated["energy_norm"]
        + active_weights["time"] * evaluated["time_norm"]
    )
    evaluated["proposed_score"] = (evaluated["u0"] - evaluated["fall_penalty"]).clip(lower=0.0)
    if np.isclose(float(evaluated["proposed_score"].max()), 0.0):
        evaluated["utility"] = 0.0
    else:
        evaluated["utility"] = evaluated["proposed_score"] / float(evaluated["proposed_score"].max())
    evaluated["rank"] = rank_descending(evaluated["proposed_score"])
    return evaluated


def evaluate_all_methods(
    data: pd.DataFrame,
    stable_margin: float,
    weights: Mapping[str, float] | None = None,
    penalty_config: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """串联四种方法并返回结果表与 AHP 摘要。"""

    ahp_summary = compute_ahp_weights()
    evaluated = compute_method1_scores(data)
    evaluated = compute_method2_scores(evaluated, ahp_summary)
    evaluated = compute_method3_scores(evaluated, ahp_summary)
    evaluated = compute_final_utility(
        evaluated,
        stable_margin=stable_margin,
        weights=weights,
        penalty_config=penalty_config,
    )
    return evaluated.sort_values(by="rank", ascending=True, ignore_index=True), ahp_summary


def sensitivity_scan(
    data: pd.DataFrame,
    stable_margin: float,
    lambda_values: np.ndarray | None = None,
    k_values: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """执行惩罚参数灵敏度分析。"""

    if lambda_values is None:
        lambda_values = np.linspace(0.20, 0.40, 5)
    if k_values is None:
        k_values = np.linspace(3.0, 4.0, 5)

    baseline = compute_final_utility(data, stable_margin=stable_margin)
    baseline_top3 = set(baseline.nlargest(3, "proposed_score")["action_id"].tolist())

    overlap_rows = []
    spinning_rows = []
    for lambda_value in lambda_values:
        overlap_row = []
        spinning_row = []
        for k_value in k_values:
            tested = compute_final_utility(
                data,
                stable_margin=stable_margin,
                penalty_config={"lambda": float(lambda_value), "k": float(k_value)},
            )
            current_top3 = set(tested.nlargest(3, "proposed_score")["action_id"].tolist())
            overlap_rate = len(baseline_top3 & current_top3) / 3.0
            spinning_score = float(
                tested.loc[tested["action_id"] == "A07", "utility"].iloc[0]
            )
            overlap_row.append(overlap_rate)
            spinning_row.append(spinning_score)
        overlap_rows.append(overlap_row)
        spinning_rows.append(spinning_row)

    lambda_index = [f"{value:.2f}" for value in lambda_values]
    k_columns = [f"{value:.2f}" for value in k_values]
    return {
        "top3_overlap": pd.DataFrame(overlap_rows, index=lambda_index, columns=k_columns),
        "spinning_utility": pd.DataFrame(spinning_rows, index=lambda_index, columns=k_columns),
    }
