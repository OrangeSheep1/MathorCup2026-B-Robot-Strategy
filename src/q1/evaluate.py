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
    "score": 0.25,
    "work": 0.20,
    "exposure": 0.20,
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


def compute_base_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """计算四方法共享的标准化指标。"""

    evaluated = data.copy()
    evaluated["impact_impulse_norm"] = normalize_series(np.log1p(evaluated["impact_impulse"]))
    evaluated["impact_kinetic_norm"] = normalize_series(np.log1p(evaluated["impact_kinetic"]))
    evaluated["impact_effect"] = 0.55 * evaluated["impact_impulse_norm"] + 0.45 * evaluated["impact_kinetic_norm"]
    evaluated["score_norm"] = normalize_series(evaluated["score_potential"])
    evaluated["work_norm"] = normalize_series(np.log1p(evaluated["work_cost"]))
    evaluated["exposure_norm"] = normalize_series(evaluated["exposure_index"])
    evaluated["stability_score"] = normalize_series(evaluated["zmp_margin_norm"])
    evaluated["cost_composite"] = 0.5 * evaluated["work_norm"] + 0.5 * evaluated["exposure_norm"]
    evaluated["execution_efficiency"] = 1.0 - evaluated["cost_composite"]

    # 兼容下游模块仍在引用的旧字段。
    evaluated["tau_norm"] = evaluated["impact_effect"]
    evaluated["delta_com_norm"] = normalize_series(evaluated["com_shift_max"])
    evaluated["energy_norm"] = evaluated["work_norm"]
    evaluated["time_norm"] = normalize_series(evaluated["exec_time"])
    evaluated["efficiency_score"] = evaluated["execution_efficiency"]
    evaluated["p_reach"] = 1.0 - normalize_series(evaluated["exposure_index"])
    return evaluated


def compute_method1_scores(data: pd.DataFrame) -> pd.DataFrame:
    """方法一：纯物理冲量排序。"""

    evaluated = data.copy()
    evaluated["method1_score"] = evaluated["impact_impulse"]
    evaluated["method1_utility"] = normalize_series(np.log1p(evaluated["method1_score"]))
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
        "w_impact": float(weight_vector[0]),
        "w_stability": float(weight_vector[1]),
        "w_score": float(weight_vector[2]),
        "w_efficiency": float(weight_vector[3]),
        "lambda_max": lambda_max,
        "ci": float(ci),
        "cr": float(cr),
    }


def compute_entropy_weights(data: pd.DataFrame) -> dict[str, float]:
    """基于指标离散度计算熵权。"""

    indicator_columns = ["impact_effect", "stability_score", "score_norm", "execution_efficiency"]
    matrix = data.loc[:, indicator_columns].to_numpy(dtype=float)
    matrix = matrix + 1e-12
    proportion = matrix / matrix.sum(axis=0, keepdims=True)
    entropy = -(proportion * np.log(proportion)).sum(axis=0) / np.log(len(data))
    divergence = 1.0 - entropy
    weights = divergence / divergence.sum()
    return {
        "w_impact": float(weights[0]),
        "w_stability": float(weights[1]),
        "w_score": float(weights[2]),
        "w_efficiency": float(weights[3]),
    }


def compute_method2_scores(data: pd.DataFrame, ahp_weights: Mapping[str, float]) -> pd.DataFrame:
    """方法二：AHP 与熵权混合评分。"""

    evaluated = data.copy()
    entropy_weights = compute_entropy_weights(evaluated)
    hybrid_weights = {
        "impact": 0.5 * ahp_weights["w_impact"] + 0.5 * entropy_weights["w_impact"],
        "stability": 0.5 * ahp_weights["w_stability"] + 0.5 * entropy_weights["w_stability"],
        "score": 0.5 * ahp_weights["w_score"] + 0.5 * entropy_weights["w_score"],
        "efficiency": 0.5 * ahp_weights["w_efficiency"] + 0.5 * entropy_weights["w_efficiency"],
    }
    evaluated["method2_score"] = (
        hybrid_weights["impact"] * evaluated["impact_effect"]
        + hybrid_weights["stability"] * evaluated["stability_score"]
        + hybrid_weights["score"] * evaluated["score_norm"]
        + hybrid_weights["efficiency"] * evaluated["execution_efficiency"]
    )
    evaluated["method2_utility"] = normalize_series(evaluated["method2_score"])
    evaluated["method2_rank"] = rank_descending(evaluated["method2_score"])
    evaluated["method2_w_impact"] = hybrid_weights["impact"]
    evaluated["method2_w_stability"] = hybrid_weights["stability"]
    evaluated["method2_w_score"] = hybrid_weights["score"]
    evaluated["method2_w_efficiency"] = hybrid_weights["efficiency"]
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
    indicator_weights = np.array(
        [
            ahp_weights["w_impact"],
            ahp_weights["w_stability"],
            ahp_weights["w_score"],
            ahp_weights["w_efficiency"],
        ],
        dtype=float,
    )
    fuzzy_scores = []
    for _, row in evaluated.iterrows():
        indicator_values = [
            row["impact_effect"],
            row["stability_score"],
            row["score_norm"],
            row["execution_efficiency"],
        ]
        membership_matrix = np.vstack([build_membership_row(value) for value in indicator_values])
        fuzzy_vector = indicator_weights @ membership_matrix
        fuzzy_scores.append(float(fuzzy_vector @ FUZZY_GRADE_VECTOR))

    evaluated["method3_score"] = fuzzy_scores
    evaluated["method3_utility"] = normalize_series(evaluated["method3_score"])
    evaluated["method3_rank"] = rank_descending(evaluated["method3_score"])
    return evaluated


def compute_final_utility(
    data: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    risk_scale: float = 1.0,
) -> pd.DataFrame:
    """方法四：风险调整效用模型。"""

    active_weights = dict(PROPOSED_WEIGHTS if weights is None else weights)
    evaluated = data.copy()
    evaluated["benefit_term"] = (
        active_weights["impact"] * evaluated["impact_effect"]
        + active_weights["score"] * evaluated["score_norm"]
    )
    evaluated["cost_term"] = (
        active_weights["work"] * evaluated["work_norm"]
        + active_weights["exposure"] * evaluated["exposure_norm"]
    )
    evaluated["u0"] = evaluated["benefit_term"] - evaluated["cost_term"]
    evaluated["fall_penalty_raw"] = evaluated["benefit_term"] * evaluated["fall_risk"] * risk_scale
    evaluated["fall_penalty"] = evaluated["fall_penalty_raw"]
    evaluated["method4_score"] = evaluated["benefit_term"] * (1.0 - evaluated["fall_risk"] * risk_scale) - evaluated["cost_term"]
    evaluated["proposed_score"] = evaluated["method4_score"]
    evaluated["utility"] = normalize_series(evaluated["method4_score"])
    evaluated["rank"] = rank_descending(evaluated["method4_score"])
    evaluated["method4_rank"] = evaluated["rank"]
    return evaluated


def evaluate_all_methods(
    data: pd.DataFrame,
    stable_margin: float | None = None,
    weights: Mapping[str, float] | None = None,
    penalty_config: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """串联四种方法并返回结果表与 AHP 摘要。"""

    del stable_margin
    risk_scale = 1.0 if penalty_config is None else float(penalty_config.get("risk_scale", 1.0))
    evaluated = compute_base_indicators(data)
    ahp_summary = compute_ahp_weights()
    evaluated = compute_method1_scores(evaluated)
    evaluated = compute_method2_scores(evaluated, ahp_summary)
    evaluated = compute_method3_scores(evaluated, ahp_summary)
    evaluated = compute_final_utility(evaluated, weights=weights, risk_scale=risk_scale)
    return evaluated.sort_values(by="rank", ascending=True, ignore_index=True), ahp_summary


def sensitivity_scan(
    data: pd.DataFrame,
    stable_margin: float | None = None,
    lambda_values: np.ndarray | None = None,
    k_values: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """执行风险强度与冲击权重的灵敏度分析。"""

    del stable_margin
    if lambda_values is None:
        lambda_values = np.linspace(0.80, 1.20, 5)
    if k_values is None:
        k_values = np.linspace(0.25, 0.45, 5)

    baseline = compute_final_utility(compute_base_indicators(data))
    baseline_top3 = set(baseline.nsmallest(3, "rank")["action_id"].tolist())
    overlap_rows = []
    spinning_rows = []
    for risk_scale in lambda_values:
        overlap_row = []
        spinning_row = []
        for impact_weight in k_values:
            adjusted_weights = {
                "impact": float(impact_weight),
                "score": 0.60 - float(impact_weight),
                "work": 0.20,
                "exposure": 0.20,
            }
            tested = compute_final_utility(compute_base_indicators(data), weights=adjusted_weights, risk_scale=float(risk_scale))
            current_top3 = set(tested.nsmallest(3, "rank")["action_id"].tolist())
            overlap_rate = len(baseline_top3 & current_top3) / 3.0
            spinning_score = float(tested.loc[tested["action_id"] == "A07", "utility"].iloc[0])
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
