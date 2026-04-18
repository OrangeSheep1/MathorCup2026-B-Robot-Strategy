"""Q3 单场竞技赛策略优化的状态、动作与转移参数构建模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd


SCORE_DIFF_VALUES = tuple(range(-5, 6))
HEALTH_LEVELS = (0, 1, 2)
RECOVERY_FLAGS = (0, 1)
HEALTH_LABELS = {
    0: "Low",
    1: "Mid",
    2: "High",
}
RECOVERY_LABELS = {
    0: "Normal",
    1: "Recover",
}
TIME_PHASE_LABELS = {
    "early": "前期",
    "mid": "中期",
    "late": "后期",
    "endgame": "决胜期",
}

Q3_ASSUMPTIONS = {
    "match_time_s": 300,
    "time_step_s": 15,
    "discount_factor": 0.95,
    "battery_capacity_mah": 10000.0,
    "battery_charge_voltage_v": 54.6,
    "battery_endurance_h": 2.0,
    "energy_unit_required": "J",
    "mid_health_ratio": 0.40,
    "low_health_ratio": 0.75,
    "impact_to_health_factor": 0.30,
    "reward_score": 1.00,
    "reward_health": 0.20,
    "reward_cost": 0.20,
    "reward_fall": 0.67,
    "terminal_reward_win": 10.0,
    "terminal_reward_loss": -10.0,
    "low_health_energy_cutoff": 0.50,
    "fall_recovery_lock_steps": 1,
}
TIME_STEP_CARD = int(Q3_ASSUMPTIONS["match_time_s"] // Q3_ASSUMPTIONS["time_step_s"])

ASSUMPTION_NOTES = {
    "match_time_s": "来自比赛规则，单回合净比赛时间 5 分钟。",
    "time_step_s": "按“覆盖单动作最大执行时间且总步数可控”的约束选取 15 秒。",
    "discount_factor": "反映单场序贯决策中的长期回报权衡。",
    "battery_capacity_mah": "来自机器人参数页中的电池容量。",
    "battery_charge_voltage_v": "由 54.6V 充电口径近似为电池工作电压。",
    "battery_endurance_h": "来自机器人参数页中的续航时间。",
    "energy_unit_required": "Q3 的疲劳模型要求 Q1 的 energy_cost 使用焦耳单位。",
    "mid_health_ratio": "将单场可用能量消耗达到 40% 作为 High 向 Mid 转档阈值。",
    "low_health_ratio": "将单场可用能量消耗达到 75% 作为 Mid 向 Low 转档阈值。",
    "impact_to_health_factor": "将有效冲击映射为机能降档概率的折算系数。",
    "reward_score": "得分事件作为基准奖励，取 1.0。",
    "reward_health": "削弱对手机能对后续回报的贡献折算为 0.2。",
    "reward_cost": "我方机能下降对后续动作空间的损失折算为 0.2。",
    "reward_fall": "倒地约损失 10 秒，相对 15 秒时间步折算为 0.67 分。",
    "terminal_reward_win": "终局获胜奖励显著高于单步奖励，用于保证策略追求最终胜利。",
    "terminal_reward_loss": "终局失败惩罚与获胜奖励对称。",
    "low_health_energy_cutoff": "Low 机能状态下仅保留能耗归一值低于 0.5 的攻击动作。",
    "fall_recovery_lock_steps": "倒地后下一决策步强制进入恢复约束，以体现倒地的后续状态影响。",
}

REQUIRED_ACTION_COLUMNS = [
    "action_id",
    "action_name",
    "category",
    "impact_score",
    "score_prob",
    "energy_cost",
    "utility",
    "rank",
    "tau_norm",
    "exec_time",
]

REQUIRED_MATCHUP_COLUMNS = [
    "action_id",
    "action_name",
    "defense_id_r1",
    "defense_id_r2",
    "defense_id_r3",
    "block_prob_r1",
    "block_prob_r2",
    "block_prob_r3",
    "counter_prob_r1",
    "counter_prob_r2",
    "counter_prob_r3",
    "counter_window_r1",
    "counter_window_r2",
    "counter_window_r3",
    "fall_risk_r1",
    "fall_risk_r2",
    "fall_risk_r3",
    "defense_score_r1",
    "defense_score_r2",
    "defense_score_r3",
]

REQUIRED_DEFENSE_FEATURE_COLUMNS = [
    "defense_id",
    "defense_name",
    "defense_category",
    "closure_group",
    "exec_time_def",
    "absorb_rate",
    "mobility_cost",
    "force_capacity",
    "contact_mode",
]

REQUIRED_PAIR_COLUMNS = [
    "action_id",
    "action_name",
    "attack_category",
    "score_prob",
    "energy_cost",
    "attack_utility",
    "tau_norm",
    "exec_time",
    "defense_id",
    "defense_name",
    "defense_category",
    "closure_group",
    "exec_time_def",
    "p_block",
    "counter_prob",
    "defense_damage",
    "counter_window",
    "p_fall",
    "proposed_score",
]


@dataclass(frozen=True)
class MatchState:
    """Q3 单场策略状态。"""

    score_diff: int
    time_step: int
    health_my: int
    health_opp: int
    recovery_lock: int = 0


@dataclass(frozen=True)
class Q3Config:
    """Q3 的有限时域 MDP 配置。"""

    match_time_s: int
    time_step_s: int
    n_time_steps: int
    gamma: float
    battery_capacity_mah: float
    battery_voltage_v: float
    battery_endurance_h: float
    battery_energy_wh: float
    battery_energy_j: float
    available_match_energy_j: float
    high_health_threshold_j: float
    mid_health_gap_j: float
    low_health_threshold_j: float
    energy_unit_required: str
    impact_to_health_factor: float
    reward_score: float
    reward_health: float
    reward_cost: float
    reward_fall: float
    terminal_reward_win: float
    terminal_reward_loss: float
    low_health_energy_cutoff: float
    fall_recovery_lock_steps: int


@dataclass(frozen=True)
class ActionKernel:
    """给定机能状态下某个动作的一步转移摘要。"""

    action_id: str
    action_name: str
    action_type: str
    macro_group: str
    health_my: int
    health_opp: int
    expected_reward: float
    p_score_for: float
    p_score_against: float
    p_self_drop: float
    p_opp_drop: float
    p_fall: float
    preferred_counter_action: str
    preferred_counter_name: str


@dataclass
class Q3Environment:
    """Q3 模型运行所需的全部输入与中间对象。"""

    config: Q3Config
    attack_table: pd.DataFrame
    defense_table: pd.DataFrame
    matchup_table: pd.DataFrame
    pair_table: pd.DataFrame
    action_table: pd.DataFrame
    state_table: pd.DataFrame
    kernel_table: pd.DataFrame
    pair_lookup: pd.DataFrame
    kernel_lookup: dict[tuple[int, int, str], ActionKernel]
    action_index_map: dict[str, int]
    attack_ids_by_health: dict[int, tuple[str, ...]]
    defense_ids: tuple[str, ...]
    recovery_defense_ids: tuple[str, ...]


def _require_columns(data: pd.DataFrame, required: list[str], label: str) -> None:
    """检查输入表字段是否齐全。"""

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{label}缺少字段: {missing}")


def load_action_features(file_path: str | Path) -> pd.DataFrame:
    """读取 Q1 输出的攻击动作特征表。"""

    data = pd.read_csv(file_path)
    _require_columns(data, REQUIRED_ACTION_COLUMNS, "Q1 动作特征表")
    return data.copy()


def load_defense_matchup(file_path: str | Path) -> pd.DataFrame:
    """读取 Q2 输出的防守匹配表。"""

    data = pd.read_csv(file_path)
    _require_columns(data, REQUIRED_MATCHUP_COLUMNS, "Q2 防守匹配表")
    return data.copy()


def load_defense_features(file_path: str | Path) -> pd.DataFrame:
    """读取 Q2 输出的防守特征表。"""

    data = pd.read_csv(file_path)
    _require_columns(data, REQUIRED_DEFENSE_FEATURE_COLUMNS, "Q2 防守特征表")
    return data.copy()


def load_defense_pair_scores(file_path: str | Path) -> pd.DataFrame:
    """读取 Q2 输出的攻防对评分明细表。"""

    data = pd.read_csv(file_path)
    _require_columns(data, REQUIRED_PAIR_COLUMNS, "Q2 攻防对评分表")
    return data.copy()


def build_q3_config() -> Q3Config:
    """构建 Q3 有限时域 MDP 配置。"""

    match_time_s = int(Q3_ASSUMPTIONS["match_time_s"])
    time_step_s = int(Q3_ASSUMPTIONS["time_step_s"])
    n_time_steps = match_time_s // time_step_s
    battery_capacity_mah = float(Q3_ASSUMPTIONS["battery_capacity_mah"])
    battery_voltage_v = float(Q3_ASSUMPTIONS["battery_charge_voltage_v"])
    battery_endurance_h = float(Q3_ASSUMPTIONS["battery_endurance_h"])
    battery_energy_wh = battery_capacity_mah / 1000.0 * battery_voltage_v
    battery_energy_j = battery_energy_wh * 3600.0
    available_match_energy_j = battery_energy_j * (match_time_s / (battery_endurance_h * 3600.0))
    high_health_threshold_j = available_match_energy_j * float(Q3_ASSUMPTIONS["mid_health_ratio"])
    low_health_threshold_j = available_match_energy_j * float(Q3_ASSUMPTIONS["low_health_ratio"])
    mid_health_gap_j = low_health_threshold_j - high_health_threshold_j

    return Q3Config(
        match_time_s=match_time_s,
        time_step_s=time_step_s,
        n_time_steps=n_time_steps,
        gamma=float(Q3_ASSUMPTIONS["discount_factor"]),
        battery_capacity_mah=battery_capacity_mah,
        battery_voltage_v=battery_voltage_v,
        battery_endurance_h=battery_endurance_h,
        battery_energy_wh=battery_energy_wh,
        battery_energy_j=battery_energy_j,
        available_match_energy_j=available_match_energy_j,
        high_health_threshold_j=high_health_threshold_j,
        mid_health_gap_j=mid_health_gap_j,
        low_health_threshold_j=low_health_threshold_j,
        energy_unit_required=str(Q3_ASSUMPTIONS["energy_unit_required"]),
        impact_to_health_factor=float(Q3_ASSUMPTIONS["impact_to_health_factor"]),
        reward_score=float(Q3_ASSUMPTIONS["reward_score"]),
        reward_health=float(Q3_ASSUMPTIONS["reward_health"]),
        reward_cost=float(Q3_ASSUMPTIONS["reward_cost"]),
        reward_fall=float(Q3_ASSUMPTIONS["reward_fall"]),
        terminal_reward_win=float(Q3_ASSUMPTIONS["terminal_reward_win"]),
        terminal_reward_loss=float(Q3_ASSUMPTIONS["terminal_reward_loss"]),
        low_health_energy_cutoff=float(Q3_ASSUMPTIONS["low_health_energy_cutoff"]),
        fall_recovery_lock_steps=int(Q3_ASSUMPTIONS["fall_recovery_lock_steps"]),
    )


def encode_state(state: MatchState) -> int:
    """将五维状态编码为一维索引。"""

    score_index = state.score_diff - SCORE_DIFF_VALUES[0]
    time_index = state.time_step - 1
    return (
        score_index * (TIME_STEP_CARD * len(HEALTH_LEVELS) * len(HEALTH_LEVELS) * len(RECOVERY_FLAGS))
        + time_index * (len(HEALTH_LEVELS) * len(HEALTH_LEVELS) * len(RECOVERY_FLAGS))
        + state.health_my * len(HEALTH_LEVELS) * len(RECOVERY_FLAGS)
        + state.health_opp * len(RECOVERY_FLAGS)
        + state.recovery_lock
    )


def decode_state(state_index: int) -> MatchState:
    """将一维索引还原为五维状态。"""

    recovery_card = len(RECOVERY_FLAGS)
    health_card = len(HEALTH_LEVELS)
    score_block = TIME_STEP_CARD * health_card * health_card * recovery_card
    score_index = state_index // score_block
    remain = state_index % score_block
    time_index = remain // (health_card * health_card * recovery_card)
    remain = remain % (health_card * health_card * recovery_card)
    health_my = remain // (health_card * recovery_card)
    remain = remain % (health_card * recovery_card)
    health_opp = remain // recovery_card
    recovery_lock = remain % recovery_card
    return MatchState(
        score_diff=SCORE_DIFF_VALUES[0] + score_index,
        time_step=time_index + 1,
        health_my=int(health_my),
        health_opp=int(health_opp),
        recovery_lock=int(recovery_lock),
    )


def build_state_table(config: Q3Config) -> pd.DataFrame:
    """构建 Q3 状态表。"""

    records: list[dict[str, object]] = []
    for score_diff in SCORE_DIFF_VALUES:
        for time_step in range(1, config.n_time_steps + 1):
            if time_step <= 5:
                time_phase = "early"
            elif time_step <= 10:
                time_phase = "mid"
            elif time_step <= 15:
                time_phase = "late"
            else:
                time_phase = "endgame"
            for health_my in HEALTH_LEVELS:
                for health_opp in HEALTH_LEVELS:
                    for recovery_lock in RECOVERY_FLAGS:
                        state = MatchState(score_diff, time_step, health_my, health_opp, recovery_lock)
                        records.append(
                            {
                                "state_index": encode_state(state),
                                "score_diff": score_diff,
                                "time_step": time_step,
                                "time_phase": time_phase,
                                "time_phase_name": TIME_PHASE_LABELS[time_phase],
                                "health_my": health_my,
                                "health_my_name": HEALTH_LABELS[health_my],
                                "health_opp": health_opp,
                                "health_opp_name": HEALTH_LABELS[health_opp],
                                "recovery_lock": recovery_lock,
                                "recovery_name": RECOVERY_LABELS[recovery_lock],
                            }
                        )
    return pd.DataFrame(records)


def _normalize_series(series: pd.Series) -> pd.Series:
    """对序列做极差标准化。"""

    minimum = float(series.min())
    maximum = float(series.max())
    if math.isclose(maximum, minimum):
        return pd.Series(np.full(len(series), 0.5), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)


def _validate_energy_cost_unit(action_features: pd.DataFrame, config: Q3Config) -> str:
    """校验 Q1 的 energy_cost 是否采用焦耳口径。"""

    energy_series = pd.to_numeric(action_features["energy_cost"], errors="coerce").astype(float)
    if energy_series.isna().any():
        raise ValueError("Q1 的 energy_cost 存在缺失或非数值记录，Q3 无法建立疲劳模型。")
    if float(energy_series.max()) <= 5.0 and float(energy_series.quantile(0.90)) <= 1.0:
        raise ValueError(
            "Q3 当前疲劳模型要求 Q1 的 energy_cost 使用焦耳单位，"
            "检测到当前值域更像归一化量，请先统一 Q1/Q3 的能耗口径。"
        )
    if float(energy_series.max()) > config.available_match_energy_j * 1.10:
        raise ValueError(
            "Q1 的 energy_cost 已明显超出单场可用能量上界，"
            "请检查 Q1 输出是否仍为焦耳口径。"
        )
    return config.energy_unit_required


def _classify_attack_macro_group(row: pd.Series) -> str:
    """将攻击动作归为宏观策略类。"""

    if (
        str(row["category"]) == "punch"
        and float(row["energy_norm"]) <= 0.10
        and float(row["attack_fall_risk"]) <= 0.10
        and float(row["tau_norm"]) <= 0.60
    ):
        return "试探攻击"
    if (
        float(row["tau_norm"]) >= 0.75
        or float(row["attack_fall_risk"]) >= 0.15
        or (
            float(row["attack_utility"]) >= 0.60
            and float(row["energy_norm"]) >= 0.12
        )
    ):
        return "激进攻击"
    return "试探攻击"


def _classify_defense_macro_group(row: pd.Series) -> str:
    """将防守动作归为宏观策略类。"""

    category = str(row["defense_category"])
    if category in {"evade", "combo"} and float(row["avg_counter_window"]) >= 0.20:
        return "反击防守"
    if category in {"balance", "ground"}:
        return "平衡恢复"
    return "保守防守"


def build_attack_table(
    action_features: pd.DataFrame,
    pair_scores: pd.DataFrame,
    config: Q3Config,
) -> pd.DataFrame:
    """构建 Q3 使用的攻击动作摘要表。"""

    attacks = action_features.copy()
    attacks["energy_unit"] = _validate_energy_cost_unit(attacks, config)
    attacks["energy_cost_j"] = attacks["energy_cost"].astype(float)
    attacks["energy_match_ratio"] = (attacks["energy_cost_j"] / config.available_match_energy_j).clip(0.0, 1.0)
    attacks["energy_norm"] = _normalize_series(attacks["energy_cost_j"])
    if "fall_penalty_raw" in attacks.columns:
        attacks["attack_fall_risk"] = _normalize_series(attacks["fall_penalty_raw"]).clip(0.0, 1.0)
    elif "fall_penalty" in attacks.columns:
        attacks["attack_fall_risk"] = _normalize_series(attacks["fall_penalty"]).clip(0.0, 1.0)
    else:
        attacks["attack_fall_risk"] = 0.0

    pair_summary = (
        pair_scores.groupby("action_id", as_index=False)
        .agg(
            avg_block_rate=("p_block", "mean"),
            avg_defense_score=("proposed_score", "mean"),
        )
    )
    attacks = attacks.merge(pair_summary, on="action_id", how="left")
    attacks["avg_block_rate"] = attacks["avg_block_rate"].fillna(0.0)
    attacks["avg_defense_score"] = attacks["avg_defense_score"].fillna(0.0)
    attacks = attacks.rename(
        columns={
            "utility": "attack_utility",
            "rank": "attack_rank",
        }
    )
    attacks["score_success_base"] = attacks["score_prob"] * (1.0 - attacks["avg_block_rate"])
    attacks["energy_drop_high"] = (attacks["energy_cost_j"] / config.high_health_threshold_j).clip(0.0, 1.0)
    attacks["energy_drop_mid"] = (attacks["energy_cost_j"] / config.mid_health_gap_j).clip(0.0, 1.0)
    attacks["low_health_available"] = attacks["energy_norm"] < config.low_health_energy_cutoff
    attacks["macro_group"] = attacks.apply(_classify_attack_macro_group, axis=1)
    return attacks.sort_values(by="action_id").reset_index(drop=True)


def _normalize_positive_weights(values: list[float]) -> list[float]:
    """将非负评分归一为权重。"""

    if not values:
        return []
    weights = np.clip(np.asarray(values, dtype=float), 0.0, None)
    if np.isclose(float(weights.sum()), 0.0):
        weights = np.full(len(values), 1.0 / len(values), dtype=float)
    else:
        weights = weights / weights.sum()
    return weights.tolist()


def _safe_float(value: object) -> float:
    """将可能为空的标量安全转为浮点数。"""

    if pd.isna(value):
        return 0.0
    return float(value)


def _pick_counter_action(counter_window: float, attack_table: pd.DataFrame) -> tuple[str, str, float, float]:
    """在给定反击窗口内选择最优反击动作。"""

    feasible = attack_table[attack_table["exec_time"] <= counter_window].copy()
    if feasible.empty:
        return "", "", 0.0, 0.0
    feasible = feasible.sort_values(
        by=["attack_utility", "impact_score", "exec_time"],
        ascending=[False, False, True],
    )
    row = feasible.iloc[0]
    return (
        str(row["action_id"]),
        str(row["action_name"]),
        float(row["score_success_base"]),
        float(row["tau_norm"]),
    )


def build_pair_table(pair_scores: pd.DataFrame, attack_table: pd.DataFrame) -> pd.DataFrame:
    """补充 Q3 使用的攻防对派生字段。"""

    pair_table = pair_scores.copy().sort_values(by=["action_id", "defense_id"]).reset_index(drop=True)
    counter_ids: list[str] = []
    counter_names: list[str] = []
    counter_success_probs: list[float] = []
    counter_tau_norms: list[float] = []
    effective_counter_probs: list[float] = []

    for _, row in pair_table.iterrows():
        counter_id, counter_name, counter_success, counter_tau_norm = _pick_counter_action(
            float(row["counter_window"]),
            attack_table,
        )
        effective_counter = float(row["p_block"]) * counter_success if counter_id else 0.0
        counter_ids.append(counter_id)
        counter_names.append(counter_name)
        counter_success_probs.append(effective_counter)
        counter_tau_norms.append(counter_tau_norm)
        effective_counter_probs.append(effective_counter)

    pair_table["counter_action_id"] = counter_ids
    pair_table["counter_action_name"] = counter_names
    pair_table["counter_score_prob_base"] = counter_success_probs
    pair_table["counter_tau_norm"] = counter_tau_norms
    pair_table["counter_prob_effective"] = effective_counter_probs
    return pair_table


def build_attack_response_profile(defense_matchup: pd.DataFrame) -> pd.DataFrame:
    """根据 Q2 的 Top3 防守集构建攻击响应画像。"""

    response_records: list[dict[str, object]] = []
    for _, row in defense_matchup.iterrows():
        candidates: list[dict[str, float | str]] = []
        defense_scores: list[float] = []
        for rank in (1, 2, 3):
            defense_id = str(row.get(f"defense_id_r{rank}", "")).strip()
            if not defense_id or defense_id.lower() == "nan":
                continue
            defense_score = _safe_float(row.get(f"defense_score_r{rank}", 0.0))
            defense_scores.append(defense_score)
            candidates.append(
                {
                    "defense_id": defense_id,
                    "block_prob": _safe_float(row.get(f"block_prob_r{rank}", 0.0)),
                    "counter_prob": _safe_float(row.get(f"counter_prob_r{rank}", 0.0)),
                    "counter_window": _safe_float(row.get(f"counter_window_r{rank}", 0.0)),
                    "fall_risk": _safe_float(row.get(f"fall_risk_r{rank}", 0.0)),
                    "defense_score": defense_score,
                }
            )

        weights = _normalize_positive_weights(defense_scores)
        if not candidates:
            response_records.append(
                {
                    "action_id": str(row["action_id"]),
                    "opponent_block_rate": 0.0,
                    "opponent_counter_prob": 0.0,
                    "opponent_counter_window": 0.0,
                    "opponent_defense_fall_risk": 0.0,
                    "opp_defense_id_r1": "",
                    "opp_defense_id_r2": "",
                    "opp_defense_id_r3": "",
                    "opp_defense_weight_r1": 0.0,
                    "opp_defense_weight_r2": 0.0,
                    "opp_defense_weight_r3": 0.0,
                }
            )
            continue

        weighted_block = 0.0
        weighted_counter = 0.0
        weighted_window = 0.0
        weighted_fall = 0.0
        record: dict[str, object] = {
            "action_id": str(row["action_id"]),
        }
        for rank in (1, 2, 3):
            if rank <= len(candidates):
                candidate = candidates[rank - 1]
                weight = weights[rank - 1]
                record[f"opp_defense_id_r{rank}"] = str(candidate["defense_id"])
                record[f"opp_defense_weight_r{rank}"] = float(weight)
                weighted_block += weight * float(candidate["block_prob"])
                weighted_counter += weight * float(candidate["counter_prob"])
                weighted_window += weight * float(candidate["counter_window"])
                weighted_fall += weight * float(candidate["fall_risk"])
            else:
                record[f"opp_defense_id_r{rank}"] = ""
                record[f"opp_defense_weight_r{rank}"] = 0.0

        record["opponent_block_rate"] = weighted_block
        record["opponent_counter_prob"] = weighted_counter
        record["opponent_counter_window"] = weighted_window
        record["opponent_defense_fall_risk"] = weighted_fall
        response_records.append(record)

    return pd.DataFrame(response_records)


def build_defense_table(defense_features: pd.DataFrame, pair_table: pd.DataFrame) -> pd.DataFrame:
    """构建 Q3 使用的防守动作摘要表。"""

    defenses = defense_features.copy().sort_values(by="defense_id").reset_index(drop=True)
    pair_summary = (
        pair_table.groupby("defense_id", as_index=False)
        .agg(
            avg_block_rate=("p_block", "mean"),
            avg_damage=("defense_damage", "mean"),
            avg_counter_window=("counter_window", "mean"),
            avg_fall_risk=("p_fall", "mean"),
            avg_score=("proposed_score", "mean"),
            avg_counter_score=("counter_prob_effective", "mean"),
        )
    )
    defenses = defenses.merge(pair_summary, on="defense_id", how="left")
    defenses["avg_block_rate"] = defenses["avg_block_rate"].fillna(0.0)
    defenses["avg_damage"] = defenses["avg_damage"].fillna(0.0)
    defenses["avg_counter_window"] = defenses["avg_counter_window"].fillna(0.0)
    defenses["avg_fall_risk"] = defenses["avg_fall_risk"].fillna(0.0)
    defenses["avg_score"] = defenses["avg_score"].fillna(0.0)
    defenses["avg_counter_score"] = defenses["avg_counter_score"].fillna(0.0)
    defenses["macro_group"] = defenses.apply(_classify_defense_macro_group, axis=1)
    return defenses


def build_action_space(attack_table: pd.DataFrame, defense_table: pd.DataFrame) -> pd.DataFrame:
    """构建统一动作空间。"""

    attack_actions = attack_table[
        [
            "action_id",
            "action_name",
            "category",
            "attack_utility",
            "macro_group",
            "exec_time",
        ]
    ].copy()
    attack_actions = attack_actions.rename(
        columns={
            "attack_utility": "base_score",
            "exec_time": "exec_time_s",
        }
    )
    attack_actions["action_type"] = "attack"

    defense_actions = defense_table[
        [
            "defense_id",
            "defense_name",
            "defense_category",
            "avg_score",
            "macro_group",
            "exec_time_def",
        ]
    ].copy()
    defense_actions = defense_actions.rename(
        columns={
            "defense_id": "action_id",
            "defense_name": "action_name",
            "defense_category": "category",
            "avg_score": "base_score",
            "exec_time_def": "exec_time_s",
        }
    )
    defense_actions["action_type"] = "defense"

    action_table = pd.concat([attack_actions, defense_actions], ignore_index=True)
    return action_table.sort_values(by=["action_type", "action_id"]).reset_index(drop=True)


def available_attack_ids(attack_table: pd.DataFrame, health_level: int, config: Q3Config) -> tuple[str, ...]:
    """返回给定机能状态下可用的攻击动作。"""

    if health_level <= 0:
        subset = attack_table[attack_table["low_health_available"]].copy()
    else:
        subset = attack_table.copy()
    return tuple(subset.sort_values("action_id")["action_id"].tolist())


def available_action_ids(env: Q3Environment, health_level: int, recovery_lock: int = 0) -> tuple[str, ...]:
    """返回给定状态下的可用动作。"""

    if recovery_lock > 0:
        return env.recovery_defense_ids
    attack_ids = env.attack_ids_by_health[health_level]
    return attack_ids + env.defense_ids


def _health_drop_from_energy(attack_row: pd.Series, health_my: int) -> float:
    """根据攻击能耗估计我方机能降档概率。"""

    if health_my <= 0:
        return 0.0
    if health_my == 2:
        return float(attack_row["energy_drop_high"])
    return float(attack_row["energy_drop_mid"])


def _build_attack_kernel(
    attack_row: pd.Series,
    health_my: int,
    health_opp: int,
    config: Q3Config,
) -> ActionKernel:
    """构建攻击动作在给定机能状态下的一步核。"""

    opponent_block_rate = float(attack_row.get("opponent_block_rate", attack_row["avg_block_rate"]))
    p_score_for = float(attack_row["score_prob"]) * (1.0 - opponent_block_rate)
    p_score_against = float(attack_row.get("opponent_counter_prob", 0.0))
    total_score_prob = p_score_for + p_score_against
    if total_score_prob > 1.0:
        p_score_for = p_score_for / total_score_prob
        p_score_against = p_score_against / total_score_prob

    p_self_drop = _health_drop_from_energy(attack_row, health_my)
    p_opp_drop = 0.0 if health_opp <= 0 else min(
        1.0,
        p_score_for * float(attack_row["tau_norm"]) * config.impact_to_health_factor,
    )
    p_fall = float(attack_row["attack_fall_risk"])
    expected_reward = (
        config.reward_score * (p_score_for - p_score_against)
        + config.reward_health * p_opp_drop
        - config.reward_cost * p_self_drop
        - config.reward_fall * p_fall
    )
    return ActionKernel(
        action_id=str(attack_row["action_id"]),
        action_name=str(attack_row["action_name"]),
        action_type="attack",
        macro_group=str(attack_row["macro_group"]),
        health_my=health_my,
        health_opp=health_opp,
        expected_reward=float(expected_reward),
        p_score_for=float(np.clip(p_score_for, 0.0, 1.0)),
        p_score_against=float(np.clip(p_score_against, 0.0, 1.0)),
        p_self_drop=float(np.clip(p_self_drop, 0.0, 1.0)),
        p_opp_drop=float(np.clip(p_opp_drop, 0.0, 1.0)),
        p_fall=float(np.clip(p_fall, 0.0, 1.0)),
        preferred_counter_action="",
        preferred_counter_name="",
    )


def _build_defense_kernel(
    defense_row: pd.Series,
    pair_table: pd.DataFrame,
    health_my: int,
    health_opp: int,
    config: Q3Config,
    opponent_attack_ids: tuple[str, ...],
) -> ActionKernel:
    """构建防守动作在给定机能状态下的一步核。"""

    subset = pair_table[
        (pair_table["defense_id"] == defense_row["defense_id"])
        & (pair_table["action_id"].isin(opponent_attack_ids))
    ].copy()
    if subset.empty:
        p_score_for = 0.0
        p_score_against = 0.0
        p_self_drop = 0.0
        p_opp_drop = 0.0
        p_fall = 0.0
        preferred_counter_action = ""
        preferred_counter_name = ""
    else:
        subset["opponent_score_prob"] = subset["score_prob"] * (1.0 - subset["p_block"])
        subset["counter_health_drop"] = (
            subset["counter_prob_effective"] * subset["counter_tau_norm"] * config.impact_to_health_factor
        )
        p_score_for = float(subset["counter_prob_effective"].mean())
        p_score_against = float(subset["opponent_score_prob"].mean())
        p_self_drop = 0.0 if health_my <= 0 else float(subset["defense_damage"].mean().clip(0.0, 1.0))
        p_opp_drop = 0.0 if health_opp <= 0 else float(subset["counter_health_drop"].mean().clip(0.0, 1.0))
        p_fall = float(subset["p_fall"].mean().clip(0.0, 1.0))
        ranked = subset.sort_values(
            by=["counter_prob_effective", "counter_window", "proposed_score"],
            ascending=[False, False, False],
        )
        preferred_counter_action = str(ranked.iloc[0]["counter_action_id"])
        preferred_counter_name = str(ranked.iloc[0]["counter_action_name"])

    total_score_prob = p_score_for + p_score_against
    if total_score_prob > 1.0:
        p_score_for = p_score_for / total_score_prob
        p_score_against = p_score_against / total_score_prob

    expected_reward = (
        config.reward_score * (p_score_for - p_score_against)
        + config.reward_health * p_opp_drop
        - config.reward_cost * p_self_drop
        - config.reward_fall * p_fall
    )
    return ActionKernel(
        action_id=str(defense_row["defense_id"]),
        action_name=str(defense_row["defense_name"]),
        action_type="defense",
        macro_group=str(defense_row["macro_group"]),
        health_my=health_my,
        health_opp=health_opp,
        expected_reward=float(expected_reward),
        p_score_for=float(np.clip(p_score_for, 0.0, 1.0)),
        p_score_against=float(np.clip(p_score_against, 0.0, 1.0)),
        p_self_drop=float(np.clip(p_self_drop, 0.0, 1.0)),
        p_opp_drop=float(np.clip(p_opp_drop, 0.0, 1.0)),
        p_fall=float(np.clip(p_fall, 0.0, 1.0)),
        preferred_counter_action=preferred_counter_action,
        preferred_counter_name=preferred_counter_name,
    )


def build_kernel_table(
    attack_table: pd.DataFrame,
    defense_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    config: Q3Config,
) -> tuple[pd.DataFrame, dict[tuple[int, int, str], ActionKernel], dict[int, tuple[str, ...]]]:
    """为全部机能状态预计算动作核。"""

    kernel_records: list[dict[str, object]] = []
    kernel_lookup: dict[tuple[int, int, str], ActionKernel] = {}
    attack_ids_by_health = {
        health_level: available_attack_ids(attack_table, health_level, config)
        for health_level in HEALTH_LEVELS
    }

    for health_my in HEALTH_LEVELS:
        for health_opp in HEALTH_LEVELS:
            opponent_attack_ids = attack_ids_by_health[health_opp]

            for _, attack_row in attack_table.iterrows():
                if health_my <= 0 and not bool(attack_row["low_health_available"]):
                    continue
                kernel = _build_attack_kernel(attack_row, health_my, health_opp, config)
                kernel_lookup[(health_my, health_opp, kernel.action_id)] = kernel
                kernel_records.append(
                    {
                        "health_my": health_my,
                        "health_my_name": HEALTH_LABELS[health_my],
                        "health_opp": health_opp,
                        "health_opp_name": HEALTH_LABELS[health_opp],
                        "action_id": kernel.action_id,
                        "action_name": kernel.action_name,
                        "action_type": kernel.action_type,
                        "macro_group": kernel.macro_group,
                        "expected_reward": kernel.expected_reward,
                        "p_score_for": kernel.p_score_for,
                        "p_score_against": kernel.p_score_against,
                        "p_self_drop": kernel.p_self_drop,
                        "p_opp_drop": kernel.p_opp_drop,
                        "p_fall": kernel.p_fall,
                        "preferred_counter_action": kernel.preferred_counter_action,
                        "preferred_counter_name": kernel.preferred_counter_name,
                    }
                )

            for _, defense_row in defense_table.iterrows():
                kernel = _build_defense_kernel(
                    defense_row,
                    pair_table,
                    health_my,
                    health_opp,
                    config,
                    opponent_attack_ids,
                )
                kernel_lookup[(health_my, health_opp, kernel.action_id)] = kernel
                kernel_records.append(
                    {
                        "health_my": health_my,
                        "health_my_name": HEALTH_LABELS[health_my],
                        "health_opp": health_opp,
                        "health_opp_name": HEALTH_LABELS[health_opp],
                        "action_id": kernel.action_id,
                        "action_name": kernel.action_name,
                        "action_type": kernel.action_type,
                        "macro_group": kernel.macro_group,
                        "expected_reward": kernel.expected_reward,
                        "p_score_for": kernel.p_score_for,
                        "p_score_against": kernel.p_score_against,
                        "p_self_drop": kernel.p_self_drop,
                        "p_opp_drop": kernel.p_opp_drop,
                        "p_fall": kernel.p_fall,
                        "preferred_counter_action": kernel.preferred_counter_action,
                        "preferred_counter_name": kernel.preferred_counter_name,
                    }
                )

    kernel_table = pd.DataFrame(kernel_records)
    return kernel_table, kernel_lookup, attack_ids_by_health


def _select_recovery_defense_ids(defense_table: pd.DataFrame) -> tuple[str, ...]:
    """选取倒地恢复阶段允许使用的防守动作。"""

    mask = (
        defense_table["closure_group"].eq("fallback")
        | defense_table["defense_category"].isin(["balance", "ground"])
    )
    subset = defense_table[mask].copy()
    if subset.empty:
        subset = defense_table.copy()
    return tuple(subset.sort_values(by="defense_id")["defense_id"].tolist())


def build_environment(
    action_feature_file: str | Path,
    defense_matchup_file: str | Path,
    defense_feature_file: str | Path,
    defense_pair_file: str | Path,
) -> Q3Environment:
    """读取 Q1/Q2 输出并构建 Q3 环境对象。"""

    config = build_q3_config()
    action_features = load_action_features(action_feature_file)
    defense_matchup = load_defense_matchup(defense_matchup_file)
    defense_features = load_defense_features(defense_feature_file)
    pair_scores = load_defense_pair_scores(defense_pair_file)

    attack_table = build_attack_table(action_features, pair_scores, config)
    pair_table = build_pair_table(pair_scores, attack_table)
    attack_response = build_attack_response_profile(defense_matchup)
    attack_table = attack_table.merge(attack_response, on="action_id", how="left")
    attack_table["opponent_block_rate"] = attack_table["opponent_block_rate"].fillna(attack_table["avg_block_rate"])
    attack_table["opponent_counter_prob"] = attack_table["opponent_counter_prob"].fillna(0.0)
    attack_table["opponent_counter_window"] = attack_table["opponent_counter_window"].fillna(0.0)
    attack_table["opponent_defense_fall_risk"] = attack_table["opponent_defense_fall_risk"].fillna(0.0)
    attack_table["score_success_response"] = attack_table["score_prob"] * (1.0 - attack_table["opponent_block_rate"])
    attack_table["score_margin_response"] = attack_table["score_success_response"] - attack_table["opponent_counter_prob"]
    for rank in (1, 2, 3):
        attack_table[f"opp_defense_id_r{rank}"] = attack_table[f"opp_defense_id_r{rank}"].fillna("")
        attack_table[f"opp_defense_weight_r{rank}"] = attack_table[f"opp_defense_weight_r{rank}"].fillna(0.0)

    defense_table = build_defense_table(defense_features, pair_table)
    action_table = build_action_space(attack_table, defense_table)
    state_table = build_state_table(config)
    kernel_table, kernel_lookup, attack_ids_by_health = build_kernel_table(
        attack_table,
        defense_table,
        pair_table,
        config,
    )
    pair_lookup = pair_table.set_index(["action_id", "defense_id"]).sort_index()
    action_index_map = {
        action_id: index
        for index, action_id in enumerate(action_table["action_id"].tolist())
    }
    defense_ids = tuple(defense_table["defense_id"].tolist())
    recovery_defense_ids = _select_recovery_defense_ids(defense_table)
    return Q3Environment(
        config=config,
        attack_table=attack_table,
        defense_table=defense_table,
        matchup_table=defense_matchup,
        pair_table=pair_table,
        action_table=action_table,
        state_table=state_table,
        kernel_table=kernel_table,
        pair_lookup=pair_lookup,
        kernel_lookup=kernel_lookup,
        action_index_map=action_index_map,
        attack_ids_by_health=attack_ids_by_health,
        defense_ids=defense_ids,
        recovery_defense_ids=recovery_defense_ids,
    )


def terminal_reward(score_diff: int, config: Q3Config) -> float:
    """返回终局奖励。"""

    if score_diff > 0:
        return config.terminal_reward_win
    if score_diff < 0:
        return config.terminal_reward_loss
    return 0.0


def clamp_score_diff(score_diff: int) -> int:
    """将分差截断到状态边界内。"""

    return int(min(max(score_diff, SCORE_DIFF_VALUES[0]), SCORE_DIFF_VALUES[-1]))


def _allocate_marginal_probability(total_prob: float, primary_prob: float, secondary_prob: float) -> tuple[float, float]:
    """把一个边际概率优先分配给主事件，不足时再分配给次事件。"""

    total = float(np.clip(total_prob, 0.0, 1.0))
    primary = float(np.clip(primary_prob, 0.0, 1.0))
    secondary = float(np.clip(secondary_prob, 0.0, 1.0))
    allocated_primary = min(total, primary)
    cond_primary = allocated_primary / primary if primary > 0.0 else 0.0
    remaining = max(total - allocated_primary, 0.0)
    allocated_secondary = min(remaining, secondary)
    cond_secondary = allocated_secondary / secondary if secondary > 0.0 else 0.0
    return float(np.clip(cond_primary, 0.0, 1.0)), float(np.clip(cond_secondary, 0.0, 1.0))


def _binary_level_outcomes(level: int, event_prob: float) -> list[tuple[int, int, float]]:
    """展开降档二元结果。"""

    probability = float(np.clip(event_prob, 0.0, 1.0))
    outcomes = [(level, 0, 1.0 - probability)]
    if level > 0 and probability > 0.0:
        outcomes.append((level - 1, 1, probability))
    return outcomes


def _binary_flag_outcomes(event_prob: float) -> list[tuple[int, float]]:
    """展开标志位二元结果。"""

    probability = float(np.clip(event_prob, 0.0, 1.0))
    return [(0, 1.0 - probability), (1, probability)] if probability > 0.0 else [(0, 1.0)]


def build_transition_events(state: MatchState, kernel: ActionKernel) -> list[dict[str, object]]:
    """按条件事件树展开一步转移事件。"""

    p_success = float(np.clip(kernel.p_score_for, 0.0, 1.0))
    p_fail = float(np.clip(kernel.p_score_against, 0.0, 1.0))
    total_score_prob = p_success + p_fail
    if total_score_prob > 1.0:
        p_success = p_success / total_score_prob
        p_fail = p_fail / total_score_prob
    p_neutral = max(0.0, 1.0 - p_success - p_fail)
    p_bad = p_fail + p_neutral

    if kernel.action_type == "attack":
        self_success = self_fail = self_neutral = float(np.clip(kernel.p_self_drop, 0.0, 1.0))
    else:
        self_bad, self_success = _allocate_marginal_probability(kernel.p_self_drop, p_bad, p_success)
        self_fail = self_bad
        self_neutral = self_bad

    opp_success = min(1.0, kernel.p_opp_drop / max(p_success, 1e-12)) if p_success > 0.0 else 0.0
    fall_bad, fall_success = _allocate_marginal_probability(kernel.p_fall, p_bad, p_success)
    fall_fail = fall_bad
    fall_neutral = fall_bad

    event_specs = [
        ("score_for", p_success, 1, self_success, opp_success, fall_success),
        ("score_against", p_fail, -1, self_fail, 0.0, fall_fail),
        ("neutral", p_neutral, 0, self_neutral, 0.0, fall_neutral),
    ]

    events: list[dict[str, object]] = []
    next_time_step = state.time_step + 1
    for event_name, event_prob, delta_score, self_drop_prob, opp_drop_prob, fall_prob in event_specs:
        if event_prob <= 0.0:
            continue
        for next_health_my, self_drop, prob_my in _binary_level_outcomes(state.health_my, self_drop_prob):
            for next_health_opp, opp_drop, prob_opp in _binary_level_outcomes(state.health_opp, opp_drop_prob):
                for fall_event, prob_fall in _binary_flag_outcomes(fall_prob):
                    probability = event_prob * prob_my * prob_opp * prob_fall
                    if probability <= 0.0:
                        continue
                    next_state = MatchState(
                        score_diff=clamp_score_diff(state.score_diff + delta_score),
                        time_step=next_time_step,
                        health_my=next_health_my,
                        health_opp=next_health_opp,
                        recovery_lock=1 if fall_event else 0,
                    )
                    events.append(
                        {
                            "event_name": event_name,
                            "probability": float(probability),
                            "delta_score": int(delta_score),
                            "self_drop": int(self_drop),
                            "opp_drop": int(opp_drop),
                            "fall_event": int(fall_event),
                            "next_state": next_state,
                        }
                    )

    total_probability = float(sum(event["probability"] for event in events))
    if total_probability > 0.0 and not math.isclose(total_probability, 1.0):
        for event in events:
            event["probability"] = float(event["probability"] / total_probability)
    return events


def iter_transition_branches(state: MatchState, kernel: ActionKernel) -> list[tuple[float, MatchState]]:
    """按条件事件树展开一步转移分支。"""

    aggregated: dict[MatchState, float] = {}
    for event in build_transition_events(state, kernel):
        next_state = event["next_state"]
        aggregated[next_state] = aggregated.get(next_state, 0.0) + float(event["probability"])
    return [(probability, next_state) for next_state, probability in aggregated.items()]


def sample_transition_event(
    state: MatchState,
    kernel: ActionKernel,
    rng: np.random.Generator,
) -> dict[str, object]:
    """从条件事件树中采样一步转移事件。"""

    events = build_transition_events(state, kernel)
    if not events:
        return {
            "event_name": "neutral",
            "probability": 1.0,
            "delta_score": 0,
            "self_drop": 0,
            "opp_drop": 0,
            "fall_event": 0,
            "next_state": MatchState(
                score_diff=state.score_diff,
                time_step=state.time_step + 1,
                health_my=state.health_my,
                health_opp=state.health_opp,
                recovery_lock=0,
            ),
        }
    probabilities = np.asarray([event["probability"] for event in events], dtype=float)
    probabilities = probabilities / probabilities.sum()
    selected_index = int(rng.choice(len(events), p=probabilities))
    return events[selected_index]


def get_kernel(env: Q3Environment, state: MatchState, action_id: str) -> ActionKernel:
    """读取给定状态与动作的一步核。"""

    return env.kernel_lookup[(state.health_my, state.health_opp, action_id)]
