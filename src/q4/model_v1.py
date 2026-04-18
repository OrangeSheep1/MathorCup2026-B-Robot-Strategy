"""Q4 BO3 资源决策的状态、参数与局内转移构建模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "leading": "领先局",
    "tied": "平局局",
    "trailing": "落后局",
}
SCENARIO_INITIAL_SCORE = {
    "leading": 1,
    "tied": 0,
    "trailing": -1,
}
HEALTH_LEVELS = (0, 1, 2)
FAULT_FLAGS = (0, 1)
DOWN_FLAGS = (0, 1)
SCORE_DIFF_VALUES = tuple(range(-5, 6))
REGULAR_MACRO_ORDER = ["试探攻击", "激进攻击", "反击防守", "保守防守", "平衡恢复"]
TACTICAL_ACTION_LAYOUT = [
    ("TACT_PROBE", "试探攻击", "attack"),
    ("TACT_AGGRESSIVE", "激进攻击", "attack"),
    ("TACT_COUNTER", "反击防守", "defense"),
    ("TACT_GUARD", "保守防守", "defense"),
    ("TACT_RECOVER", "平衡恢复", "defense"),
]
RESOURCE_ACTION_LAYOUT = [
    ("USE_RESET", "人工复位", "resource"),
    ("USE_PAUSE", "战术暂停", "resource"),
    ("USE_REPAIR", "紧急维修", "resource"),
    ("WAIT_DOWN", "倒地等待", "resource"),
    ("WAIT_FAULT", "故障等待", "resource"),
]
HEALTH_RATIO_MAP = {
    2: 1.0,
    1: 0.67,
    0: 0.33,
}
REQUIRED_Q1_COLUMNS = [
    "action_id",
    "action_name",
    "impact_score",
    "balance_cost",
    "score_prob",
    "energy_cost",
    "utility",
]
REQUIRED_Q2_PAIR_COLUMNS = [
    "action_id",
    "action_name",
    "defense_id",
    "defense_name",
    "defense_category",
    "closure_group",
    "p_block",
    "counter_prob",
    "counter_window",
    "defense_damage",
    "p_fall",
    "proposed_utility",
]
REQUIRED_Q3_KERNEL_COLUMNS = [
    "health_my",
    "health_opp",
    "counter_ready",
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
]
REQUIRED_Q3_METRIC_COLUMNS = [
    "scenario",
    "method",
    "win_rate",
]


@dataclass(frozen=True)
class Q4Config:
    """Q4 故障与资源模型配置。"""

    match_time_s: int
    time_bucket_s: int
    n_time_buckets: int
    lambda_0: float
    k_fault: float
    delta_h_pause: float
    delta_h_reset: float
    p_score_loss_per_step: float
    tau_repair_buckets: int
    tau_pause_buckets: int
    tau_reset_buckets: int
    base_win_scale: float
    max_reset: int
    max_pause: int
    max_repair: int
    fall_down_weight: float
    health_full_threshold: float
    health_partial_threshold: float
    resource_time_mode: str


@dataclass(frozen=True)
class RoundState:
    """Q4 局内状态。"""

    score_diff: int
    time_bucket: int
    health_my: int
    health_opp: int
    fault: int
    down_flag: int
    reset_left: int
    pause_left: int
    repair_left: int


@dataclass(frozen=True)
class TacticalAction:
    """Q4 局内动作定义。"""

    tactical_id: str
    tactical_name: str
    action_type: str
    macro_group: str
    source_action_id: str
    source_action_name: str


@dataclass(frozen=True)
class MicroKernel:
    """局内常规战术动作的转移核摘要。"""

    tactical_id: str
    tactical_name: str
    action_type: str
    macro_group: str
    source_action_id: str
    source_action_name: str
    expected_reward: float
    p_score_for: float
    p_score_against: float
    p_self_drop: float
    p_opp_drop: float
    p_fall: float


@dataclass
class Q4Context:
    """Q4 全部输入数据与派生对象。"""

    config: Q4Config
    q1_action_table: pd.DataFrame
    q2_pair_table: pd.DataFrame
    q3_kernel_table: pd.DataFrame
    q3_metric_table: pd.DataFrame
    tactical_actions: pd.DataFrame
    micro_kernel_lookup: dict[tuple[str, int, int], MicroKernel]
    base_win_prob: dict[str, float]
    action_id_to_name: dict[str, str]
    action_id_to_macro: dict[str, str]
    all_action_ids: tuple[str, ...]
    regular_action_ids: tuple[str, ...]
    resource_action_ids: tuple[str, ...]


def _require_columns(data: pd.DataFrame, required: list[str], label: str) -> None:
    """检查输入表字段是否完整。"""

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{label} 缺少字段: {missing}")


def _clip_score_diff(score_diff: int) -> int:
    """将分差截断到 Q4 统一的边界范围。"""

    return int(max(min(score_diff, SCORE_DIFF_VALUES[-1]), SCORE_DIFF_VALUES[0]))


def health_ratio(level: int) -> float:
    """将离散机能档映射为归一化机能比例。"""

    return float(HEALTH_RATIO_MAP[int(level)])


def health_level_from_ratio(ratio: float, config: Q4Config) -> int:
    """将归一化机能比例映射回离散机能档。"""

    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio > config.health_full_threshold:
        return 2
    if ratio > config.health_partial_threshold:
        return 1
    return 0


def apply_health_gain(level: int, gain: float, config: Q4Config) -> int:
    """资源动作带来的机能恢复。"""

    ratio = health_ratio(level) + float(gain)
    return health_level_from_ratio(ratio, config)


def apply_health_drop(level: int, drop_flag: int) -> int:
    """动作命中或自损带来的离散降档。"""

    return max(0, int(level) - int(drop_flag))


def compute_fault_rate(health_my: int, config: Q4Config) -> float:
    """根据当前机能档计算故障率。"""

    ratio = health_ratio(health_my)
    fault_rate = config.lambda_0 * math.exp(config.k_fault * (1.0 - ratio))
    return float(np.clip(fault_rate, 0.0, 0.95))


def scenario_from_series_state(wins_my: int, wins_opp: int) -> str:
    """根据 BO3 当前比分判断单局情景。"""

    if wins_my > wins_opp:
        return "leading"
    if wins_my < wins_opp:
        return "trailing"
    return "tied"


def allocation_label(reset_alloc: int, pause_alloc: int, repair_alloc: int) -> str:
    """资源分配标签。"""

    return f"R{reset_alloc}-P{pause_alloc}-M{repair_alloc}"


def _default_fault_params() -> dict[str, float]:
    """返回 Q4 默认故障与资源参数。"""

    return {
        "match_time_s": 420,
        "time_bucket_s": 20,
        "lambda_0": 0.02,
        "k_fault": 3.0,
        "delta_H_pause": 0.20,
        "delta_H_reset": 0.10,
        "p_score_loss_per_step": 0.15,
        "tau_repair_buckets": 15,
        "tau_pause_buckets": 3,
        "tau_reset_buckets": 1,
        "base_win_scale": 1.0,
        "fall_down_weight": 0.35,
        "resource_time_mode": "opportunity_bucket",
        "max_reset": 2,
        "max_pause": 2,
        "max_repair": 1,
    }


def load_fault_params(file_path: str | Path) -> dict[str, float]:
    """读取 Q4 原始参数文件。"""

    with Path(file_path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    params = _default_fault_params()
    params.update(raw)
    return params


def ensure_fault_param_file(file_path: str | Path) -> Path:
    """若 Q4 参数文件不存在，则写入默认模板。"""

    path = Path(file_path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_default_fault_params(), file, ensure_ascii=False, indent=2)
    return path


def build_q4_config(params: dict[str, float]) -> Q4Config:
    """构建 Q4 配置对象。"""

    match_time_s = int(params["match_time_s"])
    time_bucket_s = int(params["time_bucket_s"])
    n_time_buckets = int(math.ceil(match_time_s / time_bucket_s))
    return Q4Config(
        match_time_s=match_time_s,
        time_bucket_s=time_bucket_s,
        n_time_buckets=n_time_buckets,
        lambda_0=float(params["lambda_0"]),
        k_fault=float(params["k_fault"]),
        delta_h_pause=float(params["delta_H_pause"]),
        delta_h_reset=float(params["delta_H_reset"]),
        p_score_loss_per_step=float(params["p_score_loss_per_step"]),
        tau_repair_buckets=int(params["tau_repair_buckets"]),
        tau_pause_buckets=int(params["tau_pause_buckets"]),
        tau_reset_buckets=int(params["tau_reset_buckets"]),
        base_win_scale=float(params.get("base_win_scale", 1.0)),
        max_reset=int(params["max_reset"]),
        max_pause=int(params["max_pause"]),
        max_repair=int(params["max_repair"]),
        fall_down_weight=float(params.get("fall_down_weight", params.get("fall_fault_weight", 0.35))),
        health_full_threshold=0.67,
        health_partial_threshold=0.33,
        resource_time_mode=str(params.get("resource_time_mode", "opportunity_bucket")),
    )


def _load_base_tables(
    action_feature_file: str | Path,
    defense_pair_file: str | Path,
    q3_kernel_file: str | Path,
    q3_metric_file: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取 Q1/Q2/Q3 基础表。"""

    q1_action_table = pd.read_csv(action_feature_file)
    q2_pair_table = pd.read_csv(defense_pair_file)
    q3_kernel_table = pd.read_csv(q3_kernel_file)
    q3_metric_table = pd.read_csv(q3_metric_file)

    _require_columns(q1_action_table, REQUIRED_Q1_COLUMNS, "Q1 动作特征表")
    _require_columns(q2_pair_table, REQUIRED_Q2_PAIR_COLUMNS, "Q2 攻防对表")
    _require_columns(q3_kernel_table, REQUIRED_Q3_KERNEL_COLUMNS, "Q3 动作核表")
    _require_columns(q3_metric_table, REQUIRED_Q3_METRIC_COLUMNS, "Q3 方法表现表")

    return q1_action_table.copy(), q2_pair_table.copy(), q3_kernel_table.copy(), q3_metric_table.copy()


def _build_base_win_prob(q3_metric_table: pd.DataFrame) -> dict[str, float]:
    """提取 Q3 无资源、无故障下的基础单局胜率。"""

    mapping = {"领先局": "leading", "平局局": "tied", "落后局": "trailing"}
    subset = q3_metric_table[q3_metric_table["method"] == "mdp"].copy()
    base_win_prob: dict[str, float] = {}
    for _, row in subset.iterrows():
        scenario = mapping.get(str(row["scenario"]))
        if scenario is None:
            continue
        base_win_prob[scenario] = float(row["win_rate"])
    missing = [scenario for scenario in SCENARIO_LABELS if scenario not in base_win_prob]
    if missing:
        raise ValueError(f"Q3 基础胜率缺失场景: {missing}")
    return base_win_prob


def _select_tactical_actions(q3_kernel_table: pd.DataFrame) -> pd.DataFrame:
    """从 Q3 常规动作核中自动筛出 5 个代表性战术动作。"""

    base = q3_kernel_table[q3_kernel_table["counter_ready"] == 0].copy()
    grouped = (
        base.groupby(["action_id", "action_name", "action_type", "macro_group"], as_index=False)["expected_reward"]
        .mean()
        .sort_values(by=["macro_group", "expected_reward", "action_id"], ascending=[True, False, True])
    )
    records: list[dict[str, str]] = []
    for tactical_id, macro_group, action_type in TACTICAL_ACTION_LAYOUT:
        local = grouped[
            (grouped["macro_group"] == macro_group)
            & (grouped["action_type"] == action_type)
        ]
        if local.empty:
            raise ValueError(f"Q3 动作核中缺少宏观动作 {macro_group}")
        row = local.iloc[0]
        records.append(
            {
                "tactical_id": tactical_id,
                "tactical_name": macro_group,
                "action_type": action_type,
                "macro_group": macro_group,
                "source_action_id": str(row["action_id"]),
                "source_action_name": str(row["action_name"]),
            }
        )
    tactical_table = pd.DataFrame(records)
    resource_table = pd.DataFrame(
        [
            {
                "tactical_id": action_id,
                "tactical_name": action_name,
                "action_type": action_type,
                "macro_group": "资源动作",
                "source_action_id": action_id,
                "source_action_name": action_name,
            }
            for action_id, action_name, action_type in RESOURCE_ACTION_LAYOUT
        ]
    )
    return pd.concat([tactical_table, resource_table], ignore_index=True)


def _build_micro_kernel_lookup(
    q3_kernel_table: pd.DataFrame,
    tactical_actions: pd.DataFrame,
    base_win_scale: float,
) -> dict[tuple[str, int, int], MicroKernel]:
    """基于 Q3 常规动作核构建 Q4 可直接调用的战术动作核。"""

    regular_actions = tactical_actions[tactical_actions["action_type"] != "resource"].copy()
    lookup: dict[tuple[str, int, int], MicroKernel] = {}

    for _, tactical_row in regular_actions.iterrows():
        source_action_id = str(tactical_row["source_action_id"])
        tactical_id = str(tactical_row["tactical_id"])
        source_rows = q3_kernel_table[
            (q3_kernel_table["counter_ready"] == 0)
            & (q3_kernel_table["action_id"] == source_action_id)
        ].copy()
        for _, kernel_row in source_rows.iterrows():
            p_for = float(kernel_row["p_score_for"])
            p_against = float(kernel_row["p_score_against"])
            neutral = max(0.0, 1.0 - p_for - p_against)
            scaled_for = float(np.clip(p_for * base_win_scale, 0.0, 1.0))
            remain = max(0.0, 1.0 - scaled_for)
            if p_against + neutral > 0:
                scaled_against = min(remain, p_against / (p_against + neutral) * remain)
            else:
                scaled_against = min(remain, p_against)
            lookup[(tactical_id, int(kernel_row["health_my"]), int(kernel_row["health_opp"]))] = MicroKernel(
                tactical_id=tactical_id,
                tactical_name=str(tactical_row["tactical_name"]),
                action_type=str(tactical_row["action_type"]),
                macro_group=str(tactical_row["macro_group"]),
                source_action_id=source_action_id,
                source_action_name=str(tactical_row["source_action_name"]),
                expected_reward=float(kernel_row["expected_reward"]),
                p_score_for=scaled_for,
                p_score_against=float(np.clip(scaled_against, 0.0, 1.0)),
                p_self_drop=float(np.clip(kernel_row["p_self_drop"], 0.0, 1.0)),
                p_opp_drop=float(np.clip(kernel_row["p_opp_drop"], 0.0, 1.0)),
                p_fall=float(np.clip(kernel_row["p_fall"], 0.0, 1.0)),
            )
    return lookup


def build_context(
    action_feature_file: str | Path,
    defense_pair_file: str | Path,
    q3_kernel_file: str | Path,
    q3_metric_file: str | Path,
    fault_param_file: str | Path,
    config_overrides: dict[str, float] | None = None,
) -> Q4Context:
    """构建 Q4 求解所需的全部上下文。"""

    q1_action_table, q2_pair_table, q3_kernel_table, q3_metric_table = _load_base_tables(
        action_feature_file,
        defense_pair_file,
        q3_kernel_file,
        q3_metric_file,
    )
    params = load_fault_params(fault_param_file)
    if config_overrides:
        params.update(config_overrides)
    config = build_q4_config(params)
    tactical_actions = _select_tactical_actions(q3_kernel_table)
    micro_kernel_lookup = _build_micro_kernel_lookup(q3_kernel_table, tactical_actions, config.base_win_scale)

    q1_action_table = q1_action_table.sort_values(by=["rank", "action_id"]).reset_index(drop=True)
    action_id_to_name = dict(zip(q1_action_table["action_id"], q1_action_table["action_name"], strict=True))
    action_id_to_macro = {
        str(row["action_id"]): str(row["macro_group"])
        for _, row in q3_kernel_table[["action_id", "macro_group"]].drop_duplicates().iterrows()
    }
    regular_ids = tuple(
        tactical_actions[tactical_actions["action_type"] != "resource"]["tactical_id"].tolist()
    )
    resource_ids = tuple(
        tactical_actions[tactical_actions["action_type"] == "resource"]["tactical_id"].tolist()
    )

    return Q4Context(
        config=config,
        q1_action_table=q1_action_table,
        q2_pair_table=q2_pair_table,
        q3_kernel_table=q3_kernel_table,
        q3_metric_table=q3_metric_table,
        tactical_actions=tactical_actions,
        micro_kernel_lookup=micro_kernel_lookup,
        base_win_prob=_build_base_win_prob(q3_metric_table),
        action_id_to_name=action_id_to_name,
        action_id_to_macro=action_id_to_macro,
        all_action_ids=tuple(tactical_actions["tactical_id"].tolist()),
        regular_action_ids=regular_ids,
        resource_action_ids=resource_ids,
    )


def get_regular_kernel(context: Q4Context, tactical_id: str, health_my: int, health_opp: int) -> MicroKernel:
    """按机能状态读取常规动作核。"""

    key = (tactical_id, int(health_my), int(health_opp))
    if key not in context.micro_kernel_lookup:
        raise KeyError(f"缺少动作核: {key}")
    return context.micro_kernel_lookup[key]


def get_feasible_actions(state: RoundState, context: Q4Context) -> tuple[str, ...]:
    """根据局内状态返回可行动作集。"""

    if state.time_bucket >= context.config.n_time_buckets:
        return tuple()

    if state.fault == 1:
        actions: list[str] = []
        if state.repair_left > 0:
            actions.append("USE_REPAIR")
        actions.append("WAIT_FAULT")
        return tuple(actions)

    if state.down_flag == 1:
        actions = []
        if state.reset_left > 0:
            actions.append("USE_RESET")
        actions.append("WAIT_DOWN")
        return tuple(actions)

    actions = list(context.regular_action_ids)
    if state.pause_left > 0:
        actions.append("USE_PAUSE")
    return tuple(actions)


def terminal_win_value(score_diff: int) -> float:
    """终局按当前分差给出胜率值。"""

    if score_diff > 0:
        return 1.0
    if score_diff == 0:
        return 0.5
    return 0.0


def transition_event_name(action_id: str) -> str:
    """资源/战术动作的事件名。"""

    if action_id.startswith("TACT_"):
        return "regular"
    if action_id == "USE_RESET":
        return "reset"
    if action_id == "USE_PAUSE":
        return "pause"
    if action_id == "USE_REPAIR":
        return "repair"
    if action_id == "WAIT_DOWN":
        return "down_wait"
    return "fault_wait"


def _resource_time_cost(state: RoundState, bucket_cost: int, context: Q4Context) -> int:
    """资源动作消耗的是战术机会桶，而非字面净比赛时钟。"""

    if context.config.resource_time_mode != "opportunity_bucket":
        return 0
    return min(context.config.n_time_buckets, state.time_bucket + bucket_cost)


def _self_recover_prob(health_my: int) -> float:
    """倒地后在一个战术机会桶内自主恢复的概率。"""

    base_prob = 0.25 + 0.40 * health_ratio(health_my)
    return float(np.clip(base_prob, 0.0, 0.90))


def _build_regular_transitions(state: RoundState, kernel: MicroKernel, context: Q4Context) -> list[tuple[float, RoundState]]:
    """构造常规战术动作的一步转移。"""

    neutral_prob = max(0.0, 1.0 - kernel.p_score_for - kernel.p_score_against)
    self_cond = min(1.0, kernel.p_self_drop / max(neutral_prob + kernel.p_score_against, 1e-9))
    opp_cond = min(1.0, kernel.p_opp_drop / max(neutral_prob + kernel.p_score_for, 1e-9))

    main_events = [
        ("score_for", kernel.p_score_for, 1),
        ("score_against", kernel.p_score_against, -1),
        ("neutral", neutral_prob, 0),
    ]
    transitions: list[tuple[float, RoundState]] = []

    for event_name, event_prob, score_delta in main_events:
        if event_prob <= 0:
            continue
        if event_name == "score_for":
            health_split = [(1 - opp_cond, 0, 0), (opp_cond, 0, 1)]
        elif event_name == "score_against":
            health_split = [(1 - self_cond, 0, 0), (self_cond, 1, 0)]
        else:
            neutral_self = self_cond * 0.35
            neutral_opp = opp_cond * 0.35
            normal = max(0.0, 1.0 - neutral_self - neutral_opp)
            health_split = [
                (normal, 0, 0),
                (neutral_self, 1, 0),
                (neutral_opp, 0, 1),
            ]

        for health_prob, self_drop, opp_drop in health_split:
            if health_prob <= 0:
                continue
            next_score = _clip_score_diff(state.score_diff + score_delta)
            next_health_my = apply_health_drop(state.health_my, self_drop)
            next_health_opp = apply_health_drop(state.health_opp, opp_drop)
            next_time = min(context.config.n_time_buckets, state.time_bucket + 1)
            fault_rate = compute_fault_rate(next_health_my, context.config)
            down_rate = float(np.clip(context.config.fall_down_weight * kernel.p_fall, 0.0, 0.95))
            fault_rate = float(np.clip(fault_rate, 0.0, 1.0))
            stable_rate = max(0.0, (1.0 - down_rate) * (1.0 - fault_rate))
            fault_only_rate = max(0.0, (1.0 - down_rate) * fault_rate)

            normal_state = RoundState(
                score_diff=next_score,
                time_bucket=next_time,
                health_my=next_health_my,
                health_opp=next_health_opp,
                fault=0,
                down_flag=0,
                reset_left=state.reset_left,
                pause_left=state.pause_left,
                repair_left=state.repair_left,
            )
            fault_state = RoundState(
                score_diff=next_score,
                time_bucket=next_time,
                health_my=next_health_my,
                health_opp=next_health_opp,
                fault=1,
                down_flag=0,
                reset_left=state.reset_left,
                pause_left=state.pause_left,
                repair_left=state.repair_left,
            )
            down_state = RoundState(
                score_diff=next_score,
                time_bucket=next_time,
                health_my=next_health_my,
                health_opp=next_health_opp,
                fault=0,
                down_flag=1,
                reset_left=state.reset_left,
                pause_left=state.pause_left,
                repair_left=state.repair_left,
            )
            weight = float(event_prob * health_prob)
            transitions.append((weight * stable_rate, normal_state))
            transitions.append((weight * fault_only_rate, fault_state))
            transitions.append((weight * down_rate, down_state))

    return _merge_transitions(transitions)


def _build_pause_transitions(state: RoundState, context: Q4Context) -> list[tuple[float, RoundState]]:
    """战术暂停的确定性转移。"""

    next_state = RoundState(
        score_diff=state.score_diff,
        time_bucket=_resource_time_cost(state, context.config.tau_pause_buckets, context),
        health_my=apply_health_gain(state.health_my, context.config.delta_h_pause, context.config),
        health_opp=state.health_opp,
        fault=0,
        down_flag=0,
        reset_left=state.reset_left,
        pause_left=max(0, state.pause_left - 1),
        repair_left=state.repair_left,
    )
    return [(1.0, next_state)]


def _build_reset_transitions(state: RoundState, context: Q4Context) -> list[tuple[float, RoundState]]:
    """人工复位的确定性转移。"""

    next_state = RoundState(
        score_diff=state.score_diff,
        time_bucket=_resource_time_cost(state, context.config.tau_reset_buckets, context),
        health_my=apply_health_gain(state.health_my, context.config.delta_h_reset, context.config),
        health_opp=state.health_opp,
        fault=state.fault,
        down_flag=0,
        reset_left=max(0, state.reset_left - 1),
        pause_left=state.pause_left,
        repair_left=state.repair_left,
    )
    return [(1.0, next_state)]


def _binomial_probabilities(n_trials: int, success_prob: float) -> list[tuple[int, float]]:
    """返回二项分布的全部概率质量。"""

    if n_trials <= 0:
        return [(0, 1.0)]
    probabilities: list[tuple[int, float]] = []
    for success_count in range(n_trials + 1):
        probability = math.comb(n_trials, success_count)
        probability *= success_prob**success_count
        probability *= (1.0 - success_prob) ** (n_trials - success_count)
        probabilities.append((success_count, float(probability)))
    return probabilities


def _build_repair_transitions(state: RoundState, context: Q4Context) -> list[tuple[float, RoundState]]:
    """紧急维修的转移：满血恢复，但时间大量流逝。"""

    remaining = max(0, context.config.n_time_buckets - state.time_bucket)
    repair_steps = min(context.config.tau_repair_buckets, remaining if context.config.resource_time_mode == "opportunity_bucket" else 0)
    next_time = _resource_time_cost(state, repair_steps, context)
    transitions: list[tuple[float, RoundState]] = []
    for score_loss, probability in _binomial_probabilities(repair_steps, context.config.p_score_loss_per_step):
        if probability <= 0:
            continue
        next_state = RoundState(
            score_diff=_clip_score_diff(state.score_diff - score_loss),
            time_bucket=next_time,
            health_my=2,
            health_opp=state.health_opp,
            fault=0,
            down_flag=0,
            reset_left=state.reset_left,
            pause_left=state.pause_left,
            repair_left=max(0, state.repair_left - 1),
        )
        transitions.append((probability, next_state))
    return _merge_transitions(transitions)


def _build_down_wait_transitions(state: RoundState, context: Q4Context) -> list[tuple[float, RoundState]]:
    """倒地后不立即复位时的被动转移。"""

    next_time = _resource_time_cost(state, context.config.tau_reset_buckets, context)
    recover_prob = _self_recover_prob(state.health_my)
    score_loss_prob = context.config.p_score_loss_per_step
    next_health_my = apply_health_drop(state.health_my, 1 if state.health_my == 0 else 0)
    transitions = [
        (
            recover_prob * score_loss_prob,
            RoundState(
                score_diff=_clip_score_diff(state.score_diff - 1),
                time_bucket=next_time,
                health_my=next_health_my,
                health_opp=state.health_opp,
                fault=0,
                down_flag=0,
                reset_left=state.reset_left,
                pause_left=state.pause_left,
                repair_left=state.repair_left,
            ),
        ),
        (
            recover_prob * (1.0 - score_loss_prob),
            RoundState(
                score_diff=state.score_diff,
                time_bucket=next_time,
                health_my=next_health_my,
                health_opp=state.health_opp,
                fault=0,
                down_flag=0,
                reset_left=state.reset_left,
                pause_left=state.pause_left,
                repair_left=state.repair_left,
            ),
        ),
        (
            (1.0 - recover_prob) * score_loss_prob,
            RoundState(
                score_diff=_clip_score_diff(state.score_diff - 1),
                time_bucket=next_time,
                health_my=next_health_my,
                health_opp=state.health_opp,
                fault=0,
                down_flag=1,
                reset_left=state.reset_left,
                pause_left=state.pause_left,
                repair_left=state.repair_left,
            ),
        ),
        (
            (1.0 - recover_prob) * (1.0 - score_loss_prob),
            RoundState(
                score_diff=state.score_diff,
                time_bucket=next_time,
                health_my=next_health_my,
                health_opp=state.health_opp,
                fault=0,
                down_flag=1,
                reset_left=state.reset_left,
                pause_left=state.pause_left,
                repair_left=state.repair_left,
            ),
        ),
    ]
    return _merge_transitions(transitions)


def _build_fault_wait_transitions(state: RoundState, context: Q4Context) -> list[tuple[float, RoundState]]:
    """故障状态下继续等待的被动转移。"""

    next_time = min(context.config.n_time_buckets, state.time_bucket + 1)
    drop_health = max(0, state.health_my - 1)
    transitions = [
        (
            context.config.p_score_loss_per_step,
            RoundState(
                score_diff=_clip_score_diff(state.score_diff - 1),
                time_bucket=next_time,
                health_my=drop_health,
                health_opp=state.health_opp,
                fault=1,
                down_flag=0,
                reset_left=state.reset_left,
                pause_left=state.pause_left,
                repair_left=state.repair_left,
            ),
        ),
        (
            1.0 - context.config.p_score_loss_per_step,
            RoundState(
                score_diff=state.score_diff,
                time_bucket=next_time,
                health_my=drop_health,
                health_opp=state.health_opp,
                fault=1,
                down_flag=0,
                reset_left=state.reset_left,
                pause_left=state.pause_left,
                repair_left=state.repair_left,
            ),
        ),
    ]
    return _merge_transitions(transitions)


def _merge_transitions(transitions: list[tuple[float, RoundState]]) -> list[tuple[float, RoundState]]:
    """合并相同下一状态的转移概率。"""

    bucket: dict[RoundState, float] = {}
    for probability, next_state in transitions:
        if probability <= 0:
            continue
        bucket[next_state] = bucket.get(next_state, 0.0) + float(probability)
    merged = [(probability, next_state) for next_state, probability in bucket.items() if probability > 0]
    total = sum(probability for probability, _ in merged)
    if total <= 0:
        return []
    return [(probability / total, next_state) for probability, next_state in merged]


def build_action_transitions(
    state: RoundState,
    action_id: str,
    context: Q4Context,
) -> list[tuple[float, RoundState]]:
    """统一构造 Q4 一步转移。"""

    if action_id in context.regular_action_ids:
        kernel = get_regular_kernel(context, action_id, state.health_my, state.health_opp)
        return _build_regular_transitions(state, kernel, context)
    if action_id == "USE_PAUSE":
        return _build_pause_transitions(state, context)
    if action_id == "USE_RESET":
        return _build_reset_transitions(state, context)
    if action_id == "USE_REPAIR":
        return _build_repair_transitions(state, context)
    if action_id == "WAIT_DOWN":
        return _build_down_wait_transitions(state, context)
    return _build_fault_wait_transitions(state, context)
