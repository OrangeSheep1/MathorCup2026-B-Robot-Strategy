"""Q2 防守动作参数推导与攻防基础矩阵模块。

本模块严格延续 Q1 的参数体系：
1. 机器人硬件参数继续读取官方参数表。
2. 攻击侧直接继承 Q1 的中间结果，不重复录入结论值。
3. 防守侧只在原始表中存储动作语义、几何标签和少量模型系数。
4. 执行时间、吸收率、位移代价、攻防基础矩阵均在代码中计算。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.q1.model_v1 import MODEL_ASSUMPTIONS, RobotParams


ACTION_FEATURE_COLUMNS = [
    "action_id",
    "action_name",
    "category",
    "impact_score",
    "balance_cost",
    "score_prob",
    "energy_cost",
    "utility",
    "rank",
    "tau_norm",
    "time_norm",
    "exec_time",
    "stable_margin",
    "stability_ratio",
]

ATTACK_SEMANTIC_COLUMNS = [
    "action_id",
    "trajectory_type",
    "direction_tag",
    "height_tag",
    "range_tag",
    "target_zone",
    "source_type",
    "description",
]

DEFENSE_COLUMNS = [
    "defense_id",
    "defense_name",
    "category",
    "closure_group",
    "coverage_direction",
    "coverage_height",
    "contact_mode",
    "primary_joint_group",
    "primary_angle_deg",
    "secondary_joint_group",
    "secondary_angle_deg",
    "sequence_ids",
    "official_time_s",
    "force_capacity_factor",
    "contact_stiffness_ratio",
    "source_type",
    "description",
]

OPTIONAL_DEFENSE_COLUMNS = [
    "force_capacity_basis",
    "stiffness_basis",
]

Q2_MODEL_ASSUMPTIONS = {
    "control_delay_s": 0.12,
    "multi_joint_sync_delay_s": 0.03,
    "lower_body_extra_delay_s": 0.05,
    "ground_motion_extra_delay_s": 0.10,
    "combo_overlap_discount_s": 0.03,
    "recover_factor_s": 0.30,
    "step_back_leg_factor": 0.30,
    "orbit_leg_factor": 0.45,
    "step_adjust_factor": 0.60,
    "micro_adjust_radius_ratio": 0.50,
    "controlled_fall_gamma": 0.12,
    "rapid_getup_gamma": 0.08,
    "ground_guard_gamma": 0.50,
    "arm_pair_link_efficiency": 0.78,
    "arm_single_link_efficiency": 0.67,
    "shoulder_single_link_efficiency": 0.72,
    "wrist_pair_link_efficiency": 0.75,
    "leg_pair_link_efficiency": 0.86,
    "leg_single_link_efficiency": 0.82,
    "torso_turn_link_efficiency": 0.74,
    "imu_link_efficiency": 0.82,
    "step_link_efficiency": 0.88,
    "ground_support_ratio": 0.80,
    "force_base_rigid": 0.85,
    "force_base_posture": 0.60,
    "force_base_balance": 0.45,
    "force_base_soft": 0.50,
    "force_base_ground": 0.30,
    "force_support_ratio_cap": 1.15,
    "elastic_mass_cap_ratio": 2.20,
    "absorb_base_rigid": 0.58,
    "absorb_base_posture": 0.76,
    "absorb_base_balance": 0.70,
    "absorb_base_soft": 0.99,
    "absorb_base_ground": 0.35,
    "absorb_noncontact": 0.95,
}

ASSUMPTION_NOTES = {
    "control_delay_s": "传感与电机控制存在基础响应时滞，用于避免极小角度动作被计算得过快。",
    "multi_joint_sync_delay_s": "双关节或同步动作需要额外的相位协调时间。",
    "lower_body_extra_delay_s": "下肢闪避和步法存在支撑切换时间。",
    "ground_motion_extra_delay_s": "倒地与地面防守动作包含姿态过渡时间。",
    "combo_overlap_discount_s": "组合防守存在少量动作重叠，因此不直接等于纯求和。",
    "recover_factor_s": "沿用题解中 IMU 响应与关节重新预载合计约 0.3 秒的口径。",
    "step_back_leg_factor": "后撤步中主要是支撑脚发力，整条腿并非完全按摆腿轨迹外摆。",
    "orbit_leg_factor": "滑步环绕以腰部转体为主，下肢仅承担辅助位移。",
    "step_adjust_factor": "步点调整是小幅迈步，不等同于完整攻击腿摆动。",
    "micro_adjust_radius_ratio": "重心补偿和卸力缓冲属于小幅关节微调，实际偏心半径低于完整躯干转体。",
    "controlled_fall_gamma": "受控倒地属于主动泄力，等效失稳位移远小于自由翻倒。",
    "rapid_getup_gamma": "快速起身虽有较大姿态变化，但为主动受控恢复动作。",
    "ground_guard_gamma": "倒地防御时腿部抬起对整体重心影响受地面支撑约束。",
    "arm_pair_link_efficiency": "双臂交叉受力时，部分冲击可通过双肩与躯干传递，等效支撑质量按传力效率折算。",
    "arm_single_link_efficiency": "单臂格挡/肘挡主要依赖单侧上肢链，等效支撑质量低于双臂硬格挡。",
    "shoulder_single_link_efficiency": "单肩下压格挡的支撑链短于双臂交叉，承载效率略低。",
    "wrist_pair_link_efficiency": "双手钳制同时依赖前臂与躯干约束，传力效率介于双臂格挡与单臂格挡之间。",
    "leg_pair_link_efficiency": "沉身防御依赖躯干与双腿共同承载，支撑链完整但存在屈膝损耗。",
    "leg_single_link_efficiency": "单腿/单髋主导的闪避与下潜存在单侧支撑损耗。",
    "torso_turn_link_efficiency": "侧身/转身防御主要通过躯干转动卸力，支撑质量以躯干为主。",
    "imu_link_efficiency": "平衡恢复类动作以微调为主，传力效率低于硬接触链。",
    "step_link_efficiency": "步法调整依赖落脚重建支撑，支撑有效性低于原地姿态防守。",
    "ground_support_ratio": "倒地与地面防御中，实际承力主体是身体与地面的复合支撑体。",
    "force_base_rigid": "刚性格挡的承载上限以 D01 十字格挡为标尺，其他刚接触动作按支撑质量和技法修正。",
    "force_base_posture": "防御姿态类以面接触分散冲击，承载上限低于硬格挡。",
    "force_base_balance": "平衡恢复类以姿态补偿为主，不宜赋予过高硬承载能力。",
    "force_base_soft": "卸力缓冲通过柔顺控制吸收冲量，承载上限取中等。",
    "force_base_ground": "地面防守主要依赖地面与身体共同承载，但不可视作稳定高效的正面格挡。",
    "force_support_ratio_cap": "支撑质量对承载能力的增益采用截断，避免整体质量过大导致承载能力失真。",
    "elastic_mass_cap_ratio": "碰撞等效质量只取局部承力链上限，避免整机质量直接代替局部接触质量。",
    "absorb_base_rigid": "刚性格挡吸收率来自“弹性传递 × 结构耗散”混合模型的基准值。",
    "absorb_base_posture": "防御姿态通过增大受力面积提高耗散，基准吸收率高于硬格挡。",
    "absorb_base_balance": "平衡恢复类动作可吸收部分冲击，但主要价值仍在姿态重建。",
    "absorb_base_soft": "柔顺控制类动作吸收率最高，接近完全卸力。",
    "absorb_base_ground": "地面接触虽刚性高，但冲击大量转移到地面，吸收率按混合模型折算。",
    "absorb_noncontact": "闪避类动作视为近似完全规避，仅保留少量残余风险。",
}

DIRECTION_PARTIAL_MAP = {
    "front": {"mixed", "all"},
    "lateral": {"mixed", "all", "spin"},
    "spin": {"mixed", "all", "lateral"},
    "low": {"mixed", "all"},
    "mixed": {"front", "lateral", "spin", "low", "all"},
    "all": {"front", "lateral", "spin", "low", "mixed"},
}

HEIGHT_PARTIAL_MAP = {
    "high": {"mixed", "all", "middle"},
    "middle": {"mixed", "all", "high", "low"},
    "low": {"mixed", "all", "middle"},
    "mixed": {"high", "middle", "low", "all"},
    "all": {"high", "middle", "low", "mixed"},
}

LOWER_BODY_GROUPS = {"hip_lateral", "knee_single", "knee_pair", "step_back", "step_adjust", "leg_raise"}
GROUND_GROUPS = {"whole_body_roll", "get_up", "leg_raise"}


def _parse_tag_tokens(tag: str) -> set[str]:
    """将复合标签拆成若干基础标签。"""

    normalized = str(tag).strip().lower().replace("|", "_")
    if not normalized:
        return set()
    if normalized in {"all", "mixed"}:
        return {normalized}
    return {part for part in normalized.split("_") if part}


def _single_tag_score(attack_token: str, defense_token: str, partial_map: dict[str, set[str]]) -> float:
    """比较单个攻击标签与单个防守标签。"""

    if defense_token == "all":
        return 1.0
    if attack_token == defense_token:
        return 1.0
    if defense_token == "mixed":
        return 0.5
    if defense_token in partial_map.get(attack_token, set()):
        return 0.5
    return 0.0


def _token_set_score(attack_tag: str, defense_tag: str, partial_map: dict[str, set[str]]) -> float:
    """比较复合攻击标签与防守标签。"""

    attack_tokens = _parse_tag_tokens(attack_tag)
    defense_tokens = _parse_tag_tokens(defense_tag)
    if not attack_tokens or not defense_tokens:
        return 0.0
    if "all" in defense_tokens:
        return 1.0
    if "mixed" in defense_tokens and len(attack_tokens) > 1:
        return 1.0

    best_score = 0.0
    for attack_token in attack_tokens:
        for defense_token in defense_tokens:
            best_score = max(best_score, _single_tag_score(attack_token, defense_token, partial_map))

    if best_score == 1.0 and len(attack_tokens) > 1 and not attack_tokens.issubset(defense_tokens):
        return 0.5
    return best_score


@dataclass(frozen=True)
class AttackSemantic:
    """攻击动作的语义标签。"""

    action_id: str
    trajectory_type: str
    direction_tag: str
    height_tag: str
    range_tag: str
    target_zone: str
    description: str


@dataclass(frozen=True)
class DefenseAction:
    """防守动作的原始元数据。"""

    defense_id: str
    defense_name: str
    category: str
    closure_group: str
    coverage_direction: str
    coverage_height: str
    contact_mode: str
    primary_joint_group: str
    primary_angle_deg: float
    secondary_joint_group: str | None
    secondary_angle_deg: float | None
    sequence_ids: tuple[str, ...]
    official_time_s: float | None
    force_capacity_factor: float | None
    contact_stiffness_ratio: float | None
    description: str
    force_capacity_basis: str
    stiffness_basis: str


def _require_columns(data: pd.DataFrame, required: list[str], label: str) -> None:
    """检查输入表字段。"""

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{label}缺少字段: {missing}")


def load_action_features(file_path: str | Path) -> pd.DataFrame:
    """读取 Q1 中间结果。"""

    data = pd.read_csv(file_path)
    _require_columns(data, ACTION_FEATURE_COLUMNS, "Q1 动作特征表")
    return data[ACTION_FEATURE_COLUMNS].copy()


def load_attack_semantics(file_path: str | Path) -> pd.DataFrame:
    """读取攻击语义标签表。"""

    data = pd.read_csv(file_path)
    _require_columns(data, ATTACK_SEMANTIC_COLUMNS, "Q2 攻击语义表")
    return data[ATTACK_SEMANTIC_COLUMNS].copy()


def load_defense_actions(file_path: str | Path) -> pd.DataFrame:
    """读取防守动作原始元数据。"""

    data = pd.read_csv(file_path)
    _require_columns(data, DEFENSE_COLUMNS, "Q2 防守动作表")
    keep_columns = DEFENSE_COLUMNS + [column for column in OPTIONAL_DEFENSE_COLUMNS if column in data.columns]
    return data[keep_columns].copy()


def build_attack_semantic(row: pd.Series) -> AttackSemantic:
    """将攻击语义表的一行转为结构化对象。"""

    return AttackSemantic(
        action_id=str(row["action_id"]),
        trajectory_type=str(row["trajectory_type"]),
        direction_tag=str(row["direction_tag"]),
        height_tag=str(row["height_tag"]),
        range_tag=str(row["range_tag"]),
        target_zone=str(row["target_zone"]),
        description=str(row["description"]),
    )


def build_defense_action(row: pd.Series) -> DefenseAction:
    """将防守表的一行转为结构化对象。"""

    secondary_group = None if pd.isna(row["secondary_joint_group"]) else str(row["secondary_joint_group"])
    secondary_angle = None if pd.isna(row["secondary_angle_deg"]) else float(row["secondary_angle_deg"])
    official_time = None if pd.isna(row["official_time_s"]) else float(row["official_time_s"])
    force_capacity = None if pd.isna(row["force_capacity_factor"]) else float(row["force_capacity_factor"])
    stiffness = None if pd.isna(row["contact_stiffness_ratio"]) else float(row["contact_stiffness_ratio"])
    sequence_ids = tuple()
    if not pd.isna(row["sequence_ids"]):
        sequence_ids = tuple(part for part in str(row["sequence_ids"]).split("|") if part)

    return DefenseAction(
        defense_id=str(row["defense_id"]),
        defense_name=str(row["defense_name"]),
        category=str(row["category"]),
        closure_group=str(row["closure_group"]),
        coverage_direction=str(row["coverage_direction"]),
        coverage_height=str(row["coverage_height"]),
        contact_mode=str(row["contact_mode"]),
        primary_joint_group=str(row["primary_joint_group"]),
        primary_angle_deg=float(row["primary_angle_deg"]) if not pd.isna(row["primary_angle_deg"]) else 0.0,
        secondary_joint_group=secondary_group,
        secondary_angle_deg=secondary_angle,
        sequence_ids=sequence_ids,
        official_time_s=official_time,
        force_capacity_factor=force_capacity,
        contact_stiffness_ratio=stiffness,
        description=str(row["description"]),
        force_capacity_basis="" if "force_capacity_basis" not in row.index or pd.isna(row["force_capacity_basis"]) else str(row["force_capacity_basis"]),
        stiffness_basis="" if "stiffness_basis" not in row.index or pd.isna(row["stiffness_basis"]) else str(row["stiffness_basis"]),
    )


def build_attack_catalog(action_features: pd.DataFrame, attack_semantics: pd.DataFrame) -> pd.DataFrame:
    """合并 Q1 数值特征与 Q2 攻击语义。"""

    catalog = action_features.merge(attack_semantics, on="action_id", how="left", validate="one_to_one")
    _require_columns(
        catalog,
        ACTION_FEATURE_COLUMNS + ATTACK_SEMANTIC_COLUMNS[1:],
        "Q2 攻击目录",
    )
    catalog = catalog.rename(
        columns={
            "category": "attack_category",
            "utility": "attack_utility",
            "rank": "attack_rank",
            "description": "attack_description",
            "source_type": "semantic_source_type",
            "stable_margin": "attack_stable_margin",
        }
    )
    catalog["opp_recover_time"] = catalog["exec_time"] + Q2_MODEL_ASSUMPTIONS["recover_factor_s"] * catalog["stability_ratio"]
    return catalog


def _arc_displacement(lever_m: float, angle_deg: float) -> float:
    """均匀刚性杆绕关节转动时的质心位移。"""

    if angle_deg <= 0:
        return 0.0
    return lever_m * math.sin(math.radians(angle_deg) / 2.0)


def _torso_turn_displacement(angle_deg: float, double_radius: bool = True) -> float:
    """腰部转体导致的躯干质心位移。"""

    radius = MODEL_ASSUMPTIONS["torso_eccentricity_m"]
    factor = 2.0 if double_radius else 1.0
    return factor * radius * math.sin(math.radians(angle_deg) / 2.0)


def _base_motion_time(angle_deg: float, robot: RobotParams) -> float:
    """按有效角速度计算基础关节运动时间。"""

    if angle_deg <= 0:
        return 0.0
    return 1.5 * math.radians(angle_deg) / robot.omega_eff_rad_s


def _joint_group_delay(defense: DefenseAction) -> float:
    """根据动作类型补充姿态切换延迟。"""

    extra = Q2_MODEL_ASSUMPTIONS["control_delay_s"]
    if defense.secondary_joint_group:
        extra += Q2_MODEL_ASSUMPTIONS["multi_joint_sync_delay_s"]
    groups = {defense.primary_joint_group}
    if defense.secondary_joint_group:
        groups.add(defense.secondary_joint_group)
    if groups & LOWER_BODY_GROUPS:
        extra += Q2_MODEL_ASSUMPTIONS["lower_body_extra_delay_s"]
    if groups & GROUND_GROUPS:
        extra += Q2_MODEL_ASSUMPTIONS["ground_motion_extra_delay_s"]
    return extra


def compute_defense_execution_time(
    defense: DefenseAction,
    robot: RobotParams,
    defense_map: dict[str, DefenseAction],
    cache: dict[str, float],
) -> float:
    """计算防守执行时间。"""

    if defense.defense_id in cache:
        return cache[defense.defense_id]

    if defense.official_time_s is not None:
        cache[defense.defense_id] = defense.official_time_s
        return defense.official_time_s

    if defense.sequence_ids:
        total_time = sum(compute_defense_execution_time(defense_map[item], robot, defense_map, cache) for item in defense.sequence_ids)
        total_time = max(
            Q2_MODEL_ASSUMPTIONS["control_delay_s"],
            total_time - Q2_MODEL_ASSUMPTIONS["combo_overlap_discount_s"],
        )
        cache[defense.defense_id] = total_time
        return total_time

    primary_time = _base_motion_time(defense.primary_angle_deg, robot)
    secondary_time = _base_motion_time(defense.secondary_angle_deg or 0.0, robot)
    motion_time = max(primary_time, secondary_time)
    total_time = motion_time + _joint_group_delay(defense)
    cache[defense.defense_id] = total_time
    return total_time


def compute_defense_balance_cost(
    defense: DefenseAction,
    robot: RobotParams,
    defense_map: dict[str, DefenseAction],
    cache: dict[str, float],
) -> float:
    """计算防守动作自身的位移代价。"""

    if defense.defense_id in cache:
        return cache[defense.defense_id]

    mass_total = robot.total_mass_kg
    forearm_mass = robot.forearm_mass_kg
    thigh_mass = robot.thigh_mass_kg
    single_leg_mass = robot.single_leg_mass_kg
    torso_mass = robot.torso_mass_kg

    if defense.sequence_ids:
        component_values = [
            compute_defense_balance_cost(defense_map[item], robot, defense_map, cache) for item in defense.sequence_ids
        ]
        value = float(np.sqrt(np.sum(np.square(component_values))))
        cache[defense.defense_id] = value
        return value

    primary = defense.primary_angle_deg
    secondary = defense.secondary_angle_deg or 0.0

    if defense.defense_id == "D01":
        value = 2.0 * forearm_mass * _arc_displacement(robot.forearm_lever_m, primary) / mass_total
    elif defense.defense_id in {"D02", "D03", "D04"}:
        value = forearm_mass * _arc_displacement(robot.forearm_lever_m, primary) / mass_total
    elif defense.defense_id == "D05":
        value = 2.0 * forearm_mass * _arc_displacement(robot.forearm_lever_m, primary) / mass_total
    elif defense.defense_id == "D06":
        value = (
            thigh_mass * _arc_displacement(robot.thigh_lever_m, primary)
            + torso_mass * _torso_turn_displacement(secondary, double_radius=False) * 0.5
        ) / mass_total
    elif defense.defense_id == "D07":
        value = thigh_mass * _arc_displacement(robot.thigh_lever_m, primary) / mass_total
    elif defense.defense_id == "D08":
        value = (
            single_leg_mass
            * _arc_displacement(robot.full_leg_lever_m, primary)
            * Q2_MODEL_ASSUMPTIONS["step_back_leg_factor"]
            + torso_mass * _torso_turn_displacement(secondary, double_radius=False)
        ) / mass_total
    elif defense.defense_id == "D09":
        value = torso_mass * _torso_turn_displacement(primary, double_radius=True) / mass_total
    elif defense.defense_id == "D10":
        value = (
            torso_mass * _torso_turn_displacement(primary, double_radius=True)
            + single_leg_mass
            * _arc_displacement(robot.thigh_lever_m, secondary)
            * Q2_MODEL_ASSUMPTIONS["orbit_leg_factor"]
        ) / mass_total
    elif defense.defense_id == "D11":
        value = 2.0 * forearm_mass * _arc_displacement(robot.forearm_lever_m, primary) / mass_total
    elif defense.defense_id == "D12":
        value = thigh_mass * _arc_displacement(robot.thigh_lever_m, primary) / mass_total
    elif defense.defense_id == "D13":
        value = torso_mass * _torso_turn_displacement(primary, double_radius=False) / mass_total
    elif defense.defense_id in {"D14", "D15"}:
        radius = MODEL_ASSUMPTIONS["torso_eccentricity_m"] * Q2_MODEL_ASSUMPTIONS["micro_adjust_radius_ratio"]
        value = torso_mass * radius * math.sin(math.radians(primary) / 2.0) / mass_total
    elif defense.defense_id == "D16":
        value = (
            single_leg_mass
            * _arc_displacement(robot.thigh_lever_m, primary)
            * Q2_MODEL_ASSUMPTIONS["step_adjust_factor"]
        ) / mass_total
    elif defense.defense_id == "D17":
        value = (
            robot.com_height_m
            * math.sin(math.radians(primary) / 2.0)
            * Q2_MODEL_ASSUMPTIONS["controlled_fall_gamma"]
        )
    elif defense.defense_id == "D18":
        value = (
            robot.com_height_m
            * math.sin(math.radians(50.0))
            * Q2_MODEL_ASSUMPTIONS["rapid_getup_gamma"]
        )
    elif defense.defense_id == "D19":
        value = (
            single_leg_mass
            * _arc_displacement(robot.thigh_lever_m, primary)
            * Q2_MODEL_ASSUMPTIONS["ground_guard_gamma"]
        ) / mass_total
    else:
        value = 0.0

    cache[defense.defense_id] = float(value)
    return float(value)


def _weighted_average_by_time(values: list[float], weights: list[float]) -> float:
    """按执行时间做稳健加权平均。"""

    if not values:
        return 0.0
    weight_array = np.array(weights, dtype=float)
    value_array = np.array(values, dtype=float)
    if np.isclose(weight_array.sum(), 0.0):
        return float(value_array.mean())
    return float(np.average(value_array, weights=weight_array))


def compute_effective_support_mass(
    defense: DefenseAction,
    defense_map: dict[str, DefenseAction],
    time_cache: dict[str, float],
    mass_cache: dict[str, float],
    robot: RobotParams,
) -> float:
    """估计防守动作用于承接或疏导冲击的等效支撑质量。"""

    if defense.defense_id in mass_cache:
        return mass_cache[defense.defense_id]

    if defense.sequence_ids:
        component_times = [
            compute_defense_execution_time(defense_map[item], robot, defense_map, time_cache) for item in defense.sequence_ids
        ]
        component_masses = [
            compute_effective_support_mass(defense_map[item], defense_map, time_cache, mass_cache, robot)
            for item in defense.sequence_ids
        ]
        support_mass = _weighted_average_by_time(component_masses, component_times)
        mass_cache[defense.defense_id] = support_mass
        return support_mass

    single_arm_mass = robot.single_arm_mass_kg
    forearm_mass = robot.forearm_mass_kg
    thigh_mass = robot.thigh_mass_kg
    single_leg_mass = robot.single_leg_mass_kg
    torso_mass = robot.torso_mass_kg

    group = defense.primary_joint_group
    if group == "shoulder_pair":
        support_mass = 2.0 * single_arm_mass / Q2_MODEL_ASSUMPTIONS["arm_pair_link_efficiency"]
    elif group == "elbow_single":
        support_mass = single_arm_mass / Q2_MODEL_ASSUMPTIONS["arm_single_link_efficiency"]
    elif group == "shoulder_single":
        support_mass = single_arm_mass / Q2_MODEL_ASSUMPTIONS["shoulder_single_link_efficiency"]
    elif group == "wrist_pair":
        support_mass = (2.0 * forearm_mass + 0.20 * torso_mass) / Q2_MODEL_ASSUMPTIONS["wrist_pair_link_efficiency"]
    elif group == "knee_pair":
        support_mass = (torso_mass + 2.0 * thigh_mass) / Q2_MODEL_ASSUMPTIONS["leg_pair_link_efficiency"]
    elif group == "knee_single":
        support_mass = (torso_mass + thigh_mass) / Q2_MODEL_ASSUMPTIONS["leg_single_link_efficiency"]
    elif group == "hip_lateral":
        support_mass = (torso_mass + 0.80 * single_leg_mass) / Q2_MODEL_ASSUMPTIONS["leg_single_link_efficiency"]
    elif group == "waist_turn":
        support_mass = torso_mass / Q2_MODEL_ASSUMPTIONS["torso_turn_link_efficiency"]
    elif group in {"imu_micro", "imu_soft"}:
        support_mass = torso_mass / Q2_MODEL_ASSUMPTIONS["imu_link_efficiency"]
    elif group in {"step_adjust", "step_back"}:
        support_mass = (torso_mass + 0.60 * single_leg_mass) / Q2_MODEL_ASSUMPTIONS["step_link_efficiency"]
    elif group in {"whole_body_roll", "get_up", "leg_raise"}:
        support_mass = robot.total_mass_kg * Q2_MODEL_ASSUMPTIONS["ground_support_ratio"]
    else:
        support_mass = torso_mass

    mass_cache[defense.defense_id] = float(support_mass)
    return float(support_mass)


def _force_base_factor(defense: DefenseAction) -> float:
    """按防守类别给出承载基准。"""

    if defense.contact_mode == "soft":
        return Q2_MODEL_ASSUMPTIONS["force_base_soft"]
    if defense.contact_mode == "ground":
        return Q2_MODEL_ASSUMPTIONS["force_base_ground"]
    if defense.category == "posture":
        return Q2_MODEL_ASSUMPTIONS["force_base_posture"]
    if defense.category == "balance":
        return Q2_MODEL_ASSUMPTIONS["force_base_balance"]
    return Q2_MODEL_ASSUMPTIONS["force_base_rigid"]


def _force_technique_multiplier(defense: DefenseAction) -> float:
    """反映不同防守技法的受力路径差异。"""

    technique_map = {
        "D01": 1.00,
        "D02": 0.85,
        "D03": 1.00,
        "D04": 0.95,
        "D05": 0.79,
        "D11": 1.08,
        "D12": 0.87,
        "D13": 0.84,
        "D14": 0.87,
        "D15": 0.87,
        "D16": 0.77,
        "D17": 1.02,
        "D18": 0.58,
        "D19": 0.87,
    }
    return technique_map.get(defense.defense_id, 1.0)


def compute_force_capacity_factor(
    defense: DefenseAction,
    defense_map: dict[str, DefenseAction],
    time_cache: dict[str, float],
    mass_cache: dict[str, float],
    factor_cache: dict[str, float],
    robot: RobotParams,
) -> float:
    """根据支撑质量与技法修正推导防守承载系数。"""

    if defense.defense_id in factor_cache:
        return factor_cache[defense.defense_id]

    if defense.contact_mode == "none":
        factor_cache[defense.defense_id] = 1.0
        return 1.0

    if defense.sequence_ids:
        component_times = [
            compute_defense_execution_time(defense_map[item], robot, defense_map, time_cache) for item in defense.sequence_ids
        ]
        component_factors = [
            compute_force_capacity_factor(defense_map[item], defense_map, time_cache, mass_cache, factor_cache, robot)
            for item in defense.sequence_ids
        ]
        derived_factor = _weighted_average_by_time(component_factors, component_times)
        factor_cache[defense.defense_id] = derived_factor
        return derived_factor

    reference_support_mass = 2.0 * robot.single_arm_mass_kg / Q2_MODEL_ASSUMPTIONS["arm_pair_link_efficiency"]
    support_mass = compute_effective_support_mass(defense, defense_map, time_cache, mass_cache, robot)
    support_ratio = math.sqrt(max(support_mass, 1e-9) / reference_support_mass)
    support_ratio = min(support_ratio, Q2_MODEL_ASSUMPTIONS["force_support_ratio_cap"])

    derived_factor = _force_base_factor(defense) * support_ratio * _force_technique_multiplier(defense)
    derived_factor = float(np.clip(derived_factor, 0.0, 1.0))
    factor_cache[defense.defense_id] = derived_factor
    return derived_factor


def _absorb_base_factor(defense: DefenseAction) -> float:
    """按接触模式给出结构耗散基准。"""

    if defense.contact_mode == "none":
        return Q2_MODEL_ASSUMPTIONS["absorb_noncontact"]
    if defense.contact_mode == "soft":
        return Q2_MODEL_ASSUMPTIONS["absorb_base_soft"]
    if defense.contact_mode == "ground":
        return Q2_MODEL_ASSUMPTIONS["absorb_base_ground"]
    if defense.category == "posture":
        return Q2_MODEL_ASSUMPTIONS["absorb_base_posture"]
    if defense.category == "balance":
        return Q2_MODEL_ASSUMPTIONS["absorb_base_balance"]
    return Q2_MODEL_ASSUMPTIONS["absorb_base_rigid"]


def compute_elastic_transfer_rate(
    defense: DefenseAction,
    defense_map: dict[str, DefenseAction],
    time_cache: dict[str, float],
    mass_cache: dict[str, float],
    elastic_cache: dict[str, float],
    robot: RobotParams,
) -> float:
    """计算局部接触链的弹性传递效率。"""

    if defense.defense_id in elastic_cache:
        return elastic_cache[defense.defense_id]

    if defense.contact_mode == "none":
        elastic_cache[defense.defense_id] = 1.0
        return 1.0

    if defense.sequence_ids:
        component_times = [
            compute_defense_execution_time(defense_map[item], robot, defense_map, time_cache) for item in defense.sequence_ids
        ]
        component_elastic = [
            compute_elastic_transfer_rate(defense_map[item], defense_map, time_cache, mass_cache, elastic_cache, robot)
            for item in defense.sequence_ids
        ]
        elastic_rate = _weighted_average_by_time(component_elastic, component_times)
        elastic_cache[defense.defense_id] = elastic_rate
        return elastic_rate

    reference_attack_mass = robot.thigh_mass_kg
    support_mass = compute_effective_support_mass(defense, defense_map, time_cache, mass_cache, robot)
    capped_support_mass = min(
        support_mass,
        Q2_MODEL_ASSUMPTIONS["elastic_mass_cap_ratio"] * reference_attack_mass,
    )
    elastic_rate = 1.0 - (
        (reference_attack_mass - capped_support_mass) / (reference_attack_mass + capped_support_mass)
    ) ** 2
    elastic_rate = float(np.clip(elastic_rate, 0.0, 1.0))
    elastic_cache[defense.defense_id] = elastic_rate
    return elastic_rate


def compute_contact_stiffness_ratio(
    defense: DefenseAction,
    defense_map: dict[str, DefenseAction],
    time_cache: dict[str, float],
    mass_cache: dict[str, float],
    elastic_cache: dict[str, float],
    stiffness_cache: dict[str, float],
    robot: RobotParams,
) -> float:
    """根据局部等效质量与结构耗散估计接触刚度比。"""

    if defense.defense_id in stiffness_cache:
        return stiffness_cache[defense.defense_id]

    if defense.contact_mode == "none":
        stiffness_ratio = 1.0 - Q2_MODEL_ASSUMPTIONS["absorb_noncontact"]
        stiffness_cache[defense.defense_id] = stiffness_ratio
        return stiffness_ratio

    if defense.sequence_ids:
        component_times = [
            compute_defense_execution_time(defense_map[item], robot, defense_map, time_cache) for item in defense.sequence_ids
        ]
        component_stiffness = [
            compute_contact_stiffness_ratio(
                defense_map[item],
                defense_map,
                time_cache,
                mass_cache,
                elastic_cache,
                stiffness_cache,
                robot,
            )
            for item in defense.sequence_ids
        ]
        stiffness_ratio = _weighted_average_by_time(component_stiffness, component_times)
        stiffness_cache[defense.defense_id] = stiffness_ratio
        return stiffness_ratio

    elastic_rate = compute_elastic_transfer_rate(defense, defense_map, time_cache, mass_cache, elastic_cache, robot)
    absorb_rate = _absorb_base_factor(defense) * elastic_rate
    stiffness_ratio = float(np.clip(1.0 - absorb_rate, 0.0, 1.0))
    stiffness_cache[defense.defense_id] = stiffness_ratio
    return stiffness_ratio


def compute_absorb_rate(
    defense: DefenseAction,
    defense_map: dict[str, DefenseAction],
    time_cache: dict[str, float],
    mass_cache: dict[str, float],
    elastic_cache: dict[str, float],
    stiffness_cache: dict[str, float],
    absorb_cache: dict[str, float],
    robot: RobotParams,
) -> float:
    """根据弹性传递与结构耗散的混合模型计算冲击吸收率。"""

    if defense.defense_id in absorb_cache:
        return absorb_cache[defense.defense_id]

    if defense.sequence_ids:
        component_times = [
            compute_defense_execution_time(defense_map[item], robot, defense_map, time_cache) for item in defense.sequence_ids
        ]
        component_absorb = [
            compute_absorb_rate(
                defense_map[item],
                defense_map,
                time_cache,
                mass_cache,
                elastic_cache,
                stiffness_cache,
                absorb_cache,
                robot,
            )
            for item in defense.sequence_ids
        ]
        absorb_rate = _weighted_average_by_time(component_absorb, component_times)
        absorb_cache[defense.defense_id] = absorb_rate
        return absorb_rate

    stiffness = compute_contact_stiffness_ratio(
        defense,
        defense_map,
        time_cache,
        mass_cache,
        elastic_cache,
        stiffness_cache,
        robot,
    )
    absorb_rate = 1.0 - stiffness
    absorb_cache[defense.defense_id] = float(absorb_rate)
    return float(absorb_rate)


def compute_force_capacity(
    defense: DefenseAction,
    defense_map: dict[str, DefenseAction],
    time_cache: dict[str, float],
    mass_cache: dict[str, float],
    factor_cache: dict[str, float],
    capacity_cache: dict[str, float],
    robot: RobotParams,
) -> float:
    """将防守承载能力映射到与攻击冲击力矩同尺度的上限。"""

    if defense.defense_id in capacity_cache:
        return capacity_cache[defense.defense_id]

    impact_ceiling = 3.0 * robot.max_joint_torque_nm
    if defense.sequence_ids:
        component_times = [
            compute_defense_execution_time(defense_map[item], robot, defense_map, time_cache) for item in defense.sequence_ids
        ]
        component_caps = [
            compute_force_capacity(defense_map[item], defense_map, time_cache, mass_cache, factor_cache, capacity_cache, robot)
            for item in defense.sequence_ids
        ]
        capacity = _weighted_average_by_time(component_caps, component_times)
        capacity_cache[defense.defense_id] = capacity
        return capacity

    factor = compute_force_capacity_factor(defense, defense_map, time_cache, mass_cache, factor_cache, robot)
    if defense.contact_mode == "none":
        capacity = float(impact_ceiling)
    else:
        capacity = float(factor * impact_ceiling)
    capacity_cache[defense.defense_id] = capacity
    return capacity


def build_defense_feature_table(defense_actions: pd.DataFrame, robot: RobotParams) -> pd.DataFrame:
    """构建防守动作特征表。"""

    defense_objects = [build_defense_action(row) for _, row in defense_actions.iterrows()]
    defense_map = {item.defense_id: item for item in defense_objects}
    time_cache: dict[str, float] = {}
    balance_cache: dict[str, float] = {}
    mass_cache: dict[str, float] = {}
    elastic_cache: dict[str, float] = {}
    stiffness_cache: dict[str, float] = {}
    absorb_cache: dict[str, float] = {}
    factor_cache: dict[str, float] = {}
    capacity_cache: dict[str, float] = {}

    records: list[dict[str, object]] = []
    for defense in defense_objects:
        exec_time = compute_defense_execution_time(defense, robot, defense_map, time_cache)
        balance_cost = compute_defense_balance_cost(defense, robot, defense_map, balance_cache)
        support_mass = compute_effective_support_mass(defense, defense_map, time_cache, mass_cache, robot)
        elastic_rate = compute_elastic_transfer_rate(defense, defense_map, time_cache, mass_cache, elastic_cache, robot)
        force_base_factor = _force_base_factor(defense)
        absorb_base_factor = _absorb_base_factor(defense)
        stiffness_ratio = compute_contact_stiffness_ratio(
            defense,
            defense_map,
            time_cache,
            mass_cache,
            elastic_cache,
            stiffness_cache,
            robot,
        )
        absorb_rate = compute_absorb_rate(
            defense,
            defense_map,
            time_cache,
            mass_cache,
            elastic_cache,
            stiffness_cache,
            absorb_cache,
            robot,
        )
        force_capacity_factor = compute_force_capacity_factor(
            defense,
            defense_map,
            time_cache,
            mass_cache,
            factor_cache,
            robot,
        )
        force_capacity = compute_force_capacity(
            defense,
            defense_map,
            time_cache,
            mass_cache,
            factor_cache,
            capacity_cache,
            robot,
        )
        mobility_cost = balance_cost / robot.stable_margin_m if robot.stable_margin_m > 0 else 0.0
        force_capacity_factor_input = np.nan if defense.contact_mode == "none" or defense.sequence_ids else defense.force_capacity_factor
        stiffness_ratio_input = np.nan if defense.sequence_ids else defense.contact_stiffness_ratio
        force_capacity_delta = (
            np.nan if pd.isna(force_capacity_factor_input) else force_capacity_factor - float(force_capacity_factor_input)
        )
        stiffness_delta = np.nan if pd.isna(stiffness_ratio_input) else stiffness_ratio - float(stiffness_ratio_input)
        force_capacity_audit = (
            "not_applicable"
            if pd.isna(force_capacity_delta)
            else ("OK" if abs(force_capacity_delta) <= 0.08 else "revisit")
        )
        stiffness_audit = (
            "not_applicable"
            if pd.isna(stiffness_delta)
            else ("OK" if abs(stiffness_delta) <= 0.10 else "revisit")
        )
        records.append(
            {
                "defense_id": defense.defense_id,
                "defense_name": defense.defense_name,
                "defense_category": defense.category,
                "closure_group": defense.closure_group,
                "coverage_direction": defense.coverage_direction,
                "coverage_height": defense.coverage_height,
                "contact_mode": defense.contact_mode,
                "primary_joint_group": defense.primary_joint_group,
                "primary_angle_deg": defense.primary_angle_deg,
                "secondary_joint_group": defense.secondary_joint_group or "",
                "secondary_angle_deg": 0.0 if defense.secondary_angle_deg is None else defense.secondary_angle_deg,
                "sequence_ids": "|".join(defense.sequence_ids),
                "exec_time_def": exec_time,
                "absorb_rate": absorb_rate,
                "balance_cost_def": balance_cost,
                "mobility_cost": mobility_cost,
                "effective_support_mass": support_mass,
                "elastic_transfer_rate": elastic_rate,
                "force_base_factor": force_base_factor,
                "absorb_base_factor": absorb_base_factor,
                "force_capacity": force_capacity,
                "force_capacity_factor": force_capacity_factor,
                "force_capacity_factor_input": force_capacity_factor_input,
                "force_capacity_factor_delta": force_capacity_delta,
                "force_capacity_factor_audit": force_capacity_audit,
                "contact_stiffness_ratio": stiffness_ratio,
                "contact_stiffness_ratio_input": stiffness_ratio_input,
                "contact_stiffness_ratio_delta": stiffness_delta,
                "contact_stiffness_ratio_audit": stiffness_audit,
                "defense_source_type": defense_actions.loc[
                    defense_actions["defense_id"] == defense.defense_id, "source_type"
                ].iloc[0],
                "defense_description": defense.description,
                "force_capacity_basis": defense.force_capacity_basis,
                "stiffness_basis": defense.stiffness_basis,
            }
        )

    feature_table = pd.DataFrame(records)
    feature_table["defense_stable_margin"] = robot.stable_margin_m
    feature_table["omega_out"] = robot.omega_out_rad_s
    feature_table["omega_eff"] = robot.omega_eff_rad_s
    return feature_table


def _direction_score(attack_direction: str, defense_direction: str) -> float:
    """计算攻防方向匹配度。"""

    return _token_set_score(attack_direction, defense_direction, DIRECTION_PARTIAL_MAP)


def _height_score(attack_height: str, defense_height: str) -> float:
    """计算攻防高度匹配度。"""

    return _token_set_score(attack_height, defense_height, HEIGHT_PARTIAL_MAP)


def compute_geo_match(pair_row: pd.Series) -> tuple[float, bool, float, float]:
    """计算几何覆盖匹配度。"""

    direction_score = _direction_score(str(pair_row["direction_tag"]), str(pair_row["coverage_direction"]))
    height_score = _height_score(str(pair_row["height_tag"]), str(pair_row["coverage_height"]))
    covered = direction_score > 0.0 and height_score > 0.0
    if direction_score >= 1.0 and height_score >= 1.0:
        return 1.0, covered, direction_score, height_score
    if direction_score >= 0.5 and height_score >= 0.5:
        return 0.5, covered, direction_score, height_score
    if covered:
        return 0.1, covered, direction_score, height_score
    return 0.0, covered, direction_score, height_score


def compute_force_match(attack_row: pd.Series, defense_row: pd.Series, robot: RobotParams) -> float:
    """计算力量匹配度。"""

    if defense_row["contact_mode"] == "none":
        return 1.0
    impact_score = float(attack_row["impact_score"])
    defense_capacity = float(defense_row["force_capacity"])
    if impact_score <= defense_capacity:
        return 1.0
    return max(0.2, 1.0 - (impact_score - defense_capacity) / (3.0 * robot.max_joint_torque_nm))


def compute_reaction_match(attack_row: pd.Series) -> float:
    """计算基于攻击速度的反应匹配度。"""

    return float(1.0 - 0.25 * (1.0 - float(attack_row["time_norm"])))


def build_pair_matrix(
    attack_catalog: pd.DataFrame,
    defense_features: pd.DataFrame,
    robot: RobotParams,
) -> pd.DataFrame:
    """构建 13×22 的攻防基础矩阵。"""

    attacks = attack_catalog.copy()
    attacks["join_key"] = 1
    defenses = defense_features.copy()
    defenses["join_key"] = 1
    pair_matrix = attacks.merge(defenses, on="join_key", how="inner").drop(columns="join_key")

    geometry = pair_matrix.apply(compute_geo_match, axis=1, result_type="expand")
    geometry.columns = ["p_geo", "geo_covered", "direction_match", "height_match"]
    pair_matrix = pd.concat([pair_matrix, geometry], axis=1)

    non_contact_mask = pair_matrix["contact_mode"].eq("none")
    force_mask = pair_matrix["impact_score"] <= pair_matrix["force_capacity"]
    pair_matrix["p_force"] = np.where(
        non_contact_mask | force_mask,
        1.0,
        np.maximum(
            0.2,
            1.0 - (pair_matrix["impact_score"] - pair_matrix["force_capacity"]) / (3.0 * robot.max_joint_torque_nm),
        ),
    )
    pair_matrix["p_react"] = 1.0 - 0.25 * (1.0 - pair_matrix["time_norm"])
    pair_matrix["p_block"] = pair_matrix["p_geo"] * pair_matrix["p_force"] * pair_matrix["p_react"]
    pair_matrix["defense_damage"] = pair_matrix["tau_norm"] * (1.0 - pair_matrix["p_block"]) * (1.0 - pair_matrix["absorb_rate"])
    pair_matrix["counter_window"] = (pair_matrix["opp_recover_time"] - pair_matrix["exec_time_def"]).clip(lower=0.0)
    pair_matrix["p_fall"] = ((1.0 - pair_matrix["p_block"]) * pair_matrix["tau_norm"] * pair_matrix["mobility_cost"]).clip(
        lower=0.0,
        upper=1.0,
    )
    if np.isclose(float(pair_matrix["counter_window"].max()), 0.0):
        pair_matrix["counter_window_norm"] = 0.0
    else:
        pair_matrix["counter_window_norm"] = pair_matrix["counter_window"] / float(pair_matrix["counter_window"].max())
    pair_matrix["counter_prob"] = pair_matrix["p_block"] * pair_matrix["counter_window_norm"]
    return pair_matrix
