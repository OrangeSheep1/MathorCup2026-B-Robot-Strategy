"""Q1 物理参数与动作特征提取模块。

本模块严格围绕题目给出的官方参数和建模方案实现：
1. 机器人基础参数来自官方数据与文中明确给出的推导口径。
2. 动作层数据不再使用主观 1-10 等级，而使用有物理含义的动作元数据。
3. 所有核心特征都通过明确公式计算，作为四种方法的共同输入。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd


ROBOT_PARAM_COLUMNS = ["param_name", "value", "unit", "source_type", "description"]
ACTION_COLUMNS = [
    "action_id",
    "action_name",
    "category",
    "eta",
    "joint_count",
    "theta_total_deg",
    "t_exec_s",
    "p_target",
    "description",
]


MODEL_ASSUMPTIONS = {
    "support_depth_m": 0.25266,
    "foot_spacing_ratio": 0.60,
    "upper_com_ratio": 0.45,
    "effective_speed_ratio": 0.60,
    "average_torque_ratio": 0.50,
    "average_speed_ratio": 0.40,
    "head_mass_ratio": 0.08,
    "torso_mass_ratio": 0.42,
    "single_arm_mass_ratio": 0.07,
    "single_leg_mass_ratio": 0.15,
    "forearm_mass_ratio": 0.45,
    "thigh_mass_ratio": 0.54,
    "torso_eccentricity_m": 0.15,
    "front_kick_support_factor": 1.10,
    "body_charge_gamma": 0.35,
    "counter_recovery_gamma": 0.15,
    "five_kick_decay_rate": 0.06,
}


ASSUMPTION_NOTES = {
    "support_depth_m": "来源于题解中的支撑多边形深度口径，不是官方直给参数。",
    "foot_spacing_ratio": "来源于题解中的双足站姿比例假设。",
    "upper_com_ratio": "来源于题解中的上体质心位置假设。",
    "effective_speed_ratio": "来源于题解中的有效角速度折减假设。",
    "average_torque_ratio": "来源于题解中的平均力矩假设。",
    "average_speed_ratio": "来源于题解中的平均角速度假设。",
    "head_mass_ratio": "来源于题解中的质量分配假设。",
    "torso_mass_ratio": "来源于题解中的质量分配假设。",
    "single_arm_mass_ratio": "来源于题解中的质量分配假设。",
    "single_leg_mass_ratio": "来源于题解中的质量分配假设。",
    "forearm_mass_ratio": "来源于题解中的前臂质量比例假设。",
    "thigh_mass_ratio": "来源于题解中的大腿质量比例假设。",
    "torso_eccentricity_m": "来源于题解中的躯干偏心距离设定。",
    "front_kick_support_factor": "来源于题解中的支撑腿补偿假设。",
    "body_charge_gamma": "来源于题解中的方向可控修正系数。",
    "counter_recovery_gamma": "来源于题解中的起身控制修正系数。",
    "five_kick_decay_rate": "来源于题解中的连续动作平衡衰减假设。",
}


@dataclass(frozen=True)
class RobotParams:
    """机器人基础参数与二级推导参数。"""

    total_height_m: float
    total_mass_kg: float
    leg_length_m: float
    arm_span_m: float
    max_joint_torque_nm: float
    max_motor_speed_rpm: float
    reducer_ratio: float
    joint_mass_kg: float
    joint_count: int
    body_width_m: float

    @property
    def arm_length_m(self) -> float:
        """单臂长度。"""

        return self.arm_span_m / 2.0

    @property
    def omega_out_rad_s(self) -> float:
        """关节输出轴角速度。"""

        return 2.0 * math.pi * self.max_motor_speed_rpm / (60.0 * self.reducer_ratio)

    @property
    def omega_eff_rad_s(self) -> float:
        """有效角速度。"""

        return MODEL_ASSUMPTIONS["effective_speed_ratio"] * self.omega_out_rad_s

    @property
    def upper_body_height_m(self) -> float:
        """上体高度。"""

        return self.total_height_m - self.leg_length_m

    @property
    def com_height_m(self) -> float:
        """整体质心高度。"""

        return self.leg_length_m + MODEL_ASSUMPTIONS["upper_com_ratio"] * self.upper_body_height_m

    @property
    def foot_spacing_m(self) -> float:
        """双足内侧间距。"""

        return MODEL_ASSUMPTIONS["foot_spacing_ratio"] * self.body_width_m

    @property
    def stable_margin_m(self) -> float:
        """保守稳定裕度。"""

        return min(self.foot_spacing_m / 2.0, MODEL_ASSUMPTIONS["support_depth_m"] / 2.0)

    @property
    def average_torque_nm(self) -> float:
        """动作平均力矩。"""

        return MODEL_ASSUMPTIONS["average_torque_ratio"] * self.max_joint_torque_nm

    @property
    def average_angular_speed_rad_s(self) -> float:
        """动作平均角速度。"""

        return MODEL_ASSUMPTIONS["average_speed_ratio"] * self.omega_out_rad_s

    @property
    def single_arm_mass_kg(self) -> float:
        """单臂质量。"""

        return MODEL_ASSUMPTIONS["single_arm_mass_ratio"] * self.total_mass_kg

    @property
    def single_leg_mass_kg(self) -> float:
        """单腿质量。"""

        return MODEL_ASSUMPTIONS["single_leg_mass_ratio"] * self.total_mass_kg

    @property
    def torso_mass_kg(self) -> float:
        """躯干与腰部质量。"""

        return MODEL_ASSUMPTIONS["torso_mass_ratio"] * self.total_mass_kg

    @property
    def forearm_mass_kg(self) -> float:
        """前臂与手的合并质量。"""

        return MODEL_ASSUMPTIONS["forearm_mass_ratio"] * self.single_arm_mass_kg

    @property
    def thigh_mass_kg(self) -> float:
        """大腿质量。"""

        return MODEL_ASSUMPTIONS["thigh_mass_ratio"] * self.single_leg_mass_kg

    @property
    def forearm_lever_m(self) -> float:
        """前臂有效力臂。"""

        return self.arm_length_m * 0.55

    @property
    def full_arm_lever_m(self) -> float:
        """全臂力臂。"""

        return self.arm_length_m

    @property
    def full_leg_lever_m(self) -> float:
        """全腿力臂。"""

        return self.leg_length_m

    @property
    def thigh_lever_m(self) -> float:
        """大腿力臂。"""

        return self.leg_length_m * 0.54

    @property
    def sweep_lever_m(self) -> float:
        """低扫腿有效力臂。"""

        return self.leg_length_m * 0.75

    @property
    def mechanical_power_factor(self) -> float:
        """能量模型中的固定乘子。"""

        return self.average_torque_nm * self.average_angular_speed_rad_s


@dataclass(frozen=True)
class AttackAction:
    """单个攻击动作的物理元数据。"""

    action_id: str
    action_name: str
    category: str
    eta: float
    joint_count: int
    theta_total_deg: float | None
    t_exec_s: float
    p_target: float
    description: str


def load_robot_params(file_path: str | Path) -> RobotParams:
    """读取机器人参数表。"""

    data = pd.read_csv(file_path)
    missing = [column for column in ROBOT_PARAM_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"机器人参数表缺少字段: {missing}")

    value_map = data.set_index("param_name")["value"].astype(float).to_dict()
    return RobotParams(
        total_height_m=value_map["total_height_m"],
        total_mass_kg=value_map["total_mass_kg"],
        leg_length_m=value_map["leg_length_m"],
        arm_span_m=value_map["arm_span_m"],
        max_joint_torque_nm=value_map["max_joint_torque_nm"],
        max_motor_speed_rpm=value_map["max_motor_speed_rpm"],
        reducer_ratio=value_map["reducer_ratio"],
        joint_mass_kg=value_map["joint_mass_kg"],
        joint_count=int(value_map["joint_count"]),
        body_width_m=value_map["body_width_m"],
    )


def load_attack_actions(file_path: str | Path) -> pd.DataFrame:
    """读取动作元数据表。"""

    data = pd.read_csv(file_path)
    missing = [column for column in ACTION_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"动作元数据表缺少字段: {missing}")
    return data[ACTION_COLUMNS].copy()


def build_action(row: pd.Series) -> AttackAction:
    """将一行动作数据转换为结构化对象。"""

    theta_value = row["theta_total_deg"]
    theta_total_deg = None if pd.isna(theta_value) else float(theta_value)
    return AttackAction(
        action_id=str(row["action_id"]),
        action_name=str(row["action_name"]),
        category=str(row["category"]),
        eta=float(row["eta"]),
        joint_count=int(row["joint_count"]),
        theta_total_deg=theta_total_deg,
        t_exec_s=float(row["t_exec_s"]),
        p_target=float(row["p_target"]),
        description=str(row["description"]),
    )


def compute_output_angular_speed(robot: RobotParams) -> float:
    """计算输出轴角速度。"""

    return robot.omega_out_rad_s


def compute_effective_angular_speed(robot: RobotParams) -> float:
    """计算有效角速度。"""

    return robot.omega_eff_rad_s


def compute_com_height(robot: RobotParams) -> float:
    """计算机器人质心高度。"""

    return robot.com_height_m


def compute_stable_margin(robot: RobotParams) -> float:
    """计算保守稳定裕度。"""

    return robot.stable_margin_m


def compute_impact_score(action: AttackAction, robot: RobotParams) -> float:
    """按方法一的刚体力学结论计算冲击力矩。"""

    return 3.0 * robot.max_joint_torque_nm * action.eta


def _sine_half_angle(length_m: float, angle_deg: float) -> float:
    """计算刚性杆质心位移公式中的位移项。"""

    return length_m * math.sin(math.radians(angle_deg / 2.0))


def _pseudo_action(action_id: str) -> AttackAction:
    """构造仅用于复用公式的占位动作。"""

    return AttackAction(action_id, "", "", 0.0, 0, None, 0.0, 0.0, "")


def compute_balance_cost(action: AttackAction, robot: RobotParams) -> float:
    """计算各动作的质心偏移量。"""

    total_mass = robot.total_mass_kg

    if action.action_id == "A01":
        displacement = _sine_half_angle(robot.forearm_lever_m, 50.0)
        return robot.forearm_mass_kg * displacement / total_mass

    if action.action_id == "A02":
        displacement = _sine_half_angle(robot.full_arm_lever_m, 70.0)
        return robot.single_arm_mass_kg * displacement / total_mass

    if action.action_id == "A03":
        return 1.2 * compute_balance_cost(_pseudo_action("A01"), robot)

    if action.action_id == "A04":
        torso_displacement = 2.0 * MODEL_ASSUMPTIONS["torso_eccentricity_m"] * math.sin(math.radians(60.0 / 2.0))
        arm_displacement = _sine_half_angle(robot.full_arm_lever_m, 90.0)
        return (
            robot.torso_mass_kg * torso_displacement
            + robot.single_arm_mass_kg * arm_displacement
        ) / total_mass

    if action.action_id == "A05":
        leg_displacement = _sine_half_angle(robot.full_leg_lever_m, 90.0)
        base_shift = robot.single_leg_mass_kg * leg_displacement / total_mass
        return base_shift * MODEL_ASSUMPTIONS["front_kick_support_factor"]

    if action.action_id == "A06":
        leg_displacement = _sine_half_angle(robot.full_leg_lever_m, 90.0)
        torso_displacement = robot.upper_body_height_m * math.sin(math.radians(30.0)) * 0.50
        return (
            robot.single_leg_mass_kg * leg_displacement
            + robot.torso_mass_kg * torso_displacement
        ) / total_mass

    if action.action_id == "A07":
        torso_displacement = 2.0 * MODEL_ASSUMPTIONS["torso_eccentricity_m"] * math.sin(math.radians(200.0 / 2.0))
        leg_displacement = _sine_half_angle(robot.full_leg_lever_m, 180.0)
        return (
            robot.torso_mass_kg * torso_displacement
            + robot.single_leg_mass_kg * leg_displacement
        ) / total_mass

    if action.action_id == "A08":
        leg_displacement = _sine_half_angle(robot.sweep_lever_m, 60.0)
        return robot.single_leg_mass_kg * leg_displacement / total_mass

    if action.action_id == "A09":
        thigh_displacement = _sine_half_angle(robot.thigh_lever_m, 70.0)
        return robot.thigh_mass_kg * thigh_displacement / total_mass

    if action.action_id == "A10":
        straight_punch_shift = compute_balance_cost(_pseudo_action("A01"), robot)
        front_kick_shift = compute_balance_cost(_pseudo_action("A05"), robot)
        return math.sqrt(straight_punch_shift**2 + front_kick_shift**2)

    if action.action_id == "A11":
        front_kick_shift = compute_balance_cost(_pseudo_action("A05"), robot)
        return front_kick_shift * (1.0 + 4.0 * MODEL_ASSUMPTIONS["five_kick_decay_rate"])

    if action.action_id == "A12":
        forward_shift = robot.com_height_m * math.sin(math.radians(15.0))
        return forward_shift * MODEL_ASSUMPTIONS["body_charge_gamma"]

    if action.action_id == "A13":
        recovery_shift = robot.com_height_m * math.sin(math.radians(50.0))
        return recovery_shift * MODEL_ASSUMPTIONS["counter_recovery_gamma"]

    raise ValueError(f"未定义的动作编号: {action.action_id}")


def compute_execution_time(action: AttackAction) -> float:
    """读取动作执行时间。"""

    return action.t_exec_s


def compute_reach_probability(action: AttackAction, t_exec_max: float) -> float:
    """计算几何可达概率。"""

    return 1.0 - 0.30 * action.t_exec_s / t_exec_max


def compute_score_prob(action: AttackAction, t_exec_max: float) -> float:
    """计算命中概率。"""

    return compute_reach_probability(action, t_exec_max) * action.p_target


def compute_energy_cost(action: AttackAction, robot: RobotParams) -> float:
    """计算动作能量消耗。"""

    return action.joint_count * robot.mechanical_power_factor * action.t_exec_s


def extract_action_features(action: AttackAction, robot: RobotParams, t_exec_max: float) -> dict[str, float | str | int]:
    """提取单个动作的结构化特征。"""

    impact_score = compute_impact_score(action, robot)
    balance_cost = compute_balance_cost(action, robot)
    score_prob = compute_score_prob(action, t_exec_max)
    energy_cost = compute_energy_cost(action, robot)
    exec_time = compute_execution_time(action)
    reach_prob = compute_reach_probability(action, t_exec_max)
    stability_ratio = balance_cost / robot.stable_margin_m

    return {
        "action_id": action.action_id,
        "action_name": action.action_name,
        "category": action.category,
        "description": action.description,
        "eta": round(action.eta, 4),
        "joint_count": action.joint_count,
        "theta_total_deg": action.theta_total_deg,
        "impact_score": round(impact_score, 4),
        "balance_cost": round(balance_cost, 6),
        "score_prob": round(score_prob, 6),
        "energy_cost": round(energy_cost, 4),
        "exec_time": round(exec_time, 4),
        "p_target": round(action.p_target, 4),
        "p_reach": round(reach_prob, 6),
        "stable_margin": round(robot.stable_margin_m, 6),
        "stability_ratio": round(stability_ratio, 6),
        "omega_out": round(robot.omega_out_rad_s, 6),
        "omega_eff": round(robot.omega_eff_rad_s, 6),
        "com_height": round(robot.com_height_m, 6),
    }


def build_feature_table(raw_actions: pd.DataFrame, robot: RobotParams) -> pd.DataFrame:
    """批量提取全部动作的基础物理特征。"""

    t_exec_max = raw_actions["t_exec_s"].max()
    records = []
    for _, row in raw_actions.iterrows():
        action = build_action(row)
        records.append(extract_action_features(action, robot, t_exec_max))
    return pd.DataFrame(records)
