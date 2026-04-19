"""Q1 模板驱动的动作特征提取模块。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROBOT_PARAM_COLUMNS = ["param_name", "value", "unit", "source_type", "description"]
SEGMENT_COLUMNS = [
    "segment_id",
    "segment_name_cn",
    "segment_name_en",
    "side",
    "parent_id",
    "mass_ratio",
    "mass_kg",
    "length_basis",
    "length_ratio",
    "length_m",
    "com_local_ratio",
    "inertia_coeff",
    "attack_capable",
    "stability_role",
    "remarks",
]
SUPPORT_MODE_COLUMNS = [
    "support_mode",
    "support_margin_ratio",
    "zmp_bias_coeff",
    "contact_type",
    "description",
]
ATTACK_ACTION_COLUMNS = [
    "action_id",
    "action_name",
    "category",
    "coordination_efficiency",
    "active_joint_count",
    "theta_total_deg",
    "exec_time_s",
    "p_target_geo",
    "description",
    "coordination_source",
    "active_joint_count_source",
    "theta_source",
    "exec_time_source",
    "p_target_source",
]
ACTION_TEMPLATE_COLUMNS = [
    "action_id",
    "laterality",
    "conditional_flag",
    "trigger_state",
    "support_mode",
    "phase_count",
    "main_plane",
    "range_tag",
    "rotation_complexity",
    "support_switch_count",
    "translation_distance_m",
    "active_segments_json",
    "support_segments_json",
    "strike_segments_json",
    "impact_weight_json",
    "stability_weight_json",
    "theta_share_json",
    "notes",
]
PHASE_TEMPLATE_COLUMNS = [
    "action_id",
    "phase_no",
    "phase_name",
    "phase_time_ratio",
    "phase_theta_deg",
    "support_mode_phase",
    "translation_share",
    "active_segments_json",
    "strike_segments_json",
    "impact_weight_json",
    "stability_weight_json",
    "theta_share_json",
    "phase_decay",
    "notes",
]
JSON_LIST_COLUMNS = ["active_segments_json", "support_segments_json", "strike_segments_json"]
JSON_DICT_COLUMNS = ["impact_weight_json", "stability_weight_json", "theta_share_json"]


MODEL_ASSUMPTIONS = {
    "support_depth_m": 0.25266,
    "foot_spacing_ratio": 0.60,
    "upper_com_ratio": 0.45,
    "effective_speed_ratio": 0.60,
    "average_torque_ratio": 0.50,
    "average_speed_ratio": 0.40,
    "torso_eccentricity_m": 0.15,
    "body_charge_gamma": 0.35,
    "counter_recovery_gamma": 0.15,
    "dynamic_gravity_m_s2": 9.81,
    "exposure_time_ref_s": 0.60,
    "exposure_phase_alpha": 0.15,
    "exposure_switch_beta": 0.10,
    "rotation_exposure_beta": 0.08,
    "reference_velocity_m_s": 6.00,
    "velocity_sigmoid_scale": 1.25,
    "zmp_logistic_slope": 8.00,
    "support_amplify_factor": 0.75,
    "translation_effect_default": 0.20,
    "phase_work_decay_default": 0.96,
    "recovery_height_m": 0.70,
    "recovery_efficiency": 0.35,
}


LEGACY_COMPATIBILITY_ASSUMPTIONS = {
    "torso_mass_ratio": 0.42,
    "single_arm_mass_ratio": 0.07,
    "single_leg_mass_ratio": 0.15,
    "forearm_mass_ratio": 0.45,
    "thigh_mass_ratio": 0.54,
}


ASSUMPTION_NOTES = {
    "support_depth_m": "前后支撑深度采用题解中的保守口径。",
    "foot_spacing_ratio": "双脚内侧间距采用机体宽度的 60%。",
    "upper_com_ratio": "整体质心高度采用上体高度的 45%。",
    "effective_speed_ratio": "动作有效角速度取理论输出上限的 60%。",
    "average_torque_ratio": "平均力矩取最大关节力矩的 50%。",
    "average_speed_ratio": "平均角速度取输出轴理论上限的 40%。",
    "torso_eccentricity_m": "躯干转体偏心半径采用 0.15m 的保守估计。",
    "body_charge_gamma": "冲撞平动对质心偏移的折减系数。",
    "counter_recovery_gamma": "倒地恢复平动对质心偏移的折减系数。",
    "five_kick_decay_rate": "五连踢的阶段衰减率。",
    "zmp_logistic_slope": "ZMP 越界比率映射为跌倒风险时的 Logistic 斜率。",
    "recovery_efficiency": "倒地恢复功采用重力势能近似后乘恢复效率。",
}


LEGACY_COMPATIBILITY_NOTES = {
    "torso_mass_ratio": "兼容 Q2 阶段中，用于树立防护和冲撞等旧问题质量口径。",
    "single_arm_mass_ratio": "兼容 Q2 防护有效支撑质量和上肢动作的遗留口径。",
    "single_leg_mass_ratio": "兼容 Q2 下肢攻防质量和步法动作的遗留口径。",
    "forearm_mass_ratio": "兼容 Q2 前臂受力质量估计的遗留参数。",
    "thigh_mass_ratio": "兼容 Q2 大腿受力质量估计的遗留参数。",
}


@dataclass(frozen=True)
class RobotParams:
    """机器人基础参数与统一推导参数。"""

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
        return self.arm_span_m / 2.0

    @property
    def upper_arm_length_m(self) -> float:
        return self.arm_length_m * 0.47

    @property
    def forearm_hand_length_m(self) -> float:
        return self.arm_length_m * 0.53

    @property
    def thigh_length_m(self) -> float:
        return self.leg_length_m * 0.54

    @property
    def shank_foot_length_m(self) -> float:
        return self.leg_length_m * 0.46

    @property
    def omega_out_rad_s(self) -> float:
        return 2.0 * math.pi * self.max_motor_speed_rpm / (60.0 * self.reducer_ratio)

    @property
    def omega_eff_rad_s(self) -> float:
        return MODEL_ASSUMPTIONS["effective_speed_ratio"] * self.omega_out_rad_s

    @property
    def upper_body_height_m(self) -> float:
        return self.total_height_m - self.leg_length_m

    @property
    def com_height_m(self) -> float:
        return self.leg_length_m + MODEL_ASSUMPTIONS["upper_com_ratio"] * self.upper_body_height_m

    @property
    def foot_spacing_m(self) -> float:
        return MODEL_ASSUMPTIONS["foot_spacing_ratio"] * self.body_width_m

    @property
    def stable_margin_m(self) -> float:
        return min(self.foot_spacing_m / 2.0, MODEL_ASSUMPTIONS["support_depth_m"] / 2.0)

    @property
    def average_torque_nm(self) -> float:
        return MODEL_ASSUMPTIONS["average_torque_ratio"] * self.max_joint_torque_nm

    @property
    def average_angular_speed_rad_s(self) -> float:
        return MODEL_ASSUMPTIONS["average_speed_ratio"] * self.omega_out_rad_s

    @property
    def single_arm_mass_kg(self) -> float:
        # 鍏煎 Q2 鏃ч€昏緫锛孮1 涓讳綋璁＄畻宸叉敼鐢ㄨ妭娈垫暟鎹〃銆?
        return LEGACY_COMPATIBILITY_ASSUMPTIONS["single_arm_mass_ratio"] * self.total_mass_kg

    @property
    def single_leg_mass_kg(self) -> float:
        return LEGACY_COMPATIBILITY_ASSUMPTIONS["single_leg_mass_ratio"] * self.total_mass_kg

    @property
    def torso_mass_kg(self) -> float:
        return LEGACY_COMPATIBILITY_ASSUMPTIONS["torso_mass_ratio"] * self.total_mass_kg

    @property
    def forearm_mass_kg(self) -> float:
        return LEGACY_COMPATIBILITY_ASSUMPTIONS["forearm_mass_ratio"] * self.single_arm_mass_kg

    @property
    def thigh_mass_kg(self) -> float:
        return LEGACY_COMPATIBILITY_ASSUMPTIONS["thigh_mass_ratio"] * self.single_leg_mass_kg

    @property
    def forearm_lever_m(self) -> float:
        return self.arm_length_m * 0.55

    @property
    def full_arm_lever_m(self) -> float:
        return self.arm_length_m

    @property
    def full_leg_lever_m(self) -> float:
        return self.leg_length_m

    @property
    def thigh_lever_m(self) -> float:
        return self.leg_length_m * 0.54

    @property
    def sweep_lever_m(self) -> float:
        return self.leg_length_m * 0.75


def _require_columns(data: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{label}缺少字段: {missing}")


def _parse_json_value(raw_value: Any, expect_dict: bool) -> Any:
    if pd.isna(raw_value):
        return {} if expect_dict else []
    parsed = json.loads(str(raw_value))
    if expect_dict and not isinstance(parsed, dict):
        raise ValueError(f"期望得到字典，实际为 {type(parsed)}")
    if not expect_dict and not isinstance(parsed, list):
        raise ValueError(f"期望得到列表，实际为 {type(parsed)}")
    return parsed


def _check_weight_sum(weights: dict[str, float], label: str) -> None:
    weight_sum = float(sum(float(value) for value in weights.values()))
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{label} 权重和不为 1，当前为 {weight_sum:.6f}")


def load_robot_params(file_path: str | Path) -> RobotParams:
    data = pd.read_csv(file_path)
    _require_columns(data, ROBOT_PARAM_COLUMNS, "机器人参数表")
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


def load_segment_params(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    _require_columns(data, SEGMENT_COLUMNS, "节段参数表")
    data["attack_capable"] = data["attack_capable"].astype(int)
    data["stability_role"] = data["stability_role"].astype(int)
    ratio_sum = float(data["mass_ratio"].sum())
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"节段质量比例和不为 1，当前为 {ratio_sum:.6f}")

    symmetry_pairs = [("S3", "S4"), ("S5", "S6"), ("S7", "S8"), ("S9", "S10")]
    index_map = data.set_index("segment_id")
    for left_id, right_id in symmetry_pairs:
        left_row = index_map.loc[left_id]
        right_row = index_map.loc[right_id]
        if not math.isclose(float(left_row["mass_kg"]), float(right_row["mass_kg"]), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{left_id} 与 {right_id} 质量不对称")
        if not math.isclose(float(left_row["length_m"]), float(right_row["length_m"]), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{left_id} 与 {right_id} 长度不对称")

    expected_parent = {
        "S1": "S2",
        "S2": "",
        "S3": "S2",
        "S4": "S2",
        "S5": "S3",
        "S6": "S4",
        "S7": "S2",
        "S8": "S2",
        "S9": "S7",
        "S10": "S8",
    }
    for segment_id, parent_id in expected_parent.items():
        current_parent = "" if pd.isna(index_map.loc[segment_id, "parent_id"]) else str(index_map.loc[segment_id, "parent_id"])
        if current_parent != parent_id:
            expect_label = parent_id if parent_id else "ROOT"
            current_label = current_parent if current_parent else "ROOT"
            raise ValueError(f"{segment_id} 的父节点应为 {expect_label}，当前为 {current_label}")
    return data


def load_support_mode_config(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    _require_columns(data, SUPPORT_MODE_COLUMNS, "支撑模式配置表")
    if ((data["support_margin_ratio"] <= 0.0) | (data["support_margin_ratio"] > 1.0)).any():
        raise ValueError("support_margin_ratio 必须位于 (0, 1] 区间")
    if (data["zmp_bias_coeff"] < 0.0).any():
        raise ValueError("zmp_bias_coeff 必须非负")
    return data


def load_attack_actions(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    _require_columns(data, ATTACK_ACTION_COLUMNS, "动作数值源表")
    if data["action_id"].nunique() != 13:
        raise ValueError("动作数值源表应包含 13 个唯一动作")
    return data.sort_values("action_id").reset_index(drop=True)


def load_action_templates(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    _require_columns(data, ACTION_TEMPLATE_COLUMNS, "动作模板表")
    for column in JSON_LIST_COLUMNS:
        data[column] = data[column].apply(lambda value: _parse_json_value(value, expect_dict=False))
    for column in JSON_DICT_COLUMNS:
        data[column] = data[column].apply(lambda value: _parse_json_value(value, expect_dict=True))

    for _, row in data.iterrows():
        _check_weight_sum(row["impact_weight_json"], f"{row['action_id']} impact_weight_json")
        _check_weight_sum(row["stability_weight_json"], f"{row['action_id']} stability_weight_json")
        _check_weight_sum(row["theta_share_json"], f"{row['action_id']} theta_share_json")
        if not set(row["strike_segments_json"]).issubset(set(row["active_segments_json"])):
            raise ValueError(f"{row['action_id']} 的 strike_segments_json 不是 active_segments_json 的子集")
        if int(row["conditional_flag"]) == 1 and str(row["trigger_state"]).strip().lower() == "standing":
            raise ValueError(f"{row['action_id']} 是条件动作，但 trigger_state 不应为 standing")
    if data["action_id"].nunique() != 13:
        raise ValueError("动作模板表应包含 13 个唯一动作")
    return data.sort_values("action_id").reset_index(drop=True)


def merge_action_definition_tables(actions: pd.DataFrame, templates: pd.DataFrame) -> pd.DataFrame:
    merged = templates.merge(actions, on="action_id", how="inner", validate="one_to_one")
    if merged["action_id"].nunique() != 13:
        raise ValueError("合并后的动作主表应包含 13 个动作")
    return merged.sort_values("action_id").reset_index(drop=True)


def load_action_phase_templates(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    _require_columns(data, PHASE_TEMPLATE_COLUMNS, "动作相位模板表")
    for column in ["active_segments_json", "strike_segments_json"]:
        data[column] = data[column].apply(lambda value: _parse_json_value(value, expect_dict=False))
    for column in ["impact_weight_json", "stability_weight_json", "theta_share_json"]:
        data[column] = data[column].apply(lambda value: _parse_json_value(value, expect_dict=True))

    for _, row in data.iterrows():
        _check_weight_sum(row["impact_weight_json"], f"{row['action_id']}-{row['phase_no']} impact_weight_json")
        _check_weight_sum(row["stability_weight_json"], f"{row['action_id']}-{row['phase_no']} stability_weight_json")
        _check_weight_sum(row["theta_share_json"], f"{row['action_id']}-{row['phase_no']} theta_share_json")
        if not set(row["strike_segments_json"]).issubset(set(row["active_segments_json"])):
            raise ValueError(f"{row['action_id']}-{row['phase_no']} 的 strike_segments_json 不是 active_segments_json 的子集")
        if float(row["phase_theta_deg"]) < 0.0:
            raise ValueError(f"{row['action_id']}-{row['phase_no']} 的 phase_theta_deg 不能为负")

    for action_id, group in data.groupby("action_id"):
        time_ratio_sum = float(group["phase_time_ratio"].sum())
        if not math.isclose(time_ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{action_id} 的 phase_time_ratio 之和不为 1，当前为 {time_ratio_sum:.6f}")
        translation_sum = float(group["translation_share"].sum())
        if translation_sum > 0.0 and not math.isclose(translation_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{action_id} 的 translation_share 之和不为 1，当前为 {translation_sum:.6f}")
    return data.sort_values(["action_id", "phase_no"]).reset_index(drop=True)


def _json_segment_keys(row: pd.Series) -> set[str]:
    keys: set[str] = set()
    keys.update(str(segment_id) for segment_id in row["active_segments_json"])
    keys.update(str(segment_id) for segment_id in row["support_segments_json"])
    keys.update(str(segment_id) for segment_id in row["strike_segments_json"])
    keys.update(str(segment_id) for segment_id in row["impact_weight_json"].keys())
    keys.update(str(segment_id) for segment_id in row["stability_weight_json"].keys())
    keys.update(str(segment_id) for segment_id in row["theta_share_json"].keys())
    return keys


def _phase_segment_keys(row: pd.Series) -> set[str]:
    keys: set[str] = set()
    keys.update(str(segment_id) for segment_id in row["active_segments_json"])
    keys.update(str(segment_id) for segment_id in row["strike_segments_json"])
    keys.update(str(segment_id) for segment_id in row["impact_weight_json"].keys())
    keys.update(str(segment_id) for segment_id in row["stability_weight_json"].keys())
    keys.update(str(segment_id) for segment_id in row["theta_share_json"].keys())
    return keys


def validate_q1_configuration(
    actions: pd.DataFrame,
    phase_templates: pd.DataFrame,
    segments: pd.DataFrame,
    support_modes: pd.DataFrame,
) -> None:
    """鍦ㄤ富璁＄畻鍓嶅 Q1 閰嶇疆鍋氭渶鍚庝竴杞竴鑷存€ф牎楠屻€?"""

    segment_ids = set(segments["segment_id"].astype(str))
    support_mode_ids = set(support_modes["support_mode"].astype(str))
    phase_count_map = phase_templates.groupby("action_id").size().to_dict()
    phase_theta_map = phase_templates.groupby("action_id")["phase_theta_deg"].sum().to_dict()
    phase_translation_map = phase_templates.groupby("action_id")["translation_share"].sum().to_dict()

    missing_phase_support_modes = set(phase_templates["support_mode_phase"].astype(str)) - support_mode_ids
    if missing_phase_support_modes:
        raise ValueError(f"相位表存在未定义的 support_mode_phase: {sorted(missing_phase_support_modes)}")

    for _, row in actions.iterrows():
        action_id = str(row["action_id"])
        support_mode = str(row["support_mode"])
        if support_mode not in support_mode_ids:
            raise ValueError(f"{action_id} 在动作主表中引用了未定义的 support_mode: {support_mode}")

        action_segments = _json_segment_keys(row)
        missing_segments = sorted(action_segments - segment_ids)
        if missing_segments:
            raise ValueError(f"{action_id} 动作模板引用了未定义的节段 ID: {missing_segments}")

        active_segments = set(str(segment_id) for segment_id in row["active_segments_json"])
        for weight_label in ("impact_weight_json", "stability_weight_json", "theta_share_json"):
            invalid_keys = sorted(set(str(key) for key in row[weight_label].keys()) - active_segments)
            if invalid_keys:
                raise ValueError(f"{action_id} 的 {weight_label} 字典键不是 active_segments_json 的子集: {invalid_keys}")

        phase_count = int(row["phase_count"])
        has_phase_rows = action_id in phase_count_map
        if has_phase_rows:
            actual_phase_count = int(phase_count_map[action_id])
            if actual_phase_count != phase_count:
                raise ValueError(f"{action_id} 的 phase_count={phase_count}，但相位表中实际有 {actual_phase_count} 条记录")

            if not pd.isna(row["theta_total_deg"]):
                phase_theta_total = float(phase_theta_map[action_id])
                if not math.isclose(float(row["theta_total_deg"]), phase_theta_total, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(
                        f"{action_id} 的 theta_total_deg={float(row['theta_total_deg']):.6f} 与相位表求和 {phase_theta_total:.6f} 不一致"
                    )

            translation_distance = float(row["translation_distance_m"])
            translation_share_sum = float(phase_translation_map[action_id])
            if translation_distance > 0.0:
                if not math.isclose(translation_share_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"{action_id} 定义了非零平移距离，但相位 translation_share 之和为 {translation_share_sum:.6f}")
            else:
                if not math.isclose(translation_share_sum, 0.0, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"{action_id} 无平移距离，但相位 translation_share 之和为 {translation_share_sum:.6f}")
        elif phase_count != 1:
            raise ValueError(f"{action_id} 的 phase_count={phase_count}，但动作相位表缺失。非单阶段动作必须显式给出相位定义。")

    for _, row in phase_templates.iterrows():
        action_id = str(row["action_id"])
        phase_keys = _phase_segment_keys(row)
        missing_segments = sorted(phase_keys - segment_ids)
        if missing_segments:
            raise ValueError(f"{action_id}-{int(row['phase_no'])} 引用了未定义的节段 ID: {missing_segments}")

        active_segments = set(str(segment_id) for segment_id in row["active_segments_json"])
        for weight_label in ("impact_weight_json", "stability_weight_json", "theta_share_json"):
            invalid_keys = sorted(set(str(key) for key in row[weight_label].keys()) - active_segments)
            if invalid_keys:
                raise ValueError(
                    f"{action_id}-{int(row['phase_no'])} 的 {weight_label} 字典键不是 active_segments_json 的子集: {invalid_keys}"
                )


def _segment_tip_radius(segment_id: str, robot: RobotParams) -> float:
    if segment_id in {"S3", "S4"}:
        return robot.upper_arm_length_m
    if segment_id in {"S5", "S6"}:
        return robot.arm_length_m
    if segment_id in {"S7", "S8"}:
        return robot.thigh_length_m
    if segment_id in {"S9", "S10"}:
        return robot.leg_length_m
    if segment_id == "S2":
        return MODEL_ASSUMPTIONS["torso_eccentricity_m"]
    if segment_id == "S1":
        return robot.upper_body_height_m * 0.34
    return 0.0


def _segment_global_com_radius(segment_row: pd.Series, robot: RobotParams) -> float:
    segment_id = str(segment_row["segment_id"])
    if segment_id in {"S3", "S4"}:
        return robot.upper_arm_length_m * float(segment_row["com_local_ratio"])
    if segment_id in {"S5", "S6"}:
        return robot.upper_arm_length_m + robot.forearm_hand_length_m * float(segment_row["com_local_ratio"])
    if segment_id in {"S7", "S8"}:
        return robot.thigh_length_m * float(segment_row["com_local_ratio"])
    if segment_id in {"S9", "S10"}:
        return robot.thigh_length_m + robot.shank_foot_length_m * float(segment_row["com_local_ratio"])
    if segment_id == "S2":
        return MODEL_ASSUMPTIONS["torso_eccentricity_m"]
    if segment_id == "S1":
        return robot.upper_body_height_m * 0.38
    return 0.0


def _plane_factor(main_plane: str) -> float:
    return {
        "sagittal": 1.00,
        "frontal": 1.05,
        "transverse": 1.10,
        "multi_plane": 1.16,
    }.get(str(main_plane), 1.00)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _parse_phase_rows(action_row: pd.Series, phase_table: pd.DataFrame) -> list[dict[str, Any]]:
    action_id = str(action_row["action_id"])
    subset = phase_table.loc[phase_table["action_id"] == action_id].sort_values("phase_no")
    if subset.empty:
        return [
            {
                "phase_no": 1,
                "phase_name": "single_phase",
                "phase_time_ratio": 1.0,
                "phase_theta_deg": float(action_row["theta_total_deg"]),
                "support_mode_phase": str(action_row["support_mode"]),
                "translation_share": 1.0 if float(action_row["translation_distance_m"]) > 0.0 else 0.0,
                "active_segments_json": list(action_row["active_segments_json"]),
                "strike_segments_json": list(action_row["strike_segments_json"]),
                "impact_weight_json": dict(action_row["impact_weight_json"]),
                "stability_weight_json": dict(action_row["stability_weight_json"]),
                "theta_share_json": dict(action_row["theta_share_json"]),
                "phase_decay": 1.0,
            }
        ]

    phases: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        phases.append(
            {
                "phase_no": int(row["phase_no"]),
                "phase_name": str(row["phase_name"]),
                "phase_time_ratio": float(row["phase_time_ratio"]),
                "phase_theta_deg": float(row["phase_theta_deg"]),
                "support_mode_phase": str(row["support_mode_phase"]),
                "translation_share": float(row["translation_share"]),
                "active_segments_json": list(row["active_segments_json"]),
                "strike_segments_json": list(row["strike_segments_json"]),
                "impact_weight_json": dict(row["impact_weight_json"]),
                "stability_weight_json": dict(row["stability_weight_json"]),
                "theta_share_json": dict(row["theta_share_json"]),
                "phase_decay": float(row["phase_decay"]),
            }
        )
    return phases


def _resolve_theta_total_deg(action_row: pd.Series, phases: list[dict[str, Any]]) -> float:
    if not pd.isna(action_row["theta_total_deg"]):
        return float(action_row["theta_total_deg"])
    return float(sum(float(phase["phase_theta_deg"]) for phase in phases))


def _support_translation_factor(support_mode_phase: str) -> float:
    if support_mode_phase == "dynamic_double_support":
        return MODEL_ASSUMPTIONS["body_charge_gamma"]
    if support_mode_phase == "recovery_transition":
        return MODEL_ASSUMPTIONS["counter_recovery_gamma"]
    return MODEL_ASSUMPTIONS["translation_effect_default"]


def _weighted_segment_mass(segment_ids: list[str], weight_map: dict[str, float], segment_map: dict[str, pd.Series]) -> float:
    total = 0.0
    for segment_id in segment_ids:
        weight = float(weight_map.get(segment_id, 0.0))
        if weight > 0.0 and segment_id in segment_map:
            total += float(segment_map[segment_id]["mass_kg"]) * weight
    return total


def _effective_reach(strike_segments: list[str], impact_weights: dict[str, float], robot: RobotParams) -> float:
    if not strike_segments:
        strike_segments = list(impact_weights.keys())
    reach = 0.0
    weight_sum = 0.0
    for segment_id in strike_segments:
        weight = float(impact_weights.get(segment_id, 0.0))
        if weight > 0.0:
            reach += _segment_tip_radius(segment_id, robot) * weight
            weight_sum += weight
    return 0.0 if weight_sum <= 0.0 else reach / weight_sum


def _phase_rotational_speed(coordination_efficiency: float, phase_theta_deg: float, phase_time_s: float, robot: RobotParams) -> tuple[float, float]:
    theta_rad = math.radians(float(phase_theta_deg))
    if phase_time_s <= 1e-9:
        return theta_rad, 0.0
    omega_eff = min(coordination_efficiency * theta_rad / phase_time_s, robot.omega_eff_rad_s)
    return theta_rad, omega_eff


def _phase_com_shift(
    phase: dict[str, Any],
    action_row: pd.Series,
    robot: RobotParams,
    segment_map: dict[str, pd.Series],
    support_map: dict[str, pd.Series],
    phase_time_s: float,
) -> tuple[float, float, float, float]:
    theta_rad = math.radians(float(phase["phase_theta_deg"]))
    rotational_shift = 0.0
    for segment_id, weight in phase["stability_weight_json"].items():
        if segment_id not in segment_map:
            continue
        segment_row = segment_map[segment_id]
        radius = _segment_global_com_radius(segment_row, robot)
        theta_share = float(phase["theta_share_json"].get(segment_id, 0.0))
        displacement = radius * math.sin(theta_rad * theta_share / 2.0)
        rotational_shift += float(segment_row["mass_kg"]) * displacement * float(weight)
    rotational_shift /= robot.total_mass_kg

    translation_shift = 0.0
    translation_distance = float(action_row["translation_distance_m"])
    if translation_distance > 0.0 and float(phase["translation_share"]) > 0.0:
        translation_shift = (
            translation_distance
            * float(phase["translation_share"])
            * _support_translation_factor(str(phase["support_mode_phase"]))
        )

    support_row = support_map[str(phase["support_mode_phase"])]
    plane_factor = _plane_factor(str(action_row["main_plane"]))
    support_factor = 1.0 + MODEL_ASSUMPTIONS["support_amplify_factor"] * (1.0 - float(support_row["support_margin_ratio"]))
    com_shift = (rotational_shift + translation_shift) * plane_factor * support_factor

    peak_acc = 0.0 if phase_time_s <= 1e-9 else 4.0 * com_shift / (phase_time_s**2)
    dynamic_extra = robot.com_height_m / MODEL_ASSUMPTIONS["dynamic_gravity_m_s2"] * peak_acc
    zmp_excursion = com_shift + dynamic_extra + float(support_row["zmp_bias_coeff"]) * com_shift
    support_margin_mode = robot.stable_margin_m * float(support_row["support_margin_ratio"])
    return com_shift, dynamic_extra, zmp_excursion, support_margin_mode


def _phase_work_and_power(
    action_row: pd.Series,
    phase: dict[str, Any],
    robot: RobotParams,
    segment_map: dict[str, pd.Series],
    omega_eff: float,
    phase_time_s: float,
    end_velocity: float,
) -> tuple[float, float]:
    theta_rad = math.radians(float(phase["phase_theta_deg"]))
    complexity_coeff = 1.0 + 0.08 * (int(action_row["active_joint_count"]) - 1) + 0.10 * (int(action_row["phase_count"]) - 1)
    rotational_work = complexity_coeff * robot.average_torque_nm * theta_rad

    translation_work = 0.0
    translation_distance = float(action_row["translation_distance_m"])
    if translation_distance > 0.0 and float(phase["translation_share"]) > 0.0 and phase_time_s > 1e-9:
        trans_speed = translation_distance * float(phase["translation_share"]) / phase_time_s
        trans_mass = _weighted_segment_mass(list(phase["active_segments_json"]), phase["impact_weight_json"], segment_map)
        translation_work = 0.5 * trans_mass * trans_speed**2

    recovery_work = 0.0
    if int(action_row["conditional_flag"]) == 1 and str(action_row["trigger_state"]) == "fallen" and "recovery" in str(phase["phase_name"]):
        recovery_work = (
            robot.total_mass_kg
            * MODEL_ASSUMPTIONS["dynamic_gravity_m_s2"]
            * MODEL_ASSUMPTIONS["recovery_height_m"]
            * MODEL_ASSUMPTIONS["recovery_efficiency"]
        )

    total_work = (rotational_work + translation_work + recovery_work) * float(phase["phase_decay"])
    peak_power = max(
        robot.average_torque_nm * omega_eff * complexity_coeff,
        total_work / phase_time_s if phase_time_s > 1e-9 else 0.0,
        end_velocity * 12.0,
    )
    return total_work, peak_power


def _phase_exposure_index(action_row: pd.Series, phase_time_s: float) -> float:
    exposure = phase_time_s / MODEL_ASSUMPTIONS["exposure_time_ref_s"]
    exposure *= 1.0 + MODEL_ASSUMPTIONS["exposure_phase_alpha"] * (int(action_row["phase_count"]) - 1)
    exposure *= 1.0 + MODEL_ASSUMPTIONS["exposure_switch_beta"] * int(action_row["support_switch_count"])
    exposure *= 1.0 + MODEL_ASSUMPTIONS["rotation_exposure_beta"] * int(action_row["rotation_complexity"])
    return exposure


def _score_potential(p_target_geo: float, end_velocity_peak: float) -> float:
    velocity_term = _sigmoid(
        (end_velocity_peak - MODEL_ASSUMPTIONS["reference_velocity_m_s"]) / MODEL_ASSUMPTIONS["velocity_sigmoid_scale"]
    )
    return float(p_target_geo) * velocity_term


def build_feature_table(
    actions: pd.DataFrame,
    robot: RobotParams,
    segments: pd.DataFrame,
    support_modes: pd.DataFrame,
    phase_templates: pd.DataFrame,
) -> pd.DataFrame:
    segment_map = {str(row["segment_id"]): row for _, row in segments.iterrows()}
    support_map = {str(row["support_mode"]): row for _, row in support_modes.iterrows()}
    records: list[dict[str, Any]] = []

    for _, action_row in actions.iterrows():
        phases = _parse_phase_rows(action_row, phase_templates)
        theta_total_deg = _resolve_theta_total_deg(action_row, phases)
        exec_time = float(action_row["exec_time_s"])
        coordination_efficiency = float(action_row["coordination_efficiency"])

        impact_impulse = 0.0
        impact_kinetic = 0.0
        work_cost = 0.0
        exposure_index = 0.0
        peak_power_proxy = 0.0
        com_shift_max = 0.0
        zmp_excursion_max = 0.0
        zmp_margin_norm_min = float("inf")
        support_margin_mode_min = float("inf")
        end_velocity_peak = 0.0

        for phase in phases:
            phase_time_s = exec_time * float(phase["phase_time_ratio"])
            _, omega_eff = _phase_rotational_speed(coordination_efficiency, float(phase["phase_theta_deg"]), phase_time_s, robot)
            effective_mass = _weighted_segment_mass(list(phase["active_segments_json"]), phase["impact_weight_json"], segment_map)
            effective_reach = _effective_reach(list(phase["strike_segments_json"]), phase["impact_weight_json"], robot)
            rotational_speed = effective_reach * omega_eff

            translational_speed = 0.0
            translation_distance = float(action_row["translation_distance_m"])
            if translation_distance > 0.0 and phase_time_s > 1e-9:
                translational_speed = translation_distance * float(phase["translation_share"]) / phase_time_s
            end_velocity = math.hypot(rotational_speed, translational_speed)
            phase_decay = float(phase["phase_decay"])

            impact_impulse += effective_mass * end_velocity * phase_decay
            impact_kinetic += 0.5 * effective_mass * end_velocity**2 * phase_decay
            end_velocity_peak = max(end_velocity_peak, end_velocity)

            com_shift_phase, _, zmp_excursion, support_margin_mode = _phase_com_shift(
                phase,
                action_row,
                robot,
                segment_map,
                support_map,
                phase_time_s,
            )
            com_shift_max = max(com_shift_max, com_shift_phase)
            zmp_excursion_max = max(zmp_excursion_max, zmp_excursion)
            support_margin_mode_min = min(support_margin_mode_min, support_margin_mode)
            if support_margin_mode > 1e-9:
                zmp_margin_norm_min = min(zmp_margin_norm_min, (support_margin_mode - zmp_excursion) / support_margin_mode)

            phase_work, phase_power = _phase_work_and_power(
                action_row,
                phase,
                robot,
                segment_map,
                omega_eff,
                phase_time_s,
                end_velocity,
            )
            work_cost += phase_work
            peak_power_proxy = max(peak_power_proxy, phase_power)
            decay_power = MODEL_ASSUMPTIONS["phase_work_decay_default"] ** max(int(phase["phase_no"]) - 1, 0)
            exposure_index += _phase_exposure_index(action_row, phase_time_s) / decay_power

        if support_margin_mode_min == float("inf"):
            support_margin_mode_min = robot.stable_margin_m
        if zmp_margin_norm_min == float("inf"):
            zmp_margin_norm_min = 0.0

        stability_ratio = zmp_excursion_max / support_margin_mode_min if support_margin_mode_min > 1e-9 else 0.0
        fall_risk = _sigmoid(MODEL_ASSUMPTIONS["zmp_logistic_slope"] * (stability_ratio - 1.0))
        p_target_geo = float(action_row["p_target_geo"])
        score_potential = _score_potential(p_target_geo, end_velocity_peak)

        records.append(
            {
                "action_id": str(action_row["action_id"]),
                "action_name": str(action_row["action_name"]),
                "category": str(action_row["category"]),
                "laterality": str(action_row["laterality"]),
                "conditional_flag": int(action_row["conditional_flag"]),
                "trigger_state": str(action_row["trigger_state"]),
                "support_mode": str(action_row["support_mode"]),
                "phase_count": int(action_row["phase_count"]),
                "main_plane": str(action_row["main_plane"]),
                "range_tag": str(action_row["range_tag"]),
                "rotation_complexity": int(action_row["rotation_complexity"]),
                "support_switch_count": int(action_row["support_switch_count"]),
                "translation_distance_m": float(action_row["translation_distance_m"]),
                "theta_total_deg": float(theta_total_deg),
                "exec_time": exec_time,
                "p_target_geo": p_target_geo,
                "coordination_efficiency": coordination_efficiency,
                "active_joint_count": int(action_row["active_joint_count"]),
                "joint_count": int(action_row["active_joint_count"]),
                "impact_impulse": impact_impulse,
                "impact_kinetic": impact_kinetic,
                "score_potential": score_potential,
                "work_cost": work_cost,
                "peak_power_proxy": peak_power_proxy,
                "exposure_index": exposure_index,
                "com_shift_max": com_shift_max,
                "zmp_excursion_max": zmp_excursion_max,
                "zmp_margin_norm": zmp_margin_norm_min,
                "fall_risk": fall_risk,
                "stable_margin_mode": support_margin_mode_min,
                "stable_margin": robot.stable_margin_m,
                "stability_ratio": stability_ratio,
                "end_velocity_peak": end_velocity_peak,
                "impact_score": impact_impulse,
                "balance_cost": com_shift_max,
                "score_prob": score_potential,
                "energy_cost": work_cost,
                "p_target": p_target_geo,
                "eta": coordination_efficiency,
                "omega_out": robot.omega_out_rad_s,
                "omega_eff": robot.omega_eff_rad_s,
                "com_height": robot.com_height_m,
                "notes": str(action_row["notes"]),
            }
        )

    return pd.DataFrame(records).sort_values("action_id").reset_index(drop=True)
