"""Q1 全流程流水线。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from src.common import INTERIM_DIR, OUTPUT_DIR, RAW_DIR, ensure_basic_dirs
from src.q1.evaluate import evaluate_all_methods, sensitivity_scan
from src.q1.model_v1 import (
    ASSUMPTION_NOTES,
    LEGACY_COMPATIBILITY_ASSUMPTIONS,
    LEGACY_COMPATIBILITY_NOTES,
    MODEL_ASSUMPTIONS,
    build_feature_table,
    load_attack_actions,
    load_action_phase_templates,
    load_action_templates,
    load_robot_params,
    load_segment_params,
    load_support_mode_config,
    merge_action_definition_tables,
    validate_q1_configuration,
)
from src.q1.plot import (
    plot_decision_atlas,
    plot_impact_balance_scatter,
    plot_method_comparison,
    plot_penalty_curve,
    plot_sensitivity_heatmap,
    plot_utility_bar,
)


ROBOT_RAW_FILE = RAW_DIR / "q1_robot_params.csv"
SEGMENT_RAW_FILE = RAW_DIR / "q1_segment_params.csv"
SUPPORT_RAW_FILE = RAW_DIR / "q1_support_mode_config.csv"
ACTION_SOURCE_FILE = RAW_DIR / "q1_attack_actions.csv"
ACTION_TEMPLATE_FILE = RAW_DIR / "q1_action_templates.csv"
ACTION_PHASE_FILE = RAW_DIR / "q1_action_phase_templates.csv"

INTERIM_FILE = INTERIM_DIR / "action_features.csv"
PARAMETER_REGISTRY_FILE = OUTPUT_DIR / "q1_parameter_registry.csv"
RUN_METADATA_FILE = OUTPUT_DIR / "q1_run_metadata.json"
UTILITY_FIGURE = OUTPUT_DIR / "q1_utility_bar.png"
TRADEOFF_FIGURE = OUTPUT_DIR / "q1_impact_balance.png"
METHOD_COMPARE_FIGURE = OUTPUT_DIR / "q1_method_comparison.png"
PENALTY_FIGURE = OUTPUT_DIR / "q1_penalty_curve.png"
SENSITIVITY_FIGURE = OUTPUT_DIR / "q1_sensitivity_heatmap.png"
DECISION_ATLAS_FIGURE = OUTPUT_DIR / "q1_decision_atlas.png"


def save_action_features(data: pd.DataFrame, output_path: str | Path = INTERIM_FILE) -> Path:
    """保存 Q1 中间结果表。"""

    stable_columns = [
        "action_id",
        "action_name",
        "category",
        "conditional_flag",
        "support_mode",
        "phase_count",
        "impact_impulse",
        "impact_kinetic",
        "score_potential",
        "work_cost",
        "peak_power_proxy",
        "exec_time",
        "exposure_index",
        "com_shift_max",
        "zmp_margin_norm",
        "fall_risk",
        "stable_margin_mode",
        "range_tag",
        "method1_score",
        "method2_score",
        "method3_score",
        "method4_score",
        "method1_rank",
        "method2_rank",
        "method3_rank",
        "method4_rank",
        "utility",
        "impact_score",
        "balance_cost",
        "score_prob",
        "energy_cost",
        "rank",
        "proposed_score",
        "method1_utility",
        "method2_utility",
        "method3_utility",
        "benefit_term",
        "cost_term",
        "fall_penalty_raw",
        "fall_penalty",
        "tau_norm",
        "delta_com_norm",
        "energy_norm",
        "time_norm",
        "stability_score",
        "efficiency_score",
        "execution_efficiency",
        "p_target_geo",
        "p_target",
        "p_reach",
        "stable_margin",
        "stability_ratio",
        "active_joint_count",
        "joint_count",
        "theta_total_deg",
        "eta",
        "coordination_efficiency",
        "omega_out",
        "omega_eff",
        "com_height",
        "laterality",
        "trigger_state",
        "main_plane",
        "rotation_complexity",
        "support_switch_count",
        "translation_distance_m",
        "end_velocity_peak",
        "zmp_excursion_max",
        "notes",
    ]
    output = Path(output_path)
    data.loc[:, stable_columns].to_csv(output, index=False, encoding="utf-8-sig")
    return output


def build_parameter_registry(
    robot_param_path: str | Path,
    robot,
    segments: pd.DataFrame,
    support_modes: pd.DataFrame,
    action_sources: pd.DataFrame,
    action_templates: pd.DataFrame,
    action_phases: pd.DataFrame,
) -> pd.DataFrame:
    """鏋勫缓 Q1 鍙傛暟鏉ユ簮鐧昏琛ㄣ€?"""

    registry_rows: list[dict[str, object]] = []
    assumption_notes = {
        "support_depth_m": "前后支撑深度采用题解中的保守口径。",
        "foot_spacing_ratio": "双脚内侧间距采用机体宽度的 60%。",
        "upper_com_ratio": "整体质心高度采用上体高度的 45%。",
        "effective_speed_ratio": "动作有效角速度取理论输出上限的 60%。",
        "average_torque_ratio": "平均力矩取最大关节力矩的 50%。",
        "average_speed_ratio": "平均角速度取输出轴理论上限的 40%。",
        "torso_eccentricity_m": "躯干转体偏心半径采用 0.15m 的保守估计。",
        "body_charge_gamma": "冲撞平动对质心偏移的折减系数。",
        "counter_recovery_gamma": "倒地恢复平动对质心偏移的折减系数。",
        "dynamic_gravity_m_s2": "重力加速度常数，用于 ZMP 近似与恢复功计算。",
        "exposure_time_ref_s": "暴露指数的参考时间尺度。",
        "exposure_phase_alpha": "多阶段动作的暴露累积系数。",
        "exposure_switch_beta": "支撑切换次数对暴露的附加系数。",
        "rotation_exposure_beta": "旋转复杂度对暴露的附加系数。",
        "reference_velocity_m_s": "速度映射为得分潜力时的参考速度。",
        "velocity_sigmoid_scale": "速度映射 Sigmoid 曲线的尺度参数。",
        "zmp_logistic_slope": "ZMP 越界比率映射为跌倒风险时的 Logistic 斜率。",
        "support_amplify_factor": "支撑模式对质心偏移的放大系数。",
        "translation_effect_default": "非冲撞类平动动作对质心偏移的默认折减。",
        "phase_work_decay_default": "多阶段动作暴露累积时的默认衰减因子。",
        "recovery_height_m": "倒地反击起身阶段的恢复高度近似。",
        "recovery_efficiency": "倒地恢复功的效率折减系数。",
    }
    robot_raw = pd.read_csv(robot_param_path)

    for _, row in robot_raw.iterrows():
        registry_rows.append(
            {
                "parameter_name": str(row["param_name"]),
                "parameter_scope": "robot",
                "source_class": "official",
                "value": row["value"],
                "unit": str(row["unit"]),
                "location": "q1_robot_params.csv",
                "source_detail": str(row["source_type"]),
                "notes": str(row["description"]),
            }
        )

    derived_rows = [
        ("arm_length_m", robot.arm_length_m, "m", "由臂展除以 2 得到"),
        ("upper_arm_length_m", robot.upper_arm_length_m, "m", "由单臂长度乘 0.47 得到"),
        ("forearm_hand_length_m", robot.forearm_hand_length_m, "m", "由单臂长度乘 0.53 得到"),
        ("thigh_length_m", robot.thigh_length_m, "m", "由腿长乘 0.54 得到"),
        ("shank_foot_length_m", robot.shank_foot_length_m, "m", "由腿长乘 0.46 得到"),
        ("omega_out_rad_s", robot.omega_out_rad_s, "rad/s", "由电机转速和减速比推导"),
        ("omega_eff_rad_s", robot.omega_eff_rad_s, "rad/s", "由输出轴角速度乘有效速度折减得到"),
        ("com_height_m", robot.com_height_m, "m", "由腿长与上体质心比例推导"),
        ("foot_spacing_m", robot.foot_spacing_m, "m", "由机体宽度和双足站姿比例推导"),
        ("stable_margin_m", robot.stable_margin_m, "m", "由双足间距与支撑深度保守口径共同确定"),
    ]
    for name, value, unit, note in derived_rows:
        registry_rows.append(
            {
                "parameter_name": name,
                "parameter_scope": "robot",
                "source_class": "derived",
                "value": value,
                "unit": unit,
                "location": "RobotParams",
                "source_detail": "derived_from_official",
                "notes": note,
            }
        )

    for _, row in segments.iterrows():
        segment_scope = f"segment:{row['segment_id']}"
        registry_rows.extend(
            [
                {
                    "parameter_name": "mass_ratio",
                    "parameter_scope": segment_scope,
                    "source_class": "derived",
                    "value": row["mass_ratio"],
                    "unit": "ratio",
                    "location": "q1_segment_params.csv",
                    "source_detail": "derived_from_mass_partition",
                    "notes": "10 刚体节段质量比例。",
                },
                {
                    "parameter_name": "mass_kg",
                    "parameter_scope": segment_scope,
                    "source_class": "derived",
                    "value": row["mass_kg"],
                    "unit": "kg",
                    "location": "q1_segment_params.csv",
                    "source_detail": "derived_from_mass_partition",
                    "notes": "按总质量乘节段比例得到。",
                },
                {
                    "parameter_name": "length_m",
                    "parameter_scope": segment_scope,
                    "source_class": "derived",
                    "value": row["length_m"],
                    "unit": "m",
                    "location": "q1_segment_params.csv",
                    "source_detail": "derived_from_length_partition",
                    "notes": "按单臂或单腿长度比例推导。",
                },
                {
                    "parameter_name": "com_local_ratio",
                    "parameter_scope": segment_scope,
                    "source_class": "assumption",
                    "value": row["com_local_ratio"],
                    "unit": "ratio",
                    "location": "q1_segment_params.csv",
                    "source_detail": "uniform_rigid_body_assumption",
                    "notes": "节段质心在局部长度上的比例位置。",
                },
                {
                    "parameter_name": "inertia_coeff",
                    "parameter_scope": segment_scope,
                    "source_class": "assumption",
                    "value": row["inertia_coeff"],
                    "unit": "-",
                    "location": "q1_segment_params.csv",
                    "source_detail": "reserved_dynamic_extension",
                    "notes": "节段转动惯量代理系数，目前作为扩展接口预留。",
                },
            ]
        )

    for _, row in support_modes.iterrows():
        support_scope = f"support_mode:{row['support_mode']}"
        registry_rows.extend(
            [
                {
                    "parameter_name": "support_margin_ratio",
                    "parameter_scope": support_scope,
                    "source_class": "assumption",
                    "value": row["support_margin_ratio"],
                    "unit": "ratio",
                    "location": "q1_support_mode_config.csv",
                    "source_detail": "support_mode_assumption",
                    "notes": "不同支撑模式对基础稳定裕度的缩减比例。",
                },
                {
                    "parameter_name": "zmp_bias_coeff",
                    "parameter_scope": support_scope,
                    "source_class": "assumption",
                    "value": row["zmp_bias_coeff"],
                    "unit": "-",
                    "location": "q1_support_mode_config.csv",
                    "source_detail": "support_mode_assumption",
                    "notes": "不同支撑模式下对动态 ZMP 偏移的附加偏置。",
                },
            ]
        )

    for _, row in action_sources.iterrows():
        action_scope = f"action:{row['action_id']}"
        for field_name, unit, source_field in (
            ("coordination_efficiency", "-", "coordination_source"),
            ("active_joint_count", "count", "active_joint_count_source"),
            ("theta_total_deg", "deg", "theta_source"),
            ("exec_time_s", "s", "exec_time_source"),
            ("p_target_geo", "ratio", "p_target_source"),
        ):
            registry_rows.append(
                {
                    "parameter_name": field_name,
                    "parameter_scope": action_scope,
                    "source_class": "assumption",
                    "value": row[field_name],
                    "unit": unit,
                    "location": "q1_attack_actions.csv",
                    "source_detail": str(row[source_field]),
                    "notes": "动作数值源表中的动作层建模设定。",
                }
            )

    template_fields = [
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
    ]
    for _, row in action_templates.iterrows():
        action_scope = f"template:{row['action_id']}"
        for field_name in template_fields:
            value = row[field_name]
            registry_rows.append(
                {
                    "parameter_name": field_name,
                    "parameter_scope": action_scope,
                    "source_class": "assumption",
                    "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value,
                    "unit": "-",
                    "location": "q1_action_templates.csv",
                    "source_detail": "action_structure_template",
                    "notes": "动作结构模板字段。",
                }
            )

    phase_fields = [
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
    ]
    for _, row in action_phases.iterrows():
        phase_scope = f"phase:{row['action_id']}-P{int(row['phase_no'])}"
        for field_name in phase_fields:
            value = row[field_name]
            registry_rows.append(
                {
                    "parameter_name": field_name,
                    "parameter_scope": phase_scope,
                    "source_class": "assumption",
                    "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value,
                    "unit": "-",
                    "location": "q1_action_phase_templates.csv",
                    "source_detail": "phase_template",
                    "notes": f"动作相位模板字段：{row['phase_name']}",
                }
            )

    for key, value in MODEL_ASSUMPTIONS.items():
        registry_rows.append(
            {
                "parameter_name": key,
                "parameter_scope": "global_assumption",
                "source_class": "assumption",
                "value": value,
                "unit": "-",
                "location": "src/q1/model_v1.py",
                "source_detail": "MODEL_ASSUMPTIONS",
                "notes": assumption_notes.get(key, ASSUMPTION_NOTES.get(key, "")),
            }
        )

    for key, value in LEGACY_COMPATIBILITY_ASSUMPTIONS.items():
        registry_rows.append(
            {
                "parameter_name": key,
                "parameter_scope": "legacy_compatibility",
                "source_class": "compatibility_assumption",
                "value": value,
                "unit": "-",
                "location": "src/q1/model_v1.py",
                "source_detail": "LEGACY_COMPATIBILITY_ASSUMPTIONS",
                "notes": LEGACY_COMPATIBILITY_NOTES.get(key, ""),
            }
        )

    return pd.DataFrame(registry_rows)


def save_parameter_registry(registry: pd.DataFrame, output_path: str | Path = PARAMETER_REGISTRY_FILE) -> Path:
    """淇濆瓨 Q1 鍙傛暟鏉ユ簮鐧昏琛ㄣ€?"""

    output = Path(output_path)
    registry.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def save_run_metadata(
    evaluated: pd.DataFrame,
    ahp_summary: dict[str, float],
    output_path: str | Path = RUN_METADATA_FILE,
) -> Path:
    """淇濆瓨 Q1 杩愯蹇収锛屼究浜庣粨鏋滃鐜般€?"""

    output = Path(output_path)
    top5 = (
        evaluated.sort_values("method4_score", ascending=False)
        .loc[:, ["action_id", "action_name", "method4_score", "method4_rank"]]
        .head(5)
        .to_dict("records")
    )
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_files": {
            "robot": str(ROBOT_RAW_FILE),
            "segments": str(SEGMENT_RAW_FILE),
            "support_modes": str(SUPPORT_RAW_FILE),
            "action_sources": str(ACTION_SOURCE_FILE),
            "action_templates": str(ACTION_TEMPLATE_FILE),
            "action_phases": str(ACTION_PHASE_FILE),
        },
        "output_files": {
            "action_features": str(INTERIM_FILE),
            "parameter_registry": str(PARAMETER_REGISTRY_FILE),
            "utility_figure": str(UTILITY_FIGURE),
            "tradeoff_figure": str(TRADEOFF_FIGURE),
            "method_compare_figure": str(METHOD_COMPARE_FIGURE),
            "penalty_figure": str(PENALTY_FIGURE),
            "sensitivity_figure": str(SENSITIVITY_FIGURE),
            "decision_atlas_figure": str(DECISION_ATLAS_FIGURE),
        },
        "top5_actions": top5,
        "ahp_summary": ahp_summary,
        "model_assumptions": MODEL_ASSUMPTIONS,
        "legacy_compatibility_assumptions": LEGACY_COMPATIBILITY_ASSUMPTIONS,
    }
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def run_pipeline() -> tuple[pd.DataFrame, dict[str, float]]:
    """执行 Q1 全流程。"""

    ensure_basic_dirs()
    robot = load_robot_params(ROBOT_RAW_FILE)
    segments = load_segment_params(SEGMENT_RAW_FILE)
    support_modes = load_support_mode_config(SUPPORT_RAW_FILE)
    action_sources = load_attack_actions(ACTION_SOURCE_FILE)
    action_templates = load_action_templates(ACTION_TEMPLATE_FILE)
    action_phases = load_action_phase_templates(ACTION_PHASE_FILE)
    action_table = merge_action_definition_tables(action_sources, action_templates)
    validate_q1_configuration(
        actions=action_table,
        phase_templates=action_phases,
        segments=segments,
        support_modes=support_modes,
    )
    parameter_registry = build_parameter_registry(
        robot_param_path=ROBOT_RAW_FILE,
        robot=robot,
        segments=segments,
        support_modes=support_modes,
        action_sources=action_sources,
        action_templates=action_templates,
        action_phases=action_phases,
    )
    save_parameter_registry(parameter_registry)

    feature_table = build_feature_table(
        actions=action_table,
        robot=robot,
        segments=segments,
        support_modes=support_modes,
        phase_templates=action_phases,
    )
    evaluated, ahp_summary = evaluate_all_methods(
        data=feature_table,
        stable_margin=robot.stable_margin_m,
    )
    save_action_features(evaluated)

    sensitivity = sensitivity_scan(
        data=feature_table,
        stable_margin=robot.stable_margin_m,
    )
    plot_utility_bar(evaluated, UTILITY_FIGURE)
    plot_impact_balance_scatter(evaluated, TRADEOFF_FIGURE)
    plot_method_comparison(evaluated, METHOD_COMPARE_FIGURE)
    plot_penalty_curve(evaluated, PENALTY_FIGURE)
    plot_decision_atlas(evaluated, DECISION_ATLAS_FIGURE)
    plot_sensitivity_heatmap(sensitivity, SENSITIVITY_FIGURE)
    save_run_metadata(evaluated, ahp_summary)
    return evaluated, ahp_summary


def main() -> tuple[pd.DataFrame, dict[str, float]]:
    """命令行友好的 Q1 入口。"""

    return run_pipeline()


if __name__ == "__main__":
    main()
