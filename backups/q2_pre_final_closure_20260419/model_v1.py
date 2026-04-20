"""Q2 attack-defense response model driven by Q1 phase features."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.q1.model_v1 import RobotParams


ACTION_FEATURE_COLUMNS = [
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
    "utility",
    "rotation_complexity",
    "support_switch_count",
    "translation_distance_m",
    "trigger_state",
    "main_plane",
    "end_velocity_peak",
]

ATTACK_SEMANTIC_COLUMNS = [
    "action_id",
    "trajectory_type",
    "direction_tag",
    "height_tag",
    "range_tag",
    "target_zone",
    "contact_mode_attack",
    "primary_plane",
    "attack_entry_state",
    "attack_trigger_state",
    "semantic_source_type",
    "description",
]

ATTACK_RESPONSE_COLUMNS = [
    "action_id",
    "contact_phase_policy",
    "contact_phase_no",
    "contact_fraction_in_phase",
    "recover_phase_policy",
    "recover_delay_base_s",
    "recover_switch_coeff",
    "recover_rotation_coeff",
    "recover_risk_coeff",
    "counter_unlock_mode",
    "counter_unlock_coeff",
    "response_policy_source",
    "notes",
]

DEFENSE_COLUMNS = [
    "defense_id",
    "defense_name",
    "category",
    "defense_role",
    "is_primary_candidate",
    "closure_group",
    "entry_state",
    "exit_state",
    "coverage_direction",
    "coverage_height",
    "coverage_range",
    "coverage_target_zone",
    "contact_mode",
    "support_mode_def",
    "route_type",
    "primary_joint_group",
    "primary_angle_deg",
    "secondary_joint_group",
    "secondary_angle_deg",
    "sequence_ids",
    "official_time_s",
    "force_capacity_factor_input",
    "contact_stiffness_ratio_input",
    "active_segments_json",
    "support_segments_json",
    "counter_readiness_tag",
    "source_type",
    "description",
    "force_capacity_basis",
    "stiffness_basis",
]

ROUTE_COLUMNS = [
    "trajectory_type",
    "contact_mode_attack",
    "direction_tag",
    "range_tag",
    "route_type",
    "route_bonus",
    "clearance_need_ratio",
    "route_source",
    "notes",
]

Q2_MODEL_ASSUMPTIONS = {
    "reaction_sigmoid_scale_s": 0.10,
    "load_sigmoid_ratio": 0.20,
    "clearance_sigmoid_scale_m": 0.07,
    "combo_overlap_discount_s": 0.04,
    "support_delay_double_s": 0.04,
    "support_delay_quasi_single_s": 0.06,
    "support_delay_dynamic_step_s": 0.08,
    "support_delay_rotational_s": 0.10,
    "support_delay_ground_s": 0.18,
    "support_delay_soft_s": 0.05,
    "mobility_plane_factor": 0.60,
    "mobility_angle_mix": 0.35,
    "mobility_scale": 0.85,
    "primary_support_mass_ratio": 0.22,
    "composite_route_weight": 0.55,
    "contact_damage_base": 0.40,
    "contact_damage_absorb_weight": 0.60,
    "counter_window_ready_threshold_s": 0.12,
    "counter_exec_tolerance_s": 0.05,
}

DIRECT_STATE_VALUES = {"standing", "any"}
DEFENSE_ROLE_VALUES = {
    "active_primary",
    "active_combo",
    "fallback_mitigation",
    "ground_only",
    "recovery_only",
    "emergency_transition",
}

COUNTER_UNLOCK_VALUES = {"immediate", "after_first_break", "after_main_contact", "after_full_chain", "disabled"}
ENTRY_STATE_VALUES = {"standing", "fallen", "any"}
EXIT_STATE_VALUES = {"standing", "off_line", "crouched", "fallen", "ground_guard", "recovering"}

JSON_LIST_COLUMNS = ["active_segments_json", "support_segments_json"]

SUPPORT_MODE_MASS_FACTOR = {
    "double_support": 1.00,
    "quasi_single_support": 0.86,
    "dynamic_step": 0.92,
    "single_support_rotational": 0.76,
    "ground_support": 1.05,
    "soft_adjust": 0.80,
}

SUPPORT_MODE_ALIAS = {
    "dynamic_step": "dynamic_double_support",
    "ground_support": "recovery_transition",
    "soft_adjust": "double_support",
}

CONTACT_ABSORB_BASE = {
    "rigid": 0.52,
    "passive": 0.63,
    "soft": 0.78,
    "ground": 0.55,
    "none": 0.00,
    "composite": 0.60,
}

CONTACT_CAPACITY_BASE = {
    "rigid": 0.95,
    "passive": 0.72,
    "soft": 0.80,
    "ground": 0.58,
    "none": 0.00,
    "composite": 0.88,
}

COUNTER_DELAY_MAP = {"fast": 0.06, "medium": 0.10, "slow": 0.18, "none": 0.35}

PLANE_LENGTH_FACTOR = {
    "sagittal": 1.00,
    "frontal": 1.04,
    "transverse": 1.08,
    "multi_plane": 1.12,
}

RANGE_NEIGHBORS = {
    "close": {"middle"},
    "middle": {"close", "long"},
    "long": {"middle"},
    "ground": {"close"},
    "all": {"close", "middle", "long", "ground"},
}

HEIGHT_NEIGHBORS = {
    "high": {"middle", "middle_high", "mixed", "all"},
    "middle": {"high", "low", "middle_high", "mixed", "all"},
    "low": {"middle", "mixed", "all"},
    "middle_high": {"high", "middle", "mixed", "all"},
    "mixed": {"high", "middle", "low", "middle_high", "all"},
    "all": {"high", "middle", "low", "middle_high", "mixed"},
}

DIRECTION_NEIGHBORS = {
    "front": {"front_lateral", "mixed", "all"},
    "lateral": {"front_lateral", "spin", "mixed", "all"},
    "spin": {"lateral", "mixed", "all"},
    "low": {"mixed", "all"},
    "front_lateral": {"front", "lateral", "mixed", "all"},
    "mixed": {"front", "lateral", "spin", "low", "front_lateral", "all"},
    "all": {"front", "lateral", "spin", "low", "front_lateral", "mixed"},
}

ZONE_TOKEN_MAP = {
    "head": {"head"},
    "torso": {"torso"},
    "limbs": {"limbs"},
    "head_torso": {"head", "torso"},
    "torso_limbs": {"torso", "limbs"},
    "lower_limbs": {"lower_limbs", "limbs"},
    "all": {"head", "torso", "limbs", "lower_limbs"},
}

NON_DIRECT_ROUTES = {
    "evade_lateral",
    "evade_duck",
    "evade_back",
    "evade_spin",
    "evade_orbit",
    "combo_lateral_parry",
    "combo_duck_slip",
    "combo_block_retreat",
    "yield_soft",
    "balance_step",
    "ground_hold",
    "ground_transfer",
}

DEFAULT_ROUTE_RULE = {
    "evade_lateral": (0.60, 0.30),
    "evade_back": (0.52, 0.42),
    "evade_orbit": (0.58, 0.34),
    "evade_spin": (0.52, 0.38),
    "evade_duck": (0.48, 0.26),
    "balance_step": (0.58, 0.24),
    "yield_soft": (0.62, 0.00),
    "ground_hold": (0.70, 0.00),
    "ground_transfer": (0.64, 0.00),
    "combo_lateral_parry": (0.68, 0.18),
    "combo_duck_slip": (0.65, 0.20),
    "combo_block_retreat": (0.64, 0.24),
}


def _require_columns(data: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _parse_json_list(raw_value: Any) -> list[str]:
    if pd.isna(raw_value) or str(raw_value).strip() == "":
        return []
    parsed = json.loads(str(raw_value))
    if not isinstance(parsed, list):
        raise ValueError("expected JSON list")
    return [str(item) for item in parsed]


def _normalize_series(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if math.isclose(minimum, maximum, rel_tol=1e-9, abs_tol=1e-9):
        return pd.Series(np.full(len(series), 0.5), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _canonical_support_mode(support_mode_def: str) -> str:
    return SUPPORT_MODE_ALIAS.get(str(support_mode_def), str(support_mode_def))


def load_action_features(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    _require_columns(data, ACTION_FEATURE_COLUMNS, "Q1 action feature table")
    return data[ACTION_FEATURE_COLUMNS].copy()


def load_attack_semantics(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    _require_columns(data, ATTACK_SEMANTIC_COLUMNS, "Q2 attack semantic table")
    if data["action_id"].nunique() != 13:
        raise ValueError("q2_attack_semantics.csv must cover 13 unique actions")
    if str(data.loc[data["action_id"] == "A13", "attack_trigger_state"].iloc[0]).strip().lower() != "fallen":
        raise ValueError("A13 must have attack_trigger_state=fallen")
    if data["contact_mode_attack"].astype(str).str.strip().eq("").any():
        raise ValueError("contact_mode_attack cannot be empty")
    return data[ATTACK_SEMANTIC_COLUMNS].copy()


def load_attack_response_policy(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    _require_columns(data, ATTACK_RESPONSE_COLUMNS, "Q2 attack response policy table")
    if data["action_id"].nunique() != 13:
        raise ValueError("q2_attack_response_policy.csv must cover 13 unique actions")
    if ((data["contact_fraction_in_phase"] <= 0.0) | (data["contact_fraction_in_phase"] > 1.0)).any():
        raise ValueError("contact_fraction_in_phase must lie in (0, 1]")
    if ((data["counter_unlock_coeff"] < 0.0) | (data["counter_unlock_coeff"] > 1.0)).any():
        raise ValueError("counter_unlock_coeff must lie in [0, 1]")
    if not data["counter_unlock_mode"].astype(str).isin(COUNTER_UNLOCK_VALUES).all():
        raise ValueError("invalid counter_unlock_mode detected")
    return data[ATTACK_RESPONSE_COLUMNS].copy()


def load_defense_actions(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    _require_columns(data, DEFENSE_COLUMNS, "Q2 defense action table")
    if data["defense_id"].nunique() != 22:
        raise ValueError("q2_defense_actions.csv must cover 22 unique defenses")
    for column in JSON_LIST_COLUMNS:
        data[column] = data[column].apply(_parse_json_list)
    if not data["defense_role"].astype(str).isin(DEFENSE_ROLE_VALUES).all():
        invalid_values = sorted(set(data.loc[~data["defense_role"].astype(str).isin(DEFENSE_ROLE_VALUES), "defense_role"]))
        raise ValueError(f"invalid defense_role values: {invalid_values}")
    if not data["entry_state"].astype(str).isin(ENTRY_STATE_VALUES).all():
        raise ValueError("invalid entry_state detected")
    if not data["exit_state"].astype(str).isin(EXIT_STATE_VALUES).all():
        raise ValueError("invalid exit_state detected")
    recovery_mask = data["defense_role"].isin(["recovery_only", "ground_only"])
    if not data.loc[recovery_mask, "is_primary_candidate"].eq(0).all():
        raise ValueError("recovery_only and ground_only defenses cannot be primary candidates")
    data["sequence_ids"] = data["sequence_ids"].fillna("").astype(str)
    return data[DEFENSE_COLUMNS].copy()


def load_route_advantage(file_path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    _require_columns(data, ROUTE_COLUMNS, "Q2 route advantage table")
    if ((data["route_bonus"] < 0.0) | (data["route_bonus"] > 1.0)).any():
        raise ValueError("route_bonus must lie in [0, 1]")
    if ((data["clearance_need_ratio"] < 0.0) | (data["clearance_need_ratio"] > 1.0)).any():
        raise ValueError("clearance_need_ratio must lie in [0, 1]")
    if data.duplicated(subset=["trajectory_type", "contact_mode_attack", "direction_tag", "range_tag", "route_type"]).any():
        raise ValueError("q2_route_advantage.csv contains duplicated route rules")
    return data[ROUTE_COLUMNS].copy()


def _token_set(tag: str) -> set[str]:
    normalized = str(tag).strip().lower()
    if normalized in {"", "nan"}:
        return set()
    if normalized in {"all", "mixed", "any"}:
        return {normalized}
    return {part for part in normalized.split("_") if part}


def _single_match(attack_token: str, defense_token: str, neighbors: dict[str, set[str]]) -> float:
    if defense_token == "all":
        return 1.0
    if attack_token == defense_token:
        return 1.0
    if defense_token == "mixed":
        return 0.5
    if defense_token in neighbors.get(attack_token, set()):
        return 0.5
    return 0.0


def _tag_score(attack_tag: str, defense_tag: str, neighbors: dict[str, set[str]]) -> float:
    attack_tokens = _token_set(attack_tag)
    defense_tokens = _token_set(defense_tag)
    if not attack_tokens or not defense_tokens:
        return 0.0
    if "all" in defense_tokens:
        return 1.0
    best = 0.0
    for attack_token in attack_tokens:
        for defense_token in defense_tokens:
            best = max(best, _single_match(attack_token, defense_token, neighbors))
    if best == 1.0 and len(attack_tokens) > 1 and not attack_tokens.issubset(defense_tokens):
        return 0.5
    return best


def _zone_score(attack_zone: str, defense_zone: str) -> float:
    attack_tokens = ZONE_TOKEN_MAP.get(str(attack_zone), {str(attack_zone)})
    defense_tokens = ZONE_TOKEN_MAP.get(str(defense_zone), {str(defense_zone)})
    if defense_zone == "all":
        return 1.0
    overlap = attack_tokens & defense_tokens
    if not overlap:
        return 0.0
    if overlap == attack_tokens:
        return 1.0
    return 0.5


def _range_score(attack_range: str, defense_range: str) -> float:
    attack = str(attack_range)
    defense = str(defense_range)
    if defense == "all":
        return 1.0
    if attack == defense:
        return 1.0
    if defense in RANGE_NEIGHBORS.get(attack, set()):
        return 0.5
    return 0.0


def _build_single_phase(action_row: pd.Series, template_row: pd.Series) -> list[dict[str, Any]]:
    return [
        {
            "phase_no": 1,
            "phase_name": "single_phase",
            "phase_time_ratio": 1.0,
            "phase_theta_deg": float(action_row.get("theta_total_deg", 0.0) or 0.0),
            "support_mode_phase": str(template_row["support_mode"]),
            "translation_share": 1.0 if float(action_row["translation_distance_m"]) > 0.0 else 0.0,
            "active_segments_json": list(template_row["active_segments_json"]),
            "strike_segments_json": list(template_row["strike_segments_json"]),
            "impact_weight_json": dict(template_row["impact_weight_json"]),
            "stability_weight_json": dict(template_row["stability_weight_json"]),
            "theta_share_json": dict(template_row["theta_share_json"]),
            "phase_decay": 1.0,
        }
    ]


def _action_phases(
    action_id: str,
    action_row: pd.Series,
    template_row: pd.Series,
    phase_templates: pd.DataFrame,
) -> list[dict[str, Any]]:
    subset = phase_templates.loc[phase_templates["action_id"] == action_id].sort_values("phase_no")
    if subset.empty:
        return _build_single_phase(action_row, template_row)
    return [
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
        for _, row in subset.iterrows()
    ]


def _select_contact_phase(phases: list[dict[str, Any]], policy_row: pd.Series) -> dict[str, Any]:
    strike_phases = [phase for phase in phases if phase["strike_segments_json"]]
    if not strike_phases:
        return phases[-1]

    policy = str(policy_row["contact_phase_policy"])
    if policy == "explicit_phase":
        phase_no = int(policy_row["contact_phase_no"])
        for phase in phases:
            if int(phase["phase_no"]) == phase_no:
                return phase
        raise ValueError(f"explicit contact phase {phase_no} does not exist")
    if policy == "first_strike":
        return strike_phases[0]
    if policy == "final_strike":
        return strike_phases[-1]
    if policy == "body_impact":
        for phase in phases:
            if "impact" in str(phase["phase_name"]):
                return phase
        return strike_phases[-1]
    if policy == "main_strike":
        return max(strike_phases, key=lambda phase: float(phase["phase_theta_deg"]))
    return strike_phases[0]


def build_attack_catalog(
    action_features: pd.DataFrame,
    robot: RobotParams,
    action_templates: pd.DataFrame,
    phase_templates: pd.DataFrame,
    attack_semantics: pd.DataFrame,
    attack_response_policy: pd.DataFrame,
) -> pd.DataFrame:
    catalog = (
        action_features.merge(attack_semantics, on="action_id", how="inner", validate="one_to_one")
        .merge(attack_response_policy, on="action_id", how="inner", validate="one_to_one")
    )
    template_map = action_templates.set_index("action_id").to_dict("index")
    phase_no_map = phase_templates.groupby("action_id")["phase_no"].apply(set).to_dict()

    records: list[dict[str, Any]] = []
    for _, row in catalog.iterrows():
        action_id = str(row["action_id"])
        template_row = template_map[action_id]
        if str(row["attack_trigger_state"]) != str(row["trigger_state"]):
            raise ValueError(f"{action_id} trigger_state mismatch between Q1 and Q2")

        if str(row["contact_phase_policy"]) == "explicit_phase":
            explicit_phase = int(row["contact_phase_no"])
            if explicit_phase not in phase_no_map.get(action_id, set()):
                raise ValueError(f"{action_id} explicit contact phase {explicit_phase} is missing in Q1 phase table")

        phases = _action_phases(action_id, row, pd.Series(template_row), phase_templates)
        contact_phase = _select_contact_phase(phases, row)
        prefix_ratio = sum(
            float(phase["phase_time_ratio"])
            for phase in phases
            if int(phase["phase_no"]) < int(contact_phase["phase_no"])
        )
        exec_time = float(row["exec_time"])
        contact_fraction = float(row["contact_fraction_in_phase"])
        contact_ratio = float(contact_phase["phase_time_ratio"])
        t_contact = exec_time * (prefix_ratio + contact_fraction * contact_ratio)
        t_recover = (
            max(exec_time - t_contact, 0.0)
            + float(row["recover_delay_base_s"])
            + float(row["recover_switch_coeff"]) * float(row["support_switch_count"])
            + float(row["recover_rotation_coeff"]) * float(row["rotation_complexity"])
            + float(row["recover_risk_coeff"]) * float(row["fall_risk"])
        )
        contact_reach_m = _contact_reach_m(contact_phase, row, robot)
        records.append(
            {
                "action_id": action_id,
                "action_name": str(row["action_name"]),
                "attack_category": str(row["category"]),
                "attack_utility": float(row["utility"]),
                "conditional_flag": int(row["conditional_flag"]),
                "trigger_state": str(row["trigger_state"]),
                "support_mode": str(row["support_mode"]),
                "phase_count": int(row["phase_count"]),
                "main_plane": str(row["main_plane"]),
                "rotation_complexity": int(row["rotation_complexity"]),
                "support_switch_count": int(row["support_switch_count"]),
                "translation_distance_m": float(row["translation_distance_m"]),
                "impact_impulse": float(row["impact_impulse"]),
                "impact_kinetic": float(row["impact_kinetic"]),
                "score_potential": float(row["score_potential"]),
                "work_cost": float(row["work_cost"]),
                "peak_power_proxy": float(row["peak_power_proxy"]),
                "exec_time": exec_time,
                "exposure_index": float(row["exposure_index"]),
                "com_shift_max": float(row["com_shift_max"]),
                "zmp_margin_norm": float(row["zmp_margin_norm"]),
                "fall_risk": float(row["fall_risk"]),
                "stable_margin_mode": float(row["stable_margin_mode"]),
                "range_tag_q1": str(row["range_tag_x"]),
                "end_velocity_peak": float(row["end_velocity_peak"]),
                "trajectory_type": str(row["trajectory_type"]),
                "direction_tag": str(row["direction_tag"]),
                "height_tag": str(row["height_tag"]),
                "range_tag": str(row["range_tag_y"]),
                "target_zone": str(row["target_zone"]),
                "contact_mode_attack": str(row["contact_mode_attack"]),
                "attack_primary_plane": str(row["primary_plane"]),
                "attack_entry_state": str(row["attack_entry_state"]),
                "attack_trigger_state": str(row["attack_trigger_state"]),
                "semantic_source_type": str(row["semantic_source_type"]),
                "contact_phase_policy": str(row["contact_phase_policy"]),
                "contact_phase_no": int(contact_phase["phase_no"]),
                "contact_phase_name": str(contact_phase["phase_name"]),
                "contact_phase_time_ratio": float(contact_phase["phase_time_ratio"]),
                "contact_fraction_in_phase": contact_fraction,
                "contact_phase_theta_deg": float(contact_phase["phase_theta_deg"]),
                "contact_translation_share": float(contact_phase["translation_share"]),
                "contact_support_mode": str(contact_phase["support_mode_phase"]),
                "contact_strike_segments_json": json.dumps(list(contact_phase["strike_segments_json"]), ensure_ascii=True),
                "contact_active_segments_json": json.dumps(list(contact_phase["active_segments_json"]), ensure_ascii=True),
                "contact_reach_m": float(contact_reach_m),
                "contact_load_family": _contact_load_family(row),
                "contact_plane": str(row["primary_plane"]),
                "recover_phase_policy": str(row["recover_phase_policy"]),
                "recover_delay_base_s": float(row["recover_delay_base_s"]),
                "recover_switch_coeff": float(row["recover_switch_coeff"]),
                "recover_rotation_coeff": float(row["recover_rotation_coeff"]),
                "recover_risk_coeff": float(row["recover_risk_coeff"]),
                "counter_unlock_mode": str(row["counter_unlock_mode"]),
                "counter_unlock_coeff": float(row["counter_unlock_coeff"]),
                "response_policy_source": str(row["response_policy_source"]),
                "t_contact": t_contact,
                "t_end": exec_time,
                "t_recover": t_recover,
                "opp_recover_time": t_contact + t_recover,
                "attack_description": str(row["description"]),
            }
        )

    attack_catalog = pd.DataFrame(records).sort_values("action_id").reset_index(drop=True)
    attack_catalog["J_attack_norm"] = _normalize_series(np.log1p(attack_catalog["impact_impulse"]))
    attack_catalog["E_attack_norm"] = _normalize_series(np.log1p(attack_catalog["impact_kinetic"]))
    attack_catalog["H_attack"] = 0.55 * attack_catalog["J_attack_norm"] + 0.45 * attack_catalog["E_attack_norm"]
    attack_catalog["R_attack"] = 0.60 * attack_catalog["fall_risk"] + 0.40 * _normalize_series(attack_catalog["exposure_index"])
    attack_catalog["tau_norm"] = _normalize_series(attack_catalog["impact_impulse"])
    attack_catalog["time_norm"] = 1.0 - _normalize_series(attack_catalog["exec_time"])
    attack_catalog["impact_score"] = attack_catalog["impact_impulse"]
    attack_catalog["score_prob"] = attack_catalog["score_potential"]
    attack_catalog["energy_cost"] = attack_catalog["work_cost"]
    attack_catalog["balance_cost"] = attack_catalog["com_shift_max"]
    return attack_catalog


def _segment_mass(segment_ids: list[str], segment_map: dict[str, dict[str, Any]]) -> float:
    return float(sum(float(segment_map[segment_id]["mass_kg"]) for segment_id in segment_ids if segment_id in segment_map))


def _segment_mean_length(segment_ids: list[str], segment_map: dict[str, dict[str, Any]]) -> float:
    lengths = [float(segment_map[segment_id]["length_m"]) for segment_id in segment_ids if segment_id in segment_map]
    if not lengths:
        return 0.0
    return float(np.mean(lengths))


def _contact_reach_m(contact_phase: dict[str, Any], attack_row: pd.Series, robot: RobotParams) -> float:
    strike_segments = set(str(segment) for segment in contact_phase["strike_segments_json"])
    if str(attack_row["contact_mode_attack"]) == "body_contact" or "S2" in strike_segments:
        return float(max(robot.body_width_m * 0.50, robot.total_height_m * 0.16))
    if strike_segments.intersection({"S5", "S6"}):
        return float(robot.full_arm_lever_m)
    if strike_segments.intersection({"S9", "S10"}):
        return float(robot.full_leg_lever_m)
    if strike_segments.intersection({"S7", "S8"}):
        return float(robot.thigh_lever_m)
    return float(max(robot.forearm_lever_m, robot.thigh_lever_m))


def _contact_load_family(attack_row: pd.Series) -> str:
    mapping = {
        "fist_contact": "upper_terminal",
        "leg_contact": "lower_terminal",
        "sweep_contact": "sweep_line",
        "body_contact": "torso_drive",
        "multi_contact": "sequence_chain",
        "conditional_contact": "ground_conditional",
    }
    return mapping.get(str(attack_row["contact_mode_attack"]), "generic")


def _weighted_average(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    total_weight = float(sum(weights))
    if math.isclose(total_weight, 0.0, rel_tol=1e-9, abs_tol=1e-9):
        return float(np.mean(values))
    return float(sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight)


def _support_delay(support_mode_def: str) -> float:
    return {
        "double_support": Q2_MODEL_ASSUMPTIONS["support_delay_double_s"],
        "quasi_single_support": Q2_MODEL_ASSUMPTIONS["support_delay_quasi_single_s"],
        "dynamic_step": Q2_MODEL_ASSUMPTIONS["support_delay_dynamic_step_s"],
        "single_support_rotational": Q2_MODEL_ASSUMPTIONS["support_delay_rotational_s"],
        "ground_support": Q2_MODEL_ASSUMPTIONS["support_delay_ground_s"],
        "soft_adjust": Q2_MODEL_ASSUMPTIONS["support_delay_soft_s"],
    }.get(str(support_mode_def), Q2_MODEL_ASSUMPTIONS["support_delay_double_s"])


def _motion_time(angle_deg: float, robot: RobotParams) -> float:
    if angle_deg <= 0.0:
        return 0.0
    return 1.5 * math.radians(angle_deg) / robot.omega_eff_rad_s


def _compute_clearance_components(
    defense_row: pd.Series,
    robot: RobotParams,
) -> tuple[float, float, float, float]:
    route_type = str(defense_row["route_type"])
    primary_angle = 0.0 if pd.isna(defense_row["primary_angle_deg"]) else float(defense_row["primary_angle_deg"])
    secondary_angle = 0.0 if pd.isna(defense_row["secondary_angle_deg"]) else float(defense_row["secondary_angle_deg"])
    clear_back_m = 0.0
    clear_lateral_m = 0.0
    clear_orbit_m = 0.0
    clear_drop_m = 0.0

    if route_type == "evade_back":
        clear_back_m = robot.full_leg_lever_m * math.sin(math.radians(primary_angle)) * 0.30
    elif route_type == "evade_lateral":
        clear_lateral_m = (
            robot.thigh_lever_m * math.sin(math.radians(primary_angle)) * 0.45
            + 0.5 * 0.15 * math.sin(math.radians(secondary_angle))
        )
    elif route_type == "evade_orbit":
        clear_orbit_m = (
            robot.stable_margin_m
            + robot.thigh_lever_m * math.sin(math.radians(secondary_angle)) * 0.35
            + 0.15 * math.sin(math.radians(primary_angle) / 2.0)
        )
    elif route_type == "evade_duck":
        clear_drop_m = robot.com_height_m * (1.0 - math.cos(math.radians(primary_angle))) * 0.35
    elif route_type == "balance_step":
        base_back = robot.full_leg_lever_m * math.sin(math.radians(primary_angle)) * 0.22
        base_lateral = (
            robot.thigh_lever_m * math.sin(math.radians(primary_angle)) * 0.30
            + 0.5 * 0.12 * math.sin(math.radians(secondary_angle))
        )
        clear_back_m = base_back
        clear_lateral_m = base_lateral
    elif route_type == "combo_lateral_parry":
        clear_lateral_m = robot.thigh_lever_m * math.sin(math.radians(max(primary_angle, 20.0))) * 0.32
    elif route_type == "combo_duck_slip":
        clear_lateral_m = robot.thigh_lever_m * math.sin(math.radians(max(primary_angle, 20.0))) * 0.24
        clear_drop_m = robot.com_height_m * (1.0 - math.cos(math.radians(max(primary_angle, 30.0)))) * 0.24
    elif route_type == "combo_block_retreat":
        clear_back_m = robot.full_leg_lever_m * math.sin(math.radians(max(primary_angle, 20.0))) * 0.18
        clear_lateral_m = robot.thigh_lever_m * math.sin(math.radians(max(primary_angle, 20.0))) * 0.16

    return (
        float(max(clear_back_m, 0.0)),
        float(max(clear_lateral_m, 0.0)),
        float(max(clear_orbit_m, 0.0)),
        float(max(clear_drop_m, 0.0)),
    )


def _capacity_base(defense_row: pd.Series) -> float:
    contact_mode = str(defense_row["contact_mode"])
    base = CONTACT_CAPACITY_BASE.get(contact_mode, 0.70)
    role = str(defense_row["defense_role"])
    if role == "fallback_mitigation":
        base *= 0.86
    elif role == "emergency_transition":
        base *= 0.74
    return base


def _absorb_base(defense_row: pd.Series) -> float:
    contact_mode = str(defense_row["contact_mode"])
    base = CONTACT_ABSORB_BASE.get(contact_mode, 0.50)
    if str(defense_row["defense_role"]) == "fallback_mitigation":
        base = max(base, 0.60)
    return base


def _compute_defense_feature(
    defense_id: str,
    raw_map: dict[str, pd.Series],
    robot: RobotParams,
    segment_map: dict[str, dict[str, Any]],
    support_mode_map: dict[str, dict[str, Any]],
    j_ref: float,
    e_ref: float,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if defense_id in cache:
        return cache[defense_id]

    row = raw_map[defense_id]
    sequence_ids = [item for item in str(row["sequence_ids"]).split("|") if item]
    if sequence_ids:
        child_features = [
            _compute_defense_feature(item, raw_map, robot, segment_map, support_mode_map, j_ref, e_ref, cache)
            for item in sequence_ids
        ]
        child_times = [float(item["exec_time_def"]) for item in child_features]
        exec_time = max(0.05, float(sum(child_times) - Q2_MODEL_ASSUMPTIONS["combo_overlap_discount_s"]))
        mobility_cost = min(1.6, _weighted_average([float(item["mobility_cost"]) for item in child_features], child_times) + 0.05)
        support_mass = _weighted_average([float(item["effective_support_mass"]) for item in child_features], child_times)
        absorb_base = _weighted_average([float(item["absorb_base"]) for item in child_features], child_times)
        j_cap = _weighted_average([float(item["J_cap"]) for item in child_features], child_times)
        e_cap = _weighted_average([float(item["E_cap"]) for item in child_features], child_times)
        clear_back_m = max(float(item.get("clear_back_m", 0.0)) for item in child_features)
        clear_lateral_m = max(float(item.get("clear_lateral_m", 0.0)) for item in child_features)
        clear_orbit_m = max(float(item.get("clear_orbit_m", 0.0)) for item in child_features)
        clear_drop_m = max(float(item.get("clear_drop_m", 0.0)) for item in child_features)
    else:
        primary_angle = 0.0 if pd.isna(row["primary_angle_deg"]) else float(row["primary_angle_deg"])
        secondary_angle = 0.0 if pd.isna(row["secondary_angle_deg"]) else float(row["secondary_angle_deg"])
        motion_time = max(_motion_time(primary_angle, robot), _motion_time(secondary_angle, robot))
        exec_time = float(row["official_time_s"]) if not pd.isna(row["official_time_s"]) else motion_time + _support_delay(str(row["support_mode_def"]))

        active_segments = list(row["active_segments_json"])
        support_segments = list(row["support_segments_json"])
        active_mass = _segment_mass(active_segments, segment_map)
        support_mass = _segment_mass(support_segments, segment_map)
        if math.isclose(support_mass, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            support_mass = max(active_mass * 0.65, robot.total_mass_kg * 0.12)
        support_mass *= SUPPORT_MODE_MASS_FACTOR.get(str(row["support_mode_def"]), 1.0)

        support_row = support_mode_map[_canonical_support_mode(str(row["support_mode_def"]))]
        dominant_angle = max(primary_angle, secondary_angle) + Q2_MODEL_ASSUMPTIONS["mobility_angle_mix"] * min(primary_angle, secondary_angle)
        mean_length = _segment_mean_length(active_segments or support_segments, segment_map)
        com_shift = (
            Q2_MODEL_ASSUMPTIONS["mobility_scale"]
            * (active_mass / robot.total_mass_kg)
            * max(mean_length, 0.10)
            * math.sin(math.radians(max(dominant_angle, 1.0)) / 2.0)
            * PLANE_LENGTH_FACTOR.get(str(row["support_mode_def"]).replace("_support", ""), 1.0)
            * (1.0 + Q2_MODEL_ASSUMPTIONS["mobility_plane_factor"] * (1.0 - float(support_row["support_margin_ratio"])))
        )
        support_margin = robot.stable_margin_m * float(support_row["support_margin_ratio"])
        mobility_cost = 0.0 if support_margin <= 1e-9 else min(1.8, com_shift / support_margin)
        clear_back_m, clear_lateral_m, clear_orbit_m, clear_drop_m = _compute_clearance_components(row, robot)

        base_capacity = _capacity_base(row)
        support_ratio = np.clip(
            math.sqrt(max(support_mass, 1e-9) / (robot.total_mass_kg * Q2_MODEL_ASSUMPTIONS["primary_support_mass_ratio"])),
            0.65,
            1.35,
        )
        j_cap = float(base_capacity * support_ratio * j_ref)
        e_cap = float(base_capacity * support_ratio * (1.05 if str(row["contact_mode"]) in {"soft", "ground", "passive"} else 0.95) * e_ref)
        absorb_base = float(np.clip(_absorb_base(row), 0.0, 0.95))

    postdef_delay = COUNTER_DELAY_MAP.get(str(row["counter_readiness_tag"]), 0.12)
    support_row = support_mode_map[_canonical_support_mode(str(row["support_mode_def"]))]
    support_margin_mode = robot.stable_margin_m * float(support_row["support_margin_ratio"])
    force_capacity_factor = 0.0 if math.isclose(j_ref, 0.0, rel_tol=1e-9, abs_tol=1e-9) else j_cap / j_ref
    contact_stiffness_ratio = 1.0 - absorb_base
    raw_force_input = np.nan if pd.isna(row["force_capacity_factor_input"]) else float(row["force_capacity_factor_input"])
    raw_stiffness_input = np.nan if pd.isna(row["contact_stiffness_ratio_input"]) else float(row["contact_stiffness_ratio_input"])
    feature = {
        "defense_id": defense_id,
        "defense_name": str(row["defense_name"]),
        "defense_category": str(row["category"]),
        "category": str(row["category"]),
        "defense_role": str(row["defense_role"]),
        "is_primary_candidate": int(row["is_primary_candidate"]),
        "closure_group": str(row["closure_group"]),
        "entry_state": str(row["entry_state"]),
        "exit_state": str(row["exit_state"]),
        "coverage_direction": str(row["coverage_direction"]),
        "coverage_height": str(row["coverage_height"]),
        "coverage_range": str(row["coverage_range"]),
        "coverage_target_zone": str(row["coverage_target_zone"]),
        "contact_mode": str(row["contact_mode"]),
        "support_mode_def": str(row["support_mode_def"]),
        "route_type": str(row["route_type"]),
        "primary_joint_group": str(row["primary_joint_group"]),
        "primary_angle_deg": 0.0 if pd.isna(row["primary_angle_deg"]) else float(row["primary_angle_deg"]),
        "secondary_joint_group": "" if pd.isna(row["secondary_joint_group"]) else str(row["secondary_joint_group"]),
        "secondary_angle_deg": 0.0 if pd.isna(row["secondary_angle_deg"]) else float(row["secondary_angle_deg"]),
        "sequence_ids": "|".join(sequence_ids),
        "exec_time_def": float(exec_time),
        "effective_support_mass": float(support_mass),
        "balance_cost_def": float(mobility_cost * support_margin_mode),
        "mobility_cost": float(mobility_cost),
        "J_cap": float(j_cap),
        "E_cap": float(e_cap),
        "absorb_base": float(absorb_base),
        "clear_back_m": float(clear_back_m),
        "clear_lateral_m": float(clear_lateral_m),
        "clear_orbit_m": float(clear_orbit_m),
        "clear_drop_m": float(clear_drop_m),
        "postdef_delay_s": float(postdef_delay),
        "counter_readiness_tag": str(row["counter_readiness_tag"]),
        "force_capacity_factor": float(force_capacity_factor),
        "force_capacity_factor_input": raw_force_input,
        "force_capacity_factor_delta": np.nan if pd.isna(raw_force_input) else float(force_capacity_factor - raw_force_input),
        "contact_stiffness_ratio": float(contact_stiffness_ratio),
        "contact_stiffness_ratio_input": raw_stiffness_input,
        "contact_stiffness_ratio_delta": np.nan if pd.isna(raw_stiffness_input) else float(contact_stiffness_ratio - raw_stiffness_input),
        "defense_source_type": str(row["source_type"]),
        "defense_description": str(row["description"]),
        "force_capacity_basis": str(row["force_capacity_basis"]),
        "stiffness_basis": str(row["stiffness_basis"]),
        "active_segments_json": json.dumps(list(row["active_segments_json"]), ensure_ascii=True),
        "support_segments_json": json.dumps(list(row["support_segments_json"]), ensure_ascii=True),
        "defense_stable_margin": float(support_margin_mode),
    }
    cache[defense_id] = feature
    return feature


def build_defense_feature_table(
    defense_actions: pd.DataFrame,
    robot: RobotParams,
    segments: pd.DataFrame,
    support_modes: pd.DataFrame,
    attack_catalog: pd.DataFrame,
) -> pd.DataFrame:
    segment_map = segments.set_index("segment_id").to_dict("index")
    support_mode_map = support_modes.set_index("support_mode").to_dict("index")
    raw_map = {str(row["defense_id"]): row for _, row in defense_actions.iterrows()}

    missing_support_modes = {
        support_mode
        for support_mode in defense_actions["support_mode_def"].astype(str)
        if _canonical_support_mode(support_mode) not in support_mode_map
    }
    if missing_support_modes:
        raise ValueError(f"undefined support_mode_def found: {sorted(missing_support_modes)}")

    segment_ids = set(segment_map)
    for _, row in defense_actions.iterrows():
        for column in JSON_LIST_COLUMNS:
            invalid_segments = sorted(set(str(item) for item in row[column]) - segment_ids)
            if invalid_segments:
                raise ValueError(f"{row['defense_id']} references unknown segments in {column}: {invalid_segments}")
        sequence_ids = [item for item in str(row["sequence_ids"]).split("|") if item]
        for sequence_id in sequence_ids:
            if sequence_id not in raw_map:
                raise ValueError(f"{row['defense_id']} references missing sequence child {sequence_id}")

    j_ref = float(attack_catalog["impact_impulse"].median())
    e_ref = float(attack_catalog["impact_kinetic"].median())
    cache: dict[str, dict[str, Any]] = {}
    records = [
        _compute_defense_feature(defense_id, raw_map, robot, segment_map, support_mode_map, j_ref, e_ref, cache)
        for defense_id in defense_actions["defense_id"].tolist()
    ]
    feature_table = pd.DataFrame(records)
    feature_table["force_capacity_factor_audit"] = np.where(
        feature_table["force_capacity_factor_delta"].abs().fillna(0.0) <= 0.12,
        "OK",
        np.where(feature_table["force_capacity_factor_delta"].isna(), "not_applicable", "revisit"),
    )
    feature_table["contact_stiffness_ratio_audit"] = np.where(
        feature_table["contact_stiffness_ratio_delta"].abs().fillna(0.0) <= 0.15,
        "OK",
        np.where(feature_table["contact_stiffness_ratio_delta"].isna(), "not_applicable", "revisit"),
    )
    return feature_table.sort_values("defense_id").reset_index(drop=True)


def _lookup_route_rule(
    attack_row: pd.Series,
    defense_row: pd.Series,
    route_advantage: pd.DataFrame,
) -> tuple[float, float]:
    route_type = str(defense_row["route_type"])
    if route_type not in NON_DIRECT_ROUTES:
        return 1.0, 0.0

    candidates = route_advantage[route_advantage["route_type"] == route_type].copy()
    default_bonus, default_clearance = DEFAULT_ROUTE_RULE.get(route_type, (0.55, 0.0))
    if candidates.empty:
        return float(default_bonus), float(default_clearance)

    best_bonus = float(default_bonus)
    best_clearance = float(default_clearance)
    best_specificity = -1
    for _, candidate in candidates.iterrows():
        specificity = 0
        matched = True
        for field in ["trajectory_type", "contact_mode_attack", "direction_tag", "range_tag"]:
            rule_value = str(candidate[field])
            pair_value = str(attack_row[field])
            if rule_value == "any":
                continue
            if rule_value != pair_value:
                matched = False
                break
            specificity += 1
        if matched and specificity >= best_specificity:
            best_specificity = specificity
            best_bonus = float(candidate["route_bonus"])
            best_clearance = float(candidate["clearance_need_ratio"])
    return float(best_bonus), float(best_clearance)


def _geo_components(attack_row: pd.Series, defense_row: pd.Series) -> tuple[float, float, float, float]:
    direction_score = _tag_score(str(attack_row["direction_tag"]), str(defense_row["coverage_direction"]), DIRECTION_NEIGHBORS)
    height_score = _tag_score(str(attack_row["height_tag"]), str(defense_row["coverage_height"]), HEIGHT_NEIGHBORS)
    range_score = _range_score(str(attack_row["range_tag"]), str(defense_row["coverage_range"]))
    zone_score = _zone_score(str(attack_row["target_zone"]), str(defense_row["coverage_target_zone"]))
    return direction_score, height_score, range_score, zone_score


def _clear_distance_for_route(row: pd.Series) -> float:
    route_type = str(row["route_type"])
    if route_type == "evade_back":
        return float(row["clear_back_m"])
    if route_type == "evade_lateral":
        return float(row["clear_lateral_m"])
    if route_type == "evade_orbit":
        return float(row["clear_orbit_m"])
    if route_type == "evade_duck":
        return float(row["clear_drop_m"])
    if route_type == "balance_step":
        return 0.8 * max(float(row["clear_back_m"]), float(row["clear_lateral_m"]))
    if route_type == "combo_lateral_parry":
        return float(row["clear_lateral_m"])
    if route_type == "combo_duck_slip":
        return max(float(row["clear_lateral_m"]), float(row["clear_drop_m"]))
    if route_type == "combo_block_retreat":
        return max(float(row["clear_back_m"]), float(row["clear_lateral_m"]))
    if route_type in {"yield_soft", "ground_hold", "ground_transfer"}:
        return 1.0
    return 0.0


def _counter_candidates(
    attack_catalog: pd.DataFrame,
    counter_window: float,
    exit_state: str,
) -> tuple[str, str, str, float, float]:
    if counter_window <= Q2_MODEL_ASSUMPTIONS["counter_window_ready_threshold_s"]:
        return "", "", "", 0.0, 0.0
    if exit_state in {"fallen", "ground_guard", "recovering"}:
        return "", "", "", 0.0, 0.0

    candidates = attack_catalog[
        (attack_catalog["exec_time"] <= counter_window + Q2_MODEL_ASSUMPTIONS["counter_exec_tolerance_s"])
        & (attack_catalog["conditional_flag"] == 0)
        & (attack_catalog["trigger_state"] == "standing")
    ].copy()
    if candidates.empty:
        return "", "", "", 0.0, 0.0

    candidates["counter_priority"] = (
        0.45 * candidates["score_potential"]
        + 0.25 * candidates["J_attack_norm"]
        + 0.20 * (1.0 - _normalize_series(candidates["work_cost"]))
        + 0.10 * (1.0 - candidates["fall_risk"])
    )
    candidates = candidates.sort_values(
        by=["counter_priority", "score_potential", "impact_impulse", "exec_time"],
        ascending=[False, False, False, True],
    ).head(3)
    best = candidates.iloc[0]
    execute_ratio = min(1.0, counter_window / float(best["exec_time"]))
    return (
        str(best["action_id"]),
        "|".join(candidates["action_id"].astype(str).tolist()),
        "|".join(candidates["action_name"].astype(str).tolist()),
        float(execute_ratio),
        float(best["tau_norm"]),
    )


def build_pair_matrix(
    attack_catalog: pd.DataFrame,
    defense_features: pd.DataFrame,
    route_advantage: pd.DataFrame,
    robot: RobotParams,
) -> pd.DataFrame:
    attacks = attack_catalog.copy()
    attacks["join_key"] = 1
    defenses = defense_features.copy()
    defenses["join_key"] = 1
    pair_matrix = attacks.merge(defenses, on="join_key", how="inner").drop(columns="join_key")

    s_j = max(float(attack_catalog["impact_impulse"].median()) * Q2_MODEL_ASSUMPTIONS["load_sigmoid_ratio"], 1e-6)
    s_e = max(float(attack_catalog["impact_kinetic"].median()) * Q2_MODEL_ASSUMPTIONS["load_sigmoid_ratio"], 1e-6)
    reaction_scale = Q2_MODEL_ASSUMPTIONS["reaction_sigmoid_scale_s"]

    records: list[dict[str, Any]] = []
    for _, row in pair_matrix.iterrows():
        direction_score, height_score, range_score, zone_score = _geo_components(row, row)
        attack_entry_state = str(row["attack_entry_state"])
        defense_entry_state = str(row["entry_state"])
        defense_role = str(row["defense_role"])
        state_feasible = defense_entry_state in {attack_entry_state, "any"}
        active_primary_feasible = (
            state_feasible
            and bool(int(row["is_primary_candidate"]))
            and defense_role in {"active_primary", "active_combo"}
        )
        if attack_entry_state == "fallen":
            active_primary_feasible = False
        fallback_feasible = state_feasible and defense_role in {"fallback_mitigation", "emergency_transition"}
        ground_feasible = state_feasible and defense_role == "ground_only"
        recovery_feasible = state_feasible and defense_role == "recovery_only"
        direct_role_feasible = active_primary_feasible or fallback_feasible or ground_feasible or recovery_feasible
        geo_score = 0.35 * direction_score + 0.25 * height_score + 0.20 * range_score + 0.20 * zone_score
        if not state_feasible:
            geo_score = 0.0

        p_react = _sigmoid((float(row["t_contact"]) - float(row["exec_time_def"])) / reaction_scale)
        p_route, clearance_need_ratio = _lookup_route_rule(row, row, route_advantage)
        p_load_j = _sigmoid((float(row["J_cap"]) - float(row["impact_impulse"])) / s_j) if float(row["J_cap"]) > 0.0 else 0.0
        p_load_e = _sigmoid((float(row["E_cap"]) - float(row["impact_kinetic"])) / s_e) if float(row["E_cap"]) > 0.0 else 0.0
        p_load = 0.5 * p_load_j + 0.5 * p_load_e
        d_need = float(clearance_need_ratio) * float(row["contact_reach_m"])
        d_clear = _clear_distance_for_route(row)
        if float(clearance_need_ratio) <= 0.0:
            p_clear = 1.0
        else:
            p_clear = _sigmoid((d_clear - d_need) / Q2_MODEL_ASSUMPTIONS["clearance_sigmoid_scale_m"])

        contact_mode = str(row["contact_mode"])
        if contact_mode == "none":
            p_success = geo_score * p_react * p_route * p_clear
            residual_damage = float(row["H_attack"]) * (1.0 - p_success)
        elif contact_mode == "composite":
            p_success = geo_score * p_react * (0.45 * p_route + 0.35 * p_load + 0.20 * p_clear)
            absorb_pair = 0.50 * float(row["absorb_base"]) + 0.30 * p_load + 0.20 * p_clear
            residual_damage = float(row["H_attack"]) * (
                1.0
                - geo_score
                * p_react
                * (0.35 + 0.65 * absorb_pair)
            )
        elif contact_mode == "rigid":
            p_success = geo_score * p_react * p_load
            absorb_pair = 0.35 * float(row["absorb_base"]) + 0.65 * p_load
            residual_damage = float(row["H_attack"]) * (
                1.0 - geo_score * p_react * (0.40 + 0.60 * absorb_pair)
            )
        elif contact_mode == "passive":
            p_success = geo_score * p_react * (0.45 * p_load + 0.55 * float(row["absorb_base"]))
            absorb_pair = 0.55 * float(row["absorb_base"]) + 0.45 * p_load
            residual_damage = float(row["H_attack"]) * (
                1.0 - geo_score * p_react * (0.35 + 0.65 * absorb_pair)
            )
        elif contact_mode == "soft":
            p_success = geo_score * p_react * (0.55 * p_route + 0.45 * float(row["absorb_base"]))
            absorb_pair = 0.80 * float(row["absorb_base"]) + 0.20 * p_load
            residual_damage = float(row["H_attack"]) * (
                1.0 - geo_score * p_react * (0.30 + 0.70 * absorb_pair)
            )
        elif contact_mode == "ground":
            p_success = geo_score * p_react * (0.40 * p_route + 0.60 * float(row["absorb_base"]))
            absorb_pair = 0.70 * float(row["absorb_base"]) + 0.30 * p_load
            residual_damage = float(row["H_attack"]) * (
                1.0 - geo_score * p_react * (0.30 + 0.70 * absorb_pair)
            )
        else:
            p_success = geo_score * p_react * p_load
            absorb_pair = 0.35 * float(row["absorb_base"]) + 0.65 * p_load
            residual_damage = float(row["H_attack"]) * (
                1.0 - geo_score * p_react * (0.40 + 0.60 * absorb_pair)
            )

        p_success = float(np.clip(p_success if direct_role_feasible else 0.0, 0.0, 1.0))
        residual_damage = float(np.clip(residual_damage, 0.0, 1.0))
        p_fall = float(
            np.clip(
                (1.0 - p_success)
                * (0.55 * float(row["R_attack"]) + 0.45 * float(row["H_attack"]))
                * float(row["mobility_cost"]),
                0.0,
                1.0,
            )
        )
        t_postdef = float(row["exec_time_def"]) + float(row["postdef_delay_s"])
        base_window = max(0.0, float(row["t_recover"]) - t_postdef)
        counter_unlock_mode = str(row["counter_unlock_mode"])
        counter_unlock_coeff = float(row["counter_unlock_coeff"])
        if counter_unlock_mode == "immediate":
            counter_window = base_window
        elif counter_unlock_mode == "after_first_break":
            counter_window = base_window * counter_unlock_coeff * p_success
        elif counter_unlock_mode == "after_main_contact":
            counter_window = base_window * counter_unlock_coeff
        elif counter_unlock_mode == "after_full_chain":
            counter_window = 0.0
        else:
            counter_window = 0.0
        best_counter_id, counter_ids, counter_names, execute_ratio, counter_tau_norm = _counter_candidates(
            attack_catalog,
            counter_window,
            str(row["exit_state"]),
        )
        counter_prob_effective = p_success * execute_ratio if best_counter_id else 0.0
        counter_ready = int(counter_window >= Q2_MODEL_ASSUMPTIONS["counter_window_ready_threshold_s"] and bool(best_counter_id))

        records.append(
            {
                **row.to_dict(),
                "state_feasible": bool(state_feasible),
                "active_primary_feasible": bool(active_primary_feasible),
                "fallback_feasible": bool(fallback_feasible),
                "ground_feasible": bool(ground_feasible),
                "recovery_feasible": bool(recovery_feasible),
                "role_feasible": bool(direct_role_feasible),
                "primary_role_feasible": bool(active_primary_feasible),
                "direction_match": float(direction_score),
                "height_match": float(height_score),
                "range_match": float(range_score),
                "zone_match": float(zone_score),
                "p_geo": float(geo_score),
                "geo_covered": bool(geo_score > 0.0),
                "p_react": float(p_react),
                "p_route": float(p_route),
                "d_need": float(d_need),
                "d_clear": float(d_clear),
                "p_clear": float(p_clear),
                "clearance_need_ratio": float(clearance_need_ratio),
                "p_load_j": float(p_load_j),
                "p_load_e": float(p_load_e),
                "p_load": float(p_load),
                "p_success": float(p_success),
                "residual_damage": float(residual_damage),
                "defense_damage": float(residual_damage),
                "p_fall": float(p_fall),
                "base_counter_window": float(base_window),
                "counter_window": float(counter_window),
                "counter_ready": int(counter_ready),
                "counter_action_id": best_counter_id,
                "counter_action_ids": counter_ids,
                "counter_action_names": counter_names,
                "counter_prob_effective": float(counter_prob_effective),
                "counter_prob": float(counter_prob_effective),
                "counter_tau_norm": float(counter_tau_norm),
                "p_block": float(p_success),
                "exit_state_after_defense": str(row["exit_state"]),
            }
        )

    evaluated = pd.DataFrame(records)
    if math.isclose(float(evaluated["counter_window"].max()), 0.0, rel_tol=1e-9, abs_tol=1e-9):
        evaluated["counter_window_norm"] = 0.0
    else:
        evaluated["counter_window_norm"] = evaluated["counter_window"] / float(evaluated["counter_window"].max())
    evaluated["defense_harm_safe"] = 1.0 - evaluated["residual_damage"]
    evaluated["stability_safe"] = 1.0 - evaluated["p_fall"]
    return evaluated
