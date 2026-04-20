# Q2 Writer Guide

## 1. 当前 Q2 的最终定位
当前 Q2 已经从“静态攻防评分表”收口为：

> 基于 Q1 相位化攻击模型的分层攻防响应模型。

它不再只输出单一的防守 Top1，而是输出一条分层响应链：

- 主防守层：`active_primary / active_combo`
- 保底层：`fallback_mitigation / emergency_transition`
- 地面层：`ground_only`
- 恢复层：`recovery_only`

因此，Q2 的结果可以直接服务 Q3/Q4，而不需要后续再重构动作语义。

---

## 2. 当前真实输入文件

### 2.1 直接复用 Q1

- `data/raw/q1_robot_params.csv`
- `data/raw/q1_segment_params.csv`
- `data/raw/q1_support_mode_config.csv`
- `data/raw/q1_action_templates.csv`
- `data/raw/q1_action_phase_templates.csv`
- `data/interim/action_features.csv`

这些文件提供：

- 机器人几何与质量参数
- 10 刚体节段参数
- 动作模板与相位模板
- Q1 输出的攻击物理量与风险量：
  - `impact_impulse`
  - `impact_kinetic`
  - `score_potential`
  - `fall_risk`
  - `exposure_index`
  - `exec_time`
  - `end_velocity_peak`

### 2.2 Q2 自身维护

- `data/raw/q2_attack_semantics.csv`
- `data/raw/q2_attack_response_policy.csv`
- `data/raw/q2_defense_actions.csv`
- `data/raw/q2_route_advantage.csv`
- `data/raw/q2_family_compatibility.csv`

这 5 张表分别负责：

- 攻击语义标签
- 接触相选择、恢复时窗、反击解锁规则
- 防守角色、状态、范围、路线、机制标签、generic scope 惩罚
- 路线优势与清距需求
- 攻击接触族类与防守机制兼容性

---

## 3. 当前运行流程

入口：

- `python scripts/run_q2.py`

执行链路：

1. `scripts/run_q2.py` 调用 `src/q2/pipeline.py`
2. `pipeline.py` 读取 Q1 输入与中间结果，同时读取 Q2 的 5 张原始表
3. `src/q2/model_v1.py` 构建：
   - `attack_catalog`
   - `defense_features`
   - `pair_matrix`
4. `src/q2/evaluate.py` 计算方法一到方法四，并输出分层响应结果
5. `pipeline.py` 保存中间表、调试表、审计表、验收表和图表

---

## 4. 当前代码文件分工

### `src/q2/model_v1.py`

负责：

- 读取并校验 Q2 原始表
- 从 Q1 相位结构中提取接触相与攻击时序
- 构建防守动作的执行时间、承载能力、吸收能力、清距能力
- 计算每个攻防 pair 的：
  - 状态可行性
  - 几何匹配
  - 时序反应
  - 路线优势
  - 清距成功率
  - family 兼容性
  - 剩余损伤
  - 跌倒风险

最关键的函数：

- `build_attack_catalog(...)`
- `build_defense_feature_table(...)`
- `build_pair_matrix(...)`

### `src/q2/evaluate.py`

负责：

- 计算方法一、二、三、四
- 在方法四中分层选取：
  - `active_top1`
  - `fallback_top1`
  - `ground_top1`
  - `recovery_if_needed`
- 输出：
  - `defense_matchup.csv`
  - `counter_chain.csv`
  - `q2_method_summary.csv`

### `src/q2/pipeline.py`

负责：

- 串联整个 Q2
- 输出主结果表、调试表、审计表、验收表
- 调用 `src/q2/plot.py` 生成最终保留图

### `src/q2/plot.py`

负责：

- 只按最终 `active_top1` 口径绘制所有“主防守”图
- 对分层响应图按类别着色
- 输出中文标题、中文坐标轴、中文图例

---

## 5. 当前模型的关键收口点

### 5.1 攻击侧真正由 Q1 驱动

Q2 当前真实使用了 Q1 的：

- 相位结构
- 接触相位
- 接触时刻 `t_contact`
- 恢复时窗 `t_recover`
- 冲量 `impact_impulse`
- 动能 `impact_kinetic`
- 暴露风险 `exposure_index`
- 倒地风险 `fall_risk`
- 支撑切换数 `support_switch_count`
- 旋转复杂度 `rotation_complexity`
- 平移距离 `translation_distance_m`

并进一步派生：

- `contact_phase_no`
- `contact_phase_name`
- `contact_reach_m`
- `contact_load_family`
- `contact_window_type`
- `counter_unlock_audit`
- `attack_response_audit`

### 5.2 `mechanism_tag`

`mechanism_tag` 是当前 Q2 最关键的结构字段之一。
它描述防守机制，而不是只给粗粒度 category。

例如：

- `D01 -> rigid_high_cross`
- `D05 -> rigid_close_clamp`
- `D06 -> evade_lateral`
- `D10 -> evade_orbit`
- `D15 -> balance_soft_yield`
- `D19 -> ground_hold`
- `D20 -> combo_lateral_parry`

这一步让 Q2 能够真正区分：

- 刚性格挡
- 近身钳制
- 横向规避
- 后撤脱距
- 环绕规避
- 柔顺卸力
- 地面压制
- 组合式防守

### 5.3 `q2_family_compatibility.csv`

这是 Q2 final 版最关键的新表之一。

固定的攻击接触族类为：

- `upper_terminal`
- `lower_terminal`
- `sweep_line`
- `torso_drive`
- `sequence_chain`
- `ground_conditional`

对每一个 `(contact_load_family, mechanism_tag)`，表中给出：

- `family_success_factor`
- `family_damage_factor`

这保证了：

- 高位格挡不会再对腿法和冲撞拿到不合理高分
- `sequence_chain` 会优先匹配 combo 防守机制
- `ground_conditional` 不会再被 standing defense 占位

### 5.4 `generic_scope_penalty` 已数据化

原来写死在 `evaluate.py` 里的 generic penalty 已经迁回数据表：

- `q2_defense_actions.csv`

新增字段：

- `generic_scope_level`
- `generic_scope_penalty`

当前关键设置：

- `D14 -> generic, 0.06`
- `D16 -> semi_generic, 0.02`
- `D15 -> local, 0.00`

因此 fallback 惩罚已经不再是代码暗箱，而是数据层显式配置。

---

## 6. 当前 pair-level 核心指标

`defense_pair_scores.csv` 当前至少保留了以下关键列：

- `state_feasible`
- `active_primary_feasible`
- `fallback_feasible`
- `ground_feasible`
- `recovery_feasible`

- `direction_match`
- `height_match`
- `range_match`
- `zone_match`
- `p_geo`

- `p_react`
- `p_route`
- `route_rule_hit`
- `route_rule_specificity`
- `route_rule_specificity_note`

- `clearance_need_ratio`
- `d_need`
- `d_clear`
- `p_clear`

- `p_load_j`
- `p_load_e`
- `p_load`

- `p_family`
- `family_rule_hit`
- `family_damage_factor`

- `geo_cap_applied`
- `geo_cap_reason`
- `pair_audit_flag`

- `success_core`
- `effective_absorb`
- `p_success`
- `residual_damage`
- `p_fall`
- `base_counter_window`
- `counter_window`

这使 Q2 已经不再是黑箱打分，而是可以逐项追踪每个攻防对为什么成立或失效。

---

## 7. 当前图层口径

这是这轮收口最重要的改动之一。
现在所有标题里写“主防守”的图：

- 一律只认最终 `active_top1`
- 不再从 `evaluated_pairs` 二次 groupby 抢 Top1

因此，图层和表层已经统一。

### 当前最终保留图

建议保留这 6 张：

1. `q2_method_comparison.png`
   - Q2 四种方法主防守 Top1 对比图
2. `q2_layered_response_overview.png`
   - Q2 分层响应总览图
3. `q2_surface.png`
   - Q2 主防守 Top1 成功率—风险散点图
4. `q2_utility_matrix.png`
   - Q2 主防守层效用矩阵
5. `q2_waterfall.png`
   - Q2 攻击动作到主防守 Top1 的映射图
6. `q2_parallel.png`
   - Q2 主防守 Top1 指标平行对比图

### 关于 `q2_parallel.png`

它已经重新接回 final 输出链路，不再是旧版本遗留图。
当前含义为：

- Q2 主防守 Top1 指标平行对比图
- 横向比较成功率、剩余伤害安全性、反击窗口、倒地风险安全性
- 用于辅助解释不同主防守动作之间的性能差异

---

## 8. 当前分层输出契约

`q2_method_summary.csv` 现在已经按 final 契约输出，不再是单一 Top1 表。

关键字段：

- `method1_top1_defense`
- `method2_top1_defense`
- `method3_top1_defense`

- `method4_has_active`
- `method4_has_ground`

- `method4_active_top1`
- `method4_active_role`
- `method4_active_category`
- `method4_active_score`
- `method4_active_confidence`

- `method4_fallback_top1`
- `method4_fallback_role`
- `method4_fallback_category`
- `method4_fallback_score`

- `method4_ground_top1`
- `method4_ground_role`
- `method4_ground_category`
- `method4_ground_score`

- `method4_recovery_if_needed`
- `method4_recovery_role`
- `method4_recovery_category`
- `method4_recovery_score`

- `method4_note`

其中：

- `method4_active_confidence` 当前按 `primary_score` 分级：
  - `strong`: `score >= 0.10`
  - `medium`: `0.03 <= score < 0.10`
  - `weak`: `0 < score < 0.03`

因此：

- 对常规 standing 场景，重点看 `active + fallback`
- 对 fallen 场景，重点看 `ground + recovery`
- `recovery` 只在确实需要恢复链时才出现

---

## 9. 当前关键结果解释

### 9.1 A11 已经收口

此前 `A11` 容易出现 `active = NA`。
当前收口后：

- `A11 -> method4_active_top1 = D22`
- `A11 -> active_role = active_combo`
- `A11 -> active_confidence = weak`

这说明：

- 五连踢不再被误压成“无主防守”
- combo 机制已经能够正确承接 `sequence_chain`
- 但它仍属于“有解但强度不高”的主防守，不应被误读为强推荐

### 9.2 A13 语义已固定

当前 A13 的最终语义是：

- `method4_active_top1 = D05`
- `method4_fallback_top1 = D16`
- `method4_note = standing_with_fallback`

即：

> A13 是“攻击方倒地后的反击动作”，但防守方并不必然处于倒地状态。因此 Q2 中应按“站立防守应对倒地反击”处理，主防守选择钳制格挡，保底响应选择步点调整。

这次修正的关键是区分：

- `attack_entry_state`：攻击方发动动作时的状态
- `defender_entry_state`：防守方进入防守动作时的状态

此前把 A13 的攻击方倒地状态误当成防守方倒地状态，会导致站立防守被全部排除，只剩 D19/D18 地面链，这是不符合题意的。

这是当前 final 版的明确口径。

### 9.3 主防守层不再被单一动作垄断

当前 `q2_active_distribution.csv` 显示：

- `D10` 出现 3 次
- `D05` 出现 3 次
- `D03` 出现 2 次
- `D01`、`D04`、`D20`、`D22`、`D06` 也都进入了主防守层

这意味着：

- 主防守层已经包含 `block / evade / combo`
- 不再被单一防守动作一统

---

## 10. 当前审计与验收

### 10.1 审计表

当前新增并保留：

- `q2_zero_tie_audit.csv`
- `q2_family_audit.csv`
- `q2_active_distribution.csv`
- `q2_rule_coverage_audit.csv`
- `q2_rule_audit_summary.csv`
- `q2_top1_audit_summary.csv`
- `q2_acceptance_checks.csv`

### 10.2 规则审计

当前 `q2_rule_audit_summary.csv` 统计：

- 总 pair 数
- family rule 命中数
- route rule 命中数
- geo cap 触发数
- missing family / route rule 数
- 非 normal 审计条目数

当前 `q2_top1_audit_summary.csv` 只针对：

- `active_top1`
- `fallback_top1`
- `ground_top1`

统计：

- Top1 中 family rule 命中占比
- Top1 中 route rule 命中占比
- Top1 中 geo cap 占比
- Top1 中 `pair_audit_flag != normal` 的占比

### 10.3 当前自动验收

`q2_acceptance_checks.csv` 当前全部通过，包括：

1. `A13_has_standing_response = True`
2. `A13_not_forced_to_ground_layer = True`
3. `A05_not_D02 = True`
4. `A12_not_D01 = True`
5. `A11_active_combo_not_blank = True`
6. `active_not_monopolized = True`
7. `active_category_diversity = True`
8. `posture_balance_outside_active = True`
9. `ground_recovery_layer_isolated = True`
10. `final_active_rule_coverage = True`
11. `A13_baseline_methods_not_blank = True`

---

## 11. 当前与 Q3/Q4 的接口关系

这版 Q2 已经能稳定向 Q3/Q4 提供：

- 主防守动作
- fallback 保底动作
- ground / recovery 分层动作
- 反击窗口 `counter_window`
- 反击候选动作
- 每个攻防 pair 的成功率、剩余损伤、跌倒风险

此外，这一轮已经顺序重跑：

- `python scripts/run_q2.py`
- `python scripts/run_q3.py`

Q3 通过，说明当前 Q2 输出接口与 Q3 兼容。

---

## 12. 论文写作建议

### 12.1 建议正文主线

建议按下面顺序写 Q2：

1. 在 Q1 相位化攻击模型基础上构建攻防响应层
2. 将防守动作划分为主防守、保底、地面、恢复四层
3. 用几何匹配、时序匹配、路线优势、清距能力、family compatibility 建立 pair-level 模型
4. 用四种方法对比
5. 以方法四作为最终主模型，输出分层响应链

### 12.2 建议强调的两点

第一：

Q2 不是在“给防守动作打分”，而是在“对每个攻击动作构造一条闭环响应链”。

第二：

Q2 不是独立拍脑袋建模，而是明确继承了 Q1 的相位时序和攻击物理量。

### 12.3 当前最适合写进正文的结果表述

- 直线拳法倾向高位刚性格挡或近身肘架
- 宽弧和旋转攻击偏向 lateral / orbit 类规避
- 腿法和冲撞更依赖 close clamp、soft yield、step reset
- 组合技更偏好 `active_combo`
- 倒地反击属于攻击方条件动作，防守方仍按站立防守建模，优先用钳制格挡限制后续反击链

---

## 13. 第二问最终答案

第二问的最终结果可以概括为：

> 对每一种攻击动作，模型给出一个第一时间应对的主防守动作，同时给出一个保底响应动作。主防守用于拦截、闪避或限制攻击，保底响应用于在主防守未完全奏效时降低剩余伤害和倒地风险。

### 13.1 十三种攻击动作的最佳主防守

| 攻击动作 | 最佳主防守 | 防守类型 | 保底响应 | 结果解释 |
|---|---|---|---|---|
| 左右直拳 | 十字格挡 | 格挡 | 护头防御 | 直拳是正面直线攻击，十字格挡能直接封住头胸正面区域，稳定性最高。 |
| 左右勾拳 | 肘挡 | 格挡 | 重心补偿 | 勾拳从侧方弧线进入，肘挡更适合近距离护头护胸，能减少侧向冲击。 |
| 组合拳 | 肘挡 | 格挡 | 卸力缓冲 | 组合拳连续性强，肘挡能维持近身防护结构，保底用卸力缓冲吸收连续冲击。 |
| 摆拳 | 滑步环绕 | 闪避 | 步点调整 | 摆拳横向扫击幅度大，硬挡风险较高，滑步环绕更适合避开攻击弧线。 |
| 前踢 | 钳制格挡 | 格挡 | 步点调整 | 前踢是直线腿法，钳制格挡可以限制腿部或后续连续动作。 |
| 侧踢 | 滑步环绕 | 闪避 | 步点调整 | 侧踢爆发力强，正面承受代价高，滑步环绕能脱离主要受力方向。 |
| 回旋踢或转身踢 | 滑步环绕 | 闪避 | 步点调整 | 回旋类攻击轨迹大、冲击强，绕开攻击弧线比正面格挡更安全。 |
| 低扫腿 | 下压格挡 | 格挡 | 步点调整 | 低扫腿主要攻击下肢，下压格挡与低位防护最匹配。 |
| 膝撞 | 钳制格挡 | 格挡 | 卸力缓冲 | 膝撞是近身直线冲击，钳制格挡可以控制近身动作并降低连续攻击风险。 |
| 拳腿组合 | 闪挡反组合 | 组合防守 | 步点调整 | 拳腿组合包含多阶段攻击，单一防守动作不足，组合防守更适合连续响应。 |
| 五连踢 | 挡撤绕组合 | 组合防守 | 卸力缓冲 | 五连踢连续性强，挡撤绕可以先阻断、再拉开距离并重新组织站位。 |
| 冲撞 | 左右侧闪 | 闪避 | 卸力缓冲 | 冲撞属于身体前冲，正面对抗容易失衡，侧闪能快速离开冲击线。 |
| 倒地反击 | 钳制格挡 | 格挡 | 步点调整 | 倒地反击是低位或近身条件攻击，钳制格挡可以限制其后续反击链。 |

### 13.2 防守类型分布

| 防守类型 | 覆盖攻击数量 | 对应攻击 |
|---|---:|---|
| 格挡类 | 7 | 左右直拳、左右勾拳、组合拳、前踢、低扫腿、膝撞、倒地反击 |
| 闪避类 | 4 | 摆拳、侧踢、回旋踢或转身踢、冲撞 |
| 组合防守类 | 2 | 拳腿组合、五连踢 |

### 13.3 结果解读

当前结果体现出三条规律：

- 对直线、近身、低位或条件反击类攻击，模型更倾向于选择格挡或钳制，因为这类攻击的接触窗口明确，直接限制对方动作更有效。
- 对大幅度、高冲击、旋转或冲撞类攻击，模型更倾向于选择闪避，因为硬挡会带来较高的失衡和剩余伤害风险。
- 对连续攻击，模型更倾向于组合防守，因为单一动作难以覆盖多个接触相，必须通过“挡、撤、绕”或“闪、挡、反”形成连续闭环。

此外，多数攻击动作都配有平衡类保底响应，说明防守结果不仅取决于是否挡住攻击，还取决于防守后能否维持站立姿态。这一点与题目要求的“完整防御闭环”一致。

---

## 14. 一句话总结

Q2 当前最终版已经完成从“静态攻防打分表”到“Q1 相位化攻击驱动的分层攻防响应模型”的收口，并且已经通过：

- 结构验收
- 规则覆盖率验收
- 边界动作验收
- Q3 兼容性验证
