# Q1 写作者说明

## 1. 本问现在在做什么

Q1 的目标已经不是“简单给 13 个动作排一个名次”，而是：

> 基于统一的 10 刚体简化模型，为 13 个攻击动作提取一组可复用的物理特征，并在此基础上完成四种方法的排序比较。

这意味着 Q1 的核心交付分成两层：

- 第一层：供 Q2/Q3/Q4 继续调用的动作物理特征
- 第二层：只服务 Q1 展示与对比的排序结果

因此，Q1 现在是全题的动作特征底座，不只是一个单独排名题。

---

## 2. 当前运行逻辑

Q1 的运行入口是：

- [scripts/run_q1.py](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/scripts/run_q1.py:1)

主流程由：

- [src/q1/pipeline.py](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/src/q1/pipeline.py:1)

统一串联。

实际执行顺序如下：

1. 读取机器人官方参数表、10 刚体节段表、支撑模式表、动作数值源表、动作模板表、动作相位表。
2. 合并 `q1_attack_actions.csv` 与 `q1_action_templates.csv`，形成完整动作定义。
3. 调用 `validate_q1_configuration()` 做最终一致性校验。
4. 基于 10 刚体模型、动作模板和相位模板计算动作物理特征。
5. 在正式计算前检查 `active_joint_count` 是否为合法正整数且不超过总关节数 23。
6. 调用四种评价方法完成打分与排序。
7. 输出 `action_features.csv`。
8. 输出参数登记表、运行元数据和图表。

一句话概括：

> 原始参数表 + 模板表 -> 统一特征提取 -> 四方法评价 -> 中间结果表与图表输出

---

## 3. 当前输入文件

Q1 当前使用 6 张输入表。

### 3.1 机器人基础参数

- [q1_robot_params.csv](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/raw/q1_robot_params.csv:1)

作用：

- 存放机器人总体尺寸、质量、关节力矩、转速、减速比、宽度等官方或直接整理参数。

### 3.2 10 刚体节段参数

- [q1_segment_params.csv](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/raw/q1_segment_params.csv:1)

作用：

- 定义 10 个刚体节段的质量、长度、质心位置、父子连接关系。
- 这是当前 Q1 动力学近似的结构骨架。

### 3.3 支撑模式配置

- [q1_support_mode_config.csv](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/raw/q1_support_mode_config.csv:1)

作用：

- 定义双足、单足、旋转单足、恢复过渡等支撑模式的稳定裕度折减与 ZMP 偏置。

### 3.4 动作数值源表

- [q1_attack_actions.csv](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/raw/q1_attack_actions.csv:1)

作用：

- 存放每个动作的数值层输入，如：
  - `coordination_efficiency`
  - `active_joint_count`
  - `theta_total_deg`
  - `exec_time_s`
  - `p_target_geo`
- 这些字段主要是动作层建模输入，不等于官方动作数据库原始测量值。

### 3.5 动作结构模板表

- [q1_action_templates.csv](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/raw/q1_action_templates.csv:1)

作用：

- 存放动作结构属性，如：
  - `support_mode`
  - `phase_count`
  - `conditional_flag`
  - `main_plane`
  - `range_tag`
  - `rotation_complexity`
  - `support_switch_count`
  - 各类节段集合 JSON
  - 各类权重 JSON

### 3.6 动作相位模板表

- [q1_action_phase_templates.csv](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/raw/q1_action_phase_templates.csv:1)

作用：

- 对组合技、多阶段动作、特殊动作进行相位拆解。
- 当前重点支持：
  - A03 组合拳
  - A07 回旋踢
  - A10 拳腿组合
  - A11 五连踢
  - A12 冲撞
  - A13 倒地反击

---

## 4. 代码文件分工

### 4.1 特征层

- [src/q1/model_v1.py](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/src/q1/model_v1.py:1)

作用：

- 读取并校验全部 Q1 输入表
- 合并动作数值源表与结构模板表
- 读取相位模板
- 建立 10 刚体参数对象
- 计算动作物理特征

其中最重要的逻辑包括：

- `load_attack_actions()`
- `load_action_templates()`
- `load_action_phase_templates()`
- `load_segment_params()`
- `load_support_modes()`
- `merge_action_definition_tables()`
- `validate_q1_configuration()`
- `build_feature_table()`

### 4.2 评价层

- [src/q1/evaluate.py](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/src/q1/evaluate.py:1)

作用：

- 在已计算的物理特征基础上，完成四种评价方法打分与排序。

### 4.3 展示层

- [src/q1/plot.py](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/src/q1/plot.py:1)

作用：

- 输出 Q1 图表，供论文展示和结果分析使用。

### 4.4 总流程层

- [src/q1/pipeline.py](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/src/q1/pipeline.py:1)

作用：

- 串联读表、校验、特征计算、评价、保存结果、绘图
- 额外输出参数登记表和运行元数据

### 4.5 独立刚体示意图脚本

- [p1/rigid_body_diagram.py](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/p1/rigid_body_diagram.py:1)
- [p1/rigid_body_layout.py](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/p1/rigid_body_layout.py:1)

作用：

- 独立绘制 Q1 的 10 刚体前视图和侧视图
- 不进入主 pipeline，但服务论文图示与结构表达

---

## 5. 当前模型是什么

Q1 当前可以概括为四层结构：

### 第一层：机器人总体参数层

利用总质量、总高度、腿长、臂展、关节力矩、输出角速度等参数，确定机器人总体尺度与动力能力。

### 第二层：10 刚体节段层

将机器人等效为 10 个节段：

- S01 头颈
- S02 躯干
- S03/S04 左右上臂
- S05/S06 左右前臂手
- S07/S08 左右大腿
- S09/S10 左右小腿足

所有动作特征的统一计算都以这个节段层为骨架。

### 第三层：动作模板层

动作不再按 `action_id` 在代码里逐个硬编码，而是由：

- 数值源表
- 结构模板表
- 相位模板表

共同驱动。

### 第四层：评价层

在统一特征基础上做四种方法对比：

1. 方法一：纯物理冲击排序
2. 方法二：AHP + 熵权混合评价
3. 方法三：模糊综合评价
4. 方法四：风险调整效用模型

其中方法四是主模型，形式为：

`U = B(1 - R) - C`

含义为：

- 收益项 `B`：打击效应 + 得分潜力
- 成本项 `C`：做功代价 + 暴露代价
- 风险项 `R`：跌倒风险

---

## 6. 当前动作特征体系

Q1 当前最重要的是特征体系，而不是某个单独分数。

### 6.1 给 Q2/Q3/Q4 复用的基础字段

这些字段是后续问题最应稳定依赖的接口：

- `action_id`
- `action_name`
- `category`
- `conditional_flag`
- `support_mode`
- `phase_count`
- `impact_impulse`
- `impact_kinetic`
- `score_potential`
- `work_cost`
- `peak_power_proxy`
- `exec_time`
- `exposure_index`
- `com_shift_max`
- `zmp_margin_norm`
- `fall_risk`
- `stable_margin_mode`
- `range_tag`
- `active_joint_count`

兼容字段：

- `joint_count`

说明：

`joint_count` 现在只是为了兼容后续模块保留的别名，真实语义应理解为：

- `active_joint_count`

### 6.2 Q1 专属评价字段

这些字段主要服务 Q1 自己的排序和图表：

- `method1_score`
- `method2_score`
- `method3_score`
- `method4_score`
- `method1_rank`
- `method2_rank`
- `method3_rank`
- `method4_rank`
- `utility`

---

## 7. 参数来源分层

这是 Q1 当前最重要的可信度基础。

### 7.1 官方/赛题直给参数

例如：

- 总高度
- 总质量
- 腿长
- 臂展
- 关节最大力矩
- 电机转速
- 减速比
- 机身宽度
- 关节数量

这些参数主要记录在：

- [q1_robot_params.csv](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/raw/q1_robot_params.csv:1)

### 7.2 由官方参数推导的参数

例如：

- 单臂长度
- 大腿长度
- 小腿足长度
- 输出轴角速度
- 有效角速度
- 基础稳定裕度
- 10 刚体质量和长度

这些参数不是题面逐字给出，但可由官方参数直接计算得到。

### 7.3 模型假设参数

例如：

- `coordination_efficiency`
- `active_joint_count`
- `theta_total_deg`
- `exec_time_s`
- `p_target_geo`
- `support_margin_ratio`
- `zmp_bias_coeff`
- `rotation_complexity`
- `phase_decay`
- `translation_distance_m`
- `dynamic_peak_acc_cap_m_s2`

这些参数不是官方真值，而是为支持统一动作建模所引入的模型输入。

### 7.4 当前如何记录参数来源

Q1 已经输出：

- [q1_parameter_registry.csv](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_parameter_registry.csv:1)

这张表会把参数按：

- `official`
- `derived`
- `model_assumption`
- `legacy_compatibility`

进行登记，便于论文和答辩解释。

---

## 8. 最终一致性校验现在检查什么

Q1 现在在正式建模前会先执行：

- `validate_q1_configuration()`

主要检查内容包括：

- 节段质量比例和是否为 1
- 左右对称节段长度与质量是否一致
- 节段树是否正确
- 动作模板中的 `support_mode` 是否存在
- 相位模板中的 `support_mode_phase` 是否存在
- 模板和相位引用的节段 ID 是否都在节段表中存在
- 权重 JSON 的 key 是否属于 `active_segments_json`
- `active_joint_count` 是否为正整数且不超过总关节数 23
- `phase_count` 与相位表条数是否一致
- 若 `theta_total_deg` 非空，则是否与各相位角度和一致
- 若 `translation_distance_m > 0`，则各相位 `translation_share` 是否和为 1
- 若 `translation_distance_m = 0`，则各相位 `translation_share` 是否全为 0

这一步的作用是把“能跑”升级成“可冻结、可复现、可检查”。

---

## 9. 当前输出文件有哪些

### 9.1 主中间结果

- [action_features.csv](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/interim/action_features.csv:1)

作用：

- 是 Q1 最核心的输出表
- 后续 Q2/Q3/Q4 都直接或间接依赖它

### 9.2 参数来源表

- [q1_parameter_registry.csv](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_parameter_registry.csv:1)

作用：

- 给写作者和答辩使用
- 明确区分哪些参数是官方、推导、假设

### 9.3 运行快照

- [q1_run_metadata.json](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_run_metadata.json:1)

作用：

- 记录本次运行的输入文件、输出文件、Top5 排名、主要参数和时间信息
- 服务复现和冻结版本管理

### 9.4 刚体示意图

- [q1_rigid_body_front.png](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_rigid_body_front.png)
- [q1_rigid_body_front.svg](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_rigid_body_front.svg)
- [q1_rigid_body_side.png](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_rigid_body_side.png)
- [q1_rigid_body_side.svg](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_rigid_body_side.svg)

作用：

- 服务论文图示
- 说明 10 刚体模型结构

### 9.5 图表

- [q1_utility_bar.png](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_utility_bar.png)
- [q1_impact_balance.png](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_impact_balance.png)
- [q1_method_comparison.png](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_method_comparison.png)
- [q1_penalty_curve.png](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_penalty_curve.png)
- [q1_decision_atlas.png](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_decision_atlas.png)
- [q1_sensitivity_heatmap.png](/E:/Python%20Project/6.Practical_project/yt/14/16th_MathorCup/MathorCup2026-B-Robot-Strategy/data/output/q1_sensitivity_heatmap.png)

---

## 10. 当前结果怎么理解

当前主模型方法四的前 5 名为：

1. `A04 swing_punch`
2. `A06 side_kick`
3. `A01 straight_punch`
4. `A02 hook_punch`
5. `A09 knee_strike`

这组结果现在可以视为：

- 基于当前统一模型、统一模板和统一校验下的**综合排序结果**
- 不是“某个动作绝对真理排序”
- 更适合写成“梯队推荐”

补充说明：

- `A12 body_charge` 在本轮最终收口中加入了动态峰值加速度上限 `dynamic_peak_acc_cap_m_s2 = 7.0`，目的是避免极短相位把动态 ZMP 项放大过头。
- 调整后，A12 仍然处于高风险、场景依赖动作区，但其风险量级解释更自然。

更稳妥的论文表达建议为：

### 第一梯队：优先推荐动作

- A04 摆拳
- A06 侧踢
- A01 直拳
- A02 勾拳
- A09 膝撞

### 第二梯队：可作为补充战术动作

- A05 前踢
- A03 组合拳
- A08 低扫腿

### 第三梯队：收益有限或场景依赖较强

- A10 拳腿组合
- A12 冲撞

### 限制使用动作

- A11 五连踢
- A07 回旋踢
- A13 倒地反击

其中：

- A13 应单列说明为条件动作，不应和常规站立攻击完全等价对待。

---

## 11. 当前模型的优点与局限

### 11.1 优点

- 已从动作编号特判升级为模板驱动模型
- 10 刚体结构明确，物理骨架统一
- 数据层、代码层和输出层现在基本对齐
- 参数来源已可追溯
- 输出字段已能稳定供 Q2/Q3/Q4 复用
- 具有最终一致性校验，结果可复现
- 对冲撞类动作的动态风险增加了加速度上限校准，避免平动相位被异常过罚

### 11.2 局限

- 动作层很多参数仍然是模型假设，而非官方逐动作测量值
- 当前动力学仍是简化、半解析近似，不是全多体数值仿真
- `support_segments_json` 与 `inertia_coeff` 当前主要承担结构表达和扩展接口作用，未进入主体高阶数值求解
- 当前排序体现的是“当前主模型下的综合结果”，不是现实比赛中的唯一绝对动作强弱顺序

---

## 12. 写作者推荐叙述主线

最推荐的 Q1 写法主线是：

> 官方参数 -> 10 刚体节段模型 -> 动作数值源表与结构模板表 -> 相位拆解 -> 统一物理特征提取 -> 四方法排序比较 -> 风险调整效用主模型 -> 分层推荐动作集

不建议把 Q1 写成“凭经验给动作打分”。
应强调：

> Q1 的核心贡献不是只给 13 个动作排个名，而是建立一套可服务后续攻防匹配、单场策略优化和资源调度的统一动作物理特征系统。

---

## 13. 给 Q2/Q3/Q4 的接口说明

Q1 当前输出的以下字段是后续问题最值得直接调用的基础输入：

- `impact_impulse`
- `impact_kinetic`
- `score_potential`
- `work_cost`
- `peak_power_proxy`
- `exec_time`
- `exposure_index`
- `com_shift_max`
- `zmp_margin_norm`
- `fall_risk`
- `stable_margin_mode`
- `range_tag`

后续问题中的典型使用方式是：

- Q2：用作攻防匹配中的攻击强度、风险、暴露与节奏输入
- Q3：用作单场 MDP 的攻击收益、能耗、风险基础
- Q4：用作故障/复位/资源使用决策的动作代价与风险输入

---

## 14. 当前封版判断

当前 Q1 已经满足以下条件：

1. 数据表与代码 schema 对齐
2. 参数来源分层可追溯
3. 输出字段已冻结
4. 刚体示意图已独立输出
5. Q1 结果已能稳定服务 Q2

因此，当前 Q1 可以视为：

> 已达到“可写进论文主模型、可继续服务 Q2/Q3/Q4”的封版状态。
