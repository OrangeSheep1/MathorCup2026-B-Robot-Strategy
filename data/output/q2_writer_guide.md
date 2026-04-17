# Q2 当前实现说明（写作者阅览版）

## 1. 当前运行逻辑

Q2 当前的实际执行链路如下：

1. 运行入口：`python scripts/run_q2.py`
2. `scripts/run_q2.py` 调用 `src/q2/pipeline.py`
3. `pipeline.py` 读取 4 份输入：
   - `data/raw/q1_robot_params.csv`
   - `data/interim/action_features.csv`
   - `data/raw/q2_attack_semantics.csv`
   - `data/raw/q2_defense_actions.csv`
4. `pipeline.py` 调用 `src/q2/model_v1.py`
   - 继承 Q1 的攻击特征
   - 读取攻击语义标签和 22 种防守动作元数据
   - 计算防守特征表 `defense_features`
   - 计算 13×22 攻防对基础矩阵 `pair_matrix`
5. `pipeline.py` 调用 `src/q2/evaluate.py`
   - 计算方法一、方法二、方法三、方法四的得分和排名
   - 进行闭环约束筛选
   - 输出供 Q3 复用的防守匹配表和反击链表
6. `pipeline.py` 保存中间结果：
   - `data/interim/defense_features.csv`
   - `data/interim/defense_pair_scores.csv`
   - `data/interim/defense_matchup.csv`
   - `data/interim/counter_chain.csv`
   - `data/interim/q2_method_summary.csv`
7. `pipeline.py` 调用 `src/q2/plot.py`
   - 输出 5 张 Q2 图表到 `data/output/`
8. `scripts/run_q2.py` 在终端打印前 5 个攻击动作的 Top1 防守建议

---

## 2. 每个代码文件的作用

### `scripts/run_q2.py`

作用：

- Q2 的唯一正式运行入口
- 调用 `pipeline.main()`
- 在终端打印前几项 Top1 防守建议，便于快速检查结果

### `src/q2/pipeline.py`

作用：

- 串联 Q2 全流程
- 负责“读取输入 → 计算特征 → 四方法评价 → 保存结果 → 绘图”
- 是 Q2 的总流程层

### `src/q2/model_v1.py`

作用：

- 定义 Q2 的输入数据结构
- 继承 Q1 输出并补充攻击语义标签
- 读取 22 种防守动作元数据
- 推导防守动作的基础物理特征
- 构造 13×22 攻防对基础矩阵

当前主要计算的量包括：

- `exec_time_def`：防守执行时间
- `balance_cost_def`：防守自身质心位移
- `mobility_cost`：防守位移代价系数
- `effective_support_mass`：防守等效支撑质量
- `elastic_transfer_rate`：局部弹性传递效率
- `absorb_rate`：冲击吸收率
- `force_capacity_factor`：防守承载系数
- `force_capacity`：防守承载上限
- `p_geo`：几何覆盖匹配度
- `p_force`：力量匹配度
- `p_react`：反应时间匹配度
- `p_block`：拦截概率
- `defense_damage`：防守耗损
- `counter_window`：反击时间窗口
- `p_fall`：防守倒地风险

### `src/q2/evaluate.py`

作用：

- 实现四种方法的评分逻辑
- 计算主模型综合防守效用
- 实现非线性倒地惩罚
- 进行闭环约束筛选
- 输出 `defense_matchup.csv`、`counter_chain.csv`、`q2_method_summary.csv`

### `src/q2/plot.py`

作用：

- 绘制 Q2 图表
- 沿用 Q1 的绘图风格
- 中文 `SimSun`，英文 `Times New Roman`
- 分辨率统一为 `520 dpi`

---

## 3. 当前模型是什么

Q2 当前实现的是“4 种方法并行”的结构：

### 方法一：规则匹配法

核心思想：

- 只按攻击轨迹类型、方向、高度与防守类别做规则查表
- 输出离散匹配等级：
  - `1.0`：强匹配
  - `0.5`：弱匹配
  - `0.0`：不匹配

特点：

- 直观
- 适合作为对比基线
- 不承担精细量化任务

### 方法二：拦截概率矩阵

核心思想：

- 对每个攻防对计算：
  - `P_block = P_geo × P_force × P_react`
- 只看单次拦截成功概率

特点：

- 比方法一更量化
- 但仍然没有把反击窗口和倒地风险纳入核心目标

### 方法三：模糊防守效用

核心思想：

- 将 4 个指标模糊化后线性合成：
  - `P_block`
  - `1 - defense_damage`
  - `counter_window_norm`
  - `1 - p_fall`

特点：

- 比方法二更平滑
- 但本质上仍是线性加权

### 方法四：我们的主模型

核心思想：

- 先计算基础防守效用：

`V_def = 0.38 * P_block - 0.12 * defense_damage + 0.25 * counter_window_norm - 0.25 * p_fall`

- 再引入非线性倒地惩罚：

`R_def_fall = lambda * exp(k * p_fall) - lambda`

- 再做量纲对齐后的惩罚缩放：

`penalty_norm = penalty_raw / penalty_max * (V_max - V_min)`

- 最终得分：

`Score_def = max(0, V_def - penalty_norm)`

并附加 3 个硬约束：

- 几何不覆盖时直接剔除
- 倒地风险过高时剔除
- Top 3 推荐集必须尽量满足“主动防守 + 保底防守”闭环

---

## 4. 当前模型的优点与缺点

### 优点

- 结构清晰，数据流分层明确：原始标签 → 防守特征 → 攻防对矩阵 → 评价输出
- 与 Q1 的参数继承关系清楚，能形成“攻击评价 → 防守匹配 → 反击接口”的链条
- 方法一到方法四层次分明，便于论文展示“为什么要引出主模型”
- 主模型已经把“防守成功”从静态拦截升级为“拦截 + 反击机会 + 倒地风险”的综合评价
- 闭环约束已经落地，不是只给单一最优答案，而是输出可供 Q3 使用的优先级列表
- 当前 `defense_features.csv` 已经附带参数审查字段，便于写作和答辩说明

### 缺点

- 攻击语义标签如 `trajectory_type`、`direction_tag`、`height_tag` 仍然是语义建模输入，不是官方直接测量量
- 防守承载链和吸收率虽然已经改为程序推导，但仍是参数化近似，不是碰撞实验标定值
- `range_tag` 仍是辅助语义字段，当前不进入主评分链
- 防守执行时间中的控制时滞、组合折减、支撑修正等仍属于建模假设
- 非线性惩罚参数 `lambda=0.30`、`k=3.50` 仍沿用 Q1 口径，尚未单独做 Q2 反标定

---

## 5. 输出文件有哪些，作用是什么

### 中间结果表

#### `data/interim/defense_features.csv`

作用：

- 保存 22 种防守动作的基础特征
- 是 Q2 的“防守侧特征接口”

核心字段包括：

- `exec_time_def`
- `balance_cost_def`
- `mobility_cost`
- `effective_support_mass`
- `elastic_transfer_rate`
- `absorb_rate`
- `force_capacity`
- `force_capacity_factor`
- `contact_stiffness_ratio`
- `force_capacity_factor_input`
- `force_capacity_factor_delta`
- `force_capacity_factor_audit`
- `contact_stiffness_ratio_input`
- `contact_stiffness_ratio_delta`
- `contact_stiffness_ratio_audit`

#### `data/interim/defense_pair_scores.csv`

作用：

- 保存 13×22 全部攻防对的评分结果
- 是 Q2 的核心计算明细表

核心字段包括：

- `p_geo`
- `direction_match`
- `height_match`
- `p_force`
- `p_react`
- `p_block`
- `defense_damage`
- `counter_window`
- `p_fall`
- `method1_score`
- `method2_score`
- `method3_score`
- `proposed_score`
- `rank`

#### `data/interim/defense_matchup.csv`

作用：

- 保存每个攻击动作对应的 Top 1-3 防守推荐
- 是 Q3 最直接使用的接口文件

核心字段包括：

- `defense_id_r1/r2/r3`
- `block_prob_r1/r2/r3`
- `counter_window_r1/r2/r3`
- `fall_risk_r1/r2/r3`
- `defense_score_r1/r2/r3`
- `counter_action_id_r1/r2/r3`
- `closure_complete`
- `closure_note`

#### `data/interim/counter_chain.csv`

作用：

- 保存“攻击动作 → 防守动作 → 可接续反击动作”的链式结构
- 便于 Q3 做状态转移和可执行反击筛选

#### `data/interim/q2_method_summary.csv`

作用：

- 保存四种方法对每个攻击动作给出的 Top1 防守动作
- 用于论文中展示方法对比

### 图表

#### `data/output/q2_utility_matrix.png`

作用：

- 展示攻防效用矩阵
- 用于看不同攻击动作的最优防守分布

#### `data/output/q2_surface.png`

作用：

- 展示综合防守效用在关键指标平面上的响应关系
- 适合解释主模型的非线性趋势

#### `data/output/q2_waterfall.png`

作用：

- 展示“攻击动作 → Top1 防守”的映射关系

#### `data/output/q2_parallel.png`

作用：

- 展示高分与低分攻防对在多维指标上的差异

#### `data/output/q2_method_comparison.png`

作用：

- 展示四种方法的 Top1 防守结果差异
- 是论文里最直接的“方法对比图”

---

## 6. 当前涉及哪些参数

Q2 的参数可以分成 4 层：

### A. 官方直接参数

这些参数直接来自题目或机器人参数表：

- `max_joint_torque_nm = 145`
- `max_motor_speed_rpm = 6400`
- `reducer_ratio = 25`
- `total_mass_kg = 42`
- `leg_length_m = 0.6865`
- `arm_span_m = 1.44`
- `body_width_m = 0.53555`
- `official_time_s = 1.00`（仅 D18 快速起身为题目明确给定）

### B. 由 Q1 继承的参数

这些参数不是 Q2 新造的，而是直接来自 Q1 输出：

- `impact_score`
- `tau_norm`
- `balance_cost`
- `stable_margin`
- `stability_ratio`
- `exec_time`
- `time_norm`
- `attack_utility`
- `attack_rank`

这些量在 Q2 中分别用于：

- 衡量攻击强度
- 计算对手恢复时间
- 计算反击窗口
- 与 Q1 的反击动作效用衔接

### C. 在 Q2 中由公式推导出的参数

这些量由程序计算，不是手填结论：

- `omega_out`
- `omega_eff`
- `exec_time_def`
- `balance_cost_def`
- `mobility_cost`
- `effective_support_mass`
- `elastic_transfer_rate`
- `force_capacity_factor`
- `force_capacity`
- `contact_stiffness_ratio`
- `absorb_rate`
- `p_geo`
- `p_force`
- `p_react`
- `p_block`
- `defense_damage`
- `counter_window`
- `counter_window_norm`
- `counter_prob`
- `p_fall`
- `v_def`
- `fall_penalty`
- `proposed_score`

### D. 建模假设参数

这些不是官方直接量，而是当前方案采用的建模设定：

#### 时间与控制相关

- `control_delay_s = 0.12`
- `multi_joint_sync_delay_s = 0.03`
- `lower_body_extra_delay_s = 0.05`
- `ground_motion_extra_delay_s = 0.10`
- `combo_overlap_discount_s = 0.03`
- `recover_factor_s = 0.30`

#### 防守位移近似

- `step_back_leg_factor = 0.30`
- `orbit_leg_factor = 0.45`
- `step_adjust_factor = 0.60`
- `micro_adjust_radius_ratio = 0.50`
- `controlled_fall_gamma = 0.12`
- `rapid_getup_gamma = 0.08`
- `ground_guard_gamma = 0.50`

#### 防守支撑链与承载链近似

- `arm_pair_link_efficiency = 0.78`
- `arm_single_link_efficiency = 0.67`
- `shoulder_single_link_efficiency = 0.72`
- `wrist_pair_link_efficiency = 0.75`
- `leg_pair_link_efficiency = 0.86`
- `leg_single_link_efficiency = 0.82`
- `torso_turn_link_efficiency = 0.74`
- `imu_link_efficiency = 0.82`
- `step_link_efficiency = 0.88`
- `ground_support_ratio = 0.80`

#### 类别基准

- `force_base_rigid = 0.85`
- `force_base_posture = 0.60`
- `force_base_balance = 0.45`
- `force_base_soft = 0.50`
- `force_base_ground = 0.30`
- `force_support_ratio_cap = 1.15`
- `elastic_mass_cap_ratio = 2.20`
- `absorb_base_rigid = 0.58`
- `absorb_base_posture = 0.76`
- `absorb_base_balance = 0.70`
- `absorb_base_soft = 0.99`
- `absorb_base_ground = 0.35`
- `absorb_noncontact = 0.95`

#### 主模型权重与惩罚参数

- `alpha = 0.38`
- `beta = 0.12`
- `gamma = 0.25`
- `delta = 0.25`
- `lambda = 0.30`
- `k = 3.50`

---

## 7. 哪些参数是有依据的，哪些是推出来的，哪些是假设

### 可以写成“有依据”

- 机器人尺寸、质量、力矩、转速、减速比
- D18 的 1 秒起身时间
- Q1 输出的攻击强度、执行时间、稳定性相关字段
- 攻击动作和防守动作的名称、类型、组合关系

### 可以写成“由公式推导得到”

- `omega_out`
- `omega_eff`
- `exec_time_def`
- `balance_cost_def`
- `mobility_cost`
- `effective_support_mass`
- `elastic_transfer_rate`
- `force_capacity_factor`
- `force_capacity`
- `contact_stiffness_ratio`
- `absorb_rate`
- `p_block`
- `defense_damage`
- `counter_window`
- `p_fall`
- `proposed_score`

### 必须写成“模型假设 / 建模设定”

- 攻击语义标签：
  - `trajectory_type`
  - `direction_tag`
  - `height_tag`
  - `range_tag`
  - `target_zone`
- 防守覆盖标签：
  - `coverage_direction`
  - `coverage_height`
  - `contact_mode`
- 各类 link efficiency、gamma、delay、base factor
- 主模型权重和非线性惩罚参数

要特别注意：

- `q2_attack_semantics.csv` 里的语义标签不是官方测量值，是对题目动作描述的结构化表达
- `q2_defense_actions.csv` 里的基础字段里，动作名称、类别、组合关系可以视为结构化动作定义；但承载能力与接触刚度相关字段只能视为模型口径，不应写成“官方给定”
- 当前代码已经把 `force_capacity_factor_input` 和 `contact_stiffness_ratio_input` 与程序推导值分开输出，因此写作时应明确“输入参考值”和“最终推导值”不是同一个概念

---

## 8. 写作者如何表述当前 Q2

建议写作者按下面的顺序叙述：

1. Q2 的目标不是单纯找“能挡住”的动作，而是找“能挡住且利于反击”的动作
2. Q2 输入分为三层：
   - 官方机器人参数
   - Q1 继承的攻击特征
   - Q2 的攻击语义与防守动作元数据
3. 先介绍三种对比方法的局限：
   - 方法一过于规则化
   - 方法二只优化拦截率
   - 方法三仍然是线性模糊合成
4. 再引出主模型：
   - 拦截概率
   - 防守耗损
   - 反击窗口
   - 倒地风险
5. 说明闭环约束：
   - 推荐集不是单一动作
   - 必须兼顾主动防守和保底防守
6. 再说明 Q2 输出如何进入 Q3：
   - `defense_matchup.csv` 给出状态转移所需概率和窗口
   - `counter_chain.csv` 给出可接续反击动作

---

## 9. 当前一句话总结

Q2 当前已经实现为一个“继承 Q1 攻击特征、基于防守物理特征推导、用四种方法并行评价、最终输出带闭环约束的攻防匹配接口”的防守建模系统，其中方法四把防守从静态拦截提升为“拦截 + 反击机会 + 倒地风险”的动态决策问题。
