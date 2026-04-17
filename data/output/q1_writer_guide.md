# Q1 当前实现说明（写作者阅读版）

## 1. 当前运行逻辑

Q1 当前的实际执行链路如下：

1. 运行入口：`python scripts/run_q1.py`
2. `scripts/run_q1.py` 调用 `src/q1/pipeline.py`
3. `pipeline.py` 读取两份原始输入：
   - `data/raw/q1_robot_params.csv`
   - `data/raw/q1_attack_actions.csv`
4. `pipeline.py` 调用 `src/q1/model_v1.py`
   - 读取机器人基础参数
   - 读取 13 种攻击动作元数据
   - 计算四种方法共用的基础特征
5. `pipeline.py` 调用 `src/q1/evaluate.py`
   - 计算方法一、方法二、方法三、方法四的分数与排名
   - 进行 AHP 一致性检验
   - 进行惩罚参数灵敏度分析
6. `pipeline.py` 保存中间结果：
   - `data/interim/action_features.csv`
7. `pipeline.py` 调用 `src/q1/plot.py`
   - 绘制 6 张 Q1 图表并保存到 `data/output/`
8. `scripts/run_q1.py` 在终端打印方法四前 5 名和 AHP 一致性结果

---

## 2. 每个代码文件的作用

### `scripts/run_q1.py`

作用：
- Q1 的唯一正式运行入口
- 调用 `pipeline.main()`
- 在终端打印前 5 名动作和 AHP 一致性结果

### `src/q1/pipeline.py`

作用：
- 串联 Q1 全流程
- 读取原始数据
- 调用特征计算、四方法评价、灵敏度分析、绘图
- 保存中间表和图表

### `src/q1/model_v1.py`

作用：
- 定义机器人参数与动作数据结构
- 读取原始参数表
- 计算基础特征

当前主要计算的基础量包括：
- `impact_score`：冲击力矩
- `balance_cost`：质心偏移量
- `score_prob`：命中概率
- `energy_cost`：能量消耗
- `exec_time`：执行时间
- `stable_margin`：稳定裕度
- `stability_ratio`：`ΔCoM / θstable`
- `omega_out`：输出轴角速度
- `omega_eff`：有效角速度
- `com_height`：质心高度

### `src/q1/evaluate.py`

作用：
- 对基础特征做标准化
- 实现四种方法的评分
- 计算最终排序
- 进行灵敏度分析

### `src/q1/plot.py`

作用：
- 绘制 Q1 图表
- 统一设置中文宋体、英文新罗马字体
- 统一图像分辨率为 520 dpi

---

## 3. 当前模型是什么

Q1 当前实现的是“四方法并行”的结构：

### 方法一：单维度物理力学评分

核心思想：
- 只看冲击力矩
- 使用公式：

`tau_impact = 3 * tau_max * eta`

其中：
- `tau_max = 145 N·m`
- `eta` 为动作协同效率系数

特点：
- 简单直接
- 只反映攻击效果，不考虑失衡、命中率和能耗

### 方法二：AHP 多准则评分

核心思想：
- 指标为：
  - 冲击力
  - 稳定性
  - 命中率
  - 效率
- 使用 AHP 判断矩阵求权重
- 当前程序实际算得：
  - `w_tau ≈ 0.2844`
  - `w_stability ≈ 0.4729`
  - `w_hit ≈ 0.1699`
  - `w_efficiency ≈ 0.0729`

特点：
- 比方法一更全面
- 但稳定性仍然是线性处理

### 方法三：模糊综合评价

核心思想：
- 在方法二基础上引入“高 / 中 / 低”隶属度
- 对边界做模糊处理
- 最终输出模糊得分

特点：
- 比 AHP 更平滑
- 但本质上仍然没有真正刻画“超出稳定阈值后的非线性突变”

### 方法四：我们的方法

核心思想：
- 先计算基础效用：

`U0 = 0.35 * tau_norm + 0.25 * P_hit - 0.20 * E_norm - 0.20 * t_norm`

- 再计算非线性倒地惩罚：

`R_fall = lambda * exp(k * ΔCoM / θstable) - lambda`

- 再做量纲对齐后的惩罚归一化
- 最终得分：

`Score = max(0, U0 - R_fall_norm)`

- 最终效用：

`utility = Score / Score_max`

特点：
- 兼顾攻击效果、命中率、能耗、时间
- 用非线性惩罚反映失稳风险
- 是当前 Q1 的主模型

---

## 4. 当前模型的优点与缺点

### 优点

- 结构清晰，四种方法分层明确，便于论文写作
- 方法四能体现“临界失稳”的非线性特征
- 已有可运行代码、可复现结果、可直接输出图表
- 中间表字段较稳定，便于后续 Q2 / Q3 / Q4 复用
- 图表已经具备论文展示基础

### 缺点

- `eta`、`p_target`、部分动作角度与时间仍属于模型参数，不是官方直接测量值
- 当前还不是严格多刚体动力学仿真
- 命中概率仍然是几何建模近似，不是对抗仿真统计结果
- 一些动作参数来自题解口径和建模设定，答辩时必须明确“这是模型参数，不是官方原始量”

---

## 5. 输出文件有哪些，作用是什么

### 中间表

#### `data/interim/action_features.csv`

作用：
- 保存 Q1 的全部基础特征、四方法分数和最终排序
- 供后续 Q2 / Q3 / Q4 复用

核心字段包括：
- `impact_score`
- `balance_cost`
- `score_prob`
- `energy_cost`
- `method1_score`
- `method2_score`
- `method3_score`
- `proposed_score`
- `utility`
- `rank`

### 图表

#### `data/output/q1_utility_bar.png`

作用：
- 展示四种方法的归一化得分对比
- 展示 Top 8 动作的跨方法排名轨迹

#### `data/output/q1_impact_balance.png`

作用：
- 展示攻击效果与平衡代价之间的权衡关系
- 直观看出哪些动作接近或超过稳定阈值

#### `data/output/q1_method_comparison.png`

作用：
- 直接展示四种方法下名次变化
- 强调用我们方法后某些动作为何被上调或下调

#### `data/output/q1_penalty_curve.png`

作用：
- 展示非线性惩罚函数
- 突出“超出稳定裕度后惩罚迅速增大”

#### `data/output/q1_decision_atlas.png`

作用：
- 展示方法四在“基础效用 - 稳定性比值”平面上的决策结构
- 适合写作时解释方法四的机制

#### `data/output/q1_sensitivity_heatmap.png`

作用：
- 展示惩罚参数 `lambda` 与 `k` 变化时的灵敏度
- 用于说明模型鲁棒性

---

## 6. 当前涉及哪些参数

参数可以分成三层：

### A. 有依据的官方参数

这些参数来自题目或你整理的机器人参数文件，可视为官方直给：

- 机器人总高度 `H = 1.4 m`
- 总质量 `M = 42 kg`
- 腿长 `L_leg = 0.6865 m`
- 双臂臂展 `1.44 m`
- 单关节最大力矩 `145 N·m`
- 电机最高转速 `6400 rpm`
- 减速比 `25`
- 单关节质量 `1.1 kg`
- 关节数 `23`
- 机器人宽度 `0.53555 m`
- 支撑深度 `0.25266 m`
  说明：当前代码中按题解口径使用，但如果后续确认它来自网页参数，则可提升为“有依据参数”

### B. 由公式推导出来的参数

这些参数不是直接填的，而是根据上面的官方参数计算出来的：

- 输出轴角速度 `omega_out`
- 有效角速度 `omega_eff`
- 质心高度 `com_height`
- 单臂长度
- 前臂力臂、全臂力臂、全腿力臂、大腿力臂、扫腿力臂
- 稳定裕度 `stable_margin`
- 各动作的：
  - `impact_score`
  - `balance_cost`
  - `score_prob`
  - `energy_cost`
  - `stability_ratio`

### C. 模型假设参数

这些参数不是官方原始量，也不是唯一可推导出的量，而是当前建模方案中采用的设定：

- `foot_spacing_ratio = 0.60`
- `upper_com_ratio = 0.45`
- `effective_speed_ratio = 0.60`
- `average_torque_ratio = 0.50`
- `average_speed_ratio = 0.40`
- 质量分配比例：
  - `single_arm_mass_ratio = 0.07`
  - `single_leg_mass_ratio = 0.15`
  - `torso_mass_ratio = 0.42`
  - `forearm_mass_ratio = 0.45`
  - `thigh_mass_ratio = 0.54`
- `torso_eccentricity_m = 0.15`
- `front_kick_support_factor = 1.10`
- `body_charge_gamma = 0.35`
- `counter_recovery_gamma = 0.15`
- `five_kick_decay_rate = 0.06`
- 惩罚参数：
  - `lambda = 0.30`
  - `k = 3.50`

### D. 动作级模型参数

这类参数写在 `q1_attack_actions.csv` 中，用于区分不同攻击动作：

- `eta`
- `joint_count`
- `theta_total_deg`
- `t_exec_s`
- `p_target`

其中：

#### `eta`

含义：
- 方法一中的多关节协同效率系数

来源：
- 来自题解方案中的动作协同效率设定表

性质：
- 模型参数
- 不是官方原始测量值

#### `joint_count`

含义：
- 动作主要参与关节数

来源：
- 按动作链结构拆分计数

性质：
- 结构化建模输入
- 不是官方直给表格项

#### `theta_total_deg`

含义：
- 动作主要旋转角度

来源：
- 来自题解中对每个动作几何过程的设定

性质：
- 动作几何参数
- 不是官方直给

#### `t_exec_s`

含义：
- 动作执行时间

来源：
- 一部分来自 `theta / omega_eff` 的推导口径
- 一部分特殊动作来自经验设定

性质：
- 混合型参数

#### `p_target`

含义：
- 命中有效区域的几何概率

来源：
- 来自题解中的覆盖区域设定

性质：
- 几何建模参数
- 不是官方实测参数

---

## 7. 写作者应如何表述这些参数

建议论文中统一分三类表述：

### 可以写成“题目已知 / 官方参数”

- 高度
- 质量
- 腿长
- 臂展
- 最大力矩
- 转速
- 减速比
- 关节质量
- 关节数量
- 宽度

### 可以写成“由官方参数推导得到”

- 输出轴角速度
- 有效角速度
- 质心高度
- 稳定裕度
- 动作冲击力矩
- 动作能量消耗
- 动作质心偏移

### 必须写成“模型设定 / 建模假设”

- 质量分配比例
- 足间距比例
- 上体质心比例
- 协同效率系数 `eta`
- 目标区域命中概率 `p_target`
- 某些动作角度与执行时间
- 非线性惩罚参数 `lambda`、`k`

---

## 8. 当前最适合写作者的叙述主线

建议写作者按下面顺序理解与写作：

1. 先说明机器人有哪些官方物理参数
2. 再说明由这些参数推导出哪些二级量
3. 然后引入 13 种动作的动作级模型参数
4. 说明方法一、二、三只是对比方法
5. 强调方法四才是主模型
6. 用 `q1_method_comparison.png` 和 `q1_decision_atlas.png` 解释“为什么方法四更合理”
7. 用 `q1_sensitivity_heatmap.png` 补充模型稳定性说明

---

## 9. 当前一句话总结

Q1 当前已经实现为“官方参数 + 推导参数 + 模型参数”三层结构下的四方法并行评价系统，其中方法四通过非线性平衡惩罚刻画高风险动作的失稳问题，并输出了可供后续问题复用的中间结果表与展示图。
