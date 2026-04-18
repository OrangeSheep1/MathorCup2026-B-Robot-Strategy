# Q3 当前实现说明（写作者阅览版）
## 1. 当前运行逻辑

Q3 当前的实际执行链路如下：

1. 运行入口：`python scripts/run_q3.py`
2. `scripts/run_q3.py` 调用 `src/q3/pipeline.py`
3. `pipeline.py` 读取 4 份中间输入：
   - `data/interim/action_features.csv`（Q1 输出）
   - `data/interim/defense_matchup.csv`（Q2 输出）
   - `data/interim/defense_features.csv`（Q2 输出）
   - `data/interim/defense_pair_scores.csv`（Q2 输出）
4. `pipeline.py` 调用 `src/q3/model_v1.py`
   - 继承 Q1/Q2 的动作与攻防参数
   - 校验能耗口径
   - 构建状态空间、动作空间、动作核与转移事件树
5. `pipeline.py` 调用 `src/q3/policy.py`
   - 生成四种方法的策略
   - 主方法使用有限时域 MDP 的 Bellman 逆向递推求解
6. `pipeline.py` 调用 `src/q3/simulate.py`
   - 对四种方法执行蒙特卡洛仿真
   - 统计胜率、均值、置信区间与两比例 z 检验
7. `pipeline.py` 保存中间结果：
   - `q3_action_kernels.csv`
   - `q3_policy_table.csv`
   - `q3_static_strategy.csv`
   - `q3_scenario_summary.csv`
   - `q3_method_metrics.csv`
   - `q3_trajectory_sample.csv`
8. `pipeline.py` 调用 `src/q3/plot.py`
   - 输出 4 张 Q3 图表到 `data/output/`

---

## 2. 每个代码文件的作用

### `scripts/run_q3.py`

作用：
- Q3 的唯一正式运行入口
- 调用 `pipeline.main()`
- 在终端打印三类典型场景下的 MDP 推荐动作摘要
- 打印 MDP 的蒙特卡洛胜率摘要

### `src/q3/pipeline.py`

作用：
- 串联 Q3 全流程
- 统一完成“读表 -> 构造环境 -> 求策略 -> 做仿真 -> 存结果 -> 画图”
- 是 Q3 的总流程层

### `src/q3/model_v1.py`

作用：
- 定义 Q3 的状态结构、配置结构、动作核结构
- 读取 Q1/Q2 输出
- 构建 Q3 的状态空间、动作空间、攻击表、防守表、攻防对表
- 将 Q1/Q2 的参数加工为 Q3 可直接使用的转移核

当前最关键的职责有 5 个：
- 校验 `energy_cost` 是否仍为焦耳口径
- 用 Q2 的 Top3 防守集构建“对手响应画像”
- 为每个动作构造 `ActionKernel`
- 将倒地影响转为显式的 `recovery_lock` 状态
- 用条件事件树替代原先的独立乘法分支

### `src/q3/policy.py`

作用：
- 实现四种策略方法
- 负责主模型求解
- 负责输出每个状态的推荐动作
- 负责提取领先/平局/落后三种典型场景摘要

四种方法分别是：
- 方法一：贪心策略
- 方法二：静态矩阵博弈
- 方法三：启发式规则树
- 方法四：有限时域 MDP + Bellman 逆向递推

说明：
- 当前代码虽然函数名仍为 `value_iteration`，但实际求解方式不是经典“反复迭代到收敛”的无限时域值迭代，而是有限时域下的逆向动态规划。

### `src/q3/simulate.py`

作用：
- 按策略执行单场比赛仿真
- 对四种方法进行蒙特卡洛对比
- 输出轨迹样例与方法统计指标

当前重点实现：
- 静态博弈法按混合策略采样，而不是取最大概率动作
- 同一场景、同一次仿真下，四种方法共享同一个随机种子，便于公平比较
- 倒地不仅有即时罚分，还会进入恢复受限状态

### `src/q3/plot.py`

作用：
- 绘制 Q3 图表
- 与 Q1/Q2 保持统一绘图风格

绘图风格：
- 中文字体：`SimSun`
- 英文字体：`Times New Roman`
- 输出精度：`520 dpi`

---

## 3. 当前模型是什么

Q3 当前实现的是“**四种方法对比 + 主模型求解**”结构。

### 方法一：贪心策略

核心思想：
- 每一步只看当前动作的即时回报
- 不考虑下一状态的长期价值

形式上相当于：

`a_greedy(s) = argmax E[R(s,a)]`

作用：
- 作为简单下界
- 用来展示“只看眼前收益”的局限

### 方法二：静态矩阵博弈

核心思想：
- 先构造静态收益矩阵
- 求解混合策略
- 仿真时按混合概率采样动作

当前实现特点：
- 不是把混合策略硬转成纯策略
- 因此比旧版更公平，也更接近真正的静态博弈基线

### 方法三：启发式规则树

核心思想：
- 领先、落后、低机能、恢复锁定等场景使用不同规则

说明：
- 这是一个可解释基线
- 规则阈值本身属于启发式设定

### 方法四：有限时域 MDP + Bellman 逆向递推

这是 Q3 的主模型。

#### 状态空间

当前状态定义为：

`S = (score_diff, time_step, health_my, health_opp, recovery_lock)`

各维含义：
- `score_diff`：分差，离散为 `-5 ~ 5`
- `time_step`：时间步，`1 ~ 20`
- `health_my`：我方机能档位，`High / Mid / Low`
- `health_opp`：对手机能档位，`High / Mid / Low`
- `recovery_lock`：倒地恢复约束标志，`0/1`

说明：
- 新增的 `recovery_lock` 是这一版的重要修正，用来保证“倒地不是只扣分，而是会影响下一步动作空间”。

#### 动作空间

动作空间由两部分组成：
- 攻击动作：Q1 的 13 种攻击动作
- 防守动作：Q2 的 22 种防守动作

并附加两类约束：
- Low 机能下，高能耗攻击动作会被剔除
- `recovery_lock=1` 时，只允许恢复/保底类防守动作

#### 转移机制

当前不是简单把“得分、降档、倒地”互相独立相乘，而是改成了**条件事件树**。

每一步先划分为 3 类事件：
- `score_for`：我方得分
- `score_against`：对手得分
- `neutral`：中性结算

然后再在每个事件分支下展开：
- 我方是否降档
- 对方是否降档
- 是否倒地

这比旧版更严谨，因为它避免了：
- “没得分但对手机能仍下降”
- “被对手反击的同时又按成功命中处理对手机能”
这类不合理转移。

#### 奖励函数

当前即时奖励采用：

`R = reward_score * delta_score + reward_health * opp_drop - reward_cost * self_drop - reward_fall * fall_event`

并在终局加入终局奖励：
- 胜：`+10`
- 负：`-10`
- 平：`0`

#### 求解方式

当前主模型求解不是无限时域迭代收敛，而是：

1. 在最后一个时间步初始化终局价值
2. 从后向前做 Bellman 递推
3. 对每个状态枚举可用动作，选取最优动作

因此论文中更准确的说法应是：

> 有限时域 MDP 的 Bellman 逆向递推求解

而不是“经典值迭代收敛求解”。

---

## 4. 当前模型的优点与缺点

### 优点

- 结构清晰，Q1 -> Q2 -> Q3 的参数链已经真正打通
- 主模型不是凭空设转移概率，而是尽量复用 Q1/Q2 输出
- 修复了旧版中最严重的四个问题：
  - 能耗量纲混乱
  - Top3 防守集退化成 Top1
  - 状态转移过度独立化
  - 倒地只罚分、不改状态
- 静态博弈基线已经改成真正的混合策略采样，对比更公平
- 蒙特卡洛比较已做随机种子对齐，噪声更小

### 缺点

- 机能状态仍然是离散三档，不是连续体能/损伤模型
- `recovery_lock=1` 只持续一个决策步，是“最小可用版本”，不是完整倒地恢复动力学
- 对手策略目前仍采用均匀攻击先验，属于信息不完全条件下的鲁棒建模，不是对真实对手的专属建模
- 宏观动作分类阈值和规则树阈值仍带有启发式成分
- 当前主模型仍然是参数化近似，不是完整的多体动力学对抗仿真

---

## 5. 输出文件有哪些，作用是什么

### 中间结果表

#### `data/interim/q3_action_kernels.csv`

作用：
- 保存不同机能状态组合下，每个动作的一步转移摘要
- 是 Q3 的“动作核接口表”

典型字段：
- `health_my`
- `health_opp`
- `action_id`
- `action_type`
- `macro_group`
- `expected_reward`
- `p_score_for`
- `p_score_against`
- `p_self_drop`
- `p_opp_drop`
- `p_fall`

#### `data/interim/q3_policy_table.csv`

作用：
- 保存每个状态下四种方法的推荐动作
- 是 Q3 最核心的策略输出表

典型字段：
- 状态字段：
  - `score_diff`
  - `time_step`
  - `health_my`
  - `health_opp`
  - `recovery_lock`
- 策略字段：
  - `greedy_action_id`
  - `static_action_id`
  - `rule_action_id`
  - `mdp_action_id`
  - `mdp_value`
  - `mdp_immediate_reward`

#### `data/interim/q3_static_strategy.csv`

作用：
- 保存静态矩阵博弈法的混合策略分布
- 是方法二的原始概率输出表

典型字段：
- `action_id`
- `static_prob`
- `static_rank`
- `static_mean_payoff`

#### `data/interim/q3_scenario_summary.csv`

作用：
- 提取三种典型场景下 MDP 的高频动作
- 方便论文中写“领先/平局/落后”策略分析

#### `data/interim/q3_method_metrics.csv`

作用：
- 保存四种方法的蒙特卡洛对比结果
- 用于方法比较图和正文表述

典型字段：
- `scenario`
- `method`
- `win_rate`
- `ci_low`
- `ci_high`
- `mean_score_diff`
- `mean_health_my`
- `mean_health_opp`
- `mean_reward`
- `p_value_vs_mdp`
- `test_method`

#### `data/interim/q3_trajectory_sample.csv`

作用：
- 保存样例对局轨迹
- 用于轨迹对比图和个案分析

典型字段：
- `time_step`
- `score_diff_before`
- `score_diff_after`
- `health_my_before`
- `health_my_after`
- `action_id`
- `opponent_action_id`
- `fall_event`
- `recovery_lock_before`
- `recovery_lock_after`

### 图表

#### `data/output/q3_policy_heatmap.png`

作用：
- 展示 MDP 在不同时间阶段、分差、机能状态下的宏观策略分布

#### `data/output/q3_value_surface.png`

作用：
- 展示高机能对高机能时，分差 × 时间步 对应的值函数分布

说明：
- 当前已改为二维填色等高图，不再使用 3D 曲面

#### `data/output/q3_method_comparison.png`

作用：
- 展示四种方法在领先/平局/落后三种局面下的胜率对比

#### `data/output/q3_trajectory.png`

作用：
- 展示 MDP 与贪心法的样例对局轨迹差异

---

## 6. 当前涉及哪些参数

Q3 的参数建议分为三层看待。

### A. 有明确依据的参数

这些参数可以写成“来自赛题、Q1、Q2 或官方参数”：

#### 来自比赛规则
- `match_time_s = 300`
- 单场无故障这一边界

#### 来自机器人参数
- `battery_capacity_mah = 10000`
- `battery_charge_voltage_v = 54.6`
- `battery_endurance_h = 2.0`

#### 来自 Q1
- `score_prob`
- `impact_score`
- `tau_norm`
- `energy_cost`
- `exec_time`
- `attack_fall_risk`
- `attack_utility`

#### 来自 Q2
- `p_block`
- `counter_window`
- `defense_damage`
- `p_fall`
- `defense_score_r1/r2/r3`
- `counter_action_id`

这些量都可以视为“有直接来源的输入量”。

### B. 由公式推导出来的参数

这些不是直接填入的，而是由已有参数进一步计算得到的：

#### 电池与单场能量相关
- `battery_energy_wh`
- `battery_energy_j`
- `available_match_energy_j`
- `high_health_threshold_j`
- `low_health_threshold_j`
- `mid_health_gap_j`

#### 攻击疲劳相关
- `energy_cost_j`
- `energy_match_ratio`
- `energy_norm`
- `energy_drop_high`
- `energy_drop_mid`
- `low_health_available`

#### Q2 Top3 防守集加权响应
- `opponent_block_rate`
- `opponent_counter_prob`
- `opponent_counter_window`
- `opponent_defense_fall_risk`

#### 反击概率统一口径
- `counter_prob_effective`

它的定义是：
- 先根据 `counter_window` 找到可执行的最佳反击动作
- 再用 `p_block × 该反击动作的基础成功率` 形成有效反击概率

这部分是当前 Q3 与 Q2 接口衔接的关键。

### C. 建模假设或算法超参数

这些参数不是“客观真值”，只能写成“有依据的建模假设”：

- `time_step_s = 15`
- `discount_factor = 0.95`
- `mid_health_ratio = 0.40`
- `low_health_ratio = 0.75`
- `impact_to_health_factor = 0.30`
- `reward_score = 1.00`
- `reward_health = 0.20`
- `reward_cost = 0.20`
- `reward_fall = 0.67`
- `terminal_reward_win = 10.0`
- `terminal_reward_loss = -10.0`
- `low_health_energy_cutoff = 0.50`
- `fall_recovery_lock_steps = 1`

这些参数可以说“有物理或规则背景支撑”，但不能写成题目唯一推出的客观参数。

### D. 启发式参数

这些参数主观性更强，应明确只用于辅助分析、基线方法或可视化分类：

- `_classify_attack_macro_group()` 中的阈值
- `_classify_defense_macro_group()` 中的阈值
- 规则树中的时间步阈值和分差阈值

这些参数不能作为主模型“硬物理真值”的依据。

---

## 7. 哪些是有依据的，哪些是推出来的，哪些是假设

可以直接这样写：

### 可以直接说“有依据”的

- 比赛总时长
- 电池容量、电压、续航
- Q1 输出的攻击命中概率、冲击强度、执行时间、能耗
- Q2 输出的拦截概率、反击窗口、防守耗损、倒地风险

### 可以说“由公式推导”的

- 单场可用能量
- 机能转档阈值
- 动作的疲劳降档概率
- 基于 Top3 防守集的加权防守响应
- 有效反击概率 `counter_prob_effective`

### 必须说“建模假设”的

- 时间步长 15 秒
- 折扣因子 0.95
- 健康状态三档离散化
- 机能转档比例阈值
- 冲击转健康损失系数
- 奖励权重
- 倒地后恢复锁定持续 1 步
- 对手攻击均匀先验

### 必须说“启发式”的

- 宏观策略分类阈值
- 规则树阈值

---

## 8. 当前版本相对旧版的重要改动

如果写作者要解释“为什么这版更严谨”，最应该强调这 5 点：

1. 统一了能耗量纲  
   Q3 现在明确要求并校验 `energy_cost` 使用焦耳。

2. Top3 防守集真正参与了 Q3  
   不再退化成只看 Top1。

3. 转移由独立分支改成条件事件树  
   避免了逻辑不一致的状态分支。

4. 倒地进入状态，而不是只罚分  
   现在倒地后下一步会进入恢复受限状态。

5. 静态博弈法改成了混合策略采样  
   方法对比更公平。

---

## 9. 写作者建议怎么写这一问

建议写作顺序如下：

1. 先交代 Q3 的任务：
   单场、无故障、状态依赖、要决定何时攻何时守。

2. 再说明为什么需要 MDP：
   贪心法只看当前，静态博弈忽略时序，规则树依赖人工阈值。

3. 再给出状态、动作、转移、奖励四部分：
   重点强调：
   - 状态加入了 `recovery_lock`
   - 转移概率由 Q1/Q2 支撑
   - 主方法使用有限时域逆向递推

4. 再写四方法对比：
   - 方法一：简单、短视
   - 方法二：理论上严谨但静态
   - 方法三：直观但阈值化
   - 方法四：能处理“时间、分差、机能、倒地恢复”共同作用

5. 最后引用图表：
   - 用 `q3_policy_heatmap.png` 解释状态依赖策略
   - 用 `q3_method_comparison.png` 解释四方法效果差异
   - 用 `q3_trajectory.png` 解释样例对局中的策略切换

---

## 10. 一句话总结当前 Q3

当前 Q3 已经不是“简单套一个 MDP 框架”，而是一个基于 Q1/Q2 参数链、带显式倒地恢复约束、使用有限时域 Bellman 逆向递推求解的单场策略优化模块。它已经具备论文级主模型雏形，但仍应把奖励权重、阈值和对手先验如实写成建模假设，而不是写成客观真值。
