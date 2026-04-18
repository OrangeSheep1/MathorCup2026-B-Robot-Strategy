# Q4 当前实现说明（写作者阅览版）

## 1. 当前运行逻辑

Q4 当前的实际执行链路如下：

1. 运行入口：`python scripts/run_q4.py`
2. `scripts/run_q4.py` 调用 `src/q4/pipeline.py`
3. `pipeline.py` 读取 Q1 / Q2 / Q3 的中间结果：
   - `data/interim/action_features.csv`
   - `data/interim/defense_pair_scores.csv`
   - `data/interim/q3_action_kernels.csv`
   - `data/interim/q3_method_metrics.csv`
4. `pipeline.py` 读取并校验 Q4 参数文件：
   - `data/raw/q4_fault_params.json`
5. `pipeline.py` 调用 `src/q4/model_v1.py`
   - 构建 Q4 上下文
   - 定义局内状态、动作、故障率、资源动作语义
   - 将 Q3 压缩为可计算的代表性战术动作集
6. `pipeline.py` 调用 `src/q4/decision.py`
   - 对 18 种局内资源配额做有限时域 DP
   - 生成单局胜率表 `P_win`
   - 额外前向传播“输赢 + 实际资源使用量”联合分布
   - 再做 BO3 宏观逆向动态规划
7. `pipeline.py` 调用 `src/q4/simulate.py`
   - 对四种方法执行 BO3 蒙特卡洛仿真
   - 对比理论胜率与仿真胜率
   - 汇总资源使用、故障曲线、批次胜率分布
8. `pipeline.py` 调用 `src/q4/plot.py`
   - 输出 6 张核心图表到 `data/output/`

---

## 2. 每个代码文件的作用

### `scripts/run_q4.py`

作用：

- Q4 的唯一正式运行入口
- 调用 `pipeline.main()`
- 在终端打印关键结果摘要

### `src/q4/pipeline.py`

作用：

- 串联 Q4 全流程
- 统一完成“读表 -> 建模 -> 单局求解 -> BO3 求解 -> 仿真 -> 出图 -> 存表”
- 是 Q4 的总流程层

### `src/q4/model_v1.py`

作用：

- 定义 Q4 的配置、状态、战术动作和动作核
- 读取 Q1/Q2/Q3 输入
- 构建局内扩展状态空间与一步转移逻辑

当前最关键的状态变量是：

- `score_diff`：当前分差
- `time_bucket`：局内时间桶
- `health_my` / `health_opp`：双方机能档
- `fault`：故障状态
- `down_flag`：倒地 / 失稳待复位状态
- `reset_left / pause_left / repair_left`：局内剩余资源

说明：

- 这一版已经把 `fault` 和 `down_flag` 拆开
- `USE_RESET` 主要处理 `down_flag=1`
- `USE_REPAIR` 主要处理 `fault=1`
- 这比旧版“把复位和故障混在一起”更贴题

### `src/q4/decision.py`

作用：

- 实现 Q4 的双层动态规划

分为两部分：

1. 局内微观层
   - 固定配额 `(reset_alloc, pause_alloc, repair_alloc)` 下
   - 用有限时域 DP 求单局最优策略
   - 输出 `p_win`
   - 额外输出“实际资源使用量联合分布”

2. 跨局宏观层
   - 以 `(W_my, W_opp, R_remain)` 为状态
   - 逆向求解 BO3 最优资源调度
   - 当前已改为按“实际使用量”扣减 BO3 资源，而不是按“分配量”直接扣减

### `src/q4/simulate.py`

作用：

- 对 Q4 的四种方法执行 BO3 蒙特卡洛仿真
- 输出方法对比摘要
- 输出资源使用画像和故障演化画像

当前实现的四种方法：

- `optimal_dp`：主模型，双层 DP
- `fixed_rule`：固定规则法
- `exhaustive_static`：穷举静态分配法
- `all_in_first`：首局全投法

说明：

- 仿真层现在也已改为按“实际使用量”扣减跨局资源
- 不再把“本局分了多少”直接当作“本局消耗了多少”

### `src/q4/plot.py`

作用：

- 绘制 Q4 图表
- 沿用 Q1/Q2/Q3 的统一绘图风格

绘图风格：

- 中文字体：`SimSun`
- 英文字体：`Times New Roman`
- 输出精度：`520 dpi`

---

## 3. 当前模型是什么

Q4 当前实现的是：

**扩展局内有限时域 MDP + BO3 宏观动态规划**

也就是一个双层模型。

### 局内微观层

目标：

- 在单局中决定何时攻防、何时使用资源
- 最大化单局获胜概率

特点：

- 故障是状态依赖随机事件
- 倒地 / 失稳单独作为 `down_flag`
- 资源动作进入可行动作集

### 跨局宏观层

目标：

- 在 BO3 中决定每一局给多少资源配额
- 最大化整个系列赛的最终获胜概率

特点：

- 使用局内层输出的 `P(win/lose, used_reset, used_pause, used_repair | allocation, scenario)`
- 递推时按实际使用量更新 BO3 剩余资源

---

## 4. 当前模型的优点与不足

### 优点

- 结构贴合题目：单局层与 BO3 层分开，逻辑清楚
- Q4 不是独立拍脑袋，直接继承 Q1/Q2/Q3 参数链
- `reset` 与 `repair` 语义已拆开，更符合规则
- BO3 资源现在按实际使用量扣减，正确性明显强于旧版
- 决胜局“资源应尽量投入”的规律能自然由宏观 DP 推出

### 不足

- 当前 Q4 局内动作层仍是 Q3 的“压缩战术动作集”，不是完整 35 个原子动作
- `counter_ready` 还没有轻量带回 Q4，所以 Q3 的“先防后攻”序列价值在 Q4 中被弱化了
- 当前故障率主要依赖 `health_my`，尚未显式加入时间因子
- `resource_time_mode` 目前采用的是“机会桶推进”口径，不是直接证明资源动作一定消耗净比赛时间

因此，当前 Q4 已经是“结构正确、可用、可解释”的版本，但如果继续提升，还可以再往更强的时序性和贴题性推进。

---

## 5. 输出文件有哪些，分别做什么

### 5.1 中间结果表

#### `data/interim/q4_base_win_prob.json`

作用：

- 保存从 Q3 提取的无资源、无故障基础单局胜率

#### `data/interim/q4_pwin_table.csv`

作用：

- 保存 18 种局内资源配额在三种情景下的单局最优胜率
- 是 BO3 宏观层和论文表格最直接的单局基准表

#### `data/interim/q4_usage_distribution.csv`

作用：

- 保存固定单局配额下，最优策略对应的：
  - 实际用了多少 `reset / pause / repair`
  - 最终赢 / 输概率
- 这是当前 Q4 最关键的新接口文件
- 宏观 DP 正是利用它按实际使用量扣减资源

#### `data/interim/q4_micro_policy.csv`

作用：

- 保存“满配额状态”下的局内最优动作表
- 便于分析局内策略切换

#### `data/interim/q4_macro_value.csv`

作用：

- 保存 BO3 各宏观状态下的最优系列赛价值

#### `data/interim/q4_macro_policy.csv`

作用：

- 保存 BO3 各宏观状态下的最优资源分配策略
- 当前也包含：
  - `expected_used_reset`
  - `expected_used_pause`
  - `expected_used_repair`

#### `data/interim/q4_exhaustive_plan.csv`

作用：

- 穷举静态分配法的 BO3 方案表
- 用于和主模型对比

#### `data/interim/q4_method_summary.csv`

作用：

- 四种方法的 BO3 蒙特卡洛胜率摘要表

#### `data/interim/q4_batch_distribution.csv`

作用：

- 将 5000 场 BO3 仿真折成批次胜率
- 供箱线图绘图使用

#### `data/interim/q4_resource_usage.csv`

作用：

- 汇总三种情景下资源使用率、首次使用时机、单局胜率和 BO3 贡献

#### `data/interim/q4_fault_profile.csv`

作用：

- 汇总最优策略下机能状态与故障率随时间的平均变化

#### `data/interim/q4_sensitivity.csv`

作用：

- 保存关键参数的单因素敏感性分析结果

### 5.2 图表文件

- `q4_policy_tree.png`：BO3 最优资源分配策略树
- `q4_pwin_heatmap.png`：三种情景下的单局胜率热力图
- `q4_fault_curve.png`：机能与故障率演化曲线
- `q4_scenario_radar.png`：三种情景资源使用风格雷达图
- `q4_method_boxplot.png`：四方法 BO3 仿真对比箱线图
- `q4_tornado.png`：参数敏感性龙卷风图

---

## 6. 涉及到哪些参数

Q4 当前用到的参数可以分成 4 层。

### 第一层：题目 / 上游模型直接继承

这些可以写成“有较强依据”：

- Q1 的：
  - `energy_cost`
  - `balance_cost`
  - `impact_score`
- Q2 的：
  - `p_block`
  - `counter_window`
  - `defense_damage`
  - `p_fall`
- Q3 的：
  - `q3_action_kernels.csv` 中的动作核
  - `q3_method_metrics.csv` 中的基础单局胜率

### 第二层：由公式和程序推导得到

这些属于“推导量”，不是原始题目直接给出的真值：

- `P_win(allocation, scenario)`
- `usage_distribution`
- `expected_used_reset / pause / repair`
- BO3 `series_value`
- 不同方法的 Monte Carlo 胜率摘要

### 第三层：有根据的建模假设

这些不是客观真值，但属于有依据的参数化设定：

- `lambda_0`
- `k_fault`
- `delta_H_pause`
- `delta_H_reset`
- `p_score_loss_per_step`
- `tau_repair_buckets`
- `tau_pause_buckets`
- `tau_reset_buckets`
- `base_win_scale`
- `fall_down_weight`

这些参数现在集中维护在：

- `data/raw/q4_fault_params.json`

### 第四层：口径性假设

这些是写论文时必须明确说明的：

- `resource_time_mode = opportunity_bucket`
  - 当前口径是：资源动作消耗的是“战术机会桶 / 节奏窗口”
  - 不是直接认定其等于净比赛时钟
- 局间机能重置
- 对手策略对称 / 由 Q3 压缩表示

---

## 7. 哪些参数“有依据”，哪些是“推出来的”，哪些是“假设”

### 有依据的

- 动作集合与资源上限
- Q1/Q2/Q3 已输出的中间参数
- BO3 三局两胜结构
- 倒地与维修两类资源的规则含义

### 推导出来的

- 单局最优胜率表 `q4_pwin_table.csv`
- 实际资源使用分布 `q4_usage_distribution.csv`
- 宏观 BO3 最优策略表 `q4_macro_policy.csv`
- 各方法仿真胜率 `q4_method_summary.csv`

### 假设参数

- 故障率函数具体形式
- 机能恢复量
- 资源动作机会桶代价
- 故障等待时的被动失分概率

写论文时，这部分必须明确写成“建模假设 / 参数设定”，不能包装成题目直接给定量。

---

## 8. 当前结果应当怎么写

当前 Q4 的正确写法不应该是：

> 本文第四问已经精确求得绝对最优真实资源方案。

更稳妥的写法应该是：

> 本文在 Q3 单局策略模型基础上，构建了“局内扩展有限时域 MDP + BO3 宏观动态规划”的双层框架。模型显式区分倒地复位与故障维修，并将单局最优策略下的实际资源使用分布向上汇总到 BO3 宏观层，从而避免了旧版本中“按分配量直接扣减资源”的偏差。

当前这版 Q4 的主要亮点有三条：

1. 资源扣减逻辑已经改为按**实际使用量**更新跨局库存
2. `reset` 和 `repair` 已经对应到两个不同状态：`down_flag` 与 `fault`
3. BO3 宏观层与单局微观层的接口已经打通

---

## 9. 当前最适合写作者提炼的几条结论

写作时，最适合提炼成论文结论的是：

- 资源价值具有明显情景差异：落后局和决胜局的资源边际价值高于领先局
- 决胜局中，宏观层倾向于更充分投入剩余资源
- 战术暂停在当前模型下使用频率最高，是最常见的节奏调整资源
- 人工复位和紧急维修的期望使用量远低于分配量，说明“配额 ≠ 实际消耗”，因此按实际使用量扣减更合理

---

## 10. 当前版本最需要在论文里如实说明的边界

下面几点不建议回避，直接说明反而更稳：

- Q4 为了保证可计算性，没有把 Q3 的全部原子动作完整搬入，而是使用了代表性战术动作集
- `counter_ready` 尚未完全带回 Q4 局内层，因此 Q3 的攻防转换结构在 Q4 中被部分压缩
- 故障率当前主要依赖机能状态，时间因子尚未显式加入

这些边界不会否定 Q4，反而能说明当前版本是一个“结构正确、接口清楚、可继续增强”的模型。

---

## 11. 写作者建议的叙述主线

如果直接写第四问，推荐按下面这个顺序展开：

1. 先说明：Q3 只解决单局、无故障、无资源的策略优化
2. 再说明：Q4 比 Q3 多了两层复杂性
   - 故障随机性
   - BO3 跨局资源权衡
3. 引出：因此需要双层模型，而不是单层继续加规则
4. 介绍局内层
   - 扩展状态
   - 三类资源动作
   - 倒地与故障分离
5. 介绍宏观层
   - BO3 状态
   - 按实际使用量扣减资源
6. 再做方法对比
   - 固定规则法
   - 穷举静态法
   - 主模型
7. 最后用图表和表格总结：
   - 资源使用规律
   - 单局胜率表
   - BO3 总体最优资源调度策略

如果后面还要继续增强 Q4，最优先的方向是：

1. 把 `counter_ready` 轻量带回 Q4
2. 故障率函数加入时间因子
3. 进一步核对“资源动作是否计入净比赛时间”的规则口径
