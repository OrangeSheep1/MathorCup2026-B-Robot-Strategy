# MathorCup 2026 B题：机器人竞技策略优化

本仓库为 2026 年第十六届 MathorCup 数学应用挑战赛 B 题的最终发布版代码与结果整理。项目围绕机器人竞技中的动作评价、防守匹配、单场策略优化和 BO3 资源调度展开，形成了从 Q1 到 Q4 的完整建模链路。

## 最终版说明

当前版本已经完成四个问题的建模、仿真、诊断和图表输出：

- Q1：构建 13 个攻击动作的基础特征、效用、风险和综合评价。
- Q2：构建 22 个防守动作与攻击动作之间的 pair-level 攻防匹配关系。
- Q3：基于 Q1/Q2 输出建立单场有限时域 MDP，输出状态依赖策略、过程指标和胜率评估。
- Q4：在 Q3 最终基线之上建立 BO3 资源调度模型，优化人工复位、战术暂停、紧急维修的使用时机与配额。

本仓库保留正式建模代码、输入数据、中间接口、最终图表和写作指南；调试脚本、临时备份和 Python 缓存已从发布版中清理。

## 项目结构

```text
MathorCup2026-B-Robot-Strategy/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── raw/          # 原始资料、题目附件、手工整理参数
│   ├── interim/      # Q1-Q4 中间结果、接口表、诊断表
│   └── output/       # 最终图表、写作指南、论文插图
├── docs/             # 题目原文、方法框架、说明文档
├── scripts/          # 一键运行入口
└── src/
    ├── common.py
    ├── io_utils.py
    ├── q1/           # 攻击动作评价
    ├── q2/           # 防守匹配建模
    ├── q3/           # 单场策略 MDP
    └── q4/           # BO3 资源调度
```

## 环境安装

建议使用虚拟环境运行：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

主要依赖包括 `numpy`、`pandas`、`matplotlib`、`seaborn`、`scipy`、`scikit-learn`。

## 运行方式

单题运行：

```bash
python scripts/run_q1.py
python scripts/run_q2.py
python scripts/run_q3.py
python scripts/run_q4.py
```

完整链路运行：

```bash
python scripts/run_all.py
```

说明：

- Q1/Q2 是 Q3/Q4 的输入基础，完整复现时建议按 Q1 -> Q2 -> Q3 -> Q4 顺序运行。
- Q3 蒙特卡洛仿真和 Q4 BO3 动态规划耗时相对较长。
- Q4 单次完整运行通常需要二十分钟量级，具体取决于机器性能。

## 建模链路

### Q1 攻击动作评价

Q1 将攻击动作拆解为执行时间、得分概率、能耗、风险、动作效用等特征，形成后续模型统一调用的攻击动作表。

关键输出：

- `data/interim/action_features.csv`
- `data/output/q1_writer_guide.md`

### Q2 防守匹配建模

Q2 在 pair-level 上评价每个防守动作对每个攻击动作的拦截、反击、恢复和风险表现，区分主防守、保底响应、倒地恢复等语义层。

关键输出：

- `data/interim/defense_pair_scores.csv`
- `data/interim/defense_matchup.csv`
- `data/interim/defense_features.csv`
- `data/output/q2_writer_guide.md`

### Q3 单场策略优化

Q3 读取 Q1/Q2 的最终接口，构建状态敏感 MDP。状态包括比分差、时间阶段、双方机能、倒地恢复锁、反击准备态等；动作空间按攻击、防守、恢复和特殊语义约束。模型统一了动作核、Bellman 求解、仿真奖励和诊断分解口径。

关键输出：

- `data/interim/q3_method_metrics.csv`
- `data/interim/q3_policy_table.csv`
- `data/interim/q3_state_qvalue_summary.csv`
- `data/interim/q3_process_metrics.csv`
- `data/interim/q3_composite_score.csv`
- `data/output/q3_main_summary.png`
- `data/output/q3_trajectory.png`
- `data/output/q3_writer_guide.md`

最终 Q3 单场基线结果：

| 场景 | MDP 胜率 |
| --- | ---: |
| 领先局 | 1.000 |
| 平局局 | 0.544 |
| 落后局 | 0.040 |

Q3 的论文表达重点不是单纯强调胜率暴涨，而是说明 MDP 在保持合理胜率的同时，提升了领先、平局、落后场景下的策略分化、攻防切换和状态适配性。

### Q4 BO3 资源调度

Q4 以 Q3 最终基线为无资源校准目标，在 BO3 赛制下同时优化局内战术动作与跨局资源分配。资源包括人工复位、战术暂停和紧急维修。模型保留“按实际使用量扣减资源”的宏观动态规划机制，并输出资源使用时机规则。

关键输出：

- `data/interim/q4_input_audit.csv`
- `data/interim/q4_zero_resource_baseline.csv`
- `data/interim/q4_method_summary.csv`
- `data/interim/q4_resource_timing_rules.csv`
- `data/interim/q4_resource_uplift_diagnostics.csv`
- `data/interim/q4_resource_value_gap_summary.csv`
- `data/interim/q4_composite_score.csv`
- `data/output/q4_main_summary.png`
- `data/output/q4_resource_policy_heatmap.png`
- `data/output/q4_policy_tree.png`
- `data/output/q4_writer_guide.md`

Q4 零资源校准结果：

| 场景 | Q3 基线 | Q4 零资源 |
| --- | ---: | ---: |
| 领先局 | 1.000 | 1.000 |
| 平局局 | 0.544 | 0.544 |
| 落后局 | 0.040 | 0.040 |

Q4 BO3 方法对比：

| 方法 | BO3 胜率 | 95% CI |
| --- | ---: | --- |
| 最优 DP | 0.6578 | [0.6446, 0.6710] |
| 穷举静态 | 0.6196 | [0.6061, 0.6331] |
| 首局全投 | 0.6160 | [0.6025, 0.6295] |
| 固定规则 | 0.4544 | [0.4406, 0.4682] |

Q4 综合表现指数：

| 方法 | 综合指数 |
| --- | ---: |
| 最优 DP | 0.9518 |
| 穷举静态 | 0.8290 |
| 首局全投 | 0.8178 |
| 固定规则 | 0.6813 |

说明：综合指数用于论文展示中的多指标比较，不等同于原始胜率。

## 主要图表

最终论文建议优先使用以下图表：

- `data/output/q3_main_summary.png`：Q3 场景动作占比、过程指标和 Q-gap 综合图。
- `data/output/q3_trajectory.png`：Q3 三场景代表轨迹对比。
- `data/output/q4_main_summary.png`：Q4 资源增益、使用时机、方法对比和综合评分主图。
- `data/output/q4_resource_policy_heatmap.png`：Q4 资源触发规则热图。
- `data/output/q4_policy_tree.png`：BO3 宏观资源调度策略树。

## 数据接口

正式接口文件主要位于 `data/interim/`：

- Q1 -> Q3/Q4：`action_features.csv`
- Q2 -> Q3/Q4：`defense_pair_scores.csv`、`defense_matchup.csv`、`defense_features.csv`
- Q3 -> Q4：`q3_method_metrics.csv`、`q3_kernel_table.csv`、`q3_policy_table.csv`、`q3_state_qvalue_summary.csv`
- Q4 解释输出：`q4_resource_timing_rules.csv`、`q4_resource_value_gap_summary.csv`、`q4_alloc_vs_actual_usage.csv`

## 结果口径

- Q3 的 MDP 胜率来自单场蒙特卡洛仿真。
- Q4 的 BO3 胜率来自系列赛仿真。
- Q4 零资源基线经过 Q3 结果校准，保证第四问没有脱离前三问。
- Q4 综合指数是为展示“胜率、领先稳定性、平局突破能力、落后韧性”的综合表现而构造的论文指标，不应替代原始胜率。

