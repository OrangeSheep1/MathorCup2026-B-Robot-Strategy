# MathorCup 2026 B题：机器人竞技策略优化

## 项目简介
本仓库用于 MathorCup 2026 B题的建模协作与代码整理。项目围绕机器人竞技策略优化展开，主线覆盖 Q1 到 Q5：攻击动作评价、 防守动作匹配、单场策略优化、BO3 资源决策，以及最终建议书整理。

题目核心对象包括 13 种攻击动作、22 种防守动作，以及 BO3 赛制下人工复位、战术暂停、紧急维修等资源约束。

## 项目目标
- 为 Q1 到 Q5 提供清晰、轻量、可协作的目录结构
- 将数据、中间结果、脚本和写作材料分开管理
- 保持比赛开发期可快速迭代，不预设过重工程框架
- 明确 Q1、Q2 对 Q3、Q4 的中间数据支撑关系

## 项目结构
```text
MathorCup2026-B-Robot-Strategy/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ output/
├─ docs/
│  ├─ notes.md
│  └─ paper_outline.md
├─ debug/
│  ├─ q1_debug.py
│  ├─ q2_debug.py
│  ├─ q3_debug.py
│  └─ q4_debug.py
├─ src/
│  ├─ common.py
│  ├─ io_utils.py
│  ├─ q1/
│  ├─ q2/
│  ├─ q3/
│  ├─ q4/
│  └─ q5/
└─ scripts/
   ├─ run_q1.py
   ├─ run_q2.py
   ├─ run_q3.py
   ├─ run_q4.py
   └─ run_all.py
```

## 建模主线
- Q1：攻击动作评价。整理 13 种攻击动作的效用、得分潜力、风险与代价。
- Q2：防守动作匹配。围绕 22 种防守动作构建对抗匹配关系与防守评分。
- Q3：单场策略优化。基于 Q1、Q2 的结果设计单场对抗中的动作与策略选择。
- Q4：BO3 资源决策。结合赛制资源约束，研究人工复位、战术暂停、紧急维修等资源如何分配。
- Q5：建议书整理。汇总模型结论、策略建议与论文表达。

说明：
- Q1 会产出攻击动作特征，供 Q3、Q4 调用。
- Q2 会产出攻防匹配结果，供 Q3、Q4 调用。
- Q3 侧重单场层面的策略选择。
- Q4 侧重系列赛层面的资源调度与决策。

## 数据接口约定
建议统一维护以下中间文件，作为问题之间的数据接口：

1. `data/interim/action_features.csv`

建议字段：
`action_id, action_name, category, impact_score, balance_cost, score_prob, energy_cost, utility`

2. `data/interim/defense_matchup.csv`

建议字段：
`attack_id, defense_id, block_prob, counter_window, fall_risk, defense_score`

约定说明：
- `data/raw/` 存放题目原始资料、动作说明、手工整理表。
- `data/interim/` 存放 Q1、Q2 等中间结果。
- `data/output/` 存放最终图表、结果汇总、论文插图等输出内容。

## 环境安装
建议使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行方式
```bash
python scripts/run_q1.py
python scripts/run_q2.py
python scripts/run_q3.py
python scripts/run_q4.py
python scripts/run_all.py
```

说明：
- 当前脚本为比赛期占位入口，后续可逐步接入各题 pipeline。
- Q5 主要偏写作整理，当前以 `src/q5/summary.py` 为汇总入口占位。

## 协作建议
- 优先把原始数据和手工整理内容放入 `data/raw/`。
- Q1、Q2 完成后，尽快统一中间字段命名，避免 Q3、Q4 反复改接口。
- 代码按“题号 + 功能角色”组织，不在顶层绑定具体算法名，便于后续换方法。
- 图表分别放在各题目录内的 `plot.py` 中维护，避免全局可视化文件混杂。
- 会议记录、建模思路、论文草稿提纲统一放在 `docs/`。

## 当前进度清单
- [x] 建立基础目录结构
- [x] 建立 Q1 到 Q5 的代码占位
- [x] 建立调试脚本入口
- [x] 建立运行脚本入口
- [x] 建立数据目录与文档目录
- [ ] 补充原始题目数据与动作整理表
- [ ] 明确 Q1 指标体系与评分口径
- [ ] 明确 Q2 攻防匹配矩阵构造方式
- [ ] 设计 Q3 单场策略模型
- [ ] 设计 Q4 BO3 资源调度模型
- [ ] 完成 Q5 论文与建议书整理
