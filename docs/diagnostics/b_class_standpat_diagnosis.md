# B 类 stand-pat 乐观度探针：有限证据收口 / STOP

日期：2026-09-19。最终证据：[`b_class_detailed_diagnosis.json`](b_class_detailed_diagnosis.json)（40 行，原始数字保留）。
复算：`python tools/diagnostics/diagnose_b_class_detailed.py`，仅读取证据，不启动引擎。

## 决策与边界

> **No useful scalar separation observed in the labeled probe.**
> **Decision: STOP scalar stand-pat heuristic work.**

本次样本与已试简单规则没有提供足够证据支持继续实现 residual/material-based
stand-pat heuristic。生产搜索不改，不新增候选。
**这不是证明不存在可分区域，也不是对整个搜索分布的否定结论。**

## 实际做了什么

- 使用 S14 模型 `329b717066082798d2f42d9e4f137385a1482a8cda03ae7849ddc654e097e4b0`，
  `current-final + NNUE`，通过 `diagnostic_eval_site_capture` 捕获 qsearch 评估站点。
- g175/g187 的 5 个分支根 + 战术语料前 4 个根，各 10,000 节点。
  日志记录候选池 **9,982 个** unique materially-behind FEN（材料亏损至少 100cp，含人工加入的根）。
  **候选池数量不等于已标注数量。**
- 第一次计划 80 个、请求 d8，300 秒超时前打印到 75/80；没有完整持久结果，不作正式样本。
- 后续完成并保存的只有 **40 个随机样本**，不是“40 个典型/代表性样本”。
  原采样从无序 set 转 list，即使固定 random seed 也无法跨进程复现；以冻结 JSON 的 FEN 为准。
- **深度口径纠错**：当时改了函数默认值与日志为 d6，但调用处仍为
  `search_ground_truth(fen, depth=8)`。因此屏幕的“depth 6”并不可信；源码请求为 d8，
  JSON 没保存逐项 completed depth / bounds / PV / binary SHA，不能补称为“已验证 d6（或 d8）真值”。
- 当前保留的同名工具已改为**证据复算器**，不是有缺陷的旧采集器，也不会悄悄生成新样本覆盖历史。

## 字段含义与未测到的东西

`material_deficit`、`residual_cp`、`full_eval`、`net_compensation`、
`ground_truth`、`optimism_gap` 保留旧字段名以便审计。

- `ground_truth` 实为**同一引擎、同一 NNUE 的有界搜索参考分**，不是独立 oracle 或博弈真值。
- `±10000` 是旧采集器将 mate 强行映射的 sentinel，**不是 cp**；不能与 cp 混算平均 gap 或作因果归因。
- `full_eval` 是旧工具 `material + round(float residual)` 的复算值；未保存原始整数输出，
  且未处理引擎的 exact mop-up override，不能保证每项等于搜索实际使用的 stand-pat。
- **未记录 alpha、beta、qply、真实 cutoff 标志、完整历史或 continuation bounds**。
  因此无法从这份 JSON 判定真实 false stand-pat cutoff。实际硬截断比较的是 beta，
  alpha raise 也不等于硬截断。
- `net_compensation = residual - deficit` 在这里基本就是 `full_eval`；
  `full_eval - material` 基本就是 residual，不能把它们当多项独立证据。

## 40 个样本能支持的观察

旧脚本的 proxy 标签为 `full_eval >= -150 && reference_score <= -300`。
**40 行中该标签的正例为 0。** “未满足标签”不能改称“已证明真实补偿”或“sound”。

| 简单规则 | TP | FP（相对 proxy 标签） | FN | TN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---|
| deficit ≥300 & residual ≥200 | 0 | 16 | 0 | 24 | 0/16 | 未定义（无正例） |
| deficit ≥300 & residual ≥300 | 0 | 14 | 0 | 26 | 0/14 | 未定义（无正例） |
| deficit ≥300 & residual ≥400 | 0 | 11 | 0 | 29 | 0/11 | 未定义（无正例） |
| deficit ≥200 & net_comp ≥−100 | 0 | 5 | 0 | 35 | 0/5 | 未定义（无正例） |
| deficit ≥300 & net_comp ≥0 | 0 | 2 | 0 | 38 | 0/2 | 未定义（无正例） |
| residual ≥300 | 0 | 15 | 0 | 25 | 0/15 | 未定义（无正例） |
| residual ≥400 | 0 | 12 | 0 | 28 | 0/12 | 未定义（无正例） |

这说明这些规则在本样本上大量触发却未命中 proxy 正例，**不是“Separability=0”的总体证明**。
无独立留出集、无按 root 分组评估；不能推断召回率或泛化性能。

有价值的反例（完整 FEN 见原始 JSON，不将以下数字外推为 >90%）：

| 材料亏损 cp | residual cp | full eval cp | 同引擎参考 cp |
|---:|---:|---:|---:|
| 290 | +796.1 | +506 | +641 |
| 720 | +607.4 | −113 | +342 |
| 520 | +386.0 | −134 | +70 |

**材料落后很多 + residual 很高 ≠ 必然错估。** 上述搜索参考分仍为正，足以警示不要凭这两个标量
直接压制 stand-pat；但并未独立证明每个局面的进攻/反杀真实成立。

旧 gap ≥500 的仅 3 条：两条参考值为 mate sentinel，另一条参考值 −1375cp、gap 562cp。
后者没有保存战术验证线，**不能写“极端 gap 全部由 mate/具体战术导致”**。

g175 `after_Qxf3` 的材料 −390、残差约 +525、完整静态 +135 是独立已知案例，
不是这 40 条中已测 cutoff 的正例。原始完整历史阶梯见
[三局诊断](../dev-log/2026-09-19-s14-knight-search-ladder.md)。

## 收口

STOP 的依据是**没有足够的正向支持证据**，不是完成了一个“证明标量不可分”的实验。
保留 40 行、小型复算工具和上述限制；删除初版重复探针和过程产物。
不重跑、扩样、扫阈值或写 heuristic。后续只有新的明确授权与测量协议才能重开。
