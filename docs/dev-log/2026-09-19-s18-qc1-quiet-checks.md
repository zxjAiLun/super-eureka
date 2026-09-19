# S18-QC1: NNUE 安静将军边界试验 — 测量、比赛与候选关闭裁决

日期：2026-09-19。计划（预声明门槛）：[s18-qc1-plan.md](../diagnostics/s18-qc1-plan.md)。
提交基准：`9db0e85`（同 commit 编译两臂，同 S14 模型）。
比赛工件：`results/s18-ab/qc1-match-report.json`、`qc1-pairs.json`（128 对小型证据）。
完整 `qc1-match.pgn` 和日志仅本地保留，不提交；报告包含其 SHA256。

---

## 一、候选定义

编译期 `experimental_nnue_qchecks`（默认 OFF）：仅 CurrentFinal + NNUE，在 qsearch
非被将军节点、stand-pat 未截断后，追加 **qply 0/1 的合法安静将军**；qply≥2 不再生成。
被将军节点、现有战术走法、MAX_QPLY/和棋/中断/NNUE 栈规则全部不变。不新增 UCI 面。

比赛两臂从同一源码 commit `9db0e85` 重建（QC1 feature OFF/ON）：
- base: `19f37de72f22bbe6659d12dd6100883cc0d0733a174bfbb53a8f6503d3883028`
- qc1: `f2b7a651c87d5e748ca4f0a5fde9f32ef8c18fc195ddd7a3824c52a9f6d8b3f7`
- S14 模型: `329b717066082798d2f42d9e4f137385a1482a8cda03ae7849ddc654e097e4b0`
- 519/519 是此前同源码工作区双构建的测试记录；提交后重建两臂做了边界 d1 冒烟，未重跑全套。
- 原成本/局面面板使用此前 base `d846def9…` / qc1 `8014d269…`，不与比赛构建 SHA 混写。

---

## 二、永久回归

`s18_game127_quiet_mate_boundary_and_unwind`（合成零残差 NNUE，不依赖 12MB 模型文件）：
- `after_Qf3+` 的两个合法应将下 `…Qg2#` 均被 `nnue_quiet_checks` 命中且不重复、不为吃子；
- 每个应将子树走完后棋盘/Zobrist/路径/NNUE 评估**逐位复原**；
- 无限预算下：qc1 报 `−(MATE−2)`；**base 保持已标注的已知缺陷**（不许静默宣称默认已修）；
- 1/2 节点预算正确中止。`s18_quiet_check_budget…` 锁预算条件（CurrentFinal+NNUE+qply<2）。
- 双臂各 519/519 全套通过（含既有 qsearch/终局/栈/TT 回归）。

---

## 三、机制与局面测量结果（Corpus Panel）

**① 127 边界（核心目标）：PASS**
`after_Qf3_check` 深度 1：base `+200cp Kg1`（走进将死）→ qc1 **`mate −1 Kh2`**。
终点 `Kg1` 深度 1 两臂同报 `Qg2# mate +1`。

**② 三局根（同模型、双臂对照）：**

| 根 | base | qc1 |
|---|---|---|
| g127 d5 | `Nxe5 +200`（坏招） | **`Kh2 +95`（提前一整层弃坏招）** |
| g127 nodes 50k | `Nxe5 +200`(d5, 23k 节点) | **`Kh2 +81`(d4, 39k 节点)** |
| g175 全档 | `Qc1` → d8 `Nh2` | 所测档位 bestmove 相同；score/nodes 并非逐位相同 |
| g187 全档 | 所测根档位保持 `Nd5`，50k 档 `Rhd1` | 所测档位 bestmove 相同；d8 改招来自此前生产阶梯 |

g175（牵制盲区）与 g187（补偿乐观）**未被安静将军修复** —— 两类问题机制不同（见根因分析）。

**③ 战术回归（23 例，S14 NNUE）：**
23-case panel **没有发现经复测确认的实质回归**：
- 修好 base 的 2 个既有失败：`kqk-mopup`（cp1435 → mate 2）、`underpromotion-options`（h1g2 → a7a8q）；
- `krk-mopup` 在固定 d3 出现 **move-order/search-horizon 差异**（qc1 选 mate 3 的另一路），d4/d5 收敛到相同 `f6g6 mate 2` —— 保留在报告里，不作为回归也不卡候选：QC1 改 qsearch move set，本来就可能改变浅层排序；
- 4 个失败两臂共有（`d10-promotion-chain-black`、`d10-unique-underpromotion`、`d10-xray-recapture`、`d10-king-recapture`）：HCE 时代语料的既有 NNUE 差距，不归因 QC1。

**④ 成本（32 局 × 深度 4 × 3 轮 AB/BA 交错）：**
- 节点比 (qc1/base)：**中位 1.26×**（门槛 ≤1.50 ✅），p90 **2.09×**（门槛 ≤3.0 ✅）；
- 墙钟比 (qc1/base)：**中位 1.32×**（门槛 ≤1.50 ✅），p90 **2.94×**；
- **p90 尾部扩张是真实警告**：是否值得付这代价由 256 局限时赛回答。
- [`qc1-cost-evidence.json`](../../results/s18-ab/qc1-cost-evidence.json) 保存 96 对节点/墙钟读数及原测量文件 SHA；
  复算沿用旧驱动的 upper-median 与 p90 索引。墙钟按 0.1ms 精度保存，零基线分母使用旧驱动的 0.1ms 下限，
  因而极小局面的比值存在量化误差；不是高精度 CPU 测量。

---

## 四、256 局限时赛结果

- **协议**：256 局，128 配对开局（`openings.epd` 321–448），双向换色，10+0.1，Hash 16，单线程，concurrency 8。
- **双方握手**：
  - A-qc1: `profile=current-final eval=nnue-v2q network=nnue-v2q evalfile=nnue-s14-datasupply-v5.bin`
  - B-base: `profile=current-final eval=nnue-v2q network=nnue-v2q evalfile=nnue-s14-datasupply-v5.bin`
- **战绩**：
  - **胜 99 / 和 42 / 负 115**（净负 16 局）
  - 得分：**120.0 / 256（46.88%）**
- **配对 Pentanomial [LL, LD, DD/WL, WD, WW]（128 个开局对）**：
  $$\mathbf{[20,\ 23,\ 53,\ 17,\ 15]}$$
  - 双负 (LL): 20 对
  - 一负一和 (LD): 23 对
  - 和/胜负对冲 (DD/WL): 53 对
  - 一胜一和 (WD): 17 对
  - 双胜 (WW): 15 对
- **Elo 统计**：
  - Cutechess 直测：**-21.7 ± 39.1 Elo**，LOS: 13.7%，DrawRatio: 16.4%
  - CI 跨零；FAIL 是冻结的 `<48%` 经济筛选门，不等于统计上证明真实 Elo 为负。
- **耗时情况**：
  - 比赛总壁钟：863.7s（约 14.4 分钟）
  - 修正后按 FEN side-to-move 归属：QC1 15,495 个有时间记录的走法、合计 3342.79s、均值 0.215733s；
    base 15,487 个、3342.34s、均值 0.215816s。旧解析器误将所有开局当白方先走；结果胜负不受影响。
    另分别有 120/136 个走法没有可解析时间注释，未把它们伪记为 0 秒。
  - PGN 时间为 rounded move-time，不是 CPU 时间；未采集比赛逐节点计数或平均有效主搜索深度。
  - 观察：QC1 明显增加了 qsearch 工作量；在固定 10+0.1 下候选最终净负 16 局，因此该额外成本没有转化成总体棋力收益。成本是合理解释，但未证明 16 局差全部由成本造成。

---

## 五、裁决：FAIL / KILL（候选关闭）

按 owner 冻结的筛选规则：
> **“比赛结果优先于这轮局面 corpus。也就是说，QC1 如果在 256 局里出现明确负向信号，就关掉；不能因为它修了 game127 就继续保送。”**
> **46.88% < 48.0% ⇒ FAIL (kill)**

- **处置**：
  1. `experimental_nnue_qchecks` 默认关闭保持不变，**不并入默认，不进入生产**；
  2. 永久回归 `s18_game127_quiet_mate_boundary_and_unwind` 与条件门保留在代码库中（已锁 base 缺陷标注与 budget 条件保护）；
  3. 彻底停止在安静将军方向上的调优或变体尝试。

---

## 六、根因结论与下一阶段路线

### 1. 结构性失配的清晰划分
- **A 类（qsearch 可见性缺失）**：QC1 修正了 game127 边界并提前弃坏招，但限时赛没有显示总体收益；不将成本与负战绩之间的合理解释写成已测因果。
- **B 类（失衡局面下的 NNUE stand-pat 乐观假设）**：g175 的 −390cp 材料被约 +525cp 残差抵消，g187 另一后续节点按白方视角为 −220+190cp。QC1 未修好这些案例，支持分开诊断，但没有实际 alpha/beta trace，不能据此证明“错误通过 alpha 截断”。硬 stand-pat cutoff 的条件是 `stand_pat >= beta`。

### 2. B 类最终收口（不是待开工计划）

见 [B 类有限证据报告](../diagnostics/b_class_standpat_diagnosis.md)：最终只保存 40 个参考评分样本，
未记录实际搜索窗口/cutoff；旧日志标 d6 与源码请求 d8 不一致，完成深度缺失。
没有足够证据支持标量 heuristic，执行 **STOP**。不把未发现有用分离能力写成不存在可分区域。

```text
mechanism diagnosis PASS
cost gate           PASS but expensive
256-game screen     FAIL
candidate           KILLED
default/production  NO
B-class scalar work STOP (no supporting evidence; no heuristic)
```

收口验证：Python runner/证据复算及负向控制；Rust 搜索和模型未改，不重跑对局或 Rust 套件。

## 相关的结论级工件

此前四个模型有效重跑的最终小报告也保留，完整 PGN/log 不进 Git。
离线按 FEN/换色复核均为 256 局/128 对，WDL 与 pentanomial 与报告一致；SHA 记录在各报告中。

| 候选 | 对 S14 的 W/D/L | score | 最终报告 |
|---|---|---|---|
| S15-short | 105/38/113 | 48.44% | [report](../../results/s15short-valid/screen-report.json) |
| S15-long | 92/31/133 | 41.99% | [report](../../results/s15long-valid/screen-report.json) |
| S16 | 78/29/149 | 36.13% | [report](../../results/s16-valid/screen-report.json) |
| S17 | 92/28/136 | 41.41% | [report](../../results/s17-valid/screen-report.json) |

旧 `results/s15{,-long}/`、`results/s16/`、`results/s17/` 下的 screen 报告保留原数字，
顶层 verdict 为 `INVALID_EVALUATOR_SELECTION`；不与上述有效结果混用。
