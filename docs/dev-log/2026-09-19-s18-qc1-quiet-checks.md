# S18-QC1: NNUE 安静将军边界试验 — 结果与决策

日期：2026-09-19。计划（预声明门槛）：[s18-qc1-plan.md](../diagnostics/s18-qc1-plan.md)。

## 单一候选

编译期 `experimental_nnue_qchecks`（默认 OFF）：仅 CurrentFinal + NNUE，在 qsearch
非被将军节点、stand-pat 未截断后，追加 **qply 0/1 的合法安静将军**；qply≥2 不再生成。
被将军节点、现有战术走法、MAX_QPLY/和棋/中断/NNUE 栈规则全部不变。不新增 UCI 面。

两臂同一工作区源码（HEAD `ccf95fb` + tracked diff）：
base `d846def9…`，qc1 `8014d269…`；S14 模型 `329b7170…`；两臂全套测试 **519/519 通过**。

## 永久回归

`s18_game127_quiet_mate_boundary_and_unwind`（合成零残差 NNUE，不依赖 12MB 模型文件）：
- `after_Qf3+` 的两个合法应将下 `…Qg2#` 均被 `nnue_quiet_checks` 命中且不重复、不为吃子；
- 每个应将子树走完后棋盘/Zobrist/路径/NNUE 评估**逐位复原**；
- 无限预算下：qc1 报 `−(MATE−2)`；**base 保持已标注的已知缺陷**（不许静默宣称默认已修）；
- 1/2 节点预算正确中止。`s18_quiet_check_budget…` 锁预算条件（CurrentFinal+NNUE+qply<2）。
- 双臂各 519/519 全套通过（含既有 qsearch/终局/栈/TT 回归）。

## 测量结果（预声明门槛）

**① 127 边界（核心目标）：PASS**
`after_Qf3_check` 深度1：base `+200cp Kg1`（走进将死）→ qc1 **`mate −1 Kh2`**。
终点 `Kg1` 深度1 两臂同报 `Qg2# mate +1`。

**② 三局根（同 HCE 根阶梯）：**

| 根 | base | qc1 |
|---|---|---|
| g127 d5 | `Nxe5 +200`（坏招） | **`Kh2 +95`（提前一整层弃坏招）** |
| g127 nodes50k | `Nxe5 +200`(d5,23k节点) | **`Kh2 +81`(d4,39k节点)** |
| g175 全档 | `Qc1`→d8 `Nh2` | 与 base 完全相同 |
| g187 全档 | `Nd5`（d8 才改） | 与 base 完全相同 |

g175（牵制盲区）与 g187（补偿乐观）**未被安静将军修复** —— 两类问题机制不同（见结论）。

**③ 战术回归（23 例，S14 NNUE）：**
23-case panel **没有发现经复测确认的实质回归**。
- 修好 base 的 2 个既有失败：`kqk-mopup`（cp1435→mate2）、`underpromotion-options`（h1g2→a7a8q）；
- `krk-mopup` 在固定 d3 出现 **move-order/search-horizon 差异**（qc1 选 mate3 的另一路），
  d4/d5 收敛到相同 `f6g6 mate 2` —— 保留在报告里，不作为回归也不卡候选：QC1 改 qsearch
  move set，本来就可能改变浅层排序；
- 4 个失败两臂共有（`d10-promotion-chain-black`、`d10-unique-underpromotion`、`d10-xray-recapture`、`d10-king-recapture`）：HCE 时代语料的既有 NNUE 差距，不归因 QC1。

**④ 成本（32 局 × 深度4 × 3 轮 AB/BA 交错）：**
节点比 qc1/base：**中位 1.26**（门槛≤1.50 ✅），p90 **2.09**（≤3.0 ✅）；
墙钟比：中位 **1.32**（≤1.50 ✅），p90 **2.94**。
**p90 尾部扩张是真实警告，不做优化**：是否值得付这代价由 256 局限时赛回答。

## 判定口径（owner 2026-09-19）

- **比赛结果优先于 corpus**：256 局里出现明确负向信号就关掉候选，不因修了 game127 保送。
- 接近 parity 也先视为“机制修复但未证明值得付成本”，**不立即并入默认**。
- QC1 只处理 A 类（qsearch 可见性）；B 类（失衡局面的 stand-pat 乐观）下一轮**先做诊断**
  （落后叶子的错误 stand-pat cutoff 是否与 residual−material-deficit 稳定相关），
  再决定是 suppress / tighten / margin-aware / 条件化，不直接跳到 heuristic。

## 结论

1. **QC1 通过全部预声明采纳门槛**（边界、无倒退、127 提前纠错、成本），唯一钉深度差异已复核为排序扰动。建议进入限时赛验证，**不是**自动进生产。
2. **根因已证实为结构性**：HCE 时代整定的一整套搜索假设（qsearch 战术集=吃子/EP/升变、
   delta 边际 500cp、futility 边际、stand-pat 语义）原样带入了 NNUE。NNUE 的材料残差评估
   在失衡局面上系统性乐观（g175 失马节点残差 +525cp vs 材料 −390；g187 +190 vs −220），
   使 qsearch 的 stand-pat 恰好在最危险的节点提前截断；安静将军又恰好在截断边界不可见。
   127 局是两件事叠加的完美风暴：叶静态 +200 == 搜索错误答案，下一步是边界外的安静杀。
3. **安静将军只修边界可见性这一类**；g175/g187 的补偿乐观类需要别的杠杆（评估侧或更深的
   失衡局面处理），留作下一候选。

## 下一步（阶段计划，非 fixture）

1. **QC1 限时赛**：256 局 / 128 配对 / 10+0.1 / 321–448 / 双臂同 S14 模型（同 s17_ab_match 流程）。
   采纳线预声明：整数一致 + PARITY 不丢（同 S17b 性能标准）。
2. **NNUE 搜索适配阶段（大方向）**：按同一纪律（候选→corpus+成本门槛→限时赛）依次检验
   HCE 遗留常数在 NNUE 下的失配：delta/futility 边际、失衡局面 stand-pat 策略、
   check 扩展（S7.5B 在 NNUE 下重开）。每项一个候选，失败即关，不做超参扫描。
3. 搜索侧排空后才回到模型/数据。

工件：`tools/diagnostics/run_s18_qc1_measure.py`、
`C:/Users/81489/AppData/Local/Temp/eureka-s18-qc1/measurements.json`（97+192 次搜索全记录）。
工作区改动待提交：`Cargo.toml`、`src/engine/search.rs`（候选+2 回归）、4 个诊断脚本、计划文档。
