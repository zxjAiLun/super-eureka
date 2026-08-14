# S7.3 — Selectivity Attribution（选择性归因，OBSERVATION ONLY）

STATUS: **APPROVED_WITH_REPAIR** — 数据有效；报告口径已按 review 修复
（proposed vs applied LMR、history 百分比、null funnel 措辞、R2 推论删除）。
主诊断修正为：**SELECTIVITY_TOO_CONSERVATIVE**，更精确机制为
**LMR_APPLICATION_SUPPRESSED_ON_NULL_WINDOW_NODES**（见 §3）。

- BASE: `d1db1af`（chess 语义 = CurrentFinal，本轮零搜索语义改动）
- Profile: `current-final`，16MB cold TT，threads 1
- Corpus: 80 位 S7（SHA `8786ffca…`）× depth 6 + depth 7；
  **aggregate 仅 80 unique S7 位**（S4 30 位是 A1 子集，不再重复加权）
- Production invariance: 30 S4 d6 nodes/score/bestmove/PV **30/30 exact**
- fmt / clippy -D warnings / debug tests / release tests 全绿
- 数据: `results/s7/s73-selectivity-attribution.json`
  （collector 曾有子键解析 bug：`prefix:sub0=v0,sub1=v1` 被 flat tokenizer
  合并；原始字符串无损，`--reparse` 修复后重算 totals/aggregate，未重跑引擎）

## 1. 研究问题

S7.2 证明大 move-loop 的主体不是"cutoff move 排太后"（late cutoff 仅
1–2%），而是 ~1.84M 个**无 beta cutoff** 的节点（mean 14.6 moves），
占 main-loop searched moves 的 ~73%（26.90M / 36.9M）。S7.3 回答这些
大树到底是什么、当前 selectivity（futility / null / LMR）在哪里留了头寸。

## 2. Aggregate（80 位 × d6+d7 合计）

### 2.1 No-beta-cutoff 节点构成（回答"大树是什么"）

| 维度 | 值 | 含义 |
|---|---|---|
| move-loop 节点 | 7,395,914 | |
| 无 cutoff loop 节点 | 1,841,811（24.903%）| 但占 ~73% searched moves |
| 其中 PV / wide-window | **7,049（0.38%）** | **PV 完全不是原因** |
| 其中 non-PV | 1,834,762（99.62%）| 主体是 non-PV fail-low |
| 其中 in-check | 119,569（6.50%）| 少数 |
| 其中尝试过 null-move | 2,011（0.11%）| null 几乎不在这类节点触发 |
| mean searched / 无cutoff节点 | 14.605 | 与 S7.2 口径一致（14.683）|

### 2.2 无 cutoff 节点里被搜 quiet 的 history 分布（搜索前读取）

| bucket | searched | 占比（分母 = 五桶合计 23,156,641）|
|---|---|---|
| ≤0 | 17,133,747 | **74.0%** |
| 1–15 | 2,039,270 | 8.81% |
| 16–63 | 1,440,587 | 6.22% |
| 64–255 | 1,206,828 | 5.21% |
| 256+ | 1,336,209 | 5.77% |

约 **74%** 在无 cutoff 节点里被搜的 quiet 是 **history ≤0** 的低价值着法
——history 信号已经"知道"它们大概率没用，但当前 selectivity 仍在搜它们。

### 2.3 当前 selectivity 的实际覆盖

| 机制 | 实测 | 评价 |
|---|---|---|
| futility prune | **26,459,593** | 真正的主力 |
| futility-eligible 节点仍被搜的 quiet | 19,591,820 | 巨大剩余头寸（margin/LMP 空间）|
| null eligible 节点 / attempts / fail-high | 9,680 / 9,680 / 4,035（41.7% FH）| 见 §3.4：coverage 需单独 funnel 归因 |
| LMR proposed（`late_move_reduction()` 理论值，d≥4 quiet）| 232,058 R1 / 0 R2 | 见 §2.4 修复口径 |
| **LMR actually applied**（`ctx.lmr_reductions`）| **26,982** | 仅为理论 R1 的 **~11.6%** |
| LMR re-search | 938 | **938 / 26,982 ≈ 3.48%（applied 口径）** |

### 2.4 depth≥4 quiet 的深度分布（LMR 交互，304,650 手）

| 维度 | 分布 |
|---|---|
| ordered idx | i0 9.8% / i1 1.8% / i2–3 5.1% / i4–7 12.0% / **i8+ 71.3%** |
| **proposed** reduction（`late_move_reduction()` 返回值，记录于 ChildWindow 选择**之前**，理论值）| R0 23.8% / **R1 76.2%** / R2 0% |
| i8+ 且 R0（full depth）| 21,702（i8+ 的 9.99%；多为 in-check/豁免路径）|
| **i8+ quiet 的 beta-cutoff** | **255 / 217,234 = 0.117%** |
| R1 scout fail-low | 26,044（applied 口径；理论 R1 中仅 ~11.6% 真正进入 scout）|

> 口径修复：`s73_q4p_quiet_red*` 系列是 **proposed/theoretical** reduction
> （在 `pvs_child_window()` 之前记录），不是实际应用分布。实际 LMR 只发生在
> `ChildWindow::Scout` 分支：26,982 次。R2=0 是 measurement contract 的机械
> 结果（R2 需 non-root remaining depth≥7，而本轮只跑 root d6/d7，非 root
> 最大 remaining depth ≈ 6），**不能**作为"R2 policy 过窄"的证据；研究 R2
> 需 root d8+。

## 3. 诊断

1. **SELECTIVITY_TOO_CONSERVATIVE（主因，成立）**
   — 无 cutoff 大树 = non-PV fail-low 节点（99.62%），其中 **74.0%** 被搜
   quiet 是 history ≤0；depth≥4 的 i8+ quiet 占 quiet 搜索的 71.3% 却只产生
   0.117% 的 cutoff。这些 move 正是 LMR/LMP 的教科书目标。
2. **更精确机制：LMR_APPLICATION_SUPPRESSED_ON_NULL_WINDOW_NODES**
   — `late_move_reduction()` 对 232,058 个 d≥4 late quiet 提出了 R1，但实际
   只应用了 26,982 次（**~11.6%**）。原因：LMR 只存在于
   `ChildWindow::Scout` 分支；当 caller 自身已是 null-window 时
   `pvs_child_window()` 返回 `Full`，reduction 被丢弃，late quiet 以
   **full depth** 搜索。结合 99.62% 的无 cutoff loop 是 non-PV（null-window）
   节点，这是结构级 PVS/LMR plumbing 缺陷，不是 R1/R2 阈值问题。
   applied-LMR re-search 率 = 938/26,982 ≈ **3.48%**（仍有余量，但不是
   0.308%）。
3. **PV / in-check 不是问题**（0.38% / 6.5%）。
4. **NULL-MOVE：NEEDS_SEPARATE_FUNNEL_ATTRIBUTION**
   — 有效数据：eligible 9,680 中 4,035 fail-high（**41.7%**）。
   "仅 0.13% loop 节点 eligible"不构成"guard 异常窄"的证据：eligibility
   本身要求 depth≥5 + null-window + 材料 + 非 mate-window，all-loop 分母
   不是 opportunity-normalized。且 fail-high 后走 full-depth verification，
   扩大 eligibility 未必免费。S7.4A 不碰 null-move；后续单独做 funnel
   （all loops → d≥5 → non-check → null-window → material → eligible）。
5. **R2 policy 本轮不可判**（见 §2.4 口径注）：需 root d8+ 才能触发。

## 4. 结论与 S7.4A 建议

- **第一刀（S7.4A，单候选）**：让**现有** LMR policy 真正作用于
  null-window caller 节点——在 caller-null-window 的 late quiet 上执行
  reduced null-window search（复用 caller 自己的窗口，不另造更窄窗口），
  fail-low 即接受，fail-high/improve 才用同窗口 full-depth 验证，且只有
  验证结果才能产生 cutoff / killer-history / PV。不改阈值、不加 LMP、
  不碰 null-move、不加 forcing extension。
- 若 S7.4A 有效，**之后**才做 S7.4B（adaptive LMR / LMP，R2 需 root d8+
  gate）与 S7.5（single-evasion / bounded check / singular extension）把
  省出的深度投回关键线——"垃圾线更浅，关键线更深"。
- 杀王 horizon 保护是硬约束：所有新 pruning 必须保留 in-check /
  tactical / mate-window 豁免，并跑 S6 teacher-challenge + mate
  regression gate。
- 单候选、预申报判据、tree-changing gate 全套（fixed-depth node +
  fixed-wall depth + tactical safety）。
