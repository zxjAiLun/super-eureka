# S7.3 — Selectivity Attribution（选择性归因，OBSERVATION ONLY）

STATUS: **COMPLETE** — 数据完整，结论支持 S7.4 进入 adaptive LMR / LMP 方向

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

| bucket | searched | 占比 |
|---|---|---|
| ≤0 | 17,133,747 | **65.70%** |
| 1–15 | 2,039,270 | 7.82% |
| 16–63 | 1,440,587 | 5.53% |
| 64–255 | 1,206,828 | 4.63% |
| 256+ | 1,336,209 | 5.13% |

绝大多数在无 cutoff 节点里被搜的 quiet 是 **history ≤0** 的低价值着法
——history 信号已经"知道"它们大概率没用，但当前 selectivity 仍在搜它们。

### 2.3 当前 selectivity 的实际覆盖

| 机制 | 实测 | 评价 |
|---|---|---|
| futility prune | **26,459,593** | 真正的主力 |
| futility-eligible 节点仍被搜的 quiet | 19,591,820 | 巨大剩余头寸（margin/LMP 空间）|
| null eligible 节点 / attempts / fail-high | 9,680 / 9,680 / 4,035（41.7% FH）| **覆盖面极窄**（仅 0.13% loop 节点）|
| LMR reductions / researches | 26,982 / 938 | **research 率 0.308%——极端保守** |

### 2.4 depth≥4 quiet 的深度分布（LMR 交互，304,650 手）

| 维度 | 分布 |
|---|---|
| ordered idx | i0 9.8% / i1 1.8% / i2–3 5.1% / i4–7 12.0% / **i8+ 71.3%** |
| reduction | R0 23.8% / **R1 76.2%** / R2 **0%**（d≥7&&i8+ 从未触发）|
| i8+ 且 R0（full depth）| 21,702（i8+ 的 9.99%；多为 in-check/豁免路径）|
| **i8+ quiet 的 beta-cutoff** | **255 / 217,234 = 0.117%** |
| R1 scout fail-low | 26,044（R1 里被证伪需要 research 的仅 938 总）|

## 3. 诊断

1. **SHALLOW_QUIET_EXPLOSION + SELECTIVITY_TOO_CONSERVATIVE（主因）**
   — 无 cutoff 大树 = non-PV fail-low 节点，其中 65.7% 被搜 quiet 是
   history ≤0；depth≥4 的 i8+ quiet 占 quiet 搜索的 71.3% 却只产生
   0.117% 的 cutoff。这些 move 正是 LMR/LMP 的教科书目标。
2. **LMR 过浅且过窄** — R1（减 1 层）对 fail-low 几乎无代价（research
   0.308%），R2 在本轮 corpus 上一次都没触发；当前公式远在安全侧。
3. **NULL-MOVE 覆盖面异常窄** — eligible 仅 9,680 节点（0.13%），
   fail-high 率 41.7% 却很高：说明 guard（深度/材料/非PV）卡掉了
   绝大多数本可 fail-high 的节点，值得在 S7.4 一并复核 eligibility
   而非只调 R。
4. **PV / in-check 不是问题**（0.38% / 6.5%）。

## 4. 结论与 S7.4 建议

- **数据方向明确**：垃圾 quiet 线（i8+、history≤0、non-PV fail-low
  语境）占大头且几乎从不产生 cutoff → **先做 adaptive LMR / LMP
  （加深 reduction、加 late-quiet pruning），再做 forcing extensions
  把省出的深度投回关键线**。
- 杀王 horizon 保护是硬约束：所有新 pruning 必须保留 in-check /
  tactical / mate-window 豁免，并跑 S6 teacher-challenge + mate
  regression gate。
- 候选设计从本轮数据出发（如 log/公式化 reduction、LMP 阈值与
  history/futility margin 联动），单候选、预申报判据、tree-changing
  gate 全套（fixed-depth node + fixed-wall depth + tactical safety）。
