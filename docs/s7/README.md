# S7 — Search Depth Attribution（搜索深度归因）

STATUS: S7.0 APPROVED；S7.1A REJECTED / CLOSED；S7.1B REJECTED / CLOSED
（Repair 1 后 -5.321%）；S7.2 CLOSED（ORDERING_NOT_PRIMARY）；S7.3
APPROVED_WITH_REPAIR（SELECTIVITY_TOO_CONSERVATIVE 成立，主机制 =
LMR_APPLICATION_SUPPRESSED_ON_NULL_WINDOW_NODES）；**S7.4A Repair 1
完成：QUALIFIED_FOR_ARENA — STRONG**（原 verdict 被 review 改判
FOLLOW_UP/P1：verification re-search 缺 `try_enter_node()` 计数合同；
repair 实现 `df7f324` 补齐 exact-once acquisition）。
修复后 evidence 全部重跑/补齐：fixed-depth gate d6 **-41.802%** /
d7 **-49.397%** / d8 子集 **-75.508%**（三档 STRONG，每行
`research_entered == research_requested`）；accounting identity
180/180 语义逐位一致，repaired candidate nodes = old candidate nodes +
successful verification-root acquisitions（+2,495）；production
CurrentFinal 30/30 exact；teacher d6 A=33/B=34、0 个 ≥100cp 分歧、
5 个 teacher_mate-labelled；R2 d8 gate 120/120 完成、0 hard-reject；
fixed-wall rerun 1000ms 35 gained/1 lost、3000ms 60 gained/0 lost
（median depth 6→7 / 7→8）。
之后：review → Arena SPRT → 若 H1 则 promote 进 CurrentFinal；然后
S7.4B adaptive LMR/LMP、S7.5 forcing extensions（关键线更深）。

本目录记录 S7 阶段的进度与结论。原始测量与证据统一放在 `results/s7/`。

---

## 1. 开发背景

Eureka 在相同时间内通常只能完成约 **depth 7**。S4/S5 阶段回答的是
"每个节点跑多快"（NPS、movegen/SEE/eval 的 wall-time 占比）；但还没人回答
"为什么需要搜这么多节点才能多完成一层"。

S7 的目标就是把这个问题量化：**搜索树的质量**（ordering / 分支因子 /
qsearch / TT / LMR / eval 稳定性）里，到底是谁在吃掉深度。这是从 depth 7
走向 depth 10–12 真正该解决的问题，而不是继续做 movegen 微优化。

前置修复（R0 Repair 2，`d71c3e7`）已让 `seldepth` 成为可信的 global-ply
口径，S7 的观测才有意义。

---

## 2. 预期开发内容（路线图）

```text
S7.0   Search Depth Attribution
       80-position corpus + 40 个 profiling-gated 计数器 + 每层归因
       → 回答"深度瓶颈在哪"（只观测，不改搜索）

S7.1A  Lazy Qsearch Materialization（tree-identical）
       非 check qsearch 把 tactical movegen 推迟到 stand-pat 之后
       → 只省被浪费的工作，树不变，快速拿 throughput

S7.1B  Conservative Delta / SEE Futility Pruning（tree-changing）
       剪掉明显抬不起 alpha 的 capture 尾巴
       → node reduction + fixed-time depth + tactical/mate 安全 gate

S7.2   Move Ordering Attribution（已完成 → ORDERING_NOT_PRIMARY）
       cutoff 机会归因：cutoff move 类别份额/成功率、quiet rank、
       history-bucket 单调性、killer/TT 成功率、cutoff 前浪费、
       no-beta-cutoff 节点单独统计、remaining-depth split、LMR 交互

S7.3   Selectivity Attribution（已完成 → SELECTIVITY_TOO_CONSERVATIVE）
       no-beta-cutoff 大树的构成：PV/wide-window vs non-PV fail-low、
       in-check、improving、LMR eligible/reduced/fail-low、futility
       eligible-unpruned、null-move 流向、late-quiet 深度分布

S7.4A  LMR-on-Null-Window（tree-changing 单候选 → QUALIFIED_FOR_ARENA STRONG）
       现有 LMR policy 真正作用于 null-window caller 的 late quiet：
       reduced null-window search + 同窗口 full-depth 验证，
       只有验证结果可 cutoff / killer-history / PV

S7.4B  （若 4A 获准 promote 或并行评审）adaptive LMR / LMP
S7.5   forcing extensions：single-evasion / bounded check / singular
```

纪律：**先观测、后动刀；每次只做一个候选；tree-identical 与 tree-changing
分开 gate；不过早下 "ordering 是第二瓶颈" 的结论。**

---

## 3. 已完成

### 3.1 S7.0 — Search Depth Attribution ✅ APPROVED

**观测来源**：`d71c3e7`（telemetry/build-only），chess baseline `Eureka v0.1.0`
（S3 + LegalityFast + SingleBuffer + SingleGeneration），profile
`current-final`，16MB cold TT，单线程。

**Corpus**（`tools/data/s7_depth_attribution_corpus.jsonl`，80 位）：
- A1 = 30 位 S4.0A compute corpus（7 类）
- A2 = 30 位真实中局（Arena self-play PGN，ply 20–80，canonical-FEN4 去重）
- A3 = 20 位 teacher-disagreement 战术局面
- SHA-256 `8786ffca6c8e6277b711c990bf9788d88eaedbb0b4b894f85fc2b18de62d5b1b`

**新增计数器**（profiling-gated，OFF 时零成本；锁定的 smoke/tree 测试证明
fixed-depth node/score/PV 完全不变）：beta-cutoff 直方图 + mover 分类、
moves-searched + 每节点 searched-moves 直方图、pv/in-check/depth 桶、
TT hit 类型拆分、LMR R1/R2、null fail-low、futility、qsearch stand-pat 等。

**核心发现**（median）：

| depth | ratio-of-medians growth | median per-pos growth | qsearch % | 1st-cutoff % | moves/main-node | seldepth−depth |
|---|---|---|---|---|---|---|
| 4 | — | — | 84.7 | 82.9 | 3.21 | 8 |
| 5 | 2.61 | 2.49 | 86.6 | 86.0 | 4.68 | 8 |
| 6 | 5.54 | 4.33 | 81.3 | 83.1 | 3.02 | 10 |
| 7 | 3.24 | 3.98 | 84.1 | 85.9 | 4.25 | 11 |
| 8 | 4.71 | 4.61 | 78.4 | 84.5 | 2.72 | 13 |

**诊断（top-3）**：

1. **QSEARCH_DOMINATED（主因）** — quiescence 占 **78–87%** 节点，主树仅
   ~15–20%；seldepth−depth 8→13（capture/promotion chains + 强制 in-check
   evasions；CurrentFinal 无 forcing-search/quiet-check extension）。
2. **ORDERING_LIMITED（次因，暂定）** — 首步 beta-cutoff ~84%（参考强引擎
   ~90%+，无本项目对照），quiet 只占 1–2% cutoff；仍需 S7.2 再确认。
3. **HIGH_EFFECTIVE_BRANCHING（结果）** — growth 2.6–5.5×/ply（参考 ~2.0，
   仅 heuristic）。

**qsearch headroom 补充**：非 check qsearch 节点里 stand-pat beta cutoff 占
**48–70%**，SEE prune ~46–51%，in-check qsearch ~9–12%。

**结论**：主树还行，是 qsearch 吃掉绝大多数预算 → 下一步先动 qsearch，不是
history/LMR。

### 3.2 S7.1A — Lazy Qsearch Materialization ❌ REJECTED

**候选** `current-final-qsearch-lazy`（`aad0413`）：非 check qsearch 把
tactical movegen/ordering 推迟到 stand-pat 之后，用 has-any 探测 stalemate。

| gate | 结果 |
|---|---|
| exact tree（400 S7 + 30 S4）| **PASS**，0 mismatch |
| throughput（550 配对 rep）| **+3.5% aggregate / +2.8% median（更慢）** |
| depth uplift（1s / 3s）| depth 不变（6 / 7），固定墙内节点 ~3% 更少 |

**根因**：`has_any_legal_move_profiled` 是 full-legal movegen，比它想推迟的
tactical movegen 还贵。实测 startpos d8：多了 100 万次 has-any，只省 37 万次
tactical movegen。当前设计本来就只在 tactical 列表为空时才调 has-any。

**含义**：stand-pat-before-movegen 不划算，除非 stalemate 探测近零成本；
不影响 S7.1B（那是 tree-changing 的 node-reduction）。

### 3.3 S7.1B — Conservative SEE-Delta Qsearch Pruning ❌ REJECTED / CLOSED

**候选** `current-final-qsearch-delta`（本轮实现）：非 check qsearch 节点
stand-pat 未 cutoff 后，对每个 eligible plain non-checking capture 只做
**一次** pruning SEE——`SEE<0` 走既有生产 prune；`SEE>=0` 且
`stand_pat + SEE + 500 <= alpha`（saturating）则 delta prune。全部安全
豁免（in-check / checking capture / promotion / EP / unsupported SEE /
mate window / 稀疏子力）fail-open。

| gate | 结果 |
|---|---|
| §12 生产不变性（30 S4 d6，nodes/score/bestmove/PV）| **PASS** 30/30 精确一致（Repair 1 后复验）|
| 单元测试（§5 十项 + 结构回归）| debug + release **311/311** 全绿 |
| fmt / clippy -D warnings | clean |
| §7 node gate（110 位 = 30 S4 + 80 S7，d6 cold）| **-5.321% 总节点**（修正候选）|
| 预申报判据 | <10% = **TOO_SMALL → REJECT / CLOSE** |

**Repair 1**（`7242d89` 之后）：首轮候选漏了 `uses_null_move()` 与
`uses_single_buffer_legal()` 两个 match arm，导致候选在无 null-move /
无 SingleBuffer 的错误配置下跑 gate（-7.945% 不可信）。修复后新增
**结构回归测试**（穷举断言候选与 CurrentFinal 在全部生产 policy 维度
一致、仅 `qsearch_delta` 为 true），并**从头**重跑 node gate。修正后：
总节点 -5.321%（qsearch 节点 -6.403%、main 节点 0.000%、110/110 位置
减少但无一 ≥10%）、delta_tests 8,712,080、delta_pruned 731,429（8.40%）。
有趣的是正确继承后 reduction 反而更小——首轮多出来的树正是无 null-move
剪枝造成的冗余 qsearch。verdict 不变：**REJECTED / CLOSED**。
详见 `docs/s7/s71b-report.md` 与 `results/s7/s71b-node-gate.json`。

### 3.4 S7.2 — Move Ordering Attribution ✅ CLOSED（ORDERING_NOT_PRIMARY）

**轮次性质**：observation-only，profile `current-final`，无任何搜索语义
改动；production 不变性 30 S4 d6 exact match。

**埋点**（profiling-gated，production ordering 完成后分类）：cutoff move
类别（TT/promotion/capture/killer0/killer1/history-quiet/other-quiet，互斥、
TT 优先）、quiet rank 直方图、history-bucket、killer/TT searched+cutoff、
cutoff 前 searched 直方图、no-beta-cutoff 节点单列、remaining-depth split、
LMR 交互。采集：80 S7 × d6/d7 + 30 S4 × d6（`f9644c4`）。

**核心数据**：

| 指标 | 值 |
|---|---|
| 首个被搜 move 即 cutoff | **80.304%** |
| ≥5 手才 cutoff / ≥9 手才 cutoff | 1.864% / 0.279% |
| remaining depth 4–5 / 6–7 的 ≥5 手才 cutoff | 1.31% / 1.87% |
| quiet cutoff 的 quiet-rank 0 / 1 / 2–3 / ≥4 | 94.683% / 3.079% / 1.069% / ~1.17% |
| history ≤0 / 1–15 / 16–63 / 64–255 / 256+ 的 cutoff 率 | 0.067% / 3.465% / 7.953% / 17.281% / 54.271% |
| killer0 searched 成功率 | 78.143% |
| no-beta-cutoff 大 move-loop 节点 | 1.91M，mean searched 14.683（≈28.07M / 36.67M 总 searched）|

**判定**：深节点 late-cutoff 同样仅 1–2%，quiet cutoff 几乎都发生在
quiet-rank 0；history 分数与 cutoff 概率严格单调（且分数在节点搜索前
读取，无循环统计）。大 move-loop 的主体**不是"存在 cutoff move 但排得
太后"**，而是真正无 beta cutoff 的节点 → 传统"把 cutoff move 前排"不是
当前主要杠杆。**ORDERING_NOT_PRIMARY，CLOSED**；不做 MovePicker /
新 history / countermove 候选。

**P2 修复**（`d1db1af`，文档措辞/口径，未重跑）：
1. "fail-low nodes" → "no-beta-cutoff move-loop nodes"（原条件含 PV/wide-window
   与有 alpha 提升但无 cutoff 的节点，不严格等于 all-moves fail-low）；
2. 类别 success rate 只保留分母口径一致的（TT/killer0/killer1/history-bucket/
   all-quiet），capture/promotion 仅报 SHARE（类别互斥优先级使分母混入
   TT/killer 重叠）；
3. 记录 aggregate 中 30 个 S4 位置 d6 被重复加权一次的 caveat（80 位 S7
   corpus 的 A1 层本就含这 30 位；对结论无影响）。

### 3.5 S7.3 — Selectivity Attribution ✅ APPROVED_WITH_REPAIR

S7.0/7.1/7.2 排除了 qsearch 省工作（1A）、qsearch 剪树（1B）、ordering
（2）。S7.3 回答：**no-beta-cutoff 大树（≈73% 的 main-loop searched
moves）到底是什么**。80 位 S7 × d6+d7，observation-only，production
不变性 30/30 exact。Review 后修复报告口径（数据本身有效，未重跑引擎）。

**核心数据**（详见 `results/s7/s73-selectivity-attribution.md`）：

| 指标 | 值 |
|---|---|
| 无 cutoff loop 节点 / 其中 PV | 1.84M（24.9% loops）/ **0.38%** |
| 无 cutoff 节点中被搜 quiet 的 history ≤0 占比 | **74.0%**（五桶分母）|
| futility pruned / eligible 节点仍搜的 quiet | 26.46M / 19.59M |
| null eligible / fail-high 率 | 9,680 / 41.7%（coverage 需单独 funnel 归因）|
| LMR proposed（理论 R1，d≥4 quiet）/ actually applied | 232,058 / **26,982（~11.6%）** |
| applied-LMR re-search 率 | 938 / 26,982 ≈ **3.48%** |
| depth≥4 quiet：i8+ 占比 / i8+ 的 cutoff 率 | 71.3% / **0.117%** |
| R2 | 本轮不可判（需 root d8+ 才能触发）|

**诊断**：**SELECTIVITY_TOO_CONSERVATIVE** 成立，主机制修正为
**LMR_APPLICATION_SUPPRESSED_ON_NULL_WINDOW_NODES**——`pvs_child_window()`
在 caller 已是 null-window 时返回 `Full`，`late_move_reduction()` 的
reduction 被丢弃，late quiet 以 full depth 搜索；而 99.62% 的无 cutoff
loop 正是 non-PV（null-window）节点。PV/in-check 不是原因；null-move
coverage 未定论（all-loop 分母不是 opportunity-normalized，且 fail-high
后有 full-depth verification）。

**S7.4A（单候选，已定）**：让**现有** LMR policy 真正作用于 null-window
caller 节点——caller-null-window 的 late quiet 用 caller 自己的窗口做
reduced null-window search；fail-low 接受，fail-high/improve 才同窗口
full-depth 验证；只有验证结果才能 cutoff / killer-history / PV。不改
阈值、不加 LMP、不碰 null-move、不加 forcing extension。之后再 S7.4B
（adaptive LMR / LMP，R2 需 d8+ gate）、S7.5 forcing extensions；杀王
horizon 保护为硬约束（S6 teacher + mate regression gate）。

### 3.6 S7.4A — LMR-on-Null-Window ✅ QUALIFIED_FOR_ARENA — STRONG

**候选**（`current-final-lmr-null-window`，实现 `d50729a`，证据
`ea1c60b`）：CurrentFinal 全维度继承（结构回归测试覆盖），唯一差异 =
caller 已是 null-window（`beta == alpha+1`）且 `reduction > 0` 的 late
quiet 改走 reduced null-window search（caller 自己的窗口，不另造更窄
窗）；fail-low 直接接受不 re-search；fail-high/improve 用同窗口
full-depth 验证，只有验证结果可 cutoff/reward/PV。LMR 阈值与 R1/R2
公式一字未动。S7.3 的理论-实际口径歧义由新计数器（proposed/applied/
suppressed + candidate 4 项 + depth/idx split）永久消除。

**首轮 Gate 结果**（pre-repair，详见 `results/s7/s74a-verdict.md`；fixed-wall 旧值因缺 deadline check 已 superseded）：

| Gate | 结果 |
|---|---|
| node gate d6（80 位）| 9.94M → 5.79M，**-41.805%** STRONG |
| node gate d7（80 位）| 43.10M → 21.81M，**-49.399%** STRONG |
| node gate d8（20 位子集，R2 首次可触发）| 50.17M → 12.29M，**-75.510%** STRONG |
| LMR proposed→applied（null-window）| d6 88.4% / d7 87.5% / d8 93.7%（对比 S7.3 的 11.6%）|
| S6 teacher challenge（178 行）| matches 33 → **34**；≥100/300/500cp 分歧 **0/0/0**；5 个 teacher_mate-labelled 完全一致 |
| search stability（d6→d7）| bestmove flips 19 → **15**；≥200cp 反转 1→1；cp↔mate 0→0 |
| fixed-wall 1000ms（旧，非正式） | median depth 7→7、mean 6.775→7.362；**44/80 gained、0 lost** |
| fixed-wall 3000ms（旧，非正式） | median depth 7→**9**、mean 7.513→8.512；**62/80 gained、0 lost** |
| production invariance | 30 S4 d6 nodes/score/bestmove/PV **30/30 exact** |
| fmt / clippy -D warnings / debug+release tests | clean |

**Repair 1 重跑证据（`df7f324`，accounting-only）**：

| Gate | 结果 |
|---|---|
| 修复形状 | 初始 reduced search 不另获节点（`probe_child_draw` 已计数）；仅 full-depth verification 前 exact-once `try_enter_node` |
| 机制回归测试 | A fail-low=0 acquisition；B exactly-one acquisition + exactly-one verification；C acquisition-fail 时 None / path / FEN / 无 cutoff / 无 killer-history reward / 无 fake PV；D nodes ≤ budget |
| fixed-depth node gate d6/d7/d8 | -41.802% / -49.397% / -75.508%，三档 STRONG；每档 `research_entered == research_requested`（344/344、1182/1182、969/969），verified cutoff 100/341/129 |
| 主/qsearch/wall/NPS/seldepth | 见 `results/s7/s74a-node-gate.json` 与 `s74a-verdict.md` Repair 1 表 |
| accounting identity（180 行 vs `ea1c60b`） | score/bestmove/PV 0 diff；每行 Δnodes == research_requested；总 Δ **+2,495 == 2,495** |
| production invariance | base `ea1c60b` vs repaired `df7f324`，CurrentFinal 30 S4 d6 **30/30 exact**（nodes/score/bestmove/PV） |
| teacher d6（重跑） | 176 evaluated + 2 TERMINAL/NOT_APPLICABLE（rows 158/164）；matches 33 → **34**；cp≥100/300/500 = **0/0/0**；mate-labelled = **5** |
| R2 tactical/horizon d8 | corpus SHA `7eeecf0e…`；120/120 完成，含 11 mate-labelled（5 teacher_mate 全含）；teacher bestmove A=58/B=59；cp≥100/300/500 = 0/0/0；mate transitions / side mismatches / distance changes / hard-reject = **0** |
| fixed-wall rerun（1000ms） | median depth 6→**7**、mean 6.412→6.850、seldepth 16.5→16.5；**35 gained / 1 lost** |
| fixed-wall rerun（3000ms） | median depth 7→**8**、mean 7.138→8.088、seldepth 18→19；**60 gained / 0 lost** |
| fmt / clippy -D warnings / debug+release tests | clean；**314/314** debug、**314/314** release |

**Verdict**：六个资格条件全部满足且超过阈值（node -41.8%/-49.4%/-75.5%
远超 ≤-15%；fixed-wall gained 35/60 vs 10 且 gained>lost；teacher 不降；
R2 d8 无任何 hard-reject；CurrentFinal 30/30 exact；测试干净）。
按预申契约 **STOP 在 offline 证据**：不 promote、不调 LMR
阈值、不加 LMP、不碰 null-move、review 前不跑 Arena。

**观察**：3s 时限 median depth +2 ply、mean +1 ply——这正是 S7 立项时
"depth 7 上不去"的直接回应；且改进来自把 S7.3 定位的结构性缺陷
（reduction 被 PVS window plumbing 丢弃）修掉，而非调参。

---

## 5. 关键产物

| 产物 | 位置 |
|---|---|
| corpus（80 位）| `tools/data/s7_depth_attribution_corpus.jsonl` |
| corpus builder | `tools/build_s7_corpus.py` |
| 归因 runner（增量+可续）| `tools/run_s70_depth_attribution.py` |
| 报告 | `results/s7/s70-depth-attribution.{json,md}` |
| tree gate / throughput / depth uplift | `tools/s71a_*.py` + `results/s7/s71a-*.{json,md}` |
| S7.1B verify / node gate | `tools/s71b_{verify_production,node_gate}.py` + `results/s7/s71b-*.{json,md}` |
| S7.1B 报告 | `docs/s7/s71b-report.md` |
| S7.2 采集 + 报告 | `tools/s72_collect.py` + `results/s7/s72-ordering-attribution.{json,md}` |
| S7.3 采集 + 报告 | `tools/s73_collect.py` + `results/s7/s73-selectivity-attribution.{json,md}` |
| S7.4A node/teacher/fixed-wall gate | `tools/s74a_{node_gate,teacher_gate,fixed_wall_gate}.py` + `results/s7/s74a-{node,teacher,fixed-wall}-gate.json` |
| S7.4A Repair 1 corpus / gates / verify | `tools/s74a_{build_r2_corpus,r2_gate,repair_verify}.py` + `data/s7/s74a-r2-tactical-corpus.jsonl` + `results/s7/s74a-{r2-gate,repair-verify}.json` |
| S7.4A 报告 | `results/s7/s74a-verdict.md` |
| S7 计数器 | `SearchContext`/`SearchStats` 中的 S7.0/S7.1A/S7.1B/S7.2/S7.3/S7.4A 字段 |

**Commit 链**：

```text
7e41d9f  test(s7): corpus (80 positions, 3 strata)
b96ddbc  feat(s7): observation-only depth-attribution counters
9ae1601  feat(s7): seldepth split + depth-attribution runner
f249f77  test(s7): S7.0 results + diagnosis
1eb5060  docs(s7): S7.0 report P2 corrections + headroom
aad0413  feat(search): S7.1A candidate (lazy qsearch materialization)
b7bc1cb  test(s7): S7.1A evidence (negative throughput)
b061d5a  feat(search): S7.1B candidate (conservative SEE-delta qsearch pruning)
7242d89  test(s7): S7.1B evidence + S7.1A wording repair
a249ffb  fix(search): S7.1B repair 1 - exact CurrentFinal policy inheritance
f9644c4  test(s7): attribute non-root move ordering quality
d1db1af  docs(s7): S7.2 P2 wording fixes
6cf4506  feat(s7): S7.3 selectivity-attribution counters + evidence
acc8f15  docs(s7): S7.3 Round-0 report repair
d50729a  feat(s7): S7.4A LMR-on-null-window candidate
ea1c60b  test(s7): S7.4A evidence (STRONG reduction + depth uplift + safety gates)
df7f324  fix(s74a): S7.4A Repair 1 - exact verification-node acquisition + accounting regression coverage
```

---

## 6. 待办决策

- [x] S7.1B：REJECTED / CLOSED（修正后 -5.321% < 10%）
- [x] S7.2：CLOSED（ORDERING_NOT_PRIMARY，无 MovePicker/history 候选）
- [x] S7.3：APPROVED_WITH_REPAIR（SELECTIVITY_TOO_CONSERVATIVE 成立；
      主机制 = LMR suppressed by null-window/PVS coupling）
- [x] **S7.4A Repair 1**：QUALIFIED_FOR_ARENA — STRONG（fixed-depth
      -41.8%/-49.4%/-75.5%；fixed-wall 明确正向；teacher/R2 安全 gate
      全绿；production 30/30 exact）；等待 review 后 Arena
- [ ] S7.4B（若 4A 有效）：adaptive LMR / LMP；R2 需 d8+ gate
- [ ] S7.5：single-evasion / bounded check / singular extension，把
      省出的深度投回关键线；null-move eligibility 单独 funnel 归因
- [ ] 若未来重开 qsearch 线，证据指向结构（quiet-check/forcing
      extension、deeper policy）而非更松的 delta margin；也仍可考虑
      near-zero-cost stalemate 探测让 S7.1A 复活
