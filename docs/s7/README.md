# S7 — Search Depth Attribution（搜索深度归因）

STATUS: **进行中** — S7.0 已 APPROVED，S7.1A 已 REJECTED，S7.1B 待开。

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

S7.2   Move Ordering Attribution（待 S7.0/S7.1 新数据再定）
       quiet move 的 rank 分布 / cutoff 机会，判断 history 是否真的弱
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

---

## 4. 未完成

### 4.1 S7.1B — Conservative Delta / SEE Futility Pruning（待开）

tree-changing，砍那 48–70% 本来就会 stand-pat-cutoff 的非 check qsearch
节点背后的 capture 尾巴。方向（候选而非定案）：

```text
stand_pat + captured_piece_value + conservative_margin <= alpha
AND non-check / non-promotion / non-EP / non-checking-capture / non-mate-window
→ 跳过该 capture
```

**Gate（重战术安全，因为最在意的是"深度不够导致晚发现王杀"）**：
- node reduction
- fixed-time depth 上升
- tactical / mate regression 不恶化
- 最终 Arena Elo

**不能**为了把 UCI depth 数字从 7 漂亮地变 10 而剪掉真正的 sacrifice/capture
线路。

### 4.2 S7.2 — Move Ordering Attribution（未开始）

待 S7.0/S7.1 数据齐后，判断是否需要：quiet move 的 cutoff 机会 / rank 分布 /
killer+history 排前后的成功率。现在不下"ordering 是第二瓶颈"的结论。

---

## 5. 关键产物

| 产物 | 位置 |
|---|---|
| corpus（80 位）| `tools/data/s7_depth_attribution_corpus.jsonl` |
| corpus builder | `tools/build_s7_corpus.py` |
| 归因 runner（增量+可续）| `tools/run_s70_depth_attribution.py` |
| 报告 | `results/s7/s70-depth-attribution.{json,md}` |
| tree gate / throughput / depth uplift | `tools/s71a_*.py` + `results/s7/s71a-*.{json,md}` |
| S7 计数器 | `SearchContext`/`SearchStats` 中的 S7.0/S7.1A 字段 |

**Commit 链**：

```text
7e41d9f  test(s7): corpus (80 positions, 3 strata)
b96ddbc  feat(s7): observation-only depth-attribution counters
9ae1601  feat(s7): seldepth split + depth-attribution runner
f249f77  test(s7): S7.0 results + diagnosis
1eb5060  docs(s7): S7.0 report P2 corrections + headroom
aad0413  feat(search): S7.1A candidate (lazy qsearch materialization)
b7bc1cb  test(s7): S7.1A evidence (negative throughput)
```

---

## 6. 待办决策

- [ ] S7.1B 开刀：先定 candidate 的 delta margin 与 SEE threshold，跑
      node-reduction + tactical 安全 gate
- [ ] S7.2 是否需要单独做 ordering attribution（等 S7.1B 收尾后）
- [ ] 若 qsearch 线收效，考虑是否回补一个 cheap stalemate 探测，让 S7.1A
      的 lazy 思路起死回生（当前 has-any 太贵）
