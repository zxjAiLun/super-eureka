# S7.1B — Conservative SEE-Delta Qsearch Pruning

STATUS: **REJECTED / CLOSED**（预申报判据：总节点 reduction < 10%）

- Baseline: Eureka v0.1.0 chess semantics
- 基线源码: `b7bc1cb29d5fb4f4b4a3ae10ba1336dd71a59208`
- 前置: S7.1A **REJECTED / CLOSED**（tree-identical 430/430、throughput
  反而 +3.5% wall 更慢、无 depth uplift；5 个 paired rep 全部负向）

## 0. Round 0 — 证据表述修复（evidence wording repair）

修复 `results/s7/s71a-verdict.md` 与 `docs/s7/readme.md` 中沿自 `b7bc1cb`
报告最后一句的概念错误：

- 移除（错误）：*"attacks the ~48-70% of non-check qsearch nodes that
  currently stand-pat-cutoff by pruning their capture tails"*
- 替换（正确）：*"S7.1B targets non-check qsearch nodes where stand-pat
  does NOT cut off and the search proceeds into the tactical move loop,
  plus the descendant capture tree created by those moves."*

根因：stand-pat 已 cutoff 的节点在 beta cutoff 处直接返回，**本来就没有
搜索 captures**，不存在可剪的 capture tail。

## 1. 候选实现（单候选，安全优先）

profile：`current-final-qsearch-delta`（bench + UCI `--profile` 均可激活）。

`CurrentFinalQsearchDelta` **精确继承** CurrentFinal 全部生产策略：
aspiration / LMR / verified null / futility / qsearch tactical movegen /
既有 qsearch SEE<0 prune / LegalityFast / SingleBuffer / SingleGeneration。
不新增 forcing search、quiet checks、threat evaluator 或其他 prune。

### 预申报 pruning rule（`DELTA_MARGIN_CP = 500`，写入
`QSEARCH_DELTA_MARGIN_CP`）

非 check qsearch 节点、stand-pat 未 cutoff 后，对每个 eligible plain
capture **只做一次** `static_exchange_eval_for_pruning`：

```text
None                          -> keep（fail open）
Some(v) if v < 0              -> 既有生产 SEE prune（不变）
Some(v) if stand_pat + v + 500 <= alpha（saturating） -> delta prune
Some(_)                       -> keep
```

node-level eligibility：`stand_pat < beta`（结构性成立，防御性保留）+
`non_pawn_material_count >= 4`（与既有 futility/null guard 同风格的
稀疏残局 fail-open）+ 非 mate-range window（沿用既有 early-return）。

安全豁免（全部继承生产版语义）：in-check node、checking capture、
promotion（含 capture-promotion 与 quiet promotion）、en-passant、
unsupported SEE。未添加任何 king-ring / queen-near-king / rook-battery
等手工启发。

### 关键实现规则：不重复计算 SEE

`prune_qsearch_captures_by_see_impl` 统一实现三条 lane
（生产 SEE / fast-SEE / delta），同一个 `see_value` 驱动 SEE<0 prune 与
delta prune。CurrentFinal 走原 path（`delta_enabled=false`），其
SEE<0 决策与改动前逐字节同义。

## 2. 单元测试（§5 十项，全部通过）

`qsearch_delta_pruning_follows_predeclared_rule_and_exemptions`（debug +
release，310/310 全绿）：

1. SEE<0 普通 capture：baseline 与 candidate 都 prune；
2. SEE>=0 且 `stand_pat+SEE+500<=alpha`：baseline keep、candidate prune
   （含 pawn 分类与 qply 0-1 桶断言）；
3. 同 capture 恰好高于阈值（-200 > alpha=-201）：candidate keep；
4. checking capture：keep；
5. capture-promotion：keep；
6. quiet promotion：keep；
7. en-passant：keep；
8. unsupported SEE：keep（fail-open 计数）；
9. mate-range alpha/beta：完全不做 SEE 测试；
10. 低非兵子力（3 子）：keep。

## 3. 机制计数器（profiling-gated）

`qsearch_delta_{tests,pruned,pruned_pawn,pruned_minor,pruned_rook,
pruned_queen,qply_0_1,qply_2_3,qply_4p}`，通过 bench profile 行输出为
`delta_tests=... delta_pruned=...` 等；既有 `qsearch_see_tests` /
`qsearch_see_pruned` / `qsearch_moves_searched` / `qsearch_nodes` 全保留。

## 4. §12 生产不变性验证 — PASS

`tools/s71b_verify_production.py`：实现前后各跑 30 位 S4 corpus、
`current-final` fixed depth 6 cold TT：

- `results/s7/s71b-verification-before.json`
- `results/s7/s71b-verification-after.json`
- **30/30 nodes / score / bestmove / PV 完全一致**（Repair 1 后用新二进制
  复验，同样 30/30）

即：仅添加候选这件事本身没有改变生产行为。fmt / clippy -D warnings /
debug tests（311 passed，含结构回归测试）/ release tests（311 passed）
全绿。

## 4a. Repair 1 — 候选 policy 继承修复与 gate 重跑

**问题**：首轮实现（`b061d5a`）的 `CurrentFinalQsearchDelta` 漏加了
`uses_null_move()` 与 `uses_single_buffer_legal()` 两个 match arm。候选
因此在 **无 null-move 剪枝、无 SingleBuffer** 的错误配置下跑了首轮
node gate，其 -7.945% 结果不可信（多出来的树主要是无 null-move 造成的
冗余 qsearch，恰好被 delta prune 砍掉，虚高了机制效果）。

**修复**：

1. 补齐两个 match arm；margin（500）与 prune 逻辑**零改动**；
2. 新增**结构回归测试**
   `qsearch_delta_profile_inherits_current_final_exactly_except_delta`：
   穷举全部生产 policy 谓词（PVS / aspiration / LMR / null move /
   futility / qsearch movegen / qsearch SEE / qsearch fast / qsearch lazy /
   root quiet history / root prev score / LegalityFast / SingleBuffer /
   SingleGeneration / Eval2 / forcing search / threat ordering /
   threat-aware qsearch），断言候选与 CurrentFinal **逐项相等**、唯
   `uses_qsearch_delta()` 不同，并显式断言 `null_move == true` 与
   SingleBuffer `== true`——不再依赖人工记忆 match arm；
3. **从头**重跑 node gate（删除旧 JSON，不用 `--resume`）。

## 5. §7 Fixed-Depth Node Gate — 结果（Repair 1 修正后，从头重跑）

工具：`tools/s71b_node_gate.py`（A/B cold TT、threads 1、depth 6）。
corpus：30 S4 + 80 S7 = **110 位，全部完成**。树按预期不同。

**完整 corpus（110 位）聚合（修正后最终）：**

| 指标 | 数值 |
|---|---|
| baseline 总节点 | 13,579,038 |
| candidate 总节点 | 12,856,552 |
| **总节点 reduction** | **-5.321%** |
| 其中 qsearch 节点 | 11,283,829 → 10,561,343（**-6.403%**）|
| 其中 main 节点 | 2,295,209 → 2,295,209（**-0.000%**）|
| delta_tests | 8,712,080 |
| delta_pruned | 731,429（tested 的 **8.396%**）|
| 位置分布 | 110 减少 / 0 增加，但**无一 ≥10%** |
| **预申报 verdict** | **TOO_SMALL → REJECT S7.1B** |

（对照：修复前错误配置下的 -7.945% 作废；正确继承 null-move +
SingleBuffer 后，被 delta 砍掉的冗余树显著缩水。）

## 6. VERDICT — REJECTED / CLOSED

按 §7/§13 预申报判据（<10% = TOO_SMALL → REJECT，不为 micro gain
Arena-test 一个 tree-changing candidate）：

1. 修正候选的完整 corpus（110 位）总节点 reduction **-5.321% < 10%**
   → 不进入 fixed-wall depth gate、teacher challenge、search stability
   gate 或 Arena；
2. 不做 `500 → 400 → 300 → 200 → 100` 的 margin 逆调参（明确禁止把
   tactical safety 换成 benchmark 数字）；
3. 生产 CurrentFinal 未改动（30/30 精确验证，Repair 1 后复验），
   `current-final-qsearch-delta` profile、计数器与结构回归测试保留为
   可复现证据。

### 结论与下一步

500cp 这么保守的第一刀在正确配置的候选上只砍掉 5.3% 总节点（tested
captures 的 8.4%，且全部集中在 qsearch 层、main 树零变化），说明在
CurrentFinal 现有 qsearch 形态下，stand-pat 未 cutoff 后真正进入
tactical loop 的 capture 树里，"抬不起 alpha +500" 的部分并不占主导
——剩余树大多是被 TT/ordering/null-move/既有 SEE<0 prune 已经压过的
有效战术搜索。这条线按预申报规则关闭。

后续方向（供 S7.2 决策参考，未启动）：
- S7.2 Move Ordering Attribution（首步 cutoff ~84% vs 参考强引擎 90%+）
- 若重开 qsearch 线，证据指向 deeper policy（如 quiet-check/forcing
  extension 的结构）而非更松的 delta margin。

## 7. 产物

| 产物 | 位置 |
|---|---|
| 候选 profile | `current-final-qsearch-delta`（bench/UCI） |
| margin 常量 | `QSEARCH_DELTA_MARGIN_CP`（src/engine/search.rs） |
| prune 实现 | `prune_qsearch_captures_by_see{,_delta,_impl}` |
| 单元测试 | `qsearch_delta_pruning_follows_predeclared_rule_and_exemptions` |
| 生产不变性 | `tools/s71b_verify_production.py` + `results/s7/s71b-verification-{before,after}.json` |
| node gate | `tools/s71b_node_gate.py` + `results/s7/s71b-node-gate.json` |
| 证据修复 | `results/s7/s71a-verdict.md`（Implication 段 NOTE） |
