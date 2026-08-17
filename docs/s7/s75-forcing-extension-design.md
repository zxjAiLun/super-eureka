# S7.5 — Forcing / Extension Lane Design

STATUS: **v1 APPROVED**（v0 APPROVED_WITH_REVISIONS，7 项修订已并入）

## 0. Direction

```text
S7.4A 已让“垃圾 quiet 线”更浅；
S7.5 研究“关键 forcing 线”更深。

同样时间：
    少把节点浪费在垃圾线       <- S7.4A 已解决
    多把节点投入低分支强制线   <- S7.5
    更早看穿 mate / only defense / king attack
```

阶段顺序：

```text
S7.5-0   forcing opportunity attribution（observation-only）
S7.5A    single-evasion extension
S7.5B    bounded checking extension
S7.5C    singular extension（最后）
```

## 1. History correction（v1）

- `uses_forcing_search()` 目前只对 `CurrentThreatAware | CurrentThreatAwareNoQchecks` 为真。
- quiet-check qsearch 是独立 predicate `uses_threat_aware_qsearch()`，`NoQchecks`
  在代码层已把 main-search forcing 与 quiet qsearch checks 分开。
- 但旧 `forcing_child_params()` 把 **checking extension + single-evasion
  extension** 绑在同一预算，并属于旧 threat-aware lineage。
- **S7.5 不复用旧 `uses_forcing_search()`，不复用旧 threat-aware profile。**

## 2. S7.5-0 observation-only contract

```text
baseline:     990aed6
profile:      PRODUCTION_PROFILE = CurrentFinal
corpora:      80 S7 × d6/d7
              120 R2 tactical × d8
semantics:    ZERO CHANGE
production:   ZERO CHANGE
```

所有 counter 只写 `profiling_enabled` 路径，生产关闭时零语义/零开销。

### 2.1 Main search funnel

```text
s75_main_nodes                    # derived: actual acquired main-tree nodes
                                  #        = nodes - qsearch_nodes
s75_main_in_check_nodes

s75_main_single_evasion_nodes_raw
s75_main_single_evasion_actionable_depth1
s75_main_single_evasion_actionable_depth2plus
s75_main_single_evasion_depth3plus
s75_main_single_evasion_chain_1
s75_main_single_evasion_chain_2
s75_main_single_evasion_chain_3plus

s75_main_checking_edges_searched

s75_main_check_child_entered
s75_main_check_child_movegen
s75_main_check_child_terminal_0
s75_main_check_child_evasions_1
s75_main_check_child_evasions_2
s75_main_check_child_evasions_3plus

s75_main_depth1_nodes
s75_main_depth1_in_check
s75_main_depth1_single_evasion
s75_main_depth1_entered_from_checking_edge
```

### 2.2 Qsearch funnel（与 main 严格分开）

```text
s75_q_nodes
s75_q_in_check_nodes
s75_q_single_evasion_nodes_raw
s75_q_single_evasion_qply0
s75_q_single_evasion_qply1plus
s75_q_checking_edges_searched
s75_q_check_child_entered
s75_q_check_child_movegen
s75_q_check_child_terminal_0
s75_q_check_child_evasions_1
s75_q_check_child_evasions_2
s75_q_check_child_evasions_3plus
```

### 2.3 Piggyback rule（v1 硬要求）

- 观察阶段**不得**为统计 checking-edge evasion count 额外生成合法着法。
- checking child 只有真正进入并自然到达 movegen 时，才记录应招数分桶。
- `terminal_0` 来自 child 自然判定为无合法应招，而不是额外探测。
- 未来若需要评估 S7.5B eligibility probe 成本，另做 profiling-only
  `count_legal_evasions_up_to_3()`，不计入 S7.5-0 wall 数据。

### 2.4 本阶段要回答的问题

1. single-evasion raw -> actionable -> depth1 horizon -> depth2+；
2. consecutive single-evasion chain length distribution；
3. checking edge -> child entered -> natural movegen -> evasions 0/1/2/3+；
4. forcing line 现在何时终止：main depth -> depth1 -> qsearch -> mate/terminal。

## 3. S7.5A — single-evasion extension（参数后置）

```text
profile:  current-final-single-evasion（候选）
base:     CurrentFinal
policy:   uses_single_evasion_extension()，NOT uses_forcing_search()

S75A_FORCING_BUDGET:
    TBD_AFTER_S75_0
initial hypothesis:
    2
```

规则：

```text
main-search node in-check 且合法着法数 == 1：
    child_depth = depth   # 而不是 depth - 1
    budget -= 1

depth == 0 或 budget == 0：不扩展
```

S7.5-0 先统计 single-evasion 的 remaining-depth 分布与连续链长度，再冻结
budget；**之后绝不拿 tactical corpus reverse-tune**。

### 3.1 TT 隔离（v1 硬要求）

- 新增独立 predicate `uses_single_evasion_extension()`。
- budget-aware TT key 只作为 extension 正确性 context。
- 优先保持 CurrentFinal 原有 TT depth-reuse 语义：

```text
same position + same extension budget
    -> normal CurrentFinal TT probe/store policy
```

### 3.2 S7.5B-0 post-A observation

S7.5A 已进入 production 后，S7.5B 必须先以新的 `CurrentFinal` 基线做
observation-only attribution。`s75b-probe` 只在 bench diagnostic 中启用，
对 checking child 使用饱和到 `3+` 的合法应招 probe；它不改变 depth、budget、
TT、node acquisition 或搜索结果。B 的 budget 语义（shared 或独立）和任何
candidate 实现都必须等待这一步数据后再冻结。

固定 corpus 仍为 `80 S7 d6/d7 + 120 R2 d8`。B-0 单独记录 checking-edge、
`evasions == 2` 的 parent depth/A-budget、与 single-evasion 的相邻关系，以及
probe pseudo-move/legality-test 成本；不得回写或混入 pre-A 的 S7.5-0 evidence。

- 仅当 same position + different remaining budget 时必须隔离。
- 若实现发现普通 `depth >= requested` TT reuse 在 extension 下有正确性问题：
  **STOP**，把 exact-depth TT policy 作为独立设计决策，不得顺手继承 S2.1。

## 4. S7.5B — bounded checking extension

第一版只做：

```text
checking child 恰有 2 个合法应招
    -> extension
```

```text
0 evasions = checkmate -> terminal，不进入 eligibility
1 evasion          -> 已由 S7.5A single-evasion 覆盖
2 evasions         -> S7.5B 第一版唯一目标
3+                 -> 暂不扩展
```

B 要回答的问题：

```text
“A 无法覆盖、但分支因子仍极低的两应招将军，值不值得 extension？”
```

若 B 成功，再单独评估 `evasions <= 2` 以及 A+B adjacent stacking。S75A/B
共用总 forcing budget；B 预算同样 TBD_AFTER_S75_0 / S7.5A evidence。

## 5. S7.5C — singular extension（设计占位）

不实现 exclusion search 之前，观测 counter 只能叫：

```text
s75_singular_precondition_nodes
```

只能回答“有多少节点值得未来尝试 singular test”，不能声称 move 真是
singular。真正的 singularity 必须等 S7.5C 做 alternate/exclusion search。

## 6. S7.5A gate chain（v1）

```text
G0  production invariance
    30 S4, depth 6
    CurrentFinal exact nodes/qsearch_nodes/score/bestmove/PV/seldepth

G1  policy isolation
    candidate == CurrentFinal 所有现有 policy 维度
    唯一差异 == S75A forcing predicate
    旧 threat-aware profiles / Current rollback 不变

G2  extension / budget / TT-context correctness
    edge 级别纯函数测试 + budget 耗尽测试 + TT key context 测试

G3  fixed-depth explosion fuse
    80 S7, d6/d7, cold TT
    aggregate nodes < 2x baseline
    aggregate wall  < 2x baseline
    任一超过 -> REJECT，不进 fixed-wall gate

G4  general fixed-wall cost characterization
    80 S7, 1s / 3s
    report completed depth / seldepth / nodes / NPS / extension count /
    budget consumption
    root completed depth 是 COST signal，不是 strength gate：
        median root depth -1 不自动 reject
        median root depth -2 或广泛系统塌陷 = hard bad sign

G5  fixed-depth tactical safety（teacher-directed，允许正确改进）
    S6 teacher challenge d6
    report:
        teacher bestmove:
            baseline-only correct / candidate-only correct / both / neither
        cp 比较: |A-teacher| vs |B-teacher|
        candidate improvement/regression at >=100/300/500cp
        mate:
            baseline sees / candidate misses = REGRESSION
            baseline misses / candidate sees = IMPROVEMENT
        mate distance:
            candidate closer to teacher = IMPROVEMENT
            candidate farther           = REGRESSION
        mate side flip wrong side       = HARD REJECT
    hard reject:
        baseline correctly sees forced mate, candidate loses it
        baseline finds the only defense, candidate loses it
        candidate invents wrong-side mate
        candidate converts teacher-confirmed correct mate -> cp/non-mate
    cp -> correct mate、mate distance closer to teacher、large cp closer to
    teacher：允许，且计为 candidate win

G5W fixed-wall tactical / horizon effectiveness（新增）
    120 R2 tactical，1s and/or 3s
    同 G5 的 teacher-directed 方向比较
    candidate tactical gains > tactical regressions
    hard reject: baseline-correct forced mate / only-defense lost

G6  depth stability / convergence（secondary diagnostic）
    80 S7 d6 -> d7
    teacher-directed，不把 raw bestmove change == bad

REVIEW

Arena only after explicit GO
```

## 7. Execution order

```text
1. S7.5-0 observation-only（当前已授权）
2. 数据 review，冻结 S75A_FORCING_BUDGET
3. S7.5A 实现
4. G0-G6 + G5W
5. verdict review
6. 只有 A 通过，才设计/实现 S7.5B
7. S7.5C 只做 precondition observation / design
```

## 8. Hard constraints

```text
- 不重新激活 S2.1 threat-aware profile
- 不改 CurrentFinal production semantics（observation/gate 阶段）
- 不改 LMR eligibility / R1 / R2
- 不加 LMP
- 不改 null move / futility / qsearch / eval / ordering
- 不拿 tactical corpus reverse-tune budget
- 不提前 promote；Arena 必须显式 review GO
```

## 9. Final review note

v0 verdict: APPROVED_WITH_REVISIONS。
v1 已并入 7 项修订：piggyback funnel、budget TBD、TT 单变量隔离、
G4 降级为 cost signal、G5 teacher-directed、B 只做 evasions==2、
singular counter 更名 precondition。
