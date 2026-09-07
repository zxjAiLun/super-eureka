# 2026-09-07 · S11-B4-B Incremental R12 Search Stack

commit: `38aaeee`(前置:`ebd95ca` B4-A / `3b9cc9f` 文档)

## 前因

B4-A 以 448 ns/edge(合法 playout corpus 上的 microcost,非 B1 真实搜索边分布的无
偏估计)过成本门。审批方冻结三项实现细节后批准 B4-B:①不改现有 `frames:
Vec<AccumulatorFor>`(baseline 热路径形状不动),另加 `Option<Vec<R12RelationState>>`;
②child relation state 每边只 recompute 一次(API 先拆 primitive);③fresh R12 oracle
profile 保留,新增 `-inc` profile,同 binary 三路径(E3 / fresh / inc)。

## 做了什么

1. **API refactor**:`r12_apply_relation_state_diff(before, child_state, pos, acc)`
   (接收已算好的 state)+ `r12_rebuild_perspective(pos, child_state, ...)`;原
   `r12_apply_relation_delta` 降为 bench 便捷封装。
2. **`NnueSearchState`**:加 `r12_relation_frames: Option<Vec<R12RelationState>>` 与
   `r12_incremental: bool`;`with_r12_incremental` 构造器(root = full combined
   accumulator + recompute(root));`push_child`(V2 delta → child state recompute →
   king: mover 视角重建 + 其他视角 diff / normal: 双视角 diff → push 双栈);
   `push_null_child`(copy 双栈);`pop`/`restore_root` 维护双栈;`evaluate_cp_i32`
   inc 路径 = 纯 dense-from-accumulator。
3. **新 profile** `current-final-nnue-v2q-material-r12-inc`(search 13 处 policy
   list + exhaustive test 48 profiles + bench/UCI 表 + 双向 feature_set fail-closed
   + 状态构造分派)。UCI `SearchNnueBackend` 增加 `r12_incremental` 位。
4. **audit 语义切换**:inc 栈直接比 full refresh(fresh-hybrid 状态保持 B2 语义),
   audit 仍返回 fresh score。

## 遇到的问题与解决

- 借用检查:push_child 里 relation 栈可变借用与 `self.top()` 冲突 → 先
  `as_ref()` 拷出 parent_state,所有计算完再 `as_mut()` push。
- Part 2b 初版想"每 11 ply 裸 pop"练 pop 路径——意识到 pop 不 unmake 会让栈顶与
  `pos` 脱同步、污染后续比较 → 移除;pop 正确性由 search 级 tree-identity 门覆盖
  (真实搜索中 pop 与 unmake 成对)。

## 结论(冻结事实)

- **Parity 全 PASS**:10k corpus lanes+raw exact;19,692 fresh 转移;10/10 定向
  fixtures;1.28M attack bits;**inc-stack 200 转移 + 29 null push(真实栈 API)
  == full refresh,0 mismatch**;**search 树一致性:24-FEN × 50k nodes,score/
  bestmove/PV/nodes/qsearch/movegen/TT 等 17 字段 0 mismatch**。
- **配对 NPS(frozen 协议,同 binary):median 0.9488 ≥ 0.90 → 门 PASS,B5 解锁**。
  分类 vs B2 fresh:opening 0.56→1.00-1.05;middlegame 0.56-0.60→0.94-1.00;
  tactical 0.64-0.70→0.90-0.91;endgame 0.75-0.94→0.80-0.91。残余 gap 在稀疏
  终局(E3 eval 极快,~450ns/边固定成本占比相对变大)。
- **工程验证**:B2 的 ~22µs/eval relation 工作被移到边频率并压成变化行 diff 后,
  完整搜索从 0.58 回到 0.95——B1 churn 数据的方向性预测成立。
- **下一步**:B5(256 roots × 100k 固定节点 search validation,lockbox scorer 已
  就绪)待审批方放行。

验证:`cargo test --release --lib` 441/441;fmt clean;结果
`results\s10\s11-b4b-{parity,tree-identity,nps,verdict}.json`。
