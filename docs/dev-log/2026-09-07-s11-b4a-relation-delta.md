# 2026-09-07 · S11-B4-A Relation-State Recompute + Delta FT Update

commit: `ebd95ca`(前置:`d03c6ce` B2/B3 closeout)

## 前因

B2/B3 判定:fresh-recompute hybrid 正确性完备但 NPS 0.58(< 0.80 门),B5 暂停。
成本归因显示每 eval 重加全部 ~24 活跃 relation 行(FT 行加法内存流量 ~12µs 主导,
attack 查询 ~7µs 次之)。审批方拍板:不做"完整局部依赖 incremental updater"(slider
ray 依赖枚举太险),改做 B4——child 侧一次生成 attack maps、全量重算 64 格 relation
state、diff、只对变化的 4-8 行做 FT add/sub。B4-A 先做 pre-gate:两件事必须证明
(attack-map 与 `is_square_attacked` 逐位 exact;state-derived rows 与 frozen emitter
逐位置 exact)+ 成本门 ≤1.5µs/edge(理想 <1µs)。

## 做了什么

1. **`Position::attack_map/attack_maps`**(chess/position.rs):每色一次棋盘遍历生成
   u64 伪攻击掩码;slider ray 穿空格 + 首个 occupied 含该格(含格!)——与
   `is_square_attacked` 的"从目标格向外第一阻塞者"视角等价。不是引擎 bitboard 化,
   只是 R12 用的局部 substrate。
2. **`R12RelationState`**([u8; 64],nnue_v2q_runtime.rs):每格一字节(none/A_N/
   A_B/A_R/A_Q/D/C + 棋子颜色位),perspective 无关;`recompute(pos)` 用 attack maps
   一次算完;`row_for_square` 在 apply 时用 child 的 king context 做 orientation/
   mirror 映射出两个 perspective 的行;`apply_diff` 跳过未变格,只对变化行 FT
   add/sub。
3. **模型公共原语**:`r12_apply_relation_delta`(边应用)、`r12_rebuild_perspective`
   (king-move 稀有路径:mover 视角 base+全量行重建)。
4. **bench `nnue-v2q-r12-delta-cost`**:确定性边 corpus(随机合法 playout + 每 250
   边从冻结 corpus FEN 重启)+ 计时边处理 + `--ratchet` 全量刷新校验重放。

## 遇到的问题与解决

- **King-move 语义 bug**(diff-chain 单测抓到):最初 king 分支只重建 mover 视角,
  漏掉另一视角的 relation diff(王走一步会同时改变两色的 attack map)→ 差恰好一行
  32(合成权重)。修复:mover 视角重建 + 其他视角照常 diff。
- **Corpus 重seed 死循环**:`corpus.len()%250==249 && seeds非空` 在不 push 的迭代里
  永真 → 无限循环(带 `--batch` 的两次运行超时被杀)。修复:推进式 `next_reseed`
  计数器,ratchet 重放逐决策镜像。
- 初版 bench 想直接解构私有 `WeightsFor` → 改为模型公共方法(r12_apply_relation_
  delta / r12_rebuild_perspective / moved_king_color / square_state),bench 只依赖
  公共面。

## 结论

- **Parity 全 PASS**:attack map 1,280,000 bits 逐位 exact(10k corpus × 2 色 ×
  64 格);state-rows == frozen emitter(10k corpus × 2 视角 + 单测 1200 playout
  位置);ratchet 10,000 边 combined(V2 delta + relation diff + king 重建路径)
  == full refresh,0 mismatch。
- **成本门 SMASH**:median **448.1 ns/edge**(p90 475 / min 427),低于 1500ns 门
  3.3×、低于 1000ns 理想线 2.2×。
- **投影**:B4 把 B2 fresh 路径 ~21.8µs/eval 的 relation 工作(扫描 + ~24 行批量
  重加)替换为 push 时的 ~0.45µs;eval 回到普通 dense-from-accumulator(~1.0µs)。
  E3 本机 ~6µs/node ⇒ 投影 NPS ratio ~0.85-0.92——落在 0.90 门附近,需要 B4-B
  真实测量裁决。
- **B4-B GO**:搜索栈集成(combined accumulator + [u8;64] state 入 frame;fresh
  R12 保留为 oracle)→ 复用 parity 资产(full==fresh==incremental;10k corpus +
  ~20k transitions + 定向 fixtures + fixed-node 树一致性)→ 重跑同一 24-FEN/
  200k-node/同 binary 配对 NPS,按 0.90/0.80 门判定。

验证:`cargo test --release --lib` 441/441(新增 3 个 s11b4a 测试);fmt clean;
结果在 `results\s10\s11-b4a-*.json`。
