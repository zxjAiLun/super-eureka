# 2026-09-07 · S11-B1 Relation-Churn Audit

commit: `3f171db`（branch `s10/nnue-production-foundation`，已推送）

## 前因

S11-A 已冻结 R12（V2 + 12-channel 关系 sidecar，输入 23,296）为运行时候选
（seed=20260819）。设计 runtime 前需要回答：一次搜索边平均改变多少条 relation
sidecar 行？该数字决定 (a) fresh-recompute hybrid 的第一版是否合理；(b) 未来
增量 relation updater 的优化空间。

## 做了什么

1. **`src\engine\nnue_search.rs`**：新增 `relation_churn` 模块（cargo feature
   `diagnostic_relation_churn`，默认关闭）——`EdgeRecord { removed, added,
   is_capture, mover_slider, phase_bucket }`、`enable(cap)` /
   `disable_and_take()` / `record_edge(before, after, mv)`。
2. **`src\engine\search.rs`**：在全部 4 个真实搜索 `make_move_profiled` 站点
   （约行 4974/7105/7293/7563）插入 feature-gated hook：make 前 clone 父局面，
   push_child 后调用 record_edge。production 构建零 churn 代码。
3. **`src\engine\bench.rs`**：`bench relation-churn --fen <fen> --nodes N
   --profile <p> --nnue-model <bin>`，输出单条 JSON（churn 分布 percentiles +
   capture/quiet、slider/non-slider、4 phase 桶）。
4. **`src\engine\nnue.rs`**：`relation_features_v2r12` 改 `pub`（diagnostic
   harness 用），语义零改动。
5. 32 个确定性 H0-C roots（8/phase）× 50k nodes，结果落
   `results\s10\s10-s11-b1-churn.json`。

## 遇到的问题与解决

- **nnue.rs 损毁**：批量字符串替换脚本把 `nnue.rs` 重写坏（815 deletions，
  `relation_features_v2r6/r12/r14` 全丢）。用 `git checkout HEAD --
  src\engine\nnue.rs` 恢复，改为最小 diff（仅 pub 化一个函数）重做。
  **教训：对该文件禁用整段脚本替换，用精确锚点 Edit。**
- `types::MoveFlag` 没有 `is_capture()`；改为
  `matches!(flag, EnPassant) || before.board()[to].is_some()`。
- Python heredoc 两次小错（exe 路径笔误、括号不匹配）浪费两轮；先小样本试跑。

## 结论

1,598,264 条真实搜索边：**median churn 4 行/边**（min 0 / max 10 per-root
median），p90≈10，p99≈18，全局 max 50；phase high 8 / mid 7 / low 4 / zero 0；
capture median 6 vs quiet 4；slider 5 vs non-slider 4。

- relation churn 温和 → **fresh-recompute hybrid（B2）第一版值得实现**；
- 若将来需要优化，增量 updater 平均只碰 4-8 行——但按协议推迟到看到真实
  benchmark 数字之后（churn ≠ fresh 成本：fresh 是全盘 64 格扫描 + attack
  查询，不是"多加几行"）。

验证：`cargo test --release --lib nnue` 76/76 PASS；production binary 无
churn 代码；诊断二进制备份于
`C:\Users\81489\AppData\Local\Temp\opencode\eureka-churn.exe`。
