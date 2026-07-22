# Chess Engine Demo

一个**小而正确**的传统 Alpha-Beta 国际象棋引擎，用 Rust 从零实现，作为学习项目。
它支持 UCI 协议，可以接入标准 GUI（Arena、Cute Chess、Banksia 等）完整对弈与分析。

## 设计原则

- **正确性优先于速度**：先用最朴素的数组棋盘 `[Option<Piece>; 64]` 和清晰的 Rust 枚举，
  不提前上 bitboard / NNUE / 多线程 / 开局库 / 残局库。等 profiling 证明 movegen 成为瓶颈后，
  再在同样的 `Position` API 背后换实现。
- **分层解耦**：`Position` 不知道 UCI，`Search` 不知道 GUI，`Evaluation` 不修改 `Position`。

## 构建与测试

```bash
cargo build --release                       # 产出可执行文件
cargo test                              # 快速：perft 1-4、FEN、搜索、UCI
cargo test --release                  # 额外跑 perft(5) = 4,865,609
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

CI（GitHub Actions）在每次 push / PR 到 `main` 时自动跑以上全部。

## 运行 Perft（自带正确性校验）

```bash
cargo run --release -- perft 5
```

会打印 `perft(5) = 4865609`，与 Stockfish 官方参考值完全一致。
这是引擎的“正确性闸门”：任何 movegen 规则 bug 都会让这个数字偏离，此时**禁止**继续做搜索。

## 运行搜索基准（M4.0，仅测量）

`bench` 子命令运行一个**确定性搜索测量框架**，只驱动既有搜索入口并逐项记录结果，不改变棋力或搜索语义。输出行以固定前缀 `bench_result` / `bench_summary` / `bench_error` 开头，可用 `grep '^bench_'` 过滤。

```bash
cargo run --release -- bench help                              # 帮助
cargo run --release -- bench smoke                           # 锁定基线的快速校验
cargo run --release -- bench standard --mode all --repeat 1
cargo run --release -- bench throughput --mode disabled --nodes 100000 --repeat 3
```

- `smoke`：两个锁定 fixture（startpos / queen-win，disabled 深度 3），精确校验 nodes/score/bestmove/PV。
- `standard`：10 个单局面 fixture（开局、战术中局、封闭中局、王暴露、高分支、车兵残局、KQK、KRK、halfmove 上下文），三种 TT 模式 `disabled`/`cold`/`warm`，默认 `repeat 1`。
- `throughput`：固定 nodes 预算测 NPS（默认 100000，默认 `repeat 3`）。
- 可选 `--mode disabled|cold|warm|all`、`--repeat N`、`--nodes N`、`--profile reference|current`。
- `--profile` 选择搜索配置：**默认 `reference`**，逐字节复现 M4.0 基线（不启用 killer / history）；`current` 启用 M4.1 的 killer + history 安静着法排序。M4.1 的 Reference / Current A/B 节点与 NPS 对照见 `docs/benchmarks/m4.1-quiet-move-ordering.md`。

完整环境、命令与数值结果见 `docs/benchmarks/m4.0-search-baseline.md`。**M4.0 只建立测量基线，未做任何搜索优化。**

## 手工 UCI 示例

通过 stdin 逐行输入：

```
uci
ucinewgame
position startpos
go depth 4
quit
```

典型输出（TT-disabled / 公开禁用路径 baseline 实测；启用持久 TT 后 `nodes` 可能变化，但 `score` / `bestmove` / `PV` 语义保持一致；PST 已让 depth 3 的分值从纯子力的 cp 0 变成 cp 50）：

```
id name ChessEngineDemo
id author Rust-learner
option name Hash type spin default 16 min 1 max 1024
uciok
info depth 1 score cp 50 nodes 20 time 1 nps 20000 pv b1c3
info depth 2 score cp 0 nodes 141 time 6 nps 23500 pv b1c3 b8c6
info depth 3 score cp 50 nodes 1149 time 54 nps 21277 pv b1c3 b8c6 g1f3
info depth 4 score cp 0 nodes 8453 time 413 nps 20467 pv b1c3 b8c6 g1f3 g8f6
bestmove b1c3
```

## 接入 GUI

1. `cargo build --release`，可执行文件在 `target/release/chess-engine-demo`（Windows 上为 `.exe`）。
2. 在 GUI 里把引擎路径指向它，协议选择 **UCI**。
3. 已支持：`uci` / `isready` / `ucinewgame` / `position startpos|fen ... moves ...` /
   `go depth N` / `go nodes N` / `go movetime MS` / `go infinite` /
   `go wtime btime [winc binc] [movestogo]` / `stop` / `quit`，外加调试用的 `perft N` /
   `setoption name Hash value N`。
   搜索在独立线程运行，`stop` 能即时中断；时间管理为基础策略（soft/hard deadline + 安全余量）。

## 当前支持的 UCI 命令

| 命令 | 状态 |
| --- | --- |
| `uci` | ✅ |
| `isready` / `readyok` | ✅ 即使搜索进行中也立即回复 |
| `ucinewgame` | ✅ 重置 GameState 并清空 TT，保留 Hash 容量 |
| `setoption name Hash value N` | ✅ 调整持久 TT；`0→1`，`>1024→1024`；resize 前停止并 join 当前搜索 |
| `position ... moves ...` | ✅ 只接受**严格合法**着法；遇到非法着法输出 `info string invalid move <uci>` 并保持原局面，绝不偷偷重置；不清空 TT（context-safe key 负责隔离 halfmove / repetition 上下文） |
| `go depth N` | ✅ |
| `go nodes N` | ✅ |
| `go movetime MS` | ✅ |
| `go infinite` | ✅ 持续搜索直到收到 `stop`（覆盖同行的 clock / movetime 参数） |
| `go wtime btime [winc binc] [movestogo]` | ✅ 按走子方时钟分配；基础策略 |
| `stop` | ✅ 即时中断搜索并输出 `bestmove` |
| `quit` / `exit` | ✅ |
| `perft N`（调试） | ✅ |

### 暂不支持（尚未实现）

`ponder`、`searchmoves`、`mate N`（其余 UCI 命令与持久 TT 均已在 Milestone 3 支持）。
当前时间分配为**基础策略**（固定比例 + 安全余量），不根据局面复杂度动态调整。

## 正确性状态（Milestone 0）

- ✅ **搜索在叶子节点正确识别将死 / 逼和**：终局判定在 `depth == 0` 的估值之前执行，
  将死返回随距离变化的 mate score，逼和返回 0（修复了“边界上的将死被当成普通子力局面”的 P0 bug）。
- ✅ **FEN 解析加固**：每个 rank 恰好 8 格、数字仅 `1..=8`、双方王唯一、`fullmove >= 1`、
  吃过路兵目标在合法 rank、多余字段报错，且**对任何字符串都不会 panic**。
- ✅ **UCI 历史着法仅接受严格合法走法**（原来用伪合法生成，会让被钉死的子或送将的棋混进来）。
- ✅ **评估已含子力差 + 基础 Piece-Square Table（M2.4）**：位置因素（PST）已叠加在子力之上；
   killer / history / tapered eval 等仍待加。quiescence 搜索（M2.1）已就位，**显著缓解**吃子 / 升变层面的 horizon effect
   （处理常规吃子、升变的战术延伸）；但仍有 `MAX_QPLY` 上限，且 counter-check
   子局面会在安全上限处使用静态估值，是**有界近似**而非完全正确解决。此外引擎对
   发展、中心、兵形等位置因素仍无概念。
- ✅ **置换表（TT，M3.2）context-safe 身份隔离**：TT 命中键不只使用 board Zobrist，
  halfmove clock 与 repetition signature 也被纳入，因此不同 halfmove / 重复上下文不会
  产生错误命中；启用 TT 与禁用 TT 保持**完全相同**的 minimax / 和棋 / 将死语义。
  公开禁用路径的回归基线仍锁定 `startpos depth3 = 1149 节点 / bestmove b1c3 / score 50`
  与 `queen-win depth3 = 963 节点 / bestmove e4a4 / score 890`。

## 开发路线

- **Milestone 0**：可信基线 —— 修复搜索终局边界、加固 FEN、UCI 仅合法着法、加 CI、加 README。
- **Milestone 1（已完成）**：真正的 UCI Demo —— 搜索在独立线程运行，`stop` 即时中断；
  时间控制 `go movetime` / `infinite` / `wtime` / `btime` / `winc` / `binc` / `movestogo` 可用
   （soft/hard deadline + 安全余量）；`info` 输出 `depth` / `score` / `nodes` / `time` / `nps` / `pv`。
- **Milestone 2（已完成）**：像在下棋 ——
  - ✅ quiescence search（吃子 + 升变；被将军时解将全部走法都搜）—— M2.1 完成；
  - ✅ 着法排序（MVV-LVA / 升变；killer、history 暂未加）—— M2.2 完成；
  - ✅ 完整主变 `info pv` + PV tracking —— M2.3 完成；
  - ✅ Piece-Square Table 评估（material + 基础 PST；King PST 留到 tapered eval）—— M2.4 完成；
- **Milestone 3（已完成）**：和棋状态与置换表（顺序已锁定，TT 在 draw context 稳定之后）——
  - **M3.0 状态与 Zobrist 基础 ✅**：`GameState` / UCI `position ... moves` 历史 / incremental
    Zobrist key / 搜索路径 hash stack / halfmove clock 正确传入搜索；已保存 UCI 对局真实历史，已维护搜索路径 hash stack。
  - **M3.1 和棋规则 ✅**：insufficient material 自动和棋；fifty-move 与 threefold 为
    claimable 0 分选项（支持 current claim 与 intended-move claim，terminal 优先）。
  - **M3.2 置换表（TT）✅**：context-safe `TtKey`（board Zobrist + halfmove clock +
    repetition signature）；Exact / Lower / Upper；depth-preferred direct-mapped 替换；
    mate score 的 ply 编解码；legal hash move 排序；持久 `Arc<Mutex<TT>>` UCI 生命周期
    （`ucinewgame` 清空但保留容量，`position` 不清空）。
- **Milestone 4**：高级增强（确认瓶颈后再加，且一次只加一个并做对照测试）——
  aspiration window、PVS、null-move pruning、LMR、SEE、futility pruning。Bitboard 不急。

## 版本变更

- **v0.2.1**（当前）：时间安全 hotfix ——
  - **P0（超时判负修复）**：时钟分配（clock mode）现在把 `allocation`
    钳制在 `usable` 之内，避免大增量（`inc`）把当前步的 hard deadline 推到
    棋钟剩余时间之外而直接超时判负；新增边界测试断言
    `hard_deadline <= now + remaining - reserve`。
  - **P1**：极小 `movetime`（如 2ms）下 `soft` 不再晚于 `hard`，
    改为 `soft_budget = hard_budget * 90%`，保证 `soft <= hard <= movetime`。
  - **P1**：UCI 时间解析上限钳制为 `u32::MAX` 毫秒，避免
    `go movetime 18446744073709551615` 这类值让 `Instant + Duration` panic；
    新增 `catch_unwind` 单测。
  - **P1**：`go infinite` 现在为最高优先级，覆盖同行的
    `wtime` / `btime` / `movetime` / `nodes`，真正持续到 `stop`；
    删除 `SearchLimits.infinite` 死字段，无限搜索唯一由
    “无 depth + 无 nodes + 无 deadline” 表达（消除两套真相）。
  - 修复测试辅助 `recv_until` 只等 200ms 就返回的 bug（区分 Timeout 与
    Disconnected），“3 秒内返回”的测试现在真的等待到 3 秒。
  - 非阻塞项：`info nps` 转 `u64` 改为饱和而非截断；`stop_and_join` 在搜索
    线程 panic 时输出 `info string search thread panicked` + `bestmove 0000`。

## License

用于学习目的，随引擎源码自由使用。
