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

## 手工 UCI 示例

通过 stdin 逐行输入：

```
uci
ucinewgame
position startpos
go depth 4
quit
```

典型输出：

```
id name ChessEngineDemo
id author Rust-learner
uciok
info depth 1 score cp 0 pv b1c3
info depth 2 score cp 0 pv b1c3
info depth 3 score cp 0 pv b1c3
info depth 4 score cp 0 pv b1c3
bestmove b1c3
```

## 接入 GUI

1. `cargo build --release`，可执行文件在 `target/release/chess-engine-demo`（Windows 上为 `.exe`）。
2. 在 GUI 里把引擎路径指向它，协议选择 **UCI**。
3. 已支持：`uci` / `isready` / `ucinewgame` / `position startpos|fen ... moves ...` /
   `go depth N` / `stop` / `quit`，外加调试用的 `perft N`。

## 当前支持的 UCI 命令

| 命令 | 状态 |
| --- | --- |
| `uci` | ✅ |
| `isready` / `readyok` | ✅ |
| `ucinewgame` | ✅ |
| `position ... moves ...` | ✅ 只接受**严格合法**着法；遇到非法着法输出 `info string invalid move <uci>` 并保持原局面，绝不偷偷重置 |
| `go depth N` | ✅ |
| `stop` | ⚠️ 占位：当前为单线程同步搜索，`stop` 不会中断搜索（见路线图 Milestone 1） |
| `quit` / `exit` | ✅ |
| `perft N`（调试） | ✅ |

### 暂不支持（尚未实现）

`go movetime` / `go wtime btime winc binc` / `go infinite` / `go nodes`
—— 当前这些参数会被忽略并退化为 `go depth 4`。时间管理与异步停止在 Milestone 1 实现。

## 正确性状态（Milestone 0）

- ✅ **搜索在叶子节点正确识别将死 / 逼和**：终局判定在 `depth == 0` 的估值之前执行，
  将死返回随距离变化的 mate score，逼和返回 0（修复了“边界上的将死被当成普通子力局面”的 P0 bug）。
- ✅ **FEN 解析加固**：每个 rank 恰好 8 格、数字仅 `1..=8`、双方王唯一、`fullmove >= 1`、
  吃过路兵目标在合法 rank、多余字段报错，且**对任何字符串都不会 panic**。
- ✅ **UCI 历史着法仅接受严格合法走法**（原来用伪合法生成，会让被钉死的子或送将的棋混进来）。
- ⚠️ 评估目前**只有子力差**，且**还没有 quiescence**，因此仍可能出现 horizon effect
  （吃子后对方能吃回，但搜索正好截止，引擎只看到自己多吃一子）。这正是下一步要修的，
  第一剂药是 quiescence，而不是暴力加深。

## 开发路线

- **Milestone 0（当前）**：可信基线 —— 修复搜索终局边界、加固 FEN、UCI 仅合法着法、加 CI、加 README。
- **Milestone 1**：真正的 UCI Demo —— 搜索移到独立线程，`stop` 与时间控制
  （`go movetime` / `infinite` / `wtime` / `btime` / `winc` / `binc`）可用，`info` 输出 `nodes` / `time` / `nps`。
- **Milestone 2**：开始像在下棋 —— quiescence search（吃子 + 升变；被将军时解将全部走法都搜）、
  着法排序（PV / MVV-LVA / 升变 / killer / history）、Piece-Square Table 评估、
  `info pv` 给出完整主变。
- **Milestone 3**：置换表与和棋规则 —— Zobrist hash、TT（Exact / Lower / Upper）、三次重复、
  五十回合、insufficient material、mate score 存取 TT 时的 ply 修正、`setoption name Hash`、`bench`。
- **Milestone 4**：高级增强（确认瓶颈后再加，且一次只加一个并做对照测试）——
  aspiration window、PVS、null-move pruning、LMR、SEE、futility pruning。Bitboard 不急。

## License

用于学习目的，随引擎源码自由使用。
