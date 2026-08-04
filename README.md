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
cargo run --release -- bench profile --nodes 100000
```

- `smoke`：两个锁定 fixture（startpos / queen-win，disabled 深度 3），精确校验 nodes/score/bestmove/PV。
- `standard`：10 个单局面 fixture（开局、战术中局、封闭中局、王暴露、高分支、车兵残局、KQK、KRK、halfmove 上下文），三种 TT 模式 `disabled`/`cold`/`warm`，默认 `repeat 1`。
- `throughput`：固定 nodes 预算测 NPS（默认 100000，默认 `repeat 3`）。
- `profile`：固定 nodes 预算记录 qsearch、evaluation、movegen、make/unmake、
  TT 以及显式候选 profile 的 aspiration/SEE/LMR/null/futility 计数；结果只用于定位
  Depth 7–8 瓶颈，不把计数或节点数解释成 Elo。
- 可选 `--mode disabled|cold|warm|all`、`--repeat N`、`--nodes N`、`--depth N`、
  `--movetime MS` 和 profile 专用的 `--fen FEN`，以及
  `--profile reference|m4.1|pvs|see|aspiration|lmr|null|futility|current`、
  `current-lmr|current-threat-aware|current-threat-aware-no-qchecks|current-threat-aware-eval-order|current-threat-aware-eval-only|current-threat-aware-order-only|current-qsearch-movegen|current-qsearch-pruning|current-qsearch-fast-pruning`，
  或一个明确的累计 `current-aspiration[-lmr[-futility[-see]]]` profile。
- `--profile` 选择搜索配置：**默认 `reference`**，使用 M4.0 的搜索路径；`m4.1` 是
  M4.1 完整窗口路径；`pvs` 是 M4.1 + PVS 的独立基线；`see`、`aspiration`、
  `lmr`、`null`、`futility` 分别只打开一个 SEARCH 1 候选；`current` 是已批准的
  M4.1 排序 + PVS + 已批准的 specialized qsearch movegen 生产路径；SEE pruning、
  fast SEE、aspiration、LMR、null 和 futility 仍关闭。`current-aspiration-*` 是
  累计 tournament candidate profile；没有现有 profile 同时包含 null 与全部累计功能。
  `current-threat-aware` 是 S2.1 的候选 profile，增加有界 king-danger 评价、检查/单一
  应将扩展、前两层 qsearch 检查和威胁感知排序；不改变 `Current`，也不打开 LMR、null、
  futility 或 SEE pruning。`current-threat-aware-no-qchecks` 与
  `current-threat-aware-eval-order`、`current-threat-aware-eval-only` 与
  `current-threat-aware-order-only` 是 S2.1b/S2.1c 仅供 `bench profile` 的成本归因变体，不能作为
  UCI 启动 profile。
  M4.1 的历史 A/B 对照见 `docs/benchmarks/m4.1-quiet-move-ordering.md`，M4.2 的批准
  lineage 见 `docs/benchmarks/m4.2-principal-variation-search.md`。

完整环境、命令与数值结果见 `docs/benchmarks/m4.0-search-baseline.md`；profiling
样本见 `docs/benchmarks/perf-profiling.md`；SEE/qsearch 收缩候选的独立记录见
`docs/benchmarks/search-see-qsearch.md`；aspiration/LMR/null/futility 候选的
记录见 `docs/benchmarks/search-pruning-candidates.md`。**M4.0 只建立测量基线；
所有这些搜索增强目前仍未通过性能/Elo 门禁，不是已接受的搜索基线。**

## E1 UCI protocol smoke

仓库内的 Python 工具现在明确只承担小规模协议与夹具 smoke，入口是
`tools/run_protocol_smoke.py`，使用 `python-chess` 校验每步合法性、终局和
和棋声明，并生成可回放 PGN、逐局 JSONL、运行 manifest 和汇总报告。它不
承担正式棋力评测，也不把短对局结果解释成 Elo。先安装开发依赖：

```bash
python -m pip install -r tools/requirements.txt
```

构建两个 UCI 引擎后运行 16 局 protocol smoke：

```bash
cargo build --release
python tools/run_protocol_smoke.py \
  --engine-a target/release/chess-engine-demo \
  --engine-b target/release/chess-engine-demo \
  --games 16 --movetime-ms 100 --move-grace-ms 25 --hash-mb 16 --seed 0 \
  --output-dir tournament-results/selfplay-smoke
```

默认规模为 16 局；smoke 夹具包含 32 条合法 UCI 开局线，每条开局
自动执行白黑换色。报告记录 baseline/candidate 的路径、文件 SHA-256、显式
git SHA、UCI 身份、Hash、movetime、host-side grace、种子、颜色、stderr、
结果、耗时和 PGN。每个开局严格跑完整白黑换色 pair；五分类 pair 计数、
候选方得分、Elo 点估计和区间都明确标记为 diagnostic。当前固定 draw-rate
模型只用于诊断，不是经过验证的 GSPRT，不提供 `alpha/beta` 错误率保证，
也不会提前停止比赛。正式 feature acceptance 交给 Fastchess/OpenBench/
Fishtest；Fastchess 的入口、PGN 书库 manifest 和固定参数见
`tools/run_fastchess.py`、`tools/fastchess_profiles.json` 与 `books/`。
最终状态只允许为 `COMPLETED`、`INCONCLUSIVE` 或
`INTEGRITY_FAIL`；该工具不会把固定局面 benchmark 或短 self-play 直接解释
成 2500+ Elo。统计背景参见
[Stockfish Fishtest Mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)。
`--sha-a`/`--sha-b` 未提供时记录为 `unknown`。

## 手工 UCI 示例

通过 stdin 逐行输入：

```
uci
ucinewgame
position startpos
go depth 4
quit
```

典型输出（TT-disabled / 公开禁用路径 baseline 实测；启用持久 TT 后 `nodes` 可能变化，但 `score` / `bestmove` / `PV` 语义保持一致；EVAL 1A 保持 startpos depth 3 为 cp 50）：

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
- ✅ **评估已含材质、基础 PST、EVAL 1A tapered King PST 与 EVAL 1B KQK/KRK 收官项**：
   EVAL 1B 只识别恰好双王加单后/单车的局面，并保留逼和边界；
   killer / history 等搜索增强不因本项自动启用。quiescence 搜索（M2.1）已就位，**显著缓解**吃子 / 升变层面的 horizon effect
   （处理常规吃子、升变的战术延伸）；但仍有 `MAX_QPLY` 上限，且 counter-check
   子局面会在安全上限处使用静态估值，是**有界近似**而非完全正确解决。此外引擎对
   发展、中心、兵形等位置因素仍无概念。
- ✅ **置换表（TT，M3.2）context-safe 身份隔离**：TT 命中键不只使用 board Zobrist，
  halfmove clock 与 repetition signature 也被纳入，因此不同 halfmove / 重复上下文不会
  产生错误命中；启用 TT 与禁用 TT 保持**完全相同**的 minimax / 和棋 / 将死语义。
  M4.0 reference 回归基线继续锁定 `startpos depth3 = 1149 节点 / bestmove b1c3 / score 50`
  与 `queen-win depth3 = 963 节点 / bestmove e4a4 / score 890`；Current 的 SEE/EVAL 候选
  结果单独记录，不改写该历史基线。

## 开发路线

- **Milestone 0**：可信基线 —— 修复搜索终局边界、加固 FEN、UCI 仅合法着法、加 CI、加 README。
- **Milestone 1（已完成）**：真正的 UCI Demo —— 搜索在独立线程运行，`stop` 即时中断；
  时间控制 `go movetime` / `infinite` / `wtime` / `btime` / `winc` / `binc` / `movestogo` 可用
   （soft/hard deadline + 安全余量）；`info` 输出 `depth` / `score` / `nodes` / `time` / `nps` / `pv`。
- **Milestone 2（已完成）**：像在下棋 ——
  - ✅ quiescence search（吃子 + 升变；被将军时解将全部走法都搜）—— M2.1 完成；
  - ✅ 着法排序（MVV-LVA / 升变；killer、history 暂未加）—— M2.2 完成；
  - ✅ 完整主变 `info pv` + PV tracking —— M2.3 完成；
  - ✅ Piece-Square Table 评估（material + 基础 PST）—— M2.4 完成；
- **Milestone 3（已完成）**：和棋状态与置换表（顺序已锁定，TT 在 draw context 稳定之后）——
  - **M3.0 状态与 Zobrist 基础 ✅**：`GameState` / UCI `position ... moves` 历史 / incremental
    Zobrist key / 搜索路径 hash stack / halfmove clock 正确传入搜索；已保存 UCI 对局真实历史，已维护搜索路径 hash stack。
  - **M3.1 和棋规则 ✅**：insufficient material 自动和棋；fifty-move 与 threefold 为
    claimable 0 分选项（支持 current claim 与 intended-move claim，terminal 优先）。
  - **M3.2 置换表（TT）✅**：context-safe `TtKey`（board Zobrist + halfmove clock +
    repetition signature；Threat-Aware candidate 另显式携带 forcing budget）；Exact / Lower /
    Upper；depth-preferred direct-mapped 替换；mate score 的 ply 编解码；legal hash move
    排序；持久 `Arc<Mutex<TT>>` UCI 生命周期（`ucinewgame` 清空但保留容量，`position` 不清空）。
- **EVAL 1（本地实现，等待独立复核）**：
  - ✅ [EVAL 1A：tapered evaluation + King PST](docs/specs/eval-1a-tapered-king.md)；
  - ✅ [EVAL 1B：exact KQK/KRK mop-up](docs/specs/eval-1b-kqk-krk.md)；
  - 当前不宣称 Elo，Depth 7–8 的瓶颈仍须 profiling、qsearch 收缩和后续独立搜索里程碑处理。
- **D1.2 qsearch movegen（已合入 Current）**：非将军 qsearch 使用 specialized
  tactical generation；规则、搜索树和 PV 保持等价，降低节点内 movegen 成本。见
  [`d1.2-qsearch-movegen.md`](docs/benchmarks/d1.2-qsearch-movegen.md)。
- **D1.3/D1.4 qsearch pruning（D1.3 已关闭）**：D1.3 保守 SEE pruning 的正确性与
  安全门禁通过，但 D1.12 固定时间深度门禁未形成稳定收益，最终
  `PERFORMANCE REJECTED / CLOSED`；保留 profile 仅用于历史复现，不再调参或晋级。
  D1.4 fast SEE 正确性通过但性能门禁失败，二者均未进入 Current。见
  [`d1.3-qsearch-see-pruning.md`](docs/benchmarks/d1.3-qsearch-see-pruning.md) 和
  [`d1.4-fast-pruning-see.md`](docs/benchmarks/d1.4-fast-pruning-see.md)。
- **D1.14 Current vs CurrentLmr**：Cutechess 进程、opening provenance、PGN/pair
  完整性和 candidate-first 统计方向的 40 局 smoke 已通过；历史 9,704 局
  `10+0.1` 运行因本地墙钟预算中止，保留 827 局原始 PGN，其中只有 826 局/413
  对可用于 paired statistics。CP-only Stockfish 赛后筛查结果混合，未授权短赛，
  `CurrentLmr` 仍为 candidate-only，`Current` 保持不变。见
  [`d1.14-current-vs-current-lmr.md`](docs/benchmarks/d1.14-current-vs-current-lmr.md)。
- **S3-PROMOTION CurrentFinal vs Current（SPRT H1 accepted）**：冻结的
  `current-final` 组合候选在新的 500-position opening slice 上完成 106 局，
  以 62-28-16（66.038%）触及 `H1=+60 Elo` 边界；PGN、manager 日志、完整命令、
  SHA-256 provenance 和独立合法性/配对 verifier 均已保存。该结果只说明候选通过
  本次正式推广门禁。随后默认 UCI 启动已切换到 `current-final`；显式
  `--profile current` 仍保留旧生产基线，便于历史复现和回退。见
  [`s3-promotion.md`](docs/benchmarks/s3-promotion.md)。
- **S2.1 Threat-Aware Search（候选开发中）**：新增显式
  `current-threat-aware` 启动 profile，包含候选 king-danger 评价、最多四个共享预算的
  check/single-evasion 主搜索扩展、qply 0/1 的 bounded checking qsearch、threat-aware
  ordering 和上一完整迭代的 root score ordering。候选 TT key 显式区分剩余 forcing
  budget，并对候选 score cutoff 使用精确 nominal-depth 门禁；`Current`、
  LMR/null/futility/SEE pruning 均保持原边界。见
  [`s2.1-threat-aware-search.md`](docs/specs/s2.1-threat-aware-search.md)。
  之后的 9 个实战保存局面、1/3/10 秒冷 TT gate 没有产生固定时间名义深度正收益，
  这是搜索成本负向信号而不是完整棋步质量结论。随后对 54 条保存决策做了
  Stockfish 18 固定 500k 节点 CP/mate 分离重评分：king-danger/defensive-resource
  组初步偏向候选，但 control 组有回归风险，因此总体为 MIXED、只授权窄范围 redesign；
  `current-threat-aware` 仍为 candidate-only，详见
  [`s2.1-practical-position-gate.md`](docs/benchmarks/s2.1-practical-position-gate.md)。
  S2.1b 在相同 9 个局面的 3 秒四路归因中发现 quiet qsearch checks 和 forcing
  extensions 的成本会压低完成深度；只保留 threat evaluation/order/root reorder 的
  bench-only 变体恢复了 Current 的总体中位深度，但仍有逐局回退，故继续做共享核心归因，
  不运行比赛、不改变 Current。见
  [`s2.1b-threat-aware-cost-attribution.md`](docs/benchmarks/s2.1b-threat-aware-cost-attribution.md)。
  S2.1c 进一步拆分共享核心：`eval-only` 在本次轮换 3 秒诊断中相对 Current 为
  1 胜、7 平、1 负的逐局完成深度，并避开 `control-237` 的有害 `g8h8`；
  `eval+order` 仍为 6 局较浅，故只保留 `current-threat-aware-eval-only` 进入下一轮
  1/3/10 秒门禁，仍不运行比赛、不改变 Current。见
  [`s2.1c-shared-core-attribution.md`](docs/benchmarks/s2.1c-shared-core-attribution.md)。
  S2.1d 的 1/3/10 秒 54 次固定时间门禁中，三档中位深度均未低于 Current、
  所有候选污染计数均为零，但 27 对仅有 2 次 EvalOnly 更深、6 次 Current 更深，且
  `control-237` 在 1 秒仍走出有害 `g8h8`，故门禁 FAIL。EvalOnly 不进入 Current，
  不进行 Stockfish 重评分或比赛。见
  [`s2.1d-king-danger-eval-fixed-time.md`](docs/benchmarks/s2.1d-king-danger-eval-fixed-time.md)。
- **D1.6 incremental base evaluation（已关闭）**：历史候选曾缓存基础 tapered
  MG/EG/phase；固定深度结果等价，但三路跨二进制性能门禁未通过，运行时代码已移除。
  仅保留文档与 Git lineage。见
  [`d1.6-incremental-base-eval.md`](docs/benchmarks/d1.6-incremental-base-eval.md)。
- **D1.7 one-pass full evaluation（已关闭）**：历史候选曾在单次 64 格扫描中同时
  计算 MG/EG/phase；固定深度搜索结果完全一致，但交错 paired wall-time 没有稳定
  改善，运行时代码已移除。仅保留文档与 Git lineage。见
  [`d1.7-one-pass-eval.md`](docs/benchmarks/d1.7-one-pass-eval.md)。
- **D1.9/D1.9.1/D1.9.2 外部搜索安全验证 harness（基础设施完成）**：使用固定版本的 EPD manifest，
  通过带绝对 search deadline、stderr 诊断和 completed-depth/PV 门禁的真实 UCI 进程比较 `current`
  与 `current-qsearch-pruning` 的合法着、终局、mate/score class 和候选相对基线安全性；
  不改变 `Current`，不代表 Elo。见
  [`d1.9-external-search-validation.md`](docs/benchmarks/d1.9-external-search-validation.md)。
- **D1.10 SEE-specific corpus expansion（验证数据完成）**：将 manifest 固定为 23 个 case、46 次
  真实 UCI 搜索，覆盖升变链、唯一升马、吃子升变、x-ray、钉住子、王吃子、普通与负 SEE 将军吃子、
  EP 发现将军和防守吃子案例，
  支持显式 forbidden-move 门禁；仍只比较 `current` 与 `current-qsearch-pruning`，不改变
  `Current`，不代表 Elo 或允许 SEE pruning 晋级。见
  [`d1.10-see-specific-corpus.md`](docs/benchmarks/d1.10-see-specific-corpus.md)。
- **D1.11/D1.11.1 外部位置切片差分 smoke（PASS，非 motif 验证完成）**：基于固定版本的
  Official Stockfish books，提交 32 个外部 case，按 `closedpos` / `stalemate-stress` /
  `endgames-a` / `endgames-cdb` 各 8 个切片，完成 64 次真实 UCI 搜索并通过合法着、completed
  depth、PV 回放、score/mate 自洽、超时、正常退出码和进程完整性门禁。离线 verifier 还会
  按固定 source line 与本地 SHA-384 已验证的上游 EPD 逐行比对。它与 D1.10 的 23 个项目自建
  case 分开统计；不代表 tactical/mate/promotion/endgame motif 已验证，不改变 `Current`，
  不代表 Elo，也不授权 D1.3 SEE pruning 晋级。见
  [`d1.11-external-validation-corpus.md`](docs/benchmarks/d1.11-external-validation-corpus.md)。
- **D1.12 D1.3 depth-uplift gate（FAIL，D1.3 已关闭，Current 未改变）**：复用 D1.10/D1.11 corpus，
  增加 depth override 以及十个标准局面的固定时间 `A/B` 交错对照；记录 completed depth、
  nodes、耗时、score、bestmove 和 PV。它不修改搜索实现或 `Current`，不把 protocol smoke
  或固定时间样本直接解释成 Elo；结果显示没有稳定的跨局面 depth uplift，因此 D1.3
  关闭且未进入 `Current` 或 Elo/SPRT。当前 UCI 未暴露 qsearch counter，因此报告明确记为 null。
  见 [`d1.12-depth-uplift-gate.md`](docs/benchmarks/d1.12-depth-uplift-gate.md)。
- **D1.13 isolated Current + LMR（PASS，Current 未改变）**：新增独立的
  `current-lmr` 启动 profile，只继承 Current 的 PVS、排序和 D1.2 qsearch movegen，
  额外开启既有保守 LMR；aspiration、null、futility、SEE pruning 和 fast SEE 均关闭。
  D1.10 depth 7 与 D1.11 depth 6 差分无回归，1s/3s 固定时间矩阵全部完成，3s
  等深耗时总体中位数达到门槛，因此候选具备单独 Elo/SPRT 资格，但没有自动晋级 Current。
  见 [`d1.13-current-lmr-depth-gate.md`](docs/benchmarks/d1.13-current-lmr-depth-gate.md)。
- **SEARCH 1（本地候选，未接受）**：独立 profile 与累计
  `current-aspiration-*` candidates 提供 aspiration、受限 LMR、验证式 null probe
  和浅层 futility；批准的 `current` 仍关闭这些开关。见
  [`search-pruning-candidates.md`](docs/benchmarks/search-pruning-candidates.md)。
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
