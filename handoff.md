# ChessEngineDemo Handoff

> 状态快照：2026-09-10
> 仓库：`E:\AUbuntuProject\project\chessenginedemo`
> 工作分支：`s10/nnue-production-foundation`（已推送至 `b3145e1`；本次 S14 closeout 文档提交随后）
> `main` HEAD：`3dae2fa`（S9-B2 closeout，2026-08-26）
> crate / 二进制名：`eureka`（旧名 `chess-engine-demo` 已废弃）

## 文档制度（2026-09-07 起，长期有效）

1. **本文件（handoff.md）纳入版本控制**（原为 `.git/info/exclude` 不跟踪，正是它
   漂移 30 天的根因），每完成一个 season / 里程碑必须更新状态快照与结论地图。
2. **handoff.md 末尾维护 append-only 更新日志**（每轮一条，含日期、轮次、commit
   SHA、改动摘要），旧条目不改写。
3. **每轮工作在 `docs/dev-log/` 落一份开发文档**，命名
   `YYYY-MM-DD-<round-id>.md`（如 `2026-09-07-s11-b1-relation-churn.md`），内容：
   前因、做了什么、遇到的问题与解决、结论、涉及 commit SHA。追踪进度时按
   commit SHA ↔ dev-log 文件双向对照即可，不追求长文。

## 一句话结论

S4-S9 已完成核心性能、搜索选择性与 Eval2 晋级（S8 正式 SPRT +71.3 Elo）；S10 全季
NNUE 生产化建成但三轮 Arena 全拒；S11 R12 链（fresh 0.58 → delta 448ns → 栈
0.9488 → B5 平手）收尾，不继续 transfer 归因。**S12 CLOSE / PARITY：Bullet 配方
（SCReLU + 8 桶 FT256 + sigmoid 得分损失）全栈落地（v5 artifact、parity 全 PASS、
NPS 持平 E3）并保留为生产级 NNUE 基础设施；R0 早期筛选（128 局）与生产
current-final 平手（W/D/L 55/17/56，49.61%，pentanomial [10,5,34,6,9]）；R1
batch-cadence 单变量修复 FAIL（val 0.009657 > 门 0.00910，形态不变）；**S13-A
真实结果 blend FAIL（256 局 12.89%，-332 Elo——0.75 权重的在线快棋真实胜负
压垮 SF teacher 信号）**。NNUE 训练线两个便宜杠杆（cadence、result blend）均
已单变量否决；剩余杠杆 = 数据供给（5M+ fresh + 双信号，需新标注算力，未授权；
S14 实做 = CP-only + 5.0M 数据池，未采用双信号）。
工程资产保留：v5 runtime + current-final-s12 profile 生产可用；current-final
仍是生产引擎。R12-inc vs SF2400 的 1+0 计 Elo live match 仍在跑（`6a07cc07…`）。**

**S14 更新（2026-09-10）**：S14 = **S12-R0 配方逐字（CP-only），只把数据池扩到 5.0M**
（779,590 canonical CP + 4,220,410 fresh Fishtest；**未用真实结果 blend**——那是 S13、已 FAIL）。
训练完成、屏幕 58.01%；唯一正式 promotion SPRT `6cd87ee8…` 终判 **ACCEPT_H1**
（LLR 2.9696 ≥ 2.9444，177/500 pairs，354 局）。用户裁定：**S14 = PROMOTION-QUALIFIED
（棋力审批已获，不因 HOLD 作废）；生产默认保持 HCE-20260825，切换 HOLD**——
阻断项为受控 rollback 控制面缺位（生产级运维安全，非棋力）；无需任何补充比赛。
详见更新日志与 2026-09-10 closeout。**

## 术语与版本命名（2026-09-10 起）

四个规范名（禁止再用模糊叫法）：

| 名称 | 含义 |
|---|---|
| **HCE-20260825** | 旧生产引擎 = CurrentFinal search + HCE Eval2 = Arena EngineVersion `ce-currentfinal-20260825` |
| **S11-R12** | 旧 NNUE（FT128）+ R12；历史候选 |
| **S12-R0** | FT256 + SCReLU + 8 桶、780k 数据；NNUE 架构基线 |
| **S14** | S12 架构 + R12 + **CP-only 配方** + 5.0M 数据池 + CurrentFinal search = **当前最强 Eureka**；promotion SPRT `ACCEPT_H1`（binary `dceacfb7…`、model `329b7170…`、源码 `b3145e1`） |

- `current-final` 有两个合法身份：**`Arena channel: current-final`**（指针，当前 → HCE-20260825）
  与 **`engine profile: current-final`**（源码里真实存在的 profile 名）。**书写强制前缀**：单独
  说 `current-final` 默认指 Arena channel；凡指 profile 必须写 `profile:` 前缀（如
  `engine profile: current-final`、`engine profile: current-final-s12`）。禁用无前缀的模糊说法
  （"current-final 很强"、"current-final-s12 版本"、"S12 production"）。
- 两层歧义备忘：`engine profile: current-final`（=旧 HCE）≠ `Arena channel: current-final`；
  `engine profile: current-final-s12 --nnue-model nnue-s14-datasupply-v5.bin` 实际加载的是 **S14**
  权重（alias 名不跟踪模型身份）。
- 对齐事实（2026-09-10 实测）：本地 / GitHub / 服务器引擎代码 = `b3145e1`（当前 HEAD `ca6ad3c`
  仅文档）。最强组合已部署服务器并过 SPRT；生产 channel 仍指向 HCE-20260825——"已部署最强"
  ≠ "线上默认在用"。

**构建身份备忘（2026-09-10 实测）**：

- promotion binary `dceacfb7…` = 本地 WSL 构建（git-archive `b3145e1` + rustc 1.94.1 +
  `cargo build --release --locked` + `EUREKA_GIT_SHA/_SHORT/_DIRTY` env、`DATE` 留空）；
  **未走 GitHub 云端 workflow**（该 workflow 历史仅 2 次运行：8/5 失败、8/29 成功 @ `9ef078f`）；
  云端**从未实际构建过 `b3145e1`**——"同 recipe 会得到等价程序"是合理推断，不是实测。
- 复现实测：同环境重建 → 与 `dceacfb7` **仅差 6 字节**（RUNPATH 内嵌的 rustc 随机临时目录名），
  其余 1,201,706 字节逐字节一致。该 RUNPATH 仅在"登录 shell（`cc`=zig cc，clang 18.1.6）"
  构建时注入；非登录/云端环境（gcc + rust-lld 21）产物无 RUNPATH、字节不同但引擎等价。
- **默认启动澄清**：最新 release binary 已**包含并能运行**当前最强 S14 配置（binary 能力/来源
  与 promotion 对齐），但其**无参数默认启动仍是 HCE**（`engine profile: current-final`）；S14 仍需
  显式 `--profile current-final-s12 --nnue-model …`。**不要写"默认产物就是 promotion 引擎"。**
  跨环境（本地 vs 云端）不要指望二进制哈希一致，逐字节复现只差构建路径元数据。

```text
当前版本关系（2026-09-10）：
引擎源码版本        b3145e1（文档 HEAD 以分支 tip 为准）
服务器最新 binary   dceacfb7…（来自 b3145e1）
当前最强运行配置    S14 NNUE + CurrentFinal Search（model 329b7170…；SPRT ACCEPT_H1）
当前 binary 默认启动 HCE（engine profile: current-final），不是 S14
```

## 当前生产行为

### UCI 启动 profile（自 S8 起）

```text
无启动参数                    -> current-final（含 Eval2 集成位置评价，S8 b2c0efe 晋级）
--profile current-final       -> current-final
--profile current             -> current（历史回退路径，仍在）
```

`current-final` 当前组合：PVS + specialized qsearch movegen + aspiration + LMR
（S7.4A null-window 版）+ NMP + futility + qsearch SEE + 单缓冲 movegen
（S4.4B）+ single-generation probe（S5.0B）+ single-evasion extension（S7.5A）
+ Eval2 集成位置评价（S8）。

### NNUE 的 UCI 交付（S10-D，默认关）

- UCI options `NnueMode`（off / nnue-v2q / nnue-v2q-full）与 `EvalFile`：
  只替换 evaluator，不改搜索策略与 profile 身份；默认 off，Arena/GUI 行为不变。
- 模型按 `eureka.exe` 所在目录自动查找 `nnue-v2-q01.bin`，绝不按 GUI 工作目录；
  无可加载模型时 fail-closed（`bestmove 0000`），不静默回退 HCE。
- `--nnue-model` 的相对路径**先按 `eureka.exe` 所在目录解析**（不存在再按原样；绝对路径不变；
  缺失仍 fail-closed）。无参数默认启动 profile 不变（不是 S14）。
- 试用指南：[results/s10/s10-nnue-tryout-guide.md](results/s10/s10-nnue-tryout-guide.md)。
- NNUE profiles（`current-final-nnue-v2q-material` 等）仅供实验，**不是生产默认**
  （三轮 Arena 均被拒绝，见下表）。

### 其余不变项

- Hash 默认 16 MB（1..=1024 可配）；单线程搜索；无热切换 profile 的 UCI 路径。
- 构建：`cargo build --release` → `target\release\eureka.exe`。

## 主线时间线（S4 → 当前，全部已关闭项）

| 季节 | 日期 | 内容 | 结论 |
|---|---|---|---|
| S4 | 08-08~11 | Core performance：归因 + movegen 深度现代化 | legality fast path（26604c4）、single-buffer movegen（710400a）等逐项晋级 |
| S5 | 08-12 | single-generation child probe | 晋级 8eb9bd6 |
| S6 | 08-17~20 | 第一代 NNUE 探索（N1-N3、dataset-v1、phase-affine C1） | C1 Arena screen FAIL；NNUE 判定"可学但当时不可产" |
| S7 | 08-12~17 | 深度/排序/选择性归因；S7.4A null-window LMR、S7.5A single-evasion | 两项晋级 current-final |
| S8 | 08-24 | Eval2 集成位置评价晋级 | **formal SPRT ACCEPT_H1 +71.3 Elo**；3+2 blitz 验证 +109.3（64 pairs） |
| S9 | 08-25~26 | Eval2Mask + 6 个 LOO ablation（水平归因） | closed（docs 校准归档） |
| S10 | 08-28~09-06 | NNUE 生产化全链（A→J2，详见下节） | runtime 建成；三轮 Arena 全拒；H0 MULTIFACTORIAL |
| S11 | 09-07~ | R12 关系 sidecar → runtime | A 完成（GO R12）；B1 完成；B2/B3 进行中 |

## S10 NNUE 计划全景（结论地图）

**建成并关闭的基础设施（这些是资产，不是失败）：**

- **A**：V2 representation 冻结（SELECT_V2，HalfKAv2_hm，22528 输入）。
- **B1-B5**：300k→1M 数据集构建、SF18 teacher 标注、三 seed 盲 holdout 选种、
  FP32 Rust 推理 parity、EUNN2Q01 PTQ 量化门（全 PASS）。
- **C1-C3**：增量 accumulator bit-exact（C1）、move-aware 更新（C2A）、搜索栈集成
  full↔incremental exact parity（C2B）、三臂 NPS 基准（C3-B）、AVX2 madd L1 kernel
  （C3-C2，**+50% incremental NPS，已接受**）。UCI NNUE 交付（S10-D）。
- **E1/E2**：嵌套 1M 数据集 `s10-eval-v2-1m01` + Windows 原生并行标注。
- **G1-B**：v3 宽度感知 runtime（FT128/FT256 双宽度门 PASS）。
- **H0 审计链**（见 [results/s10/s10-h0-master-closeout.md](results/s10/s10-h0-master-closeout.md)）：
  teacher budget、representation collision、search-site 分布漂移（真实且稳健，
  R=1.18-1.25）、calibration、决策质量五路单变量实验。
- **J1-0**：统一 H0-E lockbox scorer（后续所有任务对齐评估的基准工具）。

**被拒绝的候选（SPRT ACCEPT_H0，不要重跑）：**

| 实验 | 结果 | 状态 |
|---|---|---|
| E3-D1 1M material-residual | 42.87%（≈-50 Elo）；MAE 138.6 未转化为棋力 | REJECTED |
| F1-D1 material-anchored residual | ACCEPT_H0 | REJECTED |
| G1-D1 FT256 宽度 | ACCEPT_H0 | REJECTED，宽度阶梯 STOPPED |
| I1-A sibling-ranking 目标（含 Repair 1） | FAIL（decisive） | REJECTED |
| J1-A dense 64-64 容量 / J2-A phase heads / J2 Repair 1 | FAIL（decisive） | REJECTED |
| H0-C2 search-site 增强、H0-D3 futility K*=+75 | FAIL | REJECTED |
| S6-C1 phase-affine | Arena screen FAIL | REJECTED（历史） |

**H0 核心结论（冻结措辞）**：剩余 NNUE 棋力缺口无法归因于单一失效点。scalar CP
精度提升**确实**部分转化为更好 sibling 排序（H0-E：regret 224.4→210.1，
pairwise +9.3pp，search 后衰减约一半）；但 teacher 预算、精确 representation
碰撞、朴素 search-site 增强、futility-only 校准各自都不充分。
**STOP/HOLD 清单**（冻结至新因果证据）：teacher 预算扩张、盲目 2M/3M 标量位置、
FT384/512 宽度钓鱼、per-gate 边际调参、同根 search-site 密度增强。

## S11：R12 关系 sidecar（当前主战场）

### S11-A 结论（已完成，GO R12）

- R6（6 通道 A/D/C）：PRIMARY 3/3 大幅超过（pairwise 67.3，+9.0pp；regret
  -67.4cp）但 dxc6 悬子补偿病理 FAIL。
- R14 Repair（victim-type-bound A）→ R1.5 归因 → **R12**（A 通道只对 N/B/R/Q
  victim，无 pawn-A；12 通道 × 64 = 768 行，总输入 23,296）。
- R12 双 seed（20260819 / 20260820）安全门（dxc6、motif、material removal）
  全 PASS；retention：pairwise +10.4~10.9pp、regret -47.7~-60.2cp、search-site
  -26cp、val MAE 127.2/-10.6cp；仅静态 acc@20（43.0/41.8）未达 44.6 bar
  （borderline，两 seed 均 ≥ E3 的 42.6）。R1.6 -A_P 推理审计 7/7 门 PASS。
- **运行时候选冻结：R12 seed=20260819**（两 seed 中 ordinary val MAE 更低者，
  选择依据独立于 H0-E lockbox）。checkpoint：
  `data\s10\s11\r2\seed-20260819\checkpoint_v2r12_s20260819.pt`。
- resolver 修复 3950c02：input-dim→feature-set 改为 exact fail-closed 映射
  （`feature_set_from_input_dim`），审计确认无已报告数字被污染。

### S11-B1 结论（已完成，3f171db）

真实搜索边的关系行 churn（32 冻结 H0-C roots × 50k nodes，1,598,264 边）：
per-root median **4 行/边**，p90≈10，p99≈18，全局 max 50；phase high 8 / mid 7 /
low 4 / zero 0。诊断 feature `diagnostic_relation_churn`（production 零代码）。
结论：churn 温和——fresh-recompute hybrid 第一版可行，增量化留作后续优化依据。

### S11-B2/B3 结论（已完成，`19c5237`..`7025c56`，详见
### docs/dev-log/2026-09-07-s11-b2b3-hybrid-runtime.md）

- **Runtime 建成且正确**：v4 artifact 格式（feature_set 字段，version-aware
  108/112 header，v1-v3 逐字节兼容）；无分配 `for_each_relation_feature_v2r12`
  单一语义核心；hybrid = 增量 V2-base 栈 + eval 时 fresh relation 行（栈绝不存
  relation 行）；profile `current-final-nnue-v2q-material-r12` 全线接线 +
  双向 fail-closed。
- **四层 parity 全 PASS**（A: FP32↔quant mean 0.13cp；B: PyInt↔Rust 10k
  exact；C: hybrid↔full 10k corpus + 19,692 transitions + 10 fixtures lanes+raw
  exact；D: scalar↔AVX2 10k exact；旧 4 artifact 逐字节回归 PASS）。
- **NPS = 0.58 median < 0.80 暂停门 → B5 PAUSED**。
- **成本归因（关键认知）**：B1 churn（4-8 行/边）是每步 delta；fresh 每次
  eval 重加全部 ~24 活跃行（median），主导成本是 FT 行加法的内存流量（~12µs，
  24 随机 256B 行 × 2 perspective 横跨 1.9MB），attack 查询次之（~7µs）。
- **决策点（待审批）**：incremental relation updater 立项与否——B1 数据表明
  它同时消掉 scan 与批量重加，预期 eval 税 ~1.5-2×；工程量为新子项目
  （slider ray 开闭、victim 联动、king move perspective 变换）。

### S11-B4（已完成）

- **B4-A（`ebd95ca`，详见 docs/dev-log/2026-09-07-s11-b4a-relation-delta.md）**：
  `Position::attack_map`（u64×2，逐位 == is_square_attacked）、`R12RelationState`
  （[u8;64]）、diff FT 应用。Parity 全 bitwise；成本门 median 448 ns/edge（合法
  playout corpus microcost——不是 B1 真实搜索边分布的无偏估计）。
- **B4-B（`38aaeee`，详见 docs/dev-log/2026-09-07-s11-b4b-incremental-stack.md）**：
  按审批冻结细节：`frames: Vec<AccumulatorFor>` 原封不动 + `Option<Vec<R12Relation
  State>>` 并行栈（非 inc 路径零额外拷贝）；API 拆分保证 child state 每边只
  recompute 一次；fresh oracle profile 保留，新增 `current-final-nnue-v2q-material-
  r12-inc`（同 binary 三路径）。Parity：inc-stack 200 转移 + 29 null push == full
  refresh；24-FEN × 50k 树一致性 17 字段 0 mismatch。**配对 NPS median 0.9488 ≥
  0.90 → 门 PASS，B5 解锁**。分类：opening/middlegame 0.94-1.05；tactical ~0.90；
  endgame 0.80-0.91（稀疏位置 E3 eval 极快，~450ns/边固定成本占比大）。
- **B5（`22c94e9`，详见 docs/dev-log/2026-09-08-s11-b5-search-validation.md）**：
  256 roots × 100k 双臂同 binary 重跑。结果双框架:历史门(mean≤64.6/acc20≥66.2/
  p90≤100)= 41.4/66.8/97 **全 PASS**;但同 harness 配对 = R12 mean **+2.0cp 差于**
  同跑 E3(39.4),仅 acc20/acc50/top1 小幅占优。anti-drift FLAGGED:本轮 E3 mean
  39.4 vs 历史 69.6 是纯尾部质量(bulk:median 0=0、p90 93=93、acc20 -1.2pp 完全
  吻合;H0-E 原搜索脚本未入库不可审计)。zero-phase 弱尾(58.2→65.2)与 S11-A
  static 一致。**绑定框架待审批方裁定;未做任何 Arena 准备(协议 STOP)。**
- **SF2400 live match(进行中)**：tournament `6a07cc07-82a4-416d-842b-eaca40a4a9a6`,
  bullet 1+0 计入 Elo,Eureka R12-inc vs SF18 Elo-2400 锚点,500 pairs(1000 局)。
  部署记录 results/s11/r12-vs-sf2400/README.md。

### S11 已知陷阱（本轮实测）

- `types::MoveFlag` 没有 `is_capture()`；手写判定。
- 不要用脚本整段替换重写 `src\engine\nnue.rs`（B1 中曾损毁 815 行，已从 HEAD
  恢复）；用精确锚点的 Edit。
- Windows 下 `rg` 的路径参数不要带 `\*.rs` glob（目录语法错误）。
- bench fixture FEN 必须先验证合法（本轮 3 个手写 FEN 王被牵制/非法）。
- **性能直觉陷阱**：per-move delta（churn）≠ per-eval 全量成本；测量分段
  （scan / scan+FT / total）才能正确归因。
- **Bench 循环纪律**：任何"条件触发重启"的 corpus 构建循环，条件必须随迭代
  推进（`len()%N==k` 在不 push 的分支里永真 → 死循环）；ratchet 重放必须逐
  决策镜像同一 RNG 序列。
- **增量语义纪律**：任何"某事件只影响 X 视角"的假设都要过双视角检查——king
  move 同时改变两色的 attack map（B4-A 的唯一真 bug）。

## 文档对齐地图（证据在哪里）

- **顶层导航**：本文件（handoff.md）——每完成一个 season 必须更新。
- **S10 及以后**：per-experiment closeout 在 `results/s10/s10-*-closeout-report.md`
  （-provenance.md 为冻结协议）；数据/判定 JSON 同目录。
- **S6/S7**：`docs/s6/`、`docs/s7/` 有 season README + `results/s6|s7/` 报告。
- **S8/S9**：证据在 `results/artifacts/2026*/` + commit message（**没有**
  season README——这是文档债，S10 起改用 results/ closeout 模式）。
- **S4 之前**：`docs/benchmarks/*.md`、`docs/specs/*.md`（冻结历史）。
- **Arena / SPRT**：独立仓库 `zxjAiLun/super-eureka-arena`；Engine 只发布
  immutable EngineArtifact。

## 当前已验证门禁（2026-09-07 本机）

```text
cargo test --release --lib      PASS（432 tests）
cargo test --release --lib nnue PASS（76 tests）
cargo build --release           PASS
production binary 无 diagnostic 代码（feature-gated）
```

## 常用操作

```powershell
cargo build --release
cargo test --release --lib
cargo test --release --lib nnue

# NNUE 诊断构建（churn recorder 等）
cargo build --release --bin eureka --features diagnostic_relation_churn

# 关系 churn 审计（示例）
target\release\eureka.exe bench relation-churn --fen <FEN> --nodes 50000 `
  --profile current-final-nnue-v2q-material `
  --nnue-model data\s10\e3\scale-1m-win\seed-20260820\nnue-v2-q01-material-v3twin.bin

# Python 工具测试
python -m unittest discover -s tools -p "test_*.py"
```

## 交接边界与下一步

### 当前不应做

- 不跑 B5（256×100k search validation）——等 B2/B3 parity+基准全 PASS 后由
  审批方放行；
- 不写 incremental relation updater（协议明确推迟，即使 B1 churn 数据诱人）；
- 不改 R12 语义 / 量化常数（FT_SHIFT=12 等全冻结）；
- 不碰 v1-v3 artifact 兼容性（必须逐字节不变）；
- 不重跑已拒绝实验（E3-D1/F1-D1/G1-D1/I1/J 系列及 STOP/HOLD 清单）；
- 不把节点/NPS/depth 直接表述为棋力；NPS 门只决定是否进入 search validation。

### 推荐下一步（按序）

1. **B5 框架裁定（审批方）**：历史门 PASS vs 同 harness 配对 parity-ish——若以
   配对为准，R12 与 E3 在完整搜索中平手，Arena SPRT 的期望优势有限；若以历史门
   为准，进入 Arena preparation（provenance/opening exclusion/build manifest 由
   审批方冻结后执行）；
2. **SF2400 match 观察**：1+0 计入 Elo 的 live 对局（无需脚本，网站上直接看）；
   结果与 B5 互为印证——R12 实战强度首读数；
3. 无论方向：handoff + dev-log 按制度更新。

## 关键文件导航

- R12 特征语义：`src\engine\nnue.rs`（`relation_features_v2r12` 行 438，
  `NNUE_INPUTS_V2R12=23296` 行 68，`active_features_for` 行 304）
- 量化 runtime / loader：`src\engine\nnue_v2q_runtime.rs`（v3 header 行 399、
  `MAX_FEATURES_PER_PERSPECTIVE` 行 235、kernels 行 879-1120）
- 搜索栈 NNUE 状态：`src\engine\nnue_search.rs`（`NnueSearchState`、churn 模块）
- 搜索 profile 开关：`src\engine\search.rs`（profile enum 行 136、eval 分派
  行 2670）
- bench 命令入口：`src\engine\bench.rs`（`relation-churn` 行 4051、
  `nnue-v2q-cost` 行 4450、`accumulator-audit` 行 4264）
- 训练 / 导出 / parity 工具：`tools\s10\train_nnue.py`、`export_quantized.py`、
  `f1_quant_parity.py`、`search_nps_c3c.py`
- 数据资产：`data\s10\s11\r2\seed-20260819\`（R12 checkpoint）、
  `data\s10\e3\scale-1m-win\seed-20260820\`（E3 基准 artifact）、
  H0-C roots 缓存 `C:\Users\81489\AppData\Local\Temp\opencode\h0c-cache\`
- H0 总结论：[results/s10/s10-h0-master-closeout.md](results/s10/s10-h0-master-closeout.md)
- NNUE 试用指南：[results/s10/s10-nnue-tryout-guide.md](results/s10/s10-nnue-tryout-guide.md)（S10 时代，历史参考）
- S14 本地 GUI 指南（En Croissant）：[results/s14/s14-en-croissant-guide.md](results/s14/s14-en-croissant-guide.md)
- 每轮开发文档：[docs/dev-log/](docs/dev-log/)（2026-09-07 起）

## 更新日志（append-only）

- **2026-09-10 · `--nnue-model` 相对路径按 exe 目录解析（GUI UX 小修复）· 本提交**
  绝对路径不变；相对路径先按 `eureka.exe` 所在目录解析，不存在再按原样，缺失仍 fail-closed
  （不回退旧模型）；无参数默认启动不变。新增 3 个单元测试（release lib 444 tests PASS）；
  clippy 无新增；WSL UCI smoke 通过（cwd 与 exe 目录不同时短参数可用、绝对路径不变、
  缺失报 startup_error）。En Croissant 配置缩成
  `--profile current-final-s12 --nnue-model nnue-s14-datasupply-v5.bin`；staging 脚本与
  S14 GUI 指南同步更新。开发代码：src/uci.rs。
- **2026-09-10 · 文档 repair（S14=CP-only 勘误；默认启动澄清；channel/profile 前缀规则）· 本提交**
  修正 S14 描述（此前误写"真实结果双信号"；实为 S12-R0 配方逐字 CP-only + 5.0M 数据池）；
  明确"默认产物 ≠ promotion 引擎"——binary 能力对齐，但无参数默认启动仍是 HCE；补记云端从未
  构建 `b3145e1`（仅 recipe 等价推断）；`current-final` 命名改为强制 `Arena channel:` /
  `engine profile:` 前缀。
- **2026-09-10 · S14 本地 GUI 工作流（staging 脚本 + En Croissant 指南）· 本提交**
  新增 `tools\stage_s14_gui.ps1`：SHA 校验冻结 S14 模型后 stage 进 `target\release\`，
  并生成 `EN-CROISSANT-S14.txt`（engine 路径 + args 行）；`-Build` 可选先跑
  `cargo build --release`（需先关闭 En Croissant，否则 exe 被锁）。新增
  `results/s14/s14-en-croissant-guide.md`；旧 S10 试用指南加过期提示。
  推荐配置 = `--profile current-final-s12 --nnue-model nnue-s14-datasupply-v5.bin`（相对名按 exe 目录解析）；该模式下
  EvalFile/NnueMode 被引擎忽略（无需手动配置）。本机 exe 已实测通过。
- **2026-09-10 · 命名规范对齐 + 构建身份复核 · `ca6ad3c` + 本提交**
  新增"术语与版本命名"一节（HCE-20260825 / S11-R12 / S12-R0 / S14 四名制；`current-final`
  仅作 Arena channel 名），closeout 状态表旧写法同步修正。构建复核：本地 WSL 重建 promotion
  二进制与 `dceacfb7` 仅差 6 字节（RUNPATH 随机段），其余逐字节一致；云端 workflow 从未构建
  `b3145e1`。
- **2026-09-10 · S14 promotion SPRT closeout — ACCEPT_H1 / PROMOTION-QUALIFIED，production HOLD · `b3145e1` + closeout 提交**
  S14（S12-R0 配方逐字、CP-only、数据池 5.0M）屏幕 58.01%；唯一正式 promotion SPRT `6cd87ee8…` 终判
  **ACCEPT_H1**（LLR 2.9696 ≥ 2.9444，177/500 pairs，354 局，W/D/L 198/62/94，
  ptnml [15,15,54,37,56]，同 binary `dceacfb7…`，非计分）。用户裁定：
  **S14 = PROMOTION-QUALIFIED；生产切换 HOLD**——部署 V2.1 控制面无可验证的受控
  rollback（旧 HCE 变 historical 后受控面拒回；duplicate-fingerprint 阻止重建），
  属生产级阻断，**非棋力问题、无需补充验证**；本轮无任何生产变更 / 新 artifact /
  规则修改 / 新比赛。checkpoint §10.2a recipe 与 rollback 措辞勘误见 closeout。
  后续：独立运维任务（建受控 rollback）→ 复核清单通过后即可直接 promote。
  开发文档：docs/dev-log/2026-09-10-s14-promotion-closeout.md（+ checkpoint）
- **2026-09-08 · S13-A real-result blend FAIL · `a0b2f4b`/`28ecf77`**
  Preflight：1M corpus 每 position 自带真实 game result（游戏-不相交）；
  [%eval] 仅 13.6% → 走 S13-A（0.75 result / 0.25 sigmoid(cp/400) blend，
  其余全冻结）。训练仍 epoch-2 即恶化（纯 CP MAE 166.5）。256 局全新
  opening screen：**W/D/L 21/24/211 = 12.89%（-331.9 ± 32.4）→ FAIL/STOP**。
  开发文档：docs/dev-log/2026-09-08-s13-result-blend.md
- **2026-09-08 · S12-R1 batch-cadence repair FAIL → S12 CLOSE / PARITY · `5b4ba04`/`55718d8`**
  单变量 batch 1024→16384（真 16×，Bullet cadence）。best epoch 4、val 0.009657
  （门 0.00910、R0 0.009204）——曲线略平但形态不变，cadence 假设否决。未导出
  未比赛；R0 128 局早期屏幕（W/D/L 55/17/56 = 49.61%）为最终对抗结果。残余差距
  记录：数据规模/刷新（~100M 数据流 vs 固定 780k）与 game-result 混合——若重开
  训练线，数据供给是首要杠杆。开发文档：docs/dev-log/2026-09-08-s12-r1-repair-close.md
- **2026-09-08 · S12 Bullet-recipe 全栈 + 生产筛选赛 · `dfe60df`/`876e4fe`**
  SCReLU+8桶 FT256+sigmoid(400) 得分损失;v5 artifact(116B header+head_kind);
  Rust loader/kernel/head-aware 化;current-final-s12 profile。Parity 全 PASS
  （PyInt↔Rust bit-exact、FP32↔quant 0.43cp、full↔inc 0 mismatch、树一致
  24/24、NPS 178.8k≈E3）。训练 204s（best epoch 2，val MAE 132.8）。筛选赛
  vs 生产 current-final：**64-64（49.6%）完全平手**——"值得续测"，不升 SPRT。
  决策点：追加训练预算 / 接受平手 / 256 局缩 CI。途中修复 B2 时代非法 castle
  fixture（34 子）。开发文档：docs/dev-log/2026-09-08-s12-bullet-recipe.md
- **2026-09-08 · S11-B5 search validation + SF2400 部署 · `475b0c9`/`22c94e9`**
  R12-inc 部署至 Arena 服务器(build `20260908-562e77c-s11b4b-r12inc-8eacd0c1`,
  manifest 补 model_artifacts 后重注册);计入 Elo 的 1+0 match
  `6a07cc07…` 启动(500 pairs vs SF2400 锚点)。B5:256×100k 双臂重跑,
  历史门全 PASS 但同 harness 配对 parity-ish(mean +2.0cp vs 同跑 E3),
  anti-drift FLAGGED(纯尾部质量,H0-E 原 harness 未入库),绑定框架待裁定。
  途中修复:UCI 启动 feature-set 门拒绝 -inc profile 的 V2R12 artifact(562e77c)。
  开发文档:docs/dev-log/2026-09-08-s11-b5-search-validation.md
- **2026-09-07 · S11-B4-B incremental R12 search stack · `38aaeee`**
  按冻结细节集成（frames 原样 + Option relation 栈；child state 单次
  recompute API；fresh oracle 保留 + `-inc` 新 profile）。Parity 全 PASS
  （含 24-FEN × 50k 树一致性 17 字段 0 mismatch、inc-stack 200 转移 +
  29 null push）。配对 NPS **median 0.9488 ≥ 0.90** → 门 PASS，**B5 解锁**。
  分类：opening/mid 0.94-1.05、tactical ~0.90、endgame 0.80-0.91。
  开发文档：docs/dev-log/2026-09-07-s11-b4b-incremental-stack.md
- **2026-09-07 · S11-B4-A relation delta · `ebd95ca`**
  审批 GO 后实现 attack-map + [u8;64] relation state + diff FT 应用
  （king-move 稀有路径）。Parity 全 bitwise（1.28M attack bits / 10k
  state-rows / 10k 边 ratchet 0 mismatch）。成本门 median 448 ns/edge
  （门 1500、理想 1000）。两个真 bug：king-move 漏另一视角 diff；
  corpus 重 seed 死循环。投影 NPS ~0.85-0.92，B4-B GO。
  开发文档：docs/dev-log/2026-09-07-s11-b4a-relation-delta.md
- **2026-09-07 · S11-B2/B3 R12 hybrid reference runtime · `19c5237`..`7025c56`**
  Phase 0: v4 export（feature_set 字段，R12 bound 61，E3 v3twin 重导
  byte-identical；发现并补提交 B1 漏掉的 search.rs churn hooks）。
  Phase 1: `for_each_relation_feature_v2r12` 无分配核心、version-aware
  loader（v1-v3 逐字节兼容）、hybrid kernels、profile 全接线（双向
  fail-closed）；关键点：栈只存 V2-base accumulator。Phase 2: 四层 parity
  全 PASS（A 0.13cp / B 10k exact / C 10k+19,692+10 fixtures lanes+raw
  exact / D AVX2↔scalar 10k exact / 旧 artifact 回归）。Phase 3: 配对
  NPS **0.58 median**（< 0.80 门）→ **B5 PAUSED**；归因：fresh 每次
  eval 重加全部 ~24 活跃行，FT 行加法内存流量主导。决策点：incremental
  relation updater 立项与否。
  开发文档：docs/dev-log/2026-09-07-s11-b2b3-hybrid-runtime.md
- **2026-09-07 · 文档制度落地 · `d1baa59`**
  handoff.md 纳入版本控制（原 `.git/info/exclude` 不跟踪是漂移根因）；
  建立 append-only 更新日志 + docs/dev-log/ 每轮开发文档制度；补 B1 与
  preflight 两份 dev-log。
- **2026-09-07 · S11-B1 relation-churn audit · `3f171db`**
  新增 cargo feature `diagnostic_relation_churn`（bench-only recorder，4 个
  push_child 站点，production 零代码）+ `bench relation-churn` 命令。32 冻结
  H0-C roots × 50k nodes = 1,598,264 边：median churn 4 行/边、p99≈18、max 50。
  途中 nnue.rs 曾被批量替换损毁（丢 815 行），`git checkout HEAD --` 恢复后改为
  最小 diff 重做。结论：fresh-recompute hybrid 第一版值得实现，增量化留作后路。
  开发文档：docs/dev-log/2026-09-07-s11-b1-relation-churn.md
- **2026-09-07 · handoff 重建 + 文档制度**
  handoff.md 从 2026-08-08（S3/S4 期）重建至 S11-B1 现状（落后 30 天 / 80+
  commits / 7 个里程碑）；发现其被 `.git/info/exclude` 不跟踪是漂移根因，移出
  exclude 纳入版本控制；建立 dev-log 制度与本更新日志。

