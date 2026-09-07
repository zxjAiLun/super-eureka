# ChessEngineDemo Handoff

> 状态快照：2026-09-07
> 仓库：`E:\AUbuntuProject\project\chessenginedemo`
> 工作分支：`s10/nnue-production-foundation`（领先 `main` 80 commits，已推送）
> 分支 HEAD：`3f171db`（S11-B1 relation-churn audit）
> `main` HEAD：`3dae2fa`（S9-B2 closeout，2026-08-26）
> 工作树：CLEAN
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
NNUE 生产化在 runtime / 量化 / 增量栈上全部建成，但三轮 Arena（E3-D1 数据规模、
F1-D1 material-residual、G1-D1 FT256 宽度）全部 SPRT ACCEPT_H0 拒绝，H0 根因审计
裁定 MULTIFACTORIAL。S11-A 关系 sidecar（R12）成为整个 NNUE 计划中离线任务对齐
收益最大的突破（lockbox pairwise +10.4~10.9pp、dxc6 安全门双 seed 通过）。当前
正在进行 **S11-B2/B3：R12 hybrid reference runtime**，B1 churn 审计已完成，下一步
是 parity 门 + 配对基准。

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

### S11-B2/B3 冻结协议（下一步，执行细节见会话 checkpoint）

- 结构：`effective_acc = incremental_V2_base_acc + fresh_R12_relation_sum`，
  双 perspective，每次 eval 现算 relation rows；**不缓存、不做 relation 增量**。
- 单一语义：exporter / full-refresh parity / runtime 共享同一 relation 实现
  （抽无分配 `for_each_relation_feature_v2r12`）；R12 pseudo-attack 近似**逐字
  保留**，不顺手"修正"。
- Artifact 契约：格式 bump **v4**，显式 `feature_set` 字段 + fail-closed
  （V2R12↔23296 精确互检；v1-v3 artifact 行为逐字节不变）。
- 已知实现要点：`MAX_FEATURES_PER_PERSPECTIVE=31`（nnue_v2q_runtime.rs:235）
  必须 raised（R12 上界 31+30）；`relation_features_v2r12` 已 pub
  （nnue.rs:438）；bench 框架复用 `nnue-v2q-cost` / `accumulator-audit` /
  `search_nps_c3c.py` 协议；parity 工具 `f1_quant_parity.py` 需参数化
  feature-set。
- Parity 硬门（benchmark 前）：A FP32↔quant；B Python quant↔Rust full-refresh；
  C Rust full↔hybrid **10k 精确 + 转移序列（capture/promotion/castling/EP/
  king-move/slider-unblock）**；D scalar↔AVX2。
- 量化门：mean ≤0.30cp / p99 ≤1.0 / max ≤2.0 / val MAE 退化 ≤0.05cp。
- 基准：同 binary 配对 B(E3 V2 FT128) vs A(R12 hybrid)：eval ns / edge+eval ns /
  NPS + 单独 relation-scan ns。runtime 门不变：NPS ≥90% → B5；80-90% →
  一次性 performance-only repair；<80% → 暂停评估增量 updater。
- **B2/B3 交付后 STOP 汇报，不顺手跑 B5。**

### S11 已知陷阱（本轮实测）

- `types::MoveFlag` 没有 `is_capture()`；手写判定。
- 不要用脚本整段替换重写 `src\engine\nnue.rs`（B1 中曾损毁 815 行，已从 HEAD
  恢复）；用精确锚点的 Edit。
- Windows 下 `rg` 的路径参数不要带 `\*.rs` glob（目录语法错误）。

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

1. **S11-B2/B3 执行**（Phase 0 Python 导出 v4 artifact → Phase 1 Rust runtime →
   Phase 2 四层 parity → Phase 3 配对基准），完成后 STOP 汇报；
2. 若 NPS ≥ 0.90：申请 B5 冻结 search validation（256 roots × 100k nodes，
   lockbox scorer 已就绪）；
3. B2/B3 无论结果，更新本 handoff 并补 results/s10/s10-s11-* closeout。

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
- NNUE 试用指南：[results/s10/s10-nnue-tryout-guide.md](results/s10/s10-nnue-tryout-guide.md)
- 每轮开发文档：[docs/dev-log/](docs/dev-log/)（2026-09-07 起）

## 更新日志（append-only）

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

