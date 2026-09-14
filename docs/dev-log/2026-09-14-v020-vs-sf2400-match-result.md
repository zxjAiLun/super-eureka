# 2026-09-14 · Eureka v0.2.0 首次生产定级 — vs Stockfish Limited 2400（bullet 1+0，500 对）

产品发布（`2026-09-12-eureka-v020-release.md`）后的第一次外部标尺对战。
本场把 v0.2.0 作为**下一轮模型升级的基线**钉死，并给出当前条件的强度位置。

## 0. 一句话

Eureka v0.2.0 在当前固定条件（1+0、SF Limited 2400 锚、8-moves 配对开局）下
**小幅领先**对手，得分率 **55.55%**（相对该对手约 **+38.7 Elo**）。
可以进入下一轮模型升级；**没有理由推翻架构**。

## 1. 参赛身份（终态，逐项核对）

```text
tournament   435e4630-7722-4ba4-a036-5b0eb2afd5c3
name         eureka-v0.2.0-vs-stockfish-2400-bullet-500pairs

engine A     ce-v020 / display_name "Eureka v0.2.0"
  source       315534991628ce084c34392d6abb98d71f3f591a
  binary       c2428a8453d871c9a6f2ff14780f077b2c2503bdb321419d4334361c54733986
  model        329b717066082798d2f42d9e4f137385a1482a8cda03ae7849ddc654e097e4b0
  build_id     20260912-3155349-eureka-v020-329b7170
  launch       --evaluation nnue
               --nnue-model /opt/chessarena/builds/<build_id>/models/nnue-s14-datasupply-v5.bin
  uci_options  {}
  fingerprint  f7353e8a19ef7b0c270907915faa4e9e8fd5934fc12a8038a84a9eb12a18618a

engine B     preset:stockfish-limited-2400 / "Stockfish Limited 2400"
  build        stockfish-18-avx2-linux-x86_64
  options      UCI_LimitStrength=true, UCI_Elo=2400   (固定锚，永不更新)
```

服务器上对 binary 与 model 现场 `sha256sum` 重算，与发布记录逐字节一致；
channel `current-final → ce-v020`，即参赛的就是生产身份本身。

**本次没有旧 q01 模型混入**（对照 `c9e4ed6` 记录的本机 GUI 事件）：
启动参数显式携带 `--nnue-model`，指向 build 内声明的 S14 artifact。

## 2. 协议（本场冻结）

| 项 | 值 |
|---|---|
| 时间控制 | `bullet_1_0` = **1+0**（cutechess `tc=60`） |
| 配对开局 | `stockfish-8moves-v3`（官方 34,700 局面） |
| 规模 | 500 对 = **1000 局**（每对严格交换黑白） |
| 每局配置 | Hash 16 / threads 1 / concurrency 1 |
| Elo | `arena_elo_enabled = true`，按 EngineVersion 记分 |

## 3. 结果

```text
status        COMPLETED
schedule      500 / 500 pairs  →  1000 games
Eureka W/D/L  476 / 159 / 365
score         55.55%
LOS           99.98%
score 95% CI  52.47% – 58.63%
duration      31h37m49s   (平均 227.7 s/对)
integrity     1000 / 1000 verified · 0 retry · 0 failure (rc≠0)
```

由得分率直接推出的相对强度：

```text
+38.7  →  2400 + 38.7 = 2438.7
```

Arena 动态积分（bullet_1_0 池）：

```text
display_name  Eureka v0.2.0
rating        2444        games = 1000        status = rated
anchor        Stockfish Limited 2400 (fixed)
```

### 3.1 终止原因（全 1000 局可解释，无超时）

```text
841  mate                (Black mates 422 / White mates 419)
 67  Draw by 3-fold repetition
 66  Draw by fifty moves rule
 22  Draw by insufficient mating material
  4  Draw by stalemate
  0  forfeits on time / loses on time / resigns
```

**零超时判负**，零判负性异常；对局全部由规则或杀棋自然终局。

## 4. 读数纪律（引用本结果时必须一并给出）

1. **主口径是得分率与对手内 Elo 差**：`55.55%` / `+38.7 Elo`。
   `2444` 是 Arena 的动态积分，与 `2438.7` 的 5 分差**不精确归因**
   （逐局 K=16 迭代，受更新顺序、初始值、对手侧是否更新共同影响）；
   两者是**不同定义**的量，不是同一量的两次测量。
2. **对手不是"2400 分的完整引擎"**。`UCI_LimitStrength=true` +
   `UCI_Elo=2400` 是**主动选择较差招法**来限制实力，不能等同于同分数的
   完整引擎，也不能等同于人类的 2400 等级分。所以本场是**项目内部标尺**，
   不能对外当作绝对棋力声明。
   官方说明：<https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html#how-do-skill-level-and-uci_elo-work>
3. **不同时间控制分别记录**，即使共用同一锚点也不做横向换算。

## 5. 与前序证据的关系

- 本场是**外部标尺**，定位 v0.2.0 在当前条件的位置。
- `+105 Elo`（S14 vs 同 binary HCE，198 胜 / 62 和 / 94 负 = 64.69%，
  bullet_1_0，双臂同 binary `dceacfb7…`）是**模型贡献**，code-to-code。
- 两者不冲突，也不能互相推导。

## 6. 对后续路线的意义（裁定）

- **继续当前 NNUE 主线**，先争取跨过 2600 档；暂时没有理由推翻架构，
  也不需要再补一场 SF2400。
- 固定版本 / 硬件 / 时限后，把 **SF2400 → 2600 → 2800 当作项目内部阶梯**；
  2600–2800 是目标区间，目前还不能判断差多少。
- **下一轮最重要的对手是 S14 自己**：新模型能否升级，先看它能否击败现任版本，
  而不是重复证明"仍然约为 2400 档"。

## 7. 成本教训（本场最大的工程信息）

本场耗时 **31 小时 38 分**——长对战是当前最昂贵的环节。因此：

```text
短对战筛候选 (256 局)  →  只有明显领先才升 1+0 验证  →  才挑战更高一级 SF
```

不再为每个训练候选跑 1000 局外部定级。

## 8. 下一轮（已收敛，待执行）

| 项目 | 安排 |
|---|---|
| 基线 | 当前发布 S14 / Eureka v0.2.0 |
| 数据 | 从已下载 Fishtest 扩到约 **2000 万独立棋局位置**，复用现成评分，不重标 |
| 模型 | 保持 **FT256 / SCReLU / 现有特征 / CP-only** 配方不变 |
| 数据读取 | 一次性生成**紧凑二进制分片缓存**，避免每次重复解析 gzip/JSON |
| 训练 | 一个 seed、一个候选；**6000 万样本呈现或 1 GPU 小时先到停**，记录实际完成量 |
| 初筛 | vs S14，**256 局，10+0.1，配对开局** |
| 升级 | 明显领先才进 1+0 验证，再挑战 SF2600 |

本轮不顺带加入 WDL / result blend / 残局特判 / 新搜索参数；数据版打完再决定
是否投入**因子化（训练时特征因子化，导出时合并回 FT 表）**与更大容量（FT512）。

## 9. 搜索优化入口

本场 **0 超时判负、0 失败对**，未暴露"频繁超时"这一类明确问题；因此**不启动**
"为什么评估收益没有传进搜索"的研究项目。搜索侧只在实战暴露明确问题时再进入。

## 10. 遗留 / 未做

- 未重训、未跑 SPRT、未改引擎/搜索/评估代码；本场是纯生产定级。
- 未做旧资格 24 FEN exact bridge；未迁移 `current-final` 的 profile 语义。
- 更长时限（blitz / rapid）与 SF2600+ 的挑战**尚未进行**，须等下一轮候选就绪。
