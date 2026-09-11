# 2026-09-11 · S14 使用端验收与发布链路诊断

引擎修复线（evaluator unification）已于 `106219a` 判定 **PASS / CLOSED**。
本文档只处理**使用端验收与发布**，不重新证明棋力、不重训、不切通道。

## 1. 本机验收（已完成）

### 1.1 目标 exe 已更新到最新代码

原先 En Croissant 指向的 `target/release/eureka.exe` 构建自 `439a564`（落后一个提交）。
已 `cargo build --release` 重建并核对：

```text
id name Eureka v0.1.0-dev+106219a0.dirty
info string source 106219a0cbfb667a9f68817a67bbe9ff715aa13a
```

### 1.2 按 GUI 实际保存的配置做完整交互验收

验收脚本 `tools/gui_acceptance_local.py` 直接采用 En Croissant
`engines.json` 中保存的**真实配置**（`Hash=64`、
`EvalFile=…\nnue-s14-datasupply-v5.bin`、`Evaluation=nnue`；
`NnueMode=nnue-v2q` 为被忽略的历史残留），走完四步：

| 步骤 | 内容 | 结果 |
|---|---|---|
| LOAD | 模型加载 + `Evaluation=nnue` 生效 | PASS |
| MOVE | 送后局面 depth 8 | `f7e6` / cp **695** PASS |
| STOP | 长搜中断返回合法着法 | `e2e4` PASS |
| NEWGAME | `ucinewgame` 后配置保留且搜索仍正确 | `f7e6` PASS |

补充控制项：NEWGAME 后先搜 startpos 得 `e2e4`（与 `f7e6` 不同），
排除"棋盘未重置导致结果陈旧"的假阳性。

模型 SHA：`329b717066082798d2f42d9e4f137385a1482a8cda03ae7849ddc654e097e4b0`
（11,936,916 B），与冻结的 S14 制品一致。

### 1.3 反向对照（证明验收有判别力）

同一局面、depth 6：

| 配置 | bestmove | 分数 | 判定 |
|---|---|---|---|
| 无任何选项（默认 HCE） | `e7e6` | cp 757 | 送后（原 bug） |
| 完整 GUI 配置（NNUE） | `f7e6` | cp 696 | 吃后（正确） |

两条路径结果不同，说明该验收确实能区分评估器，不是恒过测试。

回归：`cargo test --release --lib` **398/398 PASS**。

## 2. 发现：bench 的 NNUE 路径被 HCE 锁定值挡住（既有问题，非本轮引入）

```text
$ eureka bench smoke --evaluation nnue --nnue-model nnue-s14-datasupply-v5.bin
bench_error fixture startpos: locked score 23 != 98
```

原因：`smoke` 套件的 startpos / queen-win 两个 fixture 带有
`locked`（nodes/score/bestmove/PV）精确锁定值，这些值属于 **HCE**。
锁定条件为 `mode == Disabled && profile == CurrentFinal`，
而 NNUE 路径下 profile 同样是 `CurrentFinal`，于是拿 NNUE 的分数去比对 HCE 的锁定值。

该问题在 `618297f` 统一入口之前就存在（`eeac999` 时 `current-final-s12`
映射为 `CurrentFinal` 且同样命中锁定），只是当时 bench 的 NNUE 入口
写法不同、未被触发到。**不是本轮三个提交引入的回归。**

影响范围：仅 bench 的 `smoke`/`standard` 套件在 NNUE 下直接报错；
不影响 CLI / GUI / UCI / Play 任何实际使用路径（已由上面的本机验收覆盖）。

建议（未执行，等 GO）：让 locked 校验按 `evaluation` 分流——
NNUE 下跳过锁定值，或在锁定结构里区分 `classical` / `nnue` 两套期望值。

## 3. 云端发布链路诊断（未做任何部署或切换）

现状核对一致：

```text
production channel current-final
  → ce-currentfinal-20260825
  → 20260825-96d1a69-linux-x86_64
  → HCE
```

Play 公开接口返回的 Eureka 对手仍是 2026-08-25。**显示名未更新不是显示问题，
而是通道确实仍在旧 HCE 上**，与此前保留的 Promotion HOLD 一致。

云端已存在 S14 实验构建 `b3145e1`（`results/s14/promotion-package/`，
manifest 记录 binary+model SHA），但早于入口修复，且未接入 Play。

两个既有阻塞仍在，均与引擎代码无关，属服务端发布规则：

1. 生产版本要求**启动参数与 UCI 配置均为空**，因此无法直接接纳
   显式 NNUE 配置（`--evaluation nnue` / `EvalFile`+`Evaluation`）。
2. 旧版本一旦变为 historical，当前 promotion 路径**拒绝切回**，
   即没有已验证的受控回滚路径。

因此**单纯上传新 exe 无法完成云端上线**，必须先解决发布配置与回滚规则。

## 4. 建议（与用户一致，待批准）

- 引擎侧**继续保持无参数默认 HCE**，不为了部署限制再开特殊入口。
- 由 Arena 侧管理固定的 NNUE 启动配置与模型身份（允许启动参数/UCI 配置）。
- 先建立并验证受控回滚路径，再安装固定身份的新构建并接入 Play。
- 接入后验证网页落子确实由该版本产生（验证发布链路，不是重做棋力实验）。

## 5. 本轮未做 / 不做

- 未部署、未切换任何生产通道。
- 未重训、未跑新的 SPRT 或棋力赛。
- 未修改服务端规则。
- 引擎修复线不再追加 Repair（`106219a` 已 CLOSED）。
