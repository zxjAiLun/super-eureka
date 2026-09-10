# S14 本地 GUI 指南 — En Croissant（2026-09-10）

目标配置：**S14 NNUE + CurrentFinal 搜索**（当前最强 Eureka；与正式 promotion SPRT 的
candidate 完全一致，`ACCEPT_H1`）。

## 一键流程

```powershell
# 1. 构建（关闭 En Croissant 后再 build —— Windows 会锁定正在使用的 eureka.exe）
cargo build --release

# 2. 把冻结的 S14 模型 stage 进 target\release\ 并生成配置说明
powershell -ExecutionPolicy Bypass -File tools\stage_s14_gui.ps1
```

两步可合并（脚本带 `-Build`）：

```powershell
powershell -ExecutionPolicy Bypass -File tools\stage_s14_gui.ps1 -Build
```

产物（都在 `target\release\`）：

- `nnue-s14-datasupply-v5.bin` —— 冻结 S14 模型（脚本做 SHA-256 校验，不符即拒绝）
- `EN-CROISSANT-S14.txt` —— 配置说明（engine 路径 + `EvalFile`/`NnueMode` 选项值 + CLI 等价形式）

## En Croissant 设置（S14 已可用 —— 需本修复之后的引擎）

En Croissant 的引擎配置 = `path` + `settings`（UCI 选项），**没有启动参数字段**；
S14 用下面两个选项启用（`EN-CROISSANT-S14.txt` 里有同样的内容）：

- Command（引擎路径）：`<repo>\target\release\eureka.exe`
- Settings：
  - `Evaluation` = `nnue`（默认 `classical` = HCE）
  - `EvalFile` = `<repo>\target\release\nnue-s14-datasupply-v5.bin`（绝对路径；staging 脚本已放好）
  - `Hash` 保持默认 16 即可

要点：

- **评估器由模型元数据决定**（material-residual 组合、R12 关系栈、增量方式都在评估器内部
  自动处理），不再需要 `NnueMode` / `nnue-v2q-full` 这类格式选项（旧配置里的 `NnueMode`
  会被忽略）。
- 选项路线与 profile 路线（`--profile current-final-s12 --nnue-model …`）使用**同一个
  评估器构建入口**，评分一致；全量重算保留为诊断/正确性对照用途。
- 切换评估器/模型会**清空置换表**（旧评估器产生的分数不得复用），并有
  `evaluator changed; transposition table cleared` 提示。
- `Evaluation=nnue` 但没有可加载的 `EvalFile` 时 fail-closed（`refusing to search`），
  不会静默回退 HCE。
- ⚠️ 以上需要**本修复之后构建的引擎**；旧 exe 仍有"残差被当绝对分"的缺陷——先
  `cargo build --release` 再用。

## CLI / 支持启动参数的 GUI（等价形式）

```text
--profile current-final-s12 --nnue-model nnue-s14-datasupply-v5.bin
```

- 相对路径的 `--nnue-model` **先按 `eureka.exe` 所在目录解析**（模型已 stage 在同目录），
  因此不依赖工作目录；绝对路径照常可用；找不到仍是 fail-closed 启动报错。
- 该模式下 `EvalFile`/`Evaluation` 选项被引擎忽略（`fixed by the startup NNUE profile`）。

## 验证

正确路径（profile 式；注意要给搜索留时间，见下）：

```bash
(printf 'uci\nposition startpos moves e2e3 e7e6 d1g4 d8e7 g4e6\ngo depth 8\n'; sleep 10; printf 'quit\n') |
  ./target/release/eureka.exe --profile current-final-s12 \
  --nnue-model target/release/nnue-s14-datasupply-v5.bin
```

应看到 depth 1 起 score 约 +690、`bestmove f7e6`（赢后）。

> ⚠️ 测试坑：不要"一次灌入 `go` + `quit`"——`quit` 会在搜索完成前中止搜索，返回的 `bestmove`
> 是中止时的第一个合法着（会误导成"引擎不会吃后"）。要么像上面留出搜索时间，要么在 GUI 里观察。

## 为什么以前"要手动配置"（背景）

- 引擎**默认** `NnueMode=off`（刻意设计：默认行为不因 NNUE 改变）；无参数时只自动发现
  与 exe 同目录、固定名为 `nnue-v2-q01.bin` 的模型（`src/uci.rs`）。
- S14 有两条接入路线：profile 式（`--profile current-final-s12 --nnue-model …`，Arena/CLI，
  当前唯一正确）与选项式（`EvalFile` + `NnueMode`，GUI 路线，**对 S14 暂不可用**，见顶部 ⚠️）。
- 本地 `cargo build` 从不自动 stage 数据产物（模型是版本化数据、不是构建输入；
  Arena 侧由部署管线装入 build 目录）——所以这里补一个 staging 脚本。
- 旧流程（把 `nnue-v2-q01.bin` 放 exe 旁 + 手动改 `NnueMode`）见
  `results/s10/s10-nnue-tryout-guide.md`（S10 时代，历史参考）。
