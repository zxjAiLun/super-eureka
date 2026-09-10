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

## En Croissant 设置（选项式）

En Croissant 的引擎配置 = `path` + `settings`（UCI 选项），**没有启动参数字段**；
S14 用下面两个选项启用（`EN-CROISSANT-S14.txt` 里有同样的内容）：

- Command（引擎路径）：`<repo>\target\release\eureka.exe`
- Settings：
  - `EvalFile` = `<repo>\target\release\nnue-s14-datasupply-v5.bin`（绝对路径；staging 脚本已放好）
  - `NnueMode` = `nnue-v2q`（默认 `off` = HCE；设为 `nnue-v2q` 后评估器换成 S14 NNUE）
  - `Hash` 保持默认 16 即可

说明：

- 选项路线与 profile 路线是**同一套 S14 权重**；选项路线对 V2R12 模型按 fresh 关系行评估
  （评估结果与增量路线一致、速度略低）——适合 GUI 对局；**正式测量请用 Arena preset / CLI profile**。
- EvalFile 缺失/加载失败时会 `EvalFile load failed` + `refusing to search`（fail-closed），
  不会静默回退 HCE。

> 注意（重要）：`target\release\` 里残留的旧 `nnue-v2-q01.bin`（S10 模型）会被引擎在握手时
> 作为 `EvalFile` 的默认值上报——**务必把 `EvalFile` 改成 `nnue-s14-datasupply-v5.bin`**，
> 否则会加载旧模型。

## CLI / 支持启动参数的 GUI（等价形式）

```text
--profile current-final-s12 --nnue-model nnue-s14-datasupply-v5.bin
```

- 相对路径的 `--nnue-model` **先按 `eureka.exe` 所在目录解析**（模型已 stage 在同目录），
  因此不依赖工作目录；绝对路径照常可用；找不到仍是 fail-closed 启动报错。
- 该模式下 `EvalFile`/`NnueMode` 选项被引擎忽略（`fixed by the startup NNUE profile`）。

## 验证

选项式（与 En Croissant 的行为一致；已在当前 exe 上实测）：

```powershell
"uci`nsetoption name EvalFile value $PWD\target\release\nnue-s14-datasupply-v5.bin`nsetoption name NnueMode value nnue-v2q`nposition startpos`ngo depth 6`nquit`n" |
  & .\target\release\eureka.exe
```

应看到 `info string EvalFile loaded: nnue-s14-datasupply-v5.bin` 与 `bestmove ...`。

profile 式（CLI）：

```powershell
echo uci | & .\target\release\eureka.exe --profile current-final-s12 --nnue-model nnue-s14-datasupply-v5.bin
```

应看到 `info string profile current-final-s12` 与 `uciok`。

## 为什么以前"要手动配置"（背景）

- 引擎**默认** `NnueMode=off`（刻意设计：默认行为不因 NNUE 改变）；无参数时只自动发现
  与 exe 同目录、固定名为 `nnue-v2-q01.bin` 的模型（`src/uci.rs`）。
- S14 有两条等价接入路线：profile 式（`--profile current-final-s12 --nnue-model …`，Arena/CLI）
  与选项式（`EvalFile` + `NnueMode`，本指南给 GUI 的路线）；两者同一套 S14 权重。
- 本地 `cargo build` 从不自动 stage 数据产物（模型是版本化数据、不是构建输入；
  Arena 侧由部署管线装入 build 目录）——所以这里补一个 staging 脚本。
- 旧流程（把 `nnue-v2-q01.bin` 放 exe 旁 + 手动改 `NnueMode`）见
  `results/s10/s10-nnue-tryout-guide.md`（S10 时代，历史参考）。
