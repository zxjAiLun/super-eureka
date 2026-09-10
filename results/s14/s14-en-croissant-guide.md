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
- `EN-CROISSANT-S14.txt` —— 粘贴用配置（engine 路径 + args 行）

## En Croissant 设置

Settings → Engines → Add Engine：

- Command：`<repo>\target\release\eureka.exe`
- Arguments（粘贴 `EN-CROISSANT-S14.txt` 里的那一行）：

  ```text
  --profile current-final-s12 --nnue-model "<repo>\target\release\nnue-s14-datasupply-v5.bin"
  ```

**不需要**配置 `EvalFile` / `NnueMode`：`current-final-s12` 启动 profile 会把评估器固定；
这两个选项在该模式下被引擎忽略（会打印 `fixed by the startup NNUE profile; option ignored`）。

> 注意：`target\release\` 里若残留旧的 `nnue-v2-q01.bin`（S10 模型），它只在"无参数"
> 模式下被自动发现；使用上面的 S14 配置不受影响。

## 验证

```powershell
echo uci | & .\target\release\eureka.exe --profile current-final-s12 --nnue-model "$PWD\target\release\nnue-s14-datasupply-v5.bin"
```

应看到 `info string profile current-final-s12` 与 `uciok`；对局时正常输出 `bestmove ...`。

## 为什么以前"要手动配置"（背景）

- 引擎**默认** `NnueMode=off`（刻意设计：默认行为不因 NNUE 改变）；无参数时只自动发现
  与 exe 同目录、固定名为 `nnue-v2-q01.bin` 的模型（`src/uci.rs`）。
- S14 走"启动参数 profile"路线：`current-final-s12` 必须显式 `--nnue-model`
  （fail-closed；与 Arena 正式 preset 相同）；该模式下 GUI 选项被忽略。
- 本地 `cargo build` 从不自动 stage 数据产物（模型是版本化数据、不是构建输入；
  Arena 侧由部署管线装入 build 目录）——所以这里补一个 staging 脚本。
- 旧流程（把 `nnue-v2-q01.bin` 放 exe 旁 + 手动改 `NnueMode`）见
  `results/s10/s10-nnue-tryout-guide.md`（S10 时代，历史参考）。
