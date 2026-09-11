# S14 本地 GUI 指南 — En Croissant（2026-09-10）

目标配置：**S14 NNUE + CurrentFinal 搜索**。Promotion 仍为 HOLD；本文只说明本地启动，
不表示部署或 production channel 切换。

## 构建与 staging

```powershell
# 关闭正在使用 eureka.exe 的 GUI 后执行
powershell -ExecutionPolicy Bypass -File tools\stage_s14_gui.ps1 -Build
```

脚本在 `target\release\` 生成：

- `eureka.exe`
- `nnue-s14-datasupply-v5.bin`（脚本校验冻结 SHA-256）
- `EN-CROISSANT-S14.txt`

## En Croissant 设置

En Croissant 没有启动参数字段，因此使用标准 UCI 选项：

- Command：`<repo>\target\release\eureka.exe`
- `Evaluation` = `nnue`
- `EvalFile` = `nnue-s14-datasupply-v5.bin`，或该文件的绝对路径
- `Hash` = `16`（默认值即可）

无参数启动和 `Evaluation=classical` 都使用 HCE。`Evaluation=nnue` 缺少可加载模型时会
fail-closed 并拒绝搜索，不会静默回退 HCE。切换评估器或模型会清空置换表。

CLI、GUI `EvalFile` 和 bench 使用相同的路径规则：绝对路径原样使用；相对路径先在
`eureka.exe` 所在目录查找，不存在时再按当前工作目录解析。

模型元数据决定 material-residual 组合、特征集和增量方式；不需要 `NnueMode`、
`nnue-v2q-full` 或其他历史 profile。

## 正常 CLI 启动（推荐）

```text
eureka.exe --evaluation nnue --nnue-model nnue-s14-datasupply-v5.bin
```

bench 使用同样的显式配置：

```text
eureka.exe bench profile --evaluation nnue --nnue-model nnue-s14-datasupply-v5.bin --nodes 50000 --hash-mb 16
```

`current-final-s12` 仅为旧 Arena preset / 脚本保留的兼容 alias，等价转换为
`current-final + --evaluation nnue`，没有独立评估逻辑：

```text
eureka.exe --profile current-final-s12 --nnue-model nnue-s14-datasupply-v5.bin
```

不要在该 alias 上显式指定 `--evaluation classical`；这是矛盾配置，会在启动时明确报错。
新配置不应依赖此历史 alias。

## 交互验证

启动引擎后按顺序发送：

```text
uci
isready
position startpos moves e2e3 e7e6 d1g4 d8e7 g4e6
go depth 8
```

必须等待引擎输出 `bestmove`，之后才能发送 `quit`。不要一次性写入 `go` 和 `quit`；
`quit` 会中止仍在执行的搜索，无法作为完成深度的验证。

## 说明

本地构建不会自动把版本化模型数据复制到输出目录，所以仍需 staging 步骤。旧的 S10
`NnueMode` 流程只保留在历史文档中，不适用于当前引擎。
