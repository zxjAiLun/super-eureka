# Eureka NNUE 试用指南（En Croissant / 任意 UCI GUI）

NNUE 版本已可以通过 UCI 直接使用。En Croissant 不需要命令行参数：
`EvalFile` 与 `NnueMode` 会出现在引擎的高级设置中。启动 profile 仍在
整个进程内固定；这些选项只替换评估器，不改变搜索策略或 profile 身份。

## En Croissant：无命令行参数（推荐）

1. 把 `nnue-v2-q01.bin` 放到 `eureka.exe` 同一目录。
2. 在 En Croissant 中添加 `eureka.exe`，启动参数留空。
3. 打开该引擎的高级设置，把 `NnueMode` 从 `off` 改成
   `nnue-v2q`，保存。

`EvalFile` 留空即可：引擎只会按 `eureka.exe` 所在目录自动查找，绝不按
GUI 的当前工作目录查找。需要使用其他位置或文件名时，在 `EvalFile`
文本框填写模型的绝对路径；`nnue-v2q-full` 是每次评估全量刷新的参照模式。

默认 `NnueMode=off`，因此不设置这些选项的 Arena/fastchess 和现有 GUI
行为不变。启用 NNUE 后若没有可加载模型，引擎会输出
`NnueMode requires a loadable EvalFile; refusing to search` 和
`bestmove 0000`，不会静默回退到手工评估。

以下命令行 profile 方式继续保留，适合 Arena、fastchess 和基准测试。

## 1. NNUE Incremental（推荐体验，S10 主力候选）

引擎路径：
```
<repo>/target/release/eureka
```

启动参数（En Croissant：Settings → Engines → Add Engine，在
Arguments/命令行参数里填）：
```
--profile current-final-nnue-v2q --nnue-model /media/bailan/DISK/AUbuntuProject/project/chessenginedemo/data/s10/b3/seed-20260818/nnue-v2-q01.bin
```

注意 `--nnue-model` 必须用**绝对路径**（GUI 的工作目录不一定在 repo 里）。

## 2. NNUE Full Refresh（每步全量重算的参照版）

```
--profile current-final-nnue-v2q-full --nnue-model /media/bailan/DISK/AUbuntuProject/project/chessenginedemo/data/s10/b3/seed-20260818/nnue-v2-q01.bin
```

## 3. 生产版 CurrentFinal（Eval2，对照组）

无参数直接注册：
```
<repo>/target/release/eureka
```

## 验证是否生效

注册后可在 GUI 的控制台里看 `uci` 握手；或命令行直接测：

```bash
echo -e "uci\ngo depth 8\nquit" | \
  target/release/eureka --profile current-final-nnue-v2q \
  --nnue-model data/s10/b3/seed-20260818/nnue-v2-q01.bin
```

输出 `bestmove ...` 即正常。忘记 `--nnue-model` 时引擎会启动失败并
打印 `startup_error ... fail closed`——这是设计行为（防止静默回退到
非 NNUE 评估）。

## 当前棋力预期（诚实预告）

- NNUE 评估器：300k Stockfish-18 教师标签训练，validation MAE
  ~165cp（量化后 +0.007cp）
- 搜索：与生产 CurrentFinal 完全相同的策略（PVS/LMR/null-move/
  aspiration/SEE 等）
- 速度：incremental NNUE 约为 Eval2 版的 0.82x（搜索 NPS）
- KQK/KRK 残局保留精确 mop-up

所以它现在大约是"中等水平引擎"——别指望它赢 Stockfish，但和
CurrentFinal 版对下应该有意思。两个都注册进 GUI 可以直接对弈观察。
