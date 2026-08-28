# Eureka NNUE 试用指南（En Croissant / 任意 UCI GUI）

NNUE 版本已可以通过 UCI 直接使用了。三个引擎形态：

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

## En Croissant 具体步骤

1. 打开 En Croissant → 设置（齿轮）→ Engines → 添加
2. Engine 选 `eureka` 二进制路径
3. 在 Arguments 一栏粘贴上面 `--profile ... --nnue-model ...` 那行
4. 保存后即可在 Analysis 或 Play 里选用

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
