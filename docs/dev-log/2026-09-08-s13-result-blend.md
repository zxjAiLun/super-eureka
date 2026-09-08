# 2026-09-08 · S13-A Real-Game-Result Blend(灾难性 FAIL)

commits: `a0b2f4b`(实现+筛选)+ `28ecf77`(结果入库)

## 前因

S12 CLOSE / PARITY 后,审批方定 S13 = Data-Supply / Game-Outcome Recipe:架构/
runtime 全冻结,改训练信号。Preflight 要求先查原始 source 的 game result、现成
score、未入库量(几十分钟,不开分支)。

## Prefflight 结论(写入 screen-report.json)

- **真实棋局结果**:E1 pool cache 与 1M corpus 的每个 position 都带
  `game_result_white`(1.0/0.5/0.0,真实对局结果,非 SF WDL);splits 游戏-不相交
  (0 crossing);779,590 train 位置同时有 teacher CP + result。
- **现成 score**:源 PGN 仅 13.6% 有 `[%eval]`——不可靠 → 按审批决策树走
  **S13-A**(现有 corpus + blend)。
- 5M fresh stream 需要新标注算力,本轮未授权。

## 做了什么

1. `s12_train.py` 加 `--wdl-proportion`(0.0 = 精确 S12 路径);blend target =
   `0.75 * real_result_stm + 0.25 * sigmoid(cp/400)`,prediction/loss 不变;val
   选择用同一 blend 目标(只含有 result 的位置)。
2. 训练(seed 20260908,batch 16384,其余同 R1):**同样的 epoch-2-即恶化形态**,
   best epoch 2,blend val loss 0.1155,纯 CP val MAE 166.5(R0 132.8)。
3. 最小 identity 验证(按协议不重跑全套):PyInt↔Rust 300/300 bit-exact、UCI
   smoke PASS、artifact SHA `770db5f4…`。
4. **256 局 screen**(真正跑满合同):128 个**全新** opening pairs(book 65-192,
   避开 R0 用的 1-64),换色,同 binary,10+0.1。

## 结果

```
S13 W/D/L = 21/24/211     33.0/256 = 12.89%
Elo -331.9 ± 32.4         pentanomial [89, 17, 18, 3, 1]
```

**S13 FAIL / STOP**(12.89% ≪ 48% 档)。

## 结论

- 0.75 结果权重把 ~2300 分在线快棋的真实胜负压过了 16k 节点 SF18 teacher——
  毁灭性劣化。Bullet 的 blend 常数来自它自己的数据管线(self-play、持续刷新);
  嫁接到我们的 corpus 上,这个便宜检验被干脆地否决。
- Blend 也没有改变 early-best 形态(仍 epoch 2 后单调恶化)。
- **剩余未测的杠杆 = 数据供给**(5M+ fresh positions + 双信号),需要新标注算力,
  按协议本轮不碰。这是未来重开训练线时的第一候选,但要有明确的算力授权。
- 工程资产不变:S12 栈(v5 runtime + profile)继续生产可用,`current-final`
  仍是生产引擎。
