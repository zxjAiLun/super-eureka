# 2026-09-08 · S12-R1 Batch-Cadence Repair(FAIL)+ S12 收尾

commit: `5b4ba04`(前置:`37e78b4` S12 全栈)

## 前因

S12-R0 筛选赛 64-64 平手。审批方复核:①R0 屏幕实为 128 局(非冻结的 256),记
"EARLY SCREEN / PARITY / INFORMATIVE";②训练曲线 epoch2 后连续六 epoch 恶化,
"多训一定有潜力"不成立;③真正的单变量差异是 optimizer cadence——batch 1024 vs
Bullet 参考 16384(同 LR 1e-3,即每遍数据 16× 的 update 数)。冻结 S12-R1:只改
batch(真 16384,VRAM 够无需累积),stop 门 best val loss < 0.00910,否则
S12 CLOSE / PARITY。

## 做了什么

`--batch-size` 默认改 16384(训练循环本就是每 batch 一次 optimizer.step +
scheduler.step,单变量成立);seed/架构/数据/LR/调度/patience 全不动;225 秒
patience 停在 epoch 10。

## 结果

| | R0 (1024) | R1 (16384) |
|---|---|---|
| best epoch | 2 | 4 |
| best val loss | **0.009204** | 0.009657 |
| best val MAE | 132.8 | 136.1 |

**R1 FAIL**(0.009657 > 门 0.00910,也差于 R0)。16× batch 让曲线略平(best
后移到 epoch 4、恶化斜率变缓)但基本形态不变:early best 后单调恶化,绝对值更差。
Cadence 假设被单变量检验否决。未导出、未跑比赛;R1 checkpoint 留档。

## 结论

- **S12 CLOSE / PARITY**:最终对抗结果 = R0 128 局早期筛选 64-64;工程栈
  (v5 artifact 格式 + SCReLU/桶 runtime + 全套 parity 设施)保留为生产级基础
  设施。
- **对未来的记录**:与 Bullet 配方的剩余差距在数据规模/刷新(~100M 数据流 vs
  固定 780k)与 game-result 混合;epoch-2-即恶化的形态与"小固定语料快速记忆"
  一致——若未来重开 NNUE 训练线,数据供给是比 cadence 更可能的杠杆。
- 统计展示规范(审批方要求)已回填:R0 早期屏幕精确数字为 S12 **W/D/L =
  55/17/56**,score points **63.5/128 = 49.61%**,pentanomial(按开局对,
  S12 得分 0..2)**[10, 5, 34, 6, 9]**。此前 "64-64" 是把 63.5 四舍五入的
  展示层误差。
