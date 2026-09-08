# 2026-09-08 · S11-B5 R12 Search Validation(双框架结果 + anti-drift 判读分歧)

commit: `22c94e9`(前置:`38aaeee` B4-B / `475b0c9` SF2400 部署)

## 前因

B4-B 过 NPS 门(0.9488)解锁 B5。B5 回答唯一问题:R12 的 offline static task gain
能否转化为完整搜索的 move quality。冻结协议:双臂同 binary、1 线程、Hash 32、
100k nodes/root、256 冻结 H0-E parents、既有 SF18-32k sibling labels(零新增
SF)、每 (root, arm) 独立进程(TT 绝对隔离);teacher 语义沿用 j0_lockbox
(-sf.cp,mate→±2000,clamp);anti-drift 要求本轮同 harness 重跑 E3。

## 做了什么

新 harness `tools\s10\s11_b5_search_validation.py`(H0-E 原搜索 harness 未随
aa44d65 提交,只有结果 JSON;按 commit message 重建协议:bench profile --fen
--nodes --hash-mb + bestmove 查 sibling teacher,regret = teacher_best −
teacher(chosen))。先单 root 双臂 smoke 再全量 256×2。

## 结果(256/256 双臂,0 失败,0 缺 teacher)

| 指标 | E3 本轮 | R12-inc | delta |
|---|---|---|---|
| mean regret | 39.4 | 41.4 | +2.0 |
| median | 0 | 0 | 0 |
| p90 | 93 | 97 | +4 |
| acc@20 | 66.0% | 66.8% | +0.8pp |
| acc@50 | 80.1% | 80.9% | +0.8pp |
| top1 | 52.3% | 54.7% | +2.4pp |
| >500cp | 3 | 4 | +1 |
| >1000cp | 0 | 0 | 0 |

Phase(mean/acc20):high 25.1/65.6→29.3/64.1;mid 37.2/70.3→33.5/73.4;low
37.2/68.8→37.6/70.3;**zero 58.2/59.4→65.2/59.4**(zero-phase 仍是 R12 弱尾,
与 S11-A static 分析一致)。

## Anti-drift:FLAGGED,已诊断

本轮 E3 mean 39.4 vs 历史 69.6(-30.2cp)——但 **median(0=0)、p90(93=93)、
acc20(-1.2pp)完全吻合**。差距是纯极端尾部质量:本轮 E3 无 >1000cp regret,
历史 mean 意味着约 8-10 个 clamp-mate 位置。H0-E 原脚本未入库,尾部差异无法
进一步审计(大概率是那个 harness 的 TT/节点计数/每根状态差异)。

## 双框架判读(核心分歧,交审批方裁定)

1. **历史门框架**(门由历史 E3=69.6 推导):41.4≤64.6 ✓、66.8≥66.2 ✓、
   97≤100 ✓ → **PASS**(且"strong pass"标签为真)。
2. **同 harness 配对框架**(协议明文要求本轮重跑 E3 的目的):R12 mean 比同跑
   E3 **差 +2.0cp**,p90 +4,仅 acc20/acc50/top1 小幅占优(+0.8/+0.8/+2.4pp)
   → **parity-ish,不是明确改善**。

两个框架给出定性不同的结论。冻结协议说"历史漂移先查 harness 不直接解释
R12"——已查(harness 差异无法审计,但 bulk 分布吻合证明协议语义一致,差异
仅在尾部);是否以同跑配对为准,需要审批方拍板。

## 结论

- 运行时(B4-B 0.9488)+ 完整搜索行为(B5)都已在案;R12 在搜索中与 E3 大致
  平手(top1/acc 类指标小幅占优,mean/p90 尾部小幅劣势,zero-phase 弱尾)。
- **未做任何**:Arena 准备、artifact 部署、A_P restore、残局特判、margin
  calibration——按协议 STOP。
- 提醒:SF2400 live match(`6a07cc07…`)仍在跑,其结果将与 B5 互相印证
  (R12 首个实战强度读数)。

验证:harness smoke → 全量;结果 `results\s10\s11-b5-search-validation.json`
(含 identity:binary/artifact SHA、corpus 来源、完整性计数)。
