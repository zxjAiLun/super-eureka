# 2026-09-09 · S14 Data-Supply(Fishtest LTC 复用现成搜索评分)— PROMISING

commits: 本条目随 S14-B 落盘提交(S14-A 前置:`3563292` 下载+转换、`5e8cdfe`
组装+编码)。

## 前因

S13(0.75 result blend)灾难性 FAIL 后,S13 dev-log 明确写下:**剩余未测的杠杆 =
数据供给**(5M+ fresh positions)。S14 就是执行这条预测,且遵守资源纪律——
**不重新支付 SF teacher 标注算力**,而是复用 Fishtest LTC PGN 里现成的引擎搜索
评分(`{-0.91/21 1.749s}` 注释的 CP)。

冻结合同:**训练配方 = S12-R0 逐字,只改训练数据**。V2+R12 / FT256 / SCReLU /
8 MaterialCount buckets / material residual / CP-only `sigmoid(cp/400)` MSE /
batch 1024 / seed 20260908 / AdamW 1e-3 wd 1e-5 cosine。无 WDL blend(S13 的
`--wdl-proportion` 路径这里不走)。

## S14-A(前置,已完成)

- 下载 `official-stockfish/fishtest_pgns` 2026 全量(600 tests、68.85 GiB、~79M
  局,**镜像 hf-mirror.com + 显式 no-proxy**)。
- `s14_convert.py`:采样落子前 FEN + mover-view CP(黑方不翻号,实测确认);
  **Base 侧 fail-closed**(PGN header 精确等于 `Base-<resolved_base>`);每局最多
  1 条(`sha1(test|game)%eligible` 确定性采样);不做 depth/time 质量筛选(按指令)。
  正确性电池 12/12 replay、200/200 三元组重解析、0 Base 未知、3 corpus 重叠校准
  无翻号/尺度病理。
- `s14_assemble.py` + `s14_encode.py`:去重后精确 **4,220,410** 条,用引擎自己的
  `bench nnue-features-batch --feature-set v2r12` + `material-batch`(单一语义源)
  编码为紧凑 JSONL.gz(`data/s14/encoded/`,0.40 GiB)。

## S14-B(本轮做了什么)

新训练器 `tools/s14/s14_train.py`(不动 `s12_train.py` 的 epoch 思维,新写流式
预算 + 双轨数据 + CSR 稀疏池,内存 ~2 GB 而非 dict-per-position 的 ~20 GB):

1. **双轨数据池 = 5,000,000**:canonical CP corpus TRAIN(779,590,teacher_cp_stm
   非 null,走 S12 精确 engine-export 路径,特征/material/bucket 与 R0 位对齐)
   + fresh Fishtest(4,220,410,直接读 encoded 分片)。`779,590 + 4,220,410 =
   5,000,000` 整。
2. **预算按 presentations**:20M(≈4 遍)或 3600s 墙钟先到停;cosine schedule 以
   `ceil(20M/1024)=19,532` 步规划。
3. **选模只用原 Y16 corpus validation(97,528)**,绝不用 Fishtest 做选模——防止
   只对新标签分布过拟合(用户明确要求)。每 500 步评 sigmoid loss,取最低者。

导出 `s12_export.py` → v5;`s14_identity.py` 做最小 identity(架构/runtime 在 S12
已证,只换 weights)。

## 结果

**训练**(20M presentations 命中、19,532 步、4 遍、1,245.7s;其中数据加载 ~955s,
训练本体 ~290s @ ~66 steps/s):

```
best_step 14000   val sigmoid loss 0.008841   val_mae 131.0
```

对照 **S12-R0**(NNUE 开发基线):val 0.009204 / MAE 132.8。S14 的 offline 代理
**双双更优**,且曲线健康(单调下降到极小后随 cosine 退火轻微回升平台),**没有**
S13 的 early-best-即恶化形态。

**Identity**:PyInt↔Rust 300/300 bit-exact PASS;UCI smoke(current-final-s12
+ S14 net,`go depth 10`)bestmove PASS。artifact sha256 `329b7170…`,11,936,916 B。

**256 局 fresh-opening screen**(book 193-320,与 R0 的 1-64、S13 的 65-192 不相交;
同 binary、10+0.1、Hash 16、单线程、concurrency 6;无 SPRT):

```
S14 W/D/L = 129 / 39 / 88        148.5 / 256 = 58.01%
Elo +56.1 ± 38.6 (pentanomial)   cutechess 自报 +56.1 ± 39.7, LOS 99.7%
pentanomial [14, 12, 52, 19, 31] (128 pairs)
```

cutechess 自报 W/L/D 与解析完全一致(Win 62+67=129、Loss 44+44=88、Draw 39)。

**判定:PROMISING(58.01% > 52% 档)。STOP —— 不升 SPRT(冻结合同)。**

## 结论与纪律

- **Data-Supply 是有效杠杆**:S13 dev-log 预测的方向被证实。同一配方(架构/loss/
  runtime 全冻结)只把数据从 0.78M canonical 扩到 5.0M(canonical + fresh Fishtest
  CP),就把 vs 生产 current-final 从 R0 的 49.61%(parity)推到 58.01%
  (+56 Elo,LOS 99.7%)。与 S13 的 12.89% 形成鲜明对比——问题从来不是"NN 学不动
  Splendor…"(此处是 chess),而是**训练信号/数据供给**。
- **证据强度 honest 定位**:这是 **screen 级**证据(256 局,pentanomial CI 下界
  约 +17.5 Elo,LOS 99.7%)——强,但不是 SPRT 确认。是否升 SPRT / 是否把 S14 net
  设为新的 NNUE 开发参考 / 是否动生产默认,**都需要用户显式授权**,本轮一律不碰
  (无 SPRT、无部署、无归因)。
- **工程资产**:S12 v5 runtime + `current-final-s12` profile 不变;`current-final`
  仍是生产引擎。S14 只新增一个更强的候选 net(local-only .bin/.pt;summary/layout/
  screen 工件入库)。
- **复现要点**:数据加载 ~16 min 是纯 Python gzip+json.loads over 4.22M 行的成本
  (训练本体只 ~5 min);未来扩 10M/20M 只是多读已下载 PGN + 拉大 presentations。
```
python tools/s14/s14_train.py --out data/s14/run --seed 20260908 \
    --presentations 20000000 --max-seconds 3600 --val-every 500
python tools/s12/s12_export.py --checkpoint data/s14/run/checkpoint_s14_v2r12_s20260908.pt \
    --out data/s14/run/seed-20260908/nnue-s14-datasupply-v5.bin --layout <...>.layout.json
python tools/s14/s14_identity.py            # 300/300 + UCI
python tools/s14/s14_screen.py --concurrency 6
```
