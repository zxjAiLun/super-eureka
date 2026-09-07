# 2026-09-07 · S11-B2/B3 R12 Hybrid Reference Runtime

commits: `19c5237` (Phase 0) → `43130d1` (manifest) → `579cdab` (Phase 1 core) →
`1cad82f` (Phase 1 integration) → `d686686` (fmt) → `03cea1b` (Phase 2 parity) →
`7025c56` (Phase 3 benchmark)

## 前因

S11-A 冻结 R12 sidecar(seed19)为运行时候选;B1 churn 审计显示 relation 行温和。
B2/B3 的任务:把 R12 做成"无分配、fresh-recompute、语义绝对可信"的 reference
runtime,证明 parity,拿到真实成本数字。云端审批修正了三处计划:v1-v3 兼容是
hard gate(version-aware 108/112 header,不是全局 bump);R12 accumulator bound
= 61 不是 46;Layer D 必须先于正式 benchmark。

## 做了什么

- **Phase 0(export)**:`export_quantized.py` / `export_nnue_v2.py` 加
  `--feature-set {v2,v2r12}`;v4 格式(feature_set u32,header 108→112);
  feature-set-specific bound(V2=31,V2R12=61)。artifact
  `nnue-v2-q01-material-r12-v4.bin`(sha256 8eacd0c1…,5,983,156 bytes)+
  manifest/layout 入库(.gitignore 目录链白名单)。
- **Phase 1(runtime)**:`for_each_relation_feature_v2r12` 单一无分配语义核心
  (Vec wrapper 仅 exporter/diagnostic);`NnueFeatureSetId` enum;version-aware
  `header_bytes()`;v4 loader fail-closed 矩阵;`full_acc` feature-set 分派;
  hybrid 路径(`add_fresh_relation_rows` + `evaluate_raw_hybrid_r12` +
  `hybrid_accumulator_r12`);`base_accumulator_v2`;新 profile
  `current-final-nnue-v2q-material-r12` 全线接线(search/bench/UCI,双向
  feature_set fail-closed);audit 路径 R12-aware;**关键正确性点:栈只存 V2-base
  accumulator(root frame 也是),否则 relation 行会被双重叠加**。
- **Phase 2(parity,四层全 PASS)**:
  - A:FP32↔quant mean 0.13cp / p99 0.511 / max 1.402(门 0.30/1.0/2.0);
  - B:Python int↔Rust raw 10,000/10,000 exact(`f1_quant_parity.py` 参数化,
    读全 dataset parts);
  - C:hybrid↔full,10k corpus + 19,692 transitions + 10/10 定向 fixtures,
    lanes+raw 全 exact(新 bench 命令 `nnue-v2q-r12-parity`);
  - D:scalar↔AVX2 10,000/10,000 raw exact(force_scalar_l1 双 build);
  - 旧 artifact 回归:4 个冻结量化 artifact(B3/E3 v2、E3 v3twin、G1 FT256)
    重构前后 raw_output 逐字节一致。
- **Phase 3(benchmark)**:`nnue-v2q-cost` R12 段 + `s11_b3_nps.py` 配对 harness
  (C3-C 冻结 24-FEN corpus、200k nodes、冷 64MiB TT、8 轮 AB/BA 交替、同 binary)。

## 遇到的问题与解决

- **B1 遗漏**:3f171db 漏提交 search.rs 的 4 个 churn hooks(数据仍有效——
  当时从工作树构建——但提交树无法重建诊断工具);Phase 0 一并补提交并验证
  重建二进制复现 smoke 数字。
- **Bench fixture FEN 三次不合法**(promotion FEN 的王被牵制等)→ 修正为合法
  FEN,一次一个。
- **Part 11 scan 计量误导**:初版只有 scan(7.4µs)vs hybrid eval(22.8µs),
  14µs 差值无法解释;加 `scan_plus_dummyft` 部分定位到 **FT 行加法**是主导,
  不是 attack 查询。
- **性能假设被推翻**:曾预期 fresh 路径"可行";实测 NPS 0.58。
- cargo fmt 揭示仓库存在 pre-existing fmt drift → 单独 style commit 收口。
- probe-batch 需要 `position_id|fen` 格式;EUNN2F32 bridge artifact 理应被
  拒载(回归脚本里的"FAIL"是脚本误报)。

## 结论(冻结事实)

1. **Runtime 正确性完备**:四层 parity 全 PASS,R12 hybrid 与训练语义逐位一致;
   v1-v3 兼容逐字节不变。
2. **NPS = 0.58 median**(opening ~0.56 / tactical ~0.67 / endgame ~0.85):
   **低于 0.80 暂停门,B5 search validation PAUSED**(按冻结协议)。
3. **成本归因**(最重要的认知修正):
   - B1 的 churn(4-8 行/边)是**每步 delta**;fresh 路径每次 eval 重加**全部
     ~24 活跃行**(median,mean 23.9,p90 44);
   - 主导成本是 **FT 行加法的内存流量**(~12µs:24 个随机 256B 行 × 双
     perspective,relation 行横跨 1.9MB ft_weights),attack 查询次之(~7µs);
   - eval-only ~22×、edge+eval ~3.8×(简单位置)、搜索整体 0.58。
4. **下一步决策点**(留给审批方):incremental relation updater 是否值得做——
   B1 数据(平均 4-8 行/边)表明它能同时消掉 scan 和批量重加,预期把 eval 税
   压到 E3 的 ~1.5-2×以内;但这是新子项目(滑动线开闭、victim 联动、king move
   perspective 变换),工作量数倍于本轮。

验证:`cargo test --release --lib` 438/438;nnue 套件 82/82;fmt clean;
所有结果 JSON 在 `results\s10\s11-b2b3-*.json`。
