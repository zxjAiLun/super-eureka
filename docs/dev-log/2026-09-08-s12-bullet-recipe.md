# 2026-09-08 · S12 Bullet-Recipe NNUE 全栈 + 生产筛选赛

commits: `dfe60df`(实现链)+ `876e4fe`(筛选赛)

## 前因

S11-B5 判定 R12 与 E3 在完整搜索中平手后,审批方裁定 S11 收尾(不做 transfer 归
因),转 S12:以 Bullet `629ee500` output-buckets 配方为参考做"一套新配方、一个
模型、一场筛选赛"。配方冻结:V2+R12 输入、双视角 FT256、SCReLU、8 个棋子数量
线性输出桶((occ含王−2)/4)、材质残差评分、sigmoid(400) 得分损失;沿用 PyTorch、
1M 数据零新标签;预算 20 epochs / 1 GPU 小时,单 seed;对手是生产 current-final。

## 做了什么

1. **训练侧**(`tools\s12\s12_train.py`):新 S12Model(FT256 + SCReLU +
   Linear(512,8) 桶选择)+ sigmoid 得分损失 + AdamW 余弦衰减;复用 train_nnue 的
   全部数据管线(特征/材质导出、划分、确定性)。桶规则逐字实现 Bullet
   MaterialCount::<8>。
2. **导出**(`s12_export.py`):v5 artifact(header v4 + head_kind u32 = 116 字节;
   FT i16@2^12、head i16@2^12、bias i32@2^24;SCReLU 整数形式
   clamp(A,0,QA)²/QA 精确;head MAC i64;proven bounds fail-closed)。
3. **Rust**:v5 loader(fail-closed:ScreluBuckets+V2R12+FT256 only)+
   `screlu_bucket_forward` 内核;`evaluate_raw` / `evaluate_raw_from_accumulator` /
   `evaluate_raw_hybrid_r12` 全部 head-aware;新 profile `current-final-s12` 全线
   接线;inc R12 栈 W256 分支补齐。
4. **验证**:PyInt↔Rust 2000/2000 bit-exact;FP32↔quant mean 0.43cp;full↔inc
   (10k corpus + 19,692 转移 + inc 栈 200 + 17 null + 10 fixtures)0 mismatch;
   树一致 24/24;NPS 178.8k ≈ E3 179.2k(SCReLU 单层头比 32-32-1 更便宜)。
5. **训练执行**:204 秒 patience 停在 epoch 8,best epoch 2;composed val MAE
   132.8(E3 138.6,R12 127.2)。
6. **筛选赛**:current-final-s12 vs 生产 current-final,10+0.1 同 binary 换色,
   128 局:**64-64(49.6%),Elo −2.7 ± 30.7 —— 完全平手**。

## 遇到的问题与解决

- **castle fixture FEN 非法**(B2 时代手写,34 子>32):被新加的 occupancy 检查
  暴露——此前一切比较自洽所以从未发现;换成合法 kiwipete + bucket 加 clamp 防御。
- v5 header 字节数算错一次(120→116:f32 计数);Python 整数参考初版残留死代码;
  embedding_bag 2D/offset 形状误用;EncodedSplit 需要 `target_scaled` 键——全部
  smoke 即抓即修。
- legacy PayloadLayout 检查先于 v5 分支执行 → 改为 head 条件化。

## 结论

- **工程结论**:Bullet 配方适配完成且全部一致性门 PASS;SCReLU+单层桶头比旧
  32-32-1 尾更快(NPS 持平 E3)。
- **训练结论**:204 秒训练的 S12 与生产 current-final **完全平手**(64-64)。
  按 S12 冻结规则:这是"值得续测",不是领先——不升 SPRT。
- **决策点(留审批方)**:(a) 追加训练预算(8/20 epochs 就 patience 停了,val
  loss 由 epoch-2 主导——LR/调度/epoch 数有明显余量);(b) 接受平手停止;
  (c) 256 局重跑缩 CI。

验证:441/441 测试;fmt clean;结果在 `results\s12\`。
