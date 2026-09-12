# 2026-09-12 · Eureka v0.2.0 正式发布（Gate A 收尾 + Gate C 全链执行）

本轮从"实验候选上线"变成**产品版本切换**：Arena `display_name = Eureka v0.2.0`
成为唯一对外的版本名，git SHA / build id / model SHA / experiment id 退回
内部 provenance。发布已完成并验证，无遗留阻塞。

## 1. 发布身份（终态）

```text
Eureka v0.2.0
  source       315534991628ce084c34392d6abb98d71f3f591a   (commit 3155349, pushed, tracked tree clean)
  binary       c2428a8453d871c9a6f2ff14780f077b2c2503bdb321419d4334361c54733986
  model        329b717066082798d2f42d9e4f137385a1482a8cda03ae7849ddc654e097e4b0   (S14, 未变)
  build_id     20260912-3155349-eureka-v020-329b7170
  Arena        2d816e66678fdc729605e880c54b0e0afd0b4e04   (release 20260912194000)

production
  current-final -> ce-v020
  display       Eureka v0.2.0
  launch        --evaluation nnue --nnue-model /opt/chessarena/builds/<build_id>/models/nnue-s14-datasupply-v5.bin
  fingerprint   f7353e8a19ef7b0c270907915faa4e9e8fd5934fc12a8038a84a9eb12a18618a

rollback
  ce-currentfinal-20260825 (HCE) historical, verified selectable

health
  ok / uci_capability_gap = 0

release gates
  Gate A  CLOSED     生产身份契约 + EvalFile artifact gate
  Gate B  PASS       本地 round-trip 测试
  Gate C  PASS       受控部署链路 1–7 全部执行
  Gate D  CANCELLED  不再为旧 promotion binary 建长期冻结体系
```

引擎提交 `3155349` 仅改 `Cargo.toml` / `Cargo.lock` / `src/version.rs` 的
版本身份（`0.1.0 → 0.2.0`），无搜索/评估行为变更；no-arg 默认仍为 HCE，
NNUE 只经显式 `--evaluation nnue` 启用。

## 2. Gate A 收尾：EvalFile artifact gate（本轮唯一安全修复）

### 2.1 漏洞（用户发现并复现）

`validate_launch_artifacts()` 只校验 `command_args` 里的 `--nnue-model`，
而 `human_engine.py` 会在启动后把 `uci_options` 逐条 `setoption` 给引擎，
包括 `EvalFile`。隔离临时库复现：声明模型 A + `EvalFile` 指向未声明的
模型 B → promotion plan `ok=True`，篡改 B 的字节仍 `ok=True`。
**校验通过的是 A，实际运行加载的是 B**——配置指纹固定了路径字符串，
没有固定字节。

### 2.2 修复（`2d816e6`，Arena）

- `validate_launch_artifacts(build, command_args, uci_options=None)`：
  `--nnue-model`（command_args）与非空 `EvalFile`（uci_options）统一过
  同一道闸——解析路径必须命中 manifest 声明的 artifact，且现场
  SHA-256 重哈希必须匹配；错误信息沿用原样式（build 内未声明 /
  声明集之外）。无新增 manifest schema。
- 三个调用点全部接入 `uci_options`：promotion 门禁（versions）、
  scheduler per-pair 预检（Popen 前 fail-closed）、formal candidate
  解析（version 候选经共享的 `validate_version_build_provenance`）。

### 2.3 回归与连带

新增回归（`tests/test_engine_versions.py`）：

```text
test_v020_evalfile_override_undeclared_model_rejected   声明 A、EvalFile 偷指未声明 B → plan 拒绝、promote 抛错、零变更
test_v020_evalfile_override_tampered_model_rejected     正向对照（双声明、字节匹配 → ok）；篡改覆盖模型 → SHA mismatch 拒绝
```

全套件连带发现两个仍断言旧契约的测试并按新契约改写：
`test_formal_experiments.py::test_promotion_candidate_must_pass_promotion_gate`
（显式参数不再触发 gate 错误；未声明模型候选在解析阶段 fail-closed）、
`test_admin_promote_ui.py::test_blocked_plan_has_no_confirm`
（阻塞场景换成 S10-D0 未声明模型；disabled build 部分保留）。

测试结果：三个发布门禁文件 **66/66 通过**（64 原有 + 2 新回归，含此前
失败的 scheduler artifact 用例）；全套件 547 收集，22 个 failed/error
**全部**为浏览器 E2E（WSL 无浏览器，既有环境限制），非浏览器失败 0。

### 2.4 Errata：cutechess"环境阻塞"结论作废

此前记录的"cutechess-cli 缺失环境阻塞"是误判：`tests/fixtures/fake_cutechess.py`
以 `#!/usr/bin/env python3` 直接启动，PATH 未前置 `.venv-wsl/bin` 时选到
venv 外解释器报 `ModuleNotFoundError: No module named 'chess'`。
正确命令（实测 66/66）：

```bash
cd /mnt/e/AUbuntuProject/project/chessenginearena
PATH="$PWD/.venv-wsl/bin:$PATH" .venv-wsl/bin/python -m pytest \
  tests/test_engine_versions.py tests/test_model_artifacts.py \
  tests/test_admin_builds_ui.py -q
```

已更正入 Arena `handoff.md`（随 `2d816e6` 入库）。

### 2.5 契约内容（Gate A 判定）

- 显式 `command_args/uci_options` 是 EngineVersion 的**冻结身份**，
  不再被"必须 no-arg default"门禁否掉；身份安全由
  immutable fingerprint + build provenance + S10-D0 artifact 校验保证。
- `historical` 不再阻塞 rollback/re-promotion；`status` 只做兼容镜像。
- 生产发布身份 = `CurrentFinal + --evaluation nnue --nnue-model <服务器
  immutable build 内绝对路径>`；引擎 no-arg 默认 HCE 不变。

## 3. Gate C 步骤 1：Arena 控制面上线（不切 channel）

`git archive 2d816e6` 打包 → SCP → `arena-deploy release-install
20260912194000`（pip + alembic，迁移本地与线上一致 0001–0010，实际
no-op）→ `release-switch` → `restart-api` / `restart-worker`。

部署前后对比：channel 指针未动（仍 `ce-currentfinal-20260825`）、
Play 对手输出逐字节未变；线上代码旧门禁 0 引用、`EVAL_FILE_OPTION`
5 处在位；health ok。

## 4. Gate C 步骤 2–4：clean 构建、受控安装、创建版本

- **构建**：`git archive 3155349` 导出（无 `.git`）→ WSL `cargo build
  --release`，provenance 由 `EUREKA_GIT_SHA` / `EUREKA_GIT_DIRTY=false`
  显式注入（build.rs 设计的 release 路径，避免 clone 换行符/脏树干扰）。
  二进制自报 `Eureka v0.2.0-dev+31553499`、`source 3155349…`、
  `dirty false`。本机用 S14 模型验证 `--evaluation nnue` 加载与
  `EvalFile` setoption 覆盖（坏路径 fail-closed：`EvalFile load failed`
  且保留原网络）。
- **安装**：`build-install` 由 wrapper 校验 binary SHA 一致后注册，
  profiles 探测 `[current-final, current-final-s12]`，S10-D0 模型声明
  验证通过。发布包留档 `results/v020-release-package/`（未跟踪）。
- **创建版本**：`POST /api/v1/engine-versions` 创建 `ce-v020`
  （display_name = `Eureka v0.2.0`，显式 NNUE + 服务器绝对路径），
  candidate / hidden / unrated 起步，fingerprint 冻结。

## 5. Gate C 步骤 5–7：plan → 执行 → 回滚验证 → Play smoke

| 步骤 | 内容 | 结果 |
|---|---|---|
| forward plan only | promotion 预览：provenance / binary SHA / model SHA / launch gate | 全过，0 Blocked，零变更 PASS |
| forward 执行 | channel → ce-v020；HCE → historical 镜像 | PASS |
| rollback plan | 立即 plan 指回 `ce-currentfinal-20260825` | clean PASS（0 错误）；旧契约下不可能 |
| 真实 round-trip | v0.2.0 → HCE → v0.2.0 | 两次 promote 全成，单 production 镜像成立 |
| Play smoke | 对手名 `Eureka v0.2.0`；真实对局 e2e4 → 引擎 956ms 应手 e7e5 | PASS（NNUE 实际参与搜索） |

smoke 局已认输关闭，服务器临时文件已清理。

## 6. 能力回填（server1）

`install_build.py` 不写 `uci_options_schema`（所有 build 均需单独回填）。
经 `server1`（ubuntu，免密 sudo）以 chessarena 身份执行标准脚本：

```text
backfilled 20260912-3155349-eureka-v020-329b7170
  uci id name: Eureka v0.2.0-dev+31553499
  uci options: 3        (EvalFile / Evaluation / Hash)
  binary sha256: c2428a84…33986 (unchanged, fail-closed 校验过)
```

health 由 `degraded / gap=1` 回到 `ok / gap=0`。

## 7. 裁定与遗留

- **UCI 自报 `v0.2.0-dev+31553499` 保持现状**：binary 来自未打 tag 的
  clean commit，`version.rs` 规则本来如此；产品身份由
  `display_name` 承担。不为去掉 `-dev+sha` 重开发布。下一版如需裸
  semver，构建前打 `eureka-vX.Y.Z` tag（build.rs tag 路径现成）。
- **bench `--evaluation nnue` locked fixture 问题维持独立 harness
  debt**：与 v0.2.0 发布无关，不影响 UCI/GUI/Play，本轮不碰。
- **身份分层方向保持**：`display_name` 是唯一对外产品名；
  `identity_fingerprint`（binary SHA + args + options）是运行真相；
  git SHA / build id / experiment id 是内部 provenance。
- 回滚链路已实测可复用：下一版发布直接走同一受控链
  （plan → forward → rollback plan → round-trip → smoke）。

## 8. 本轮未做 / 不做

- 未重训、未跑新的 SPRT 或棋力实验；发布资格不依赖新的棋力数据
  （旧资格桥接的 24 FEN exact bridge 仍为后续可选项，未做）。
- 未绕过任何控制面：全程 SCP → `arena-deploy` 受控子命令 +
  控制面自身 API；能力回填走标准脚本（server1 sudo）。
- 未新增 manifest schema；未重构路径语义（`--nnue-model` 生产身份
  用安装后绝对路径的既定裁定不变）。
- 未动 `bench` harness debt；未打 git tag；未改 `current-final` 的
  搜索 profile 语义（其退回内部 profile 名的方向不变，本厂未迁移）。
