# S4 Arena v1 — 服务器 Bootstrap 最终验收证据（reviewer 版）

日期：2026-08-06（服务器 CST，UTC+8）· 服务器：`pearllover.site` = `150.158.50.58`（Ubuntu 24.04.4，2 vCPU / 2 GB）
部署方式：**本地主导部署**（本机 SSH/SCP + 受限 `arena-deploy` wrapper）；GitHub Actions 不持有服务器 SSH 密钥。

---

## 1. Frozen baselines and provenance

```text
当前 Arena 运行时 release SHA: 96b4963107e7e9f57838faaf1dd21063b36a310d
  current 路径:               /opt/chessarena/releases/20260806215940
  DEPLOY_SOURCE_SHA:          96b4963107e7e9f57838faaf1dd21063b36a310d

前代回滚 release SHA:         431571ca64c2865c4d24cad235e0f411dfddd29b
  release 路径:               /opt/chessarena/releases/20260806020824
  DEPLOY_SOURCE_SHA:          431571ca64c2865c4d24cad235e0f411dfddd29b

Engine build git SHA:         51a629f04ea00bb841c7ebe5fc867b2e770668ef
  build_id:                   20260806-51a629f-linux-x86_64
  engine 路径:                /opt/chessarena/builds/20260806-51a629f-linux-x86_64/engine
  binary SHA-256:             9b1d394dbcefc2f3c30ebe768de7bfbdd2a88725c06318ad21bf1149bc4dae04
  rustc:                      rustc 1.94.1 (e408947bf 2026-03-25)
  Cargo.lock SHA-256:         370d4ebcacd639d0bad97efc2d7c4f1eb419ef13ce9a518d1d8b2f836d64e9ee
  profiles:                   ['current-final','current']

Deploy-arena 受控验收 commit:  96b4963107e7e9f57838faaf1dd21063b36a310d
  新 release:                /opt/chessarena/releases/20260806215940
  DEPLOY_SOURCE_SHA:          96b4963107e7e9f57838faaf1dd21063b36a310d（精确匹配 head_sha）

Verification-script 基线:    15dde7c6373d36e759bd7e26f85576def6e2c0be
  verify_install.py SHA-256:  788e4a0c9b518d4a484e57e5a040a7bf99fd72c186633ba5aaaef01ff44e5745

部署 wrapper（root-owned）:    /opt/chessarena/bin/arena-deploy
  SHA-256:                    b7f84470bc51e30f5baa028bdea3c24ee383955d7e98839c4595a9418ea52125
  owner/mode:                 root:root 0755

Opening sets:
  v1-openings:  disabled（6 字段 FEN，对 Cute Chess EPD 无效）
  v2-openings:  enabled，12 位置，标准 4 字段 EPD，sha256 1d96a70aa4031ac1665e27d32c7f7031a1b639c4c77b4269ef8ee1732d5869e7
```

引擎可复现性：同一提交 + rustc 1.94.1，GitHub runner 与本地 WSL2 构建**字节一致**（binary SHA 均 `9b1d394d…`）。

依赖与工具：cutechess-cli 1.5.1（AppImage 官方资产，SHA 校验 d9448693…；tag 为 annotated 未签名，未声称签名验证）；venv SQLAlchemy 2.0.51、fastapi 0.141.1、alembic 1.19.0、chess 1.11.2。

## 2. Six-gate verdict matrix

| Gate | 验证项 | 结果 |
|---|---|---|
| 1 | systemd 双服务 active + 启动日志 | PASS（API unit 经 `create_app --factory`；unit SHA 9600104d… 服务器与仓库一致） |
| 2 | 迁移 `0004_return_code_attempt (head)` + health 四字段 | PASS |
| 3 | Basic Auth / CSRF / 管理 API / UI | PASS（无认证 401；带认证 health+UI 200；错误 Origin 403 `cross-origin request rejected`） |
| 4 | 引擎 SHA 注册 / 开赛前复验 | PASS（manifest、安装 sha256sum、DB、每局 verification、开赛前复验均为 `9b1d394d…`） |
| 5 | HTTPS / 域名 / 反向代理 / 防火墙 | PASS（证书 2026-06-09→09-07；blog 不受影响 200；ufw 22/80/443） |
| 6 | 短赛：正常完成 / force-cancel / worker 重启恢复 / artifact 属主权限 / 无残留进程 / 三档时间控制 | PASS（详见第 3 段） |

## 3. Smoke tournament evidence

### 生命周期（bullet_1_0，pairs=4）

| 场景 | tournament | 结果 |
|---|---|---|
| 正常完成 | f73be112-058d-4e50-8945-21862ed5f208 | COMPLETED 4/4 pairs；verification verified=true、moves_legal=true、rc=0；candidate 2W/5L/1D=31.25% |
| force-cancel | b5fc4a16-12a5-4677-8aa4-b34eaa9b793d | CANCELLED；pair0 INTERRUPTED（reason=force-cancelled）；无残留进程 |
| worker 重启恢复 | 96f14997-d477-4c54-a503-6812fa03df14 | COMPLETED；pair0 attempt 2 重跑成功（attempt1 记录 worker shutdown），rc=0 |

DB 终态证据：pairs COMPLETED，attempt/return_code/return_code_attempt 齐全（如 96f14997 pair0: attempt=2 rc=0 rc_attempt=2）。
Artifact 路径：`/var/lib/chessarena/runs/<tid>/{combined.pgn,artifact-manifest.json,summary.json,pairs/NNNNNN/attempt-NN/{command.json,command.txt,match.pgn,opening.epd,stderr.log,stdout.log,verification.json}}`；全部 chessarena:chessarena，目录 755 / 文件 644。PGN 含 FEN header、TimeControl、引擎评分注释。
进程终态：各场景后均无 cutechess/engine 残留。

### 三档时间控制（各 2 pairs，短 smoke）

| 时间控制 | tournament | 结果 |
|---|---|---|
| 1+0（bullet_1_0） | f73be112…（pairs=4） | COMPLETED（4/4） |
| 3+2（blitz_3_2） | d2961609-8482-48d4-9049-788a4e6753ac | COMPLETED（2/2，1W/1L/2D） |
| 5+3（rapid_5_3） | c09f9a4a-f0db-41f9-9768-cd17d1c16e12 | COMPLETED（2/2，3W/1L/0D） |

三档 DB 终态一致：pairs COMPLETED、attempt=1、return_code=0、return_code_attempt=1；每对 7 个 artifact 齐全；verification verified=true / moves_legal=true / rc=0；结束后无残留进程。

### deploy-arena 受控验收（run 31108376281，head 96b4963，workflow_dispatch）

```text
clean venv install + tests:  PASS
test-and-deploy（含 package/upload）: PASS
release-install:              PASS
migration:                    0004 (head)
current switch:              /opt/chessarena/releases/20260806215940
services restart:            active/active
health（结构化 JSON 解析）:   四字段全 ok
workflow conclusion:         success；rollback 未触发
新 release DEPLOY_SOURCE_SHA: 96b4963107e7e9f57838faaf1dd21063b36a310d（精确匹配）
engine/v2-openings/tournaments/artifacts: 完整未重置
外部 HTTPS/Basic Auth/CSRF:   401/200/错误 Origin 403
```

失败 run 分类记录（31091490998，b56a1bb）：
```text
release installation: PASS · migration: PASS · release switch: PASS
service restart: PASS · actual API health: PASS
workflow health evaluation: FALSE NEGATIVE（grep 与紧凑 JSON 不匹配）
rollback execution: PASS · final workflow result: FAIL · acceptance status: NOT COMPLETE
```
（positive 证据：主部署路径与回滚路径均正确执行；已由后续成功 run 闭合。）

## 4. Document-external deviations and resolutions

| 问题 | 根因 | 修复 | commit |
|---|---|---|---|
| API unit 启动失败 | unit 用 `chessarena.main:app`，模块只有 `create_app()` | ExecStart 改 factory 形式 + 回归测试 | 431571c |
| engine 部署 workflow 无法构建 | `dtolnay/rust-toolchain@stable` clippy 漂移 | 钉死 1.94.1 | 11d53a5 |
| scp 上传空归档 | Docker action 看不到 runner host /tmp | 归档写 workspace 相对路径 + 校验 | 51a629f |
| verify_install 数据库探针失败 | 裸字符串 SQL 不兼容 SQLAlchemy 2.x | `text("SELECT 1")` + 回归测试 | 15dde7c |
| clean-install 测试失败 | fake cutechess 经 `env python3` 用了无 chess 的外层解释器 | venv bin 加入 PATH + 回归测试 | 28a8d52 |
| wrapper pip PermissionError | sudo 调用继承 cwd=/home/deploy(750)，pip sys.path[0] 扫描失败 | wrapper 归一化 cwd（cd /opt/chessarena；release-install 内 cd "$dest"）+ 回归测试 | b56a1bb |
| health gate 误判回滚 | grep `"status": "ok"` 不匹配紧凑 JSON | python3 json.load 结构化解析 + 回归测试 | c4668df |
| deploy-arena 丢失 workflow_dispatch | 嵌入 python 块列 0 缩进使 YAML 无效 | block scalar 正确缩进 + 测试（dedent 提取 + 缩进守卫） | 96b4963 |
| EPD opening mismatch | 注册行 6 字段 FEN，Cute Chess 只认 4 字段 | 重建 v2-openings（4 字段 EPD），v1 停用 | 服务器侧（无 repo 变更） |

v1-openings: disabled, invalid six-field input for Cute Chess EPD.
v2-openings: enabled, canonical four-field EPD, smoke-validated.

## 5. Security handoff and remaining authorization

### 密钥收尾（deploy-arena 验收成功后完成，2026-08-06）

```text
旧 GitHub 部署 key（chessarena-deploy）：指纹 SHA256:lmKoxIPX9NCxr2RALISZrUkiqofaH44jr5jdWfY9J2s
  → 已从 deploy 用户 authorized_keys 删除（撤销时间约 14:0x CST）
新本地主导部署 key（chessarena-deploy-local）：指纹 SHA256:yUWecZLz/s4QfL+CCaLksHjoRFOfEachp3696aCkFuE
  → 唯一保留在 authorized_keys（共 1 把）
GitHub repository secrets：ARENA_DEPLOY_KEY / ARENA_DEPLOY_HOST / ARENA_DEPLOY_USER / ARENA_SERVER_HOST_KEY 已全部删除（0 个 ARENA secrets 剩余）
验证：新 key 可登录且可读 env / 可写 incoming；旧 key 已被拒（Permission denied）
```

`deploy-arena.yml` / `deploy-engine-build.yml` 仍保留为已验证实现，但 GitHub 侧无服务器私钥 secret，不再构成有效远程入口。

### 说明与遗留

- 四个失败/复现产物 release（`20260806174010`、`20260806174313`、`20260806180327`、`20260806180524`）已在最终裁决后按完整路径删除。
- `20260806014113`：**历史无效 release、非 current、非正式 rollback target**（其 API unit 使用错误的 `chessarena.main:app` 目标，无法启动），仅作为最早失败的审计残留保留，**不可用作回滚**。当前唯一正式回滚目标是 `20260806020824`。
- 一个 DRAFT tournament `3d996bfc…`（CSRF/origin 验证副产品，未启动）保留为审计记录；不得用 SQLite 手工删除，待应用层 archive/hide 功能可用后隐藏。
- `deploy-engine-build` 工作流（含 workspace 归档修复）在 51a629f 已实测过 upload 成功；其 SSH 部署路径在 15:02 的 51a629f run 已安装构建文件（字节一致），随后本地主导部署路径负责构建产物交付。
- public-production 已授权（当前 Basic Auth 模型下）；长赛、Elo/SPRT、引擎晋级与解除 Basic Auth 需另行预声明与授权。

### 归档说明

- 本文档为证据记录，包含服务器 IP、目录结构、用户名与密钥指纹等运维信息（非直接凭据），**不建议提交到公开仓库**。
- 归档建议：本地只读归档一份 + 加密备份/受控私有存储一份，归档时记录文件 SHA-256 以校验完整性。

### Remaining authorization

```text
FINAL VERDICT:                    APPROVED
S4 ARENA V1:                      PRODUCTION READY
CURRENT APPLICATION SHA:          96b4963107e7e9f57838faaf1dd21063b36a310d
CURRENT RELEASE:                  20260806215940
ROLLBACK RELEASE:                 20260806020824
ENGINE BUILD:                     20260806-51a629f-linux-x86_64
ENGINE PROFILE:                   current-final
OPENING SET:                      v2-openings
DEPLOYMENT MODEL:                 LOCAL-LED
GITHUB SERVER SSH ACCESS:         REVOKED
FAILED RELEASE CLEANUP:           COMPLETE
DRAFT TOURNAMENT:                 RETAINED
TLS RENEWAL DRY-RUN:              PASS
LONG TOURNAMENT:                  NOT AUTHORIZED
```

---

## P4.1 — Public Replay 生产部署记录（2026-08-07）

### 部署
```text
CI（run 31124533022 rerun，事故恢复后）: success（两 job）
release:          /opt/chessarena/releases/20260807161440
current:          /opt/chessarena/releases/20260807161440
DEPLOY_SOURCE_SHA: 7b064e11a062642166cbbab84ac873a430cae474（精确匹配）
migration:        0004 (head)
services:         active/active
health:           四字段 ok
```
部署模型：本地主导部署（打包 main `7b064e1` 的 arena/ → scp → `arena-deploy release-install` + `release-switch` + restart）；GitHub Actions 部署工作流保持禁用。

### Nginx 鉴权拆分
```text
location /chessarena/admin/   → Basic Auth（.htpasswd）
location /chessarena/api/v1/  → Basic Auth
location /chessarena/static/  → 公开（样式 + 自托管 htmx/pgn-viewer）
location /chessarena/         → 公开（首页 /matches/ /games/ /public-api/v1）
nginx -t: PASS · reload: OK · 旧 snippet 已备份（chessarena.conf.bak-*）
```

### 验收矩阵（生产，全部通过）
```text
匿名 GET /chessarena/                    200
匿名 GET /chessarena/matches/            200
匿名 GET /public-api/v1/matches           200（仅 COMPLETED，4 个，DRAFT 3d996bfc 不出现）
匿名 GET /public-api/v1/matches/<id>      200（8 局 verified games）
匿名 GET /games/<gid>                     200（PGN viewer 回放页）
匿名 GET /public-api/v1/games/<gid>/pgn   200（application/x-chess-pgn，game-<n>.pgn）
匿名 GET /chessarena/admin/              401
匿名 GET /chessarena/api/v1/health        401
匿名 GET /chessarena/api/v1/tournaments   401
匿名 POST /chessarena/api/v1/tournaments  401
带 Basic Auth /admin/                     200
非法 tournament/game ID                   404
公开响应内部字段泄漏                      无（binary_sha256/config_snapshot/pgn_path/run_root/manifest 均未出现）
blog https://pearllover.site/             200
PGN 引擎名可自解释                       [White "EngineA"]/[Black "EngineB"]（旧 tournament 审计名）
```

---

## P4.2 — Engine Presets + Stockfish 接入（CLOSED，2026-08-07）

### 部署链
```text
PR #1 merged → main e6b9549（merge commit）
hotfix 1 (Threads -each) → fc1f113（合并进 main）
hotfix 2 (verifier command_args=[]) → d8da67c（合并进 main）
releases: 20260807164358 → 20260807171724 → 20260807174210（current）
CURRENT APP SOURCE: d8da67c354932562314327f511f75779be4fc02a
CURRENT APP RELEASE: 20260807174210
```

### 两个生产 bug（已修）
1. `-each option.Threads=1`：ChessEngine 未声明 Threads option → cutechess warning → verifier stderr 拒绝。修复：`-each` 只保留 Hash；Threads 不强制（双方默认 1 线程）。→ 合同：`-each tc + Hash`，各引擎专属 options 在自己的 `-engine` block。
2. verifier 重建 `snap.get("command_args") or [...]`：`[]`（Stockfish 无启动参数，合法冻结值）被误判为字段缺失 → 重建出 `--profile`。修复：key 存在即原样使用（含空列表），仅 key 缺失才 legacy fallback，与 scheduler 一致。

### Stockfish 18 接入
```text
Version: 18 | Upstream release commit: cb3d4ee | Binary: Linux x86-64 AVX2
UCI identity: Stockfish 18 | UCI_Elo range: 1320..3190
asset: stockfish-ubuntu-x86-64-avx2.tar SHA 536c0c2c…5417964
binary: SHA 6b087694…455b9f9 | GPLv3 COPYING 随二进制保留
presets: stockfish-limited-1800/2000/2200/2400（UCI_LimitStrength + UCI_Elo）
```

### 四档 smoke（ChessEngine Production vs Stockfish Limited N，3+2，2 pairs）
```text
1800 4-0 | 2000 0-4 | 2200 1-3 | 2400 2-2
```
**重要（修正）**：比分仅证明"配置验证"（正确 UCI_Elo + UCI_LimitStrength + 真实对局 + verifier + PGN/artifacts 全通），**不构成经验强度梯度结论**——每档仅 4 局，且 `UCI_LimitStrength` 随机化次优着法，小样本波动大。经验强度顺序需更大样本（正式的长期 ladder）才可宣称。

### 验收（全 PASS）
```text
8 pairs: COMPLETED, attempt=1, rc=0, rc_attempt=1, verified, moves_legal
artifacts 齐全 | 无残留进程 | PGN 引擎名自解释 | 公开回放 200
旧 tournament preset_id 保持 NULL 兼容 | 服务 active | health ok
```

### 非阻断技术债（记录，不重开 P4.2）
- `snapshot.threads` 现仅为 metadata（实际依赖双方默认单线程）。未来扩展通用 UCI engine 时：每个 build 记录 UCI capabilities → 支持 Threads 的引擎在其 `-engine` block 设置 `Threads=1`，不支持的引擎不发送，使"单线程是 Arena 合同"而非"碰巧默认一致"。

---

## P4.1 Replay Repair（2026-08-08）

### Bug
`public_game.html` 传给 LichessPgnViewer 的是 CSS selector 字符串 `"#board"`，而官方 API 要求 HTMLElement（它执行 `element.innerHTML = ''`）。页面 200 + JS 加载均正常，但初始化在运行时失败 → 棋盘空白。此前"公开游戏页 200"验收未执行 JS，无法捕获。

### 修复（main `7b03386`，release `20260808014735`）
- 改传 `document.getElementById("board")`，含 null 守卫 + 初始化异常内联可见 fallback。
- 新增 Playwright 浏览器 E2E（`test_browser_replay.py`）：启动真实 uvicorn，打开 verified game，断言 lpv viewer 挂载（chessground + move buttons）、点击 next 位置前进、console 无 error。
- Playwright + Chromium 集成进 CI arena pytest job。HTTP 200 不再作为前端功能验收。

### 生产验证（真实 smoke3 game，Playwright 访问 pearllover.site）
```text
game: /chessarena/games/becf7ced-d962-401d-aa71-f22faf4eb2e1
HTTP 200 | .lpv__board 挂载 | chessground 棋盘 | moves=341 | next 按钮存在
prev 初始 disabled → 点击 next 后 enabled（位置前进）
console errors: NONE
```

---

## Admin dashboard active-tournament 500 hotfix（2026-08-08）

### Bug
`/admin/` 在 worker 有 RUNNING tournament 时 HTTP 500：dashboard include `_tournament_status.html`（需要 `tournament` + `pairs`），但路由只传了 `active` → Jinja UndefinedError。idle 时 `active=None` 不 include → 正常。另外 `hx-get` 属性在 `{% if active %}` 外，idle 时生成 `/admin/tournaments//status` 每 5s 无效轮询。

### 修复（main `6bf3655`，release `20260808145418`）
- dashboard 路由补齐 contract：`tournament=active`、`pairs=sorted(active.pair_jobs)`、`score_percent`。
- dashboard.html：`hx-get/hx-trigger/hx-swap` 移入 `{% if active %}`，idle 无轮询目标。
- 回归：`test_admin_dashboard_renders_active_tournament`（RUNNING + WorkerState → /admin/ 200 + Running match + name + Current pair）；`test_admin_dashboard_idle_has_no_poll_target`（idle 200 + 无双斜杠 poll）。

### 生产验收（release 20260808145418，DEPLOY_SOURCE_SHA=6bf3655…）
```text
idle:      /admin/ 200 · "No match currently running" · 无 /tournaments//status poll
active:    启动 bullet smoke → RUNNING → /admin/ 200（修复前 500）
           "Running match" + tournament name + "Current pair:" 均渲染
HTMX 5s:   /admin/tournaments/<id>/status → 200
清理:      force-cancel → CANCELLED · health 全 ok · active_tournament_id=null
```

---

## P4.UI-1 生产接入修复（2026-08-08，main `d41ee7b`，release `20260808150246`）

P4.UI-1 demo 生产验证发现并修复 2 个前端 bug：

1. **SAN 重解析歧义**（`Invalid move: Bxc6`）：导航用 `move(san)` 重放真实对局 PGN 失败。修复：改用 verbose history 的 from/to/promotion 方格应用走法（无歧义）。E2E 换用含吃子的 PGN（1.e4 d5 2.exd5 Qxd5 3.Nc3 Qd8）断言 ply FEN。
2. **FEN header 起始局面**：开局库产生的 PGN 带 `[FEN]` header（非初始局面），导航从标准阵列重放错位。修复：解析 `[FEN]` header，导航从该局面初始化。新增 E2E（FEN-header PGN 断言初始局面 = FEN header + ply 导航）。

### 生产验收（release 20260808150246，DEPLOY_SOURCE_SHA=d41ee7b…）
```text
真实 Stockfish 2200 对局（68 ply，带 FEN header）：
HTTP 200 · 初始局面 = [FEN] header · moves 68
ply 点击导航生效 · last 68/68 · first 回到初始
console/pageerror: NONE（此前 Invalid move bug 已消除）
```
