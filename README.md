# dsv-mcp

DeepSeek 网页版识图模式（Vision）的 MCP 服务器。通过逆向网页版协议，把 DeepSeek 识图能力暴露为一个 MCP 工具，支持基于视觉原语（bounding box / point）的思考模式。

> **重要免责声明**
> 
> 本仓库仅供学习、研究、个人实验和内部验证使用，不提供任何形式的商业授权、适用性保证或结果保证。
> 
> 本项目通过逆向 DeepSeek 网页版接口实现，使用过程中存在账号被风控、封禁的风险。作者及仓库维护者不对因使用、修改、分发、部署或依赖本项目而产生的任何直接或间接损失、账号封禁、数据丢失、法律风险或第三方索赔负责。
> 
> 请勿将本项目用于违反服务条款、协议、法律法规或平台规则的场景。

## 目录

- [dsv-mcp](#dsv-mcp)
  - [目录](#目录)
  - [核心能力](#核心能力)
  - [快速开始](#快速开始)
  - [HTTP 部署](#http-部署)
  - [配置到 Codex](#配置到-codex)
  - [配置说明](#配置说明)
    - [accounts](#accounts)
    - [proxy](#proxy)
    - [server](#server)
    - [auto\_delete](#auto_delete)
  - [思考模式](#思考模式)
  - [测试](#测试)
  - [目录结构](#目录结构)
  - [免责声明](#免责声明)
  - [参考项目](#参考项目)

## 核心能力

| 能力          | 说明                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| 识图 MCP 工具 | `describe_image`：本地图片路径 + 可选问题 + 可选思考模式                                             |
| 三种思考模式  | `grounding`（bbox 锚定对象）/ `pointing`（点坐标锚定）/ `none`（无模式提示词）                       |
| 视觉原语提取  | grounding 自动从思考链提取 `<｜｜ref｜｜>obj｜｜/ref｜｜><｜｜box｜｜>[[x1,y1,x2,y1]]<｜｜/box｜｜>` |
| 多账号轮询    | 账号 round-robin 调度、token 缓存复用、上传风控冷却、验证码挑战检测（30 分钟冷却）                   |
| 会话管理      | 自动清理可配置（`none` / `single` / `all`），删除失败自动入队补删                                    |
| 代理三模式    | `none` 直连 / `manual` 显式代理 / `managed` 自动下载 mihomo 内核 + 订阅转配置                        |
| PoW           | wasmtime 运行官方 wasm 求解 DeepSeekHashV1，challenge 严格一次性                                     |

## 快速开始

```bash
pip install -e .
```

复制示例配置并填写账号：

```bash
cp config.example.json config.json
```

`config.json` 包含真实凭据，已在 `.gitignore` 中忽略、不会提交；仓库内只保留 `config.example.json` 占位示例。

启动 MCP streamable HTTP 服务器：

```bash
dsv-mcp --host 127.0.0.1 --port 8765
```

默认读取项目根目录（与 `src/` 同级）的 `config.json`；也可显式传入其他路径：`dsv-mcp other.json`。

然后在支持 MCP 的客户端中以 `http://127.0.0.1:8765/mcp` 接入即可，工具名为 `describe_image`。

## HTTP 部署

服务器以 streamable HTTP 单实例常驻运行，所有客户端共享同一账号池：

```bash
dsv-mcp config.json --host 127.0.0.1 --port 8765
```

- `--host` / `--port`：监听地址与端口（默认 `127.0.0.1:8765`）
- `--token <secret>`：可选；配置后所有请求需带 `Authorization: Bearer <secret>`，否则返回 401。多客户端部署建议设置
- MCP 端点路径固定为 `/mcp`，客户端按 `http://<host>:<port>/mcp` 接入

## 配置到 Codex

在 `~/.codex/config.toml` 里加一段（路径换成你自己的）：

```toml
[mcp_servers.dsv]
command = '${your_path}\.venv\Scripts\python.exe'
args = ['-m', 'dsv_mcp', '${your_path}\dsv-mcp\config.json', '--autostart']
tool_timeout_sec = 900
startup_timeout_sec = 180
default_tools_approval_mode = "auto"
```

说明：

- 用 `python.exe -m dsv_mcp` 而不是 `dsv-mcp.exe`：Codex 在 Windows 上直接 spawn
  console script 可执行文件不稳定，`python -m` 最稳
- `--autostart`：检测到共享 HTTP 实例没跑时自动后台拉起（默认 `127.0.0.1:8765`），
  多个 Codex 会话共享同一账号池；实例空闲 10 分钟自动退出，不留孤儿进程
- `startup_timeout_sec`：Codex 对 MCP 服务器默认只有 10 秒启动预算，调大避免冷启动被掐
- Windows 下 Codex 只给子进程传最小环境变量，本项目已在入口自修复
  （`SYSTEMROOT` / `PROCESSOR_ARCHITECTURE`），无需额外配置 `env_vars`
- 监听地址 / 端口 / 鉴权 token 也可写在 `config.json` 的 `server` 段
  （`--host` / `--port` / `--token` 命令行参数优先）
- `tool_timeout_sec`：Codex 调用工具的总体超时。深度思考阶段可能数分钟不返回
  数据，300 秒容易被掐，建议 900 秒
- 改完配置后需**新开** Codex 会话才生效

## 配置说明

### accounts

账号列表，支持邮箱或手机号登录，可配置多个账号用于轮询：

```json
{
  "accounts": [
    { "email": "a@example.com", "password": "pass1" },
    { "mobile": "+8613800138000", "password": "pass2" }
  ]
}
```

`banned` 字段（可选，默认 `false`）：账号被上游永久停用后自动标记为 `true` 并写回配置文件，停用账号不参与调度；若之后登录成功则自动清除标记。

### proxy

代理三模式：

```json
{
  "proxy": {
    "mode": "none"
  }
}
```

- `none`：直连
- `manual`：使用已有代理，需配置 `url`，如 `socks5://127.0.0.1:1080`
- `managed`：自动调度 mihomo 内核，需配置 `managed_subscription`（vless 订阅 URL）与 `managed_node`（订阅中节点序号，从 0 开始）

managed 示例：

```json
{
  "proxy": {
    "mode": "managed",
    "managed_subscription": "https://example.com/sub?token=xxx",
    "managed_node": 0
  }
}
```

mihomo 内核与生成的配置缓存在 `%LOCALAPPDATA%/dsv-mcp/managed`，不污染项目目录。

### server

HTTP 服务监听配置，命令行参数未指定时从这里取：

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    "token": ""
  }
}
```

- `host` / `port`：监听地址与端口（默认 `127.0.0.1:8765`），`--host` / `--port` 参数优先
- `token`：可选；配置后所有请求需带 `Authorization: Bearer <token>`，否则返回 401，
  `--token` 参数优先

### auto\_delete

识图结束后的会话清理策略（默认 `single`）：

```json
{
  "auto_delete": {
    "mode": "single"
  }
}
```

- `none`：不清理，会话保留在 DeepSeek 聊天记录中
- `single`：只删除本次识图创建的会话（用后即删）
- `all`：清空该账号全部历史会话（调 DeepSeek `chat_session/delete_all` 接口）

删除失败会自动入队，下次调用时补删，直到成功。

## 思考模式

`thinking_style` 参数（默认 `grounding`）：

- `grounding`：在问题前附加 `[Think with Grounding]` 标题，引导模型思考时用边界框锚定对象；返回时从思考链提取 `<|ref|>对象<|/ref|><|box|>[[x1,y1,x2,y2],...]<|/box|>` 行追加到文本后
- `pointing`：附加 `[Think with Pointing]` 标题，引导模型用点坐标锚定位置（适合轨迹/空间推理）；思考链出现 point 标记时返回完整思考链
- `none`：不加模式提示词，只返回最终文本

坐标均为 0-999 归一化整数（与图片像素无关），画框时按
`x * width / 999`、`y * height / 999` 换算为像素坐标。

## 测试

```bash
pytest tests
```

测试全部离线：协议解析、PoW 求解（官方 wasm 向量）、mihomo 配置生成、账号调度与冷却逻辑，以及录制自真实调用的 SSE case 重放（`tests/cases/`）。

## 目录结构

```
src/dsv_mcp/
  server.py     MCP 服务器装配、模式返回逻辑、视觉原语提取
  client.py     DeepSeek 网页版客户端（登录/会话/上传/识图/PoW/风控检测）
  protocol.py   协议常量与请求头
  http.py       HTTP 层（curl_cffi 浏览器伪装）
  pow.py        PoW 求解（wasmtime + 官方 wasm）
  sse.py        SSE 流解析（正文/思考分离、续写信号）
  mihomo.py     订阅解析、mihomo 内核下载与配置生成
  proxy.py      代理管理（三模式 + Job Object 防孤儿）
  config.py     配置模型
tests/          单元测试与录制 case
```

## 免责声明

本项目仅供学习与研究，不保证可用性与稳定性。使用本项目即表示你已知晓并同意承担所有风险，包括但不限于账号风控与封禁。请合理、节制地使用，遵守 DeepSeek 的服务条款与相关法律法规。

## 参考项目

本项目的协议实现参考了 [CJackHwang/ds2api](https://github.com/CJackHwang/ds2api)（AGPL-3.0），感谢原作者与社区的工作。