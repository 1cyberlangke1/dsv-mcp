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

- [核心能力](#核心能力)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [思考模式](#思考模式)
- [代理模式](#代理模式)
- [测试](#测试)
- [目录结构](#目录结构)

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 识图 MCP 工具 | `dsv_describe_image`：本地图片路径 + 可选问题 + 可选思考模式 |
| 三种思考模式 | `grounding`（bbox 锚定对象）/ `pointing`（点坐标锚定）/ `none`（无模式提示词） |
| 视觉原语提取 | grounding 自动从思考链提取 \\`<<｜｜ref｜｜>>obj｜｜/ref｜｜>><<｜｜box｜｜>>[[x1,y1,x2,y1]]<<｜｜/box｜｜>>\\` |
| 多账号轮询 | 账号 round-robin 调度、token 缓存复用、上传风控冷却、验证码挑战检测（30 分钟冷却） |
| 会话管理 | 单轮会话自动清理，删除失败自动入队补删 |
| 代理三模式 | `none` 直连 / `manual` 显式代理 / `managed` 自动下载 mihomo 内核 + 订阅转配置 |
| PoW | wasmtime 运行官方 wasm 求解 DeepSeekHashV1，challenge 严格一次性 |

## 快速开始

```bash
pip install -e .
```

准备配置文件 `config.json`：

```json
{
  "accounts": [
    { "email": "your@email.com", "password": "your_password" }
  ],
  "proxy": {
    "mode": "none"
  }
}
```

启动 MCP stdio 服务器：

```bash
dsv-mcp config.json
```

然后在支持 MCP 的客户端中接入即可，工具名为 `dsv_describe_image`。

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

## 思考模式

`thinking_style` 参数（默认 `grounding`）：

- `grounding`：在问题前附加 `[Think with Grounding]` 标题，引导模型思考时用边界框锚定对象；返回时从思考链提取 `<|ref|>对象<|/ref|><|box|>[[x1,y1,x2,y2],...]<|/box|>` 行追加到文本后
- `pointing`：附加 `[Think with Pointing]` 标题，引导模型用点坐标锚定位置（适合轨迹/空间推理）；思考链出现 point 标记时返回完整思考链
- `none`：不加模式提示词，只返回最终文本

坐标均为 0-999 归一化整数，可用 `dsv_mcp.server.denormalize(coord, width, height)` 换算为像素坐标。

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