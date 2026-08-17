"""MCP 服务器装配：describe_image 工具（mcp>=2.0 MCPServer）。"""

from __future__ import annotations

import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer

from dsv_mcp.client import DeepSeekClient, DeepSeekError
from dsv_mcp.config import DsvConfig
from dsv_mcp.http import HttpClient
from dsv_mcp.proxy import ProxyError, ProxyManager


MAX_IMAGE_EDGE = 1024
# 项目根目录（与 src/ 同级），默认配置放这里
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# 上传被风控（40301）后账号的冷却时长（秒）
UPLOAD_COOLDOWN = 300.0
# 验证码/风控挑战后账号的冷却时长（秒）
CAPTCHA_COOLDOWN = 1800.0
# 模式标题：提示词 = 标题 + 换行 + 用户问题
GROUNDING_TITLE = "[Think with Grounding]"
POINTING_TITLE = "[Think with Pointing]"
THINKING_STYLES = ("grounding", "pointing", "none")
# autostart 模式的共享 HTTP 实例默认参数
AUTOSTART_HOST = "127.0.0.1"
AUTOSTART_PORT = 8765
AUTOSTART_PATH = "/mcp"
TOKEN_ENV = "DSV_MCP_TOKEN"
# 共享 HTTP 实例启动等待上限：冷启动可能被杀软首扫拖慢，太短会导致 Codex 握手失败
HTTP_STARTUP_TIMEOUT = 90.0
HTTP_TOOL_TIMEOUT = 300.0
# HTTP 实例空闲多久（秒）后自动退出，避免 Codex 关闭后残留孤儿进程
IDLE_SHUTDOWN_SECONDS = 600.0
# 最近一次 HTTP 请求时间（monotonic），供空闲退出判断
_LAST_ACTIVE: dict[str, float] = {"t": 0.0}
# 后台拉起 HTTP 实例失败的错误记录（url -> 错误），供工具调用前检查
_LAUNCH_FAILED: dict[str, str] = {}


def _child_interpreter() -> str:
    """返回子进程应使用的解释器路径。

    Windows 上 venv 的 python.exe 是 launcher，sys.executable 可能被解析成
    base 解释器路径（base 环境没有 dsv_mcp 包，子进程会启动失败）。显式用
    sys.prefix 下的解释器，保证子进程与当前进程同环境；非 venv 回退
    sys.executable。
    """
    if sys.prefix != sys.base_prefix:
        exe_dir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
        exe = exe_dir / ("python.exe" if os.name == "nt" else "python")
        if exe.exists():
            return str(exe)
    return sys.executable


def _normalize_primitives(text: str) -> str:
    """把思考链里的视觉原语标记归一化为标准形态。

    实测输出：竖线为每边两个全角 U+FF5C（如 <｜｜ref｜｜>）。
    归一化后为 <|ref|>，便于解析。
    """
    text = text.replace("\uff5c", "|")
    return re.sub(r"<\|+\s*(/?ref|/?box|/?point)\s*\|+>", r"<|\1|>", text)


def extract_groundings(thinking: str) -> list[dict]:
    """从思考链提取 ref+box 配对，返回 [{ref, boxes}]。"""
    norm = _normalize_primitives(thinking)
    results: list[dict] = []
    pattern = re.compile(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|box\|>(.*?)<\|/box\|>", re.S
    )
    for m in pattern.finditer(norm):
        ref = re.sub(r"\s+", " ", m.group(1)).strip()
        try:
            boxes = json.loads(m.group(2))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(boxes, list) and boxes:
            results.append({"ref": ref, "boxes": boxes})
    return results


def has_point_primitive(thinking: str) -> bool:
    """思考链里是否出现 point 标记。"""
    norm = _normalize_primitives(thinking)
    return bool(re.search(r"<\|point\|>", norm))


def _format_groundings(groundings: list[dict]) -> str:
    """把 ref+box 配对格式化为标准标记行（标记临时转义构造）。"""
    pipe = "\x7c"  # ASCII 半角竖线，转义构造避免手写
    tag = lambda name: f"<{pipe}{name}{pipe}>"
    lines = []
    for g in groundings:
        boxes = json.dumps(g["boxes"], ensure_ascii=False, separators=(",", ":"))
        lines.append(
            f"{tag('ref')}{g['ref']}{tag('/ref')}{tag('box')}{boxes}{tag('/box')}"
        )
    return "\n".join(lines)


def compress_image(image_bytes: bytes, max_edge: int = MAX_IMAGE_EDGE) -> tuple[bytes, str]:
    """压缩图片到 max_edge 边长内（Pillow 懒加载），返回 (bytes, content_type)。"""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.thumbnail((max_edge, max_edge))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
        return out.getvalue(), "image/jpeg"


class DsvServer:
    """MCP 服务器：账号轮询 + 识图调用。"""

    def __init__(self, config_path: str | Path):
        self._config_path = str(config_path)
        self.config = DsvConfig.load(config_path)
        if not self.config.accounts:
            raise ValueError("config 中至少需要一个账号")
        self.proxy = ProxyManager(self.config.proxy)
        try:
            proxy_url = self.proxy.proxy_url()
        except ProxyError as exc:
            self.proxy.close()
            raise ValueError(f"代理启动失败: {exc}") from exc
        self.http = HttpClient(proxy=proxy_url)
        self.client = DeepSeekClient(self.http)
        self._tokens: dict[str, str] = {}
        self._busy: dict[str, bool] = {}
        self._cooldown: dict[str, float] = {}
        self._order: list[str] = []
        self._rr = -1  # round-robin 游标
        for acc in self.config.accounts:
            ident = acc.identifier()
            self._busy[ident] = False
            self._order.append(ident)

    def close(self) -> None:
        self.http.close()
        self.proxy.close()

    def _acquire(self) -> tuple[str, object]:
        now = time.monotonic()
        for _ in range(len(self._order)):
            self._rr += 1
            ident = self._order[self._rr % len(self._order)]
            account = next(a for a in self.config.accounts if a.identifier() == ident)
            if account.banned:
                continue
            if not self._busy[ident] and now >= self._cooldown.get(ident, 0.0):
                self._busy[ident] = True
                return ident, account
        raise DeepSeekError("rate_limited", "所有账号忙或冷却中，请稍后重试")

    def _token_for(self, ident: str) -> str:
        """取账号 token：内存缓存 → 磁盘缓存 → 登录，成功后写盘复用。"""
        token = self._tokens.get(ident, "")
        if token:
            return token
        disk = self.config.tokens.get(ident, "")
        if disk:
            self._tokens[ident] = disk
            return disk
        account = next(a for a in self.config.accounts if a.identifier() == ident)
        token = self.client.login(account)
        if account.banned:
            # 登录成功说明停用已解除，清掉持久化标记
            account.banned = False
        self._tokens[ident] = token
        self.config.tokens[ident] = token
        self.config.save(self._config_path)
        return token

    def describe_image(
        self,
        image_path: str,
        question: str = "请详细描述这张图片的内容。",
        thinking_style: str = "grounding",
    ) -> str:
        """识图主流程：取账号 → 登录态 → 压缩 → 单轮识图 → 语义化返回。"""
        if thinking_style not in THINKING_STYLES:
            return f"无效 thinking_style: {thinking_style}（可选 grounding/pointing/none）"
        image_bytes = Path(image_path).read_bytes()
        data, content_type = compress_image(image_bytes)
        try:
            ident, account = self._acquire()
        except DeepSeekError as exc:
            return f"账号不可用: {exc.code} {exc}"
        try:
            token = self._token_for(ident)
            if thinking_style == "grounding":
                prompt = f"{GROUNDING_TITLE}\n{question}"
            elif thinking_style == "pointing":
                prompt = f"{POINTING_TITLE}\n{question}"
            else:
                prompt = question
            result = self.client.describe_image(
                account,
                token,
                data,
                prompt=prompt,
                filename="image.jpg",
                content_type=content_type,
                thinking_enabled=True,
                auto_delete=self.config.auto_delete.mode,
            )
            text = result["text"]
            thinking = result["thinking"]
            if thinking_style == "grounding":
                groundings = extract_groundings(thinking)
                if groundings:
                    return text + "\n\n" + _format_groundings(groundings)
                return text
            if thinking_style == "pointing" and thinking and has_point_primitive(thinking):
                return _normalize_primitives(thinking)
            return text
        except DeepSeekError as exc:
            if exc.code == "auth_failed":
                self._tokens.pop(ident, None)
                self.config.tokens.pop(ident, None)
                self.config.save(self._config_path)
            elif exc.code == "account_banned":
                self._mark_banned(ident)
            elif exc.code == "upload_rate_limited":
                self._cooldown[ident] = time.monotonic() + UPLOAD_COOLDOWN
            elif exc.code == "captcha_required":
                self._cooldown[ident] = time.monotonic() + CAPTCHA_COOLDOWN
            elif exc.code == "account_muted":
                self._cooldown[ident] = time.monotonic() + self._muted_seconds(exc)
            return f"识图失败: {exc.code} {exc}"
        except Exception as exc:
            return f"识图失败: {type(exc).__name__}: {exc}"
        finally:
            self._busy[ident] = False

    def _muted_seconds(self, exc: DeepSeekError) -> float:
        """account_muted 剩余冷却秒数：有到期时间按差值，未知按验证码冷却兜底。"""
        until = exc.until or 0.0
        if until > 0:
            return max(0.0, until - time.time())
        return CAPTCHA_COOLDOWN

    def _mark_banned(self, ident: str) -> None:
        """账号被永久停用：标记并写回配置文件，调度时直接跳过。"""
        for account in self.config.accounts:
            if account.identifier() == ident:
                account.banned = True
                break
        self.config.save(self._config_path)


def build_mcp(
    server: DsvServer,
    token: str = "",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> MCPServer:
    mcp_kwargs: dict = {"name": "dsv"}
    if token:
        from mcp.server.auth.provider import AccessToken
        from mcp.server.auth.settings import AuthSettings

        class StaticTokenVerifier:
            """SDK TokenVerifier 协议实现：固定 Bearer token 校验。"""

            async def verify_token(self, t: str) -> AccessToken | None:
                if t == token:
                    return AccessToken(token=t, client_id="dsv", scopes=[])
                return None

        mcp_kwargs["auth"] = AuthSettings(
            issuer_url=f"http://{host}:{port}",
            resource_server_url=f"http://{host}:{port}",
        )
        mcp_kwargs["token_verifier"] = StaticTokenVerifier()
    mcp = MCPServer(**mcp_kwargs)

    @mcp.tool()
    def describe_image(
        image_path: str,
        question: str = "请详细描述这张图片的内容。",
        thinking_style: str = "grounding",
    ) -> str:
        """Describe an image via DeepSeek vision mode.

        Args:
            image_path: local image file path.
            question: optional prompt; defaults to asking for a detailed description.
            thinking_style: "grounding" (default) anchors objects with bounding boxes in
                thinking, "pointing" anchors positions with point coordinates,
                "none" adds no mode prompt.

        Returns:
            Plain-text description, plus grounding boxes when available. Coordinates
            are normalized 0-999 values relative to the image, independent of its
            pixel size: scale them by image_width/999 and image_height/999 to draw.
            Grounding boxes are returned as "<|ref|>label<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|>"
            lines; pointing returns the thinking chain containing "<|point|>[[x,y]]<|/point|>".
        """
        return server.describe_image(image_path, question, thinking_style)

    return mcp


def build_http_app(
    server: DsvServer,
    token: str = "",
    host: str = "127.0.0.1",
    port: int = 8765,
):
    """构造 streamable HTTP ASGI app（单实例多客户端共享账号池）。"""
    mcp = build_mcp(server, token=token, host=host, port=port)
    app = mcp.streamable_http_app(streamable_http_path="/mcp", host=host)
    return app


def serve_http(
    server: DsvServer,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str = "",
) -> None:
    """以 streamable HTTP 常驻单实例（多客户端共享账号池），空闲自动退出。"""
    mcp = build_mcp(server, token=token, host=host, port=port)
    app = _touch_middleware(mcp.streamable_http_app(streamable_http_path="/mcp", host=host))
    _LAST_ACTIVE["t"] = time.monotonic()
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    uvicorn_server = uvicorn.Server(config)
    threading.Thread(
        target=_idle_watchdog,
        args=(uvicorn_server, IDLE_SHUTDOWN_SECONDS),
        daemon=True,
    ).start()
    uvicorn_server.run()


def _touch_middleware(app):
    """每个 HTTP 请求刷新 _LAST_ACTIVE，供空闲退出判断。"""

    async def wrapped(scope, receive, send):
        if scope["type"] == "http":
            _LAST_ACTIVE["t"] = time.monotonic()
        await app(scope, receive, send)

    return wrapped


def _idle_watchdog(
    server,
    idle_seconds: float,
    now=time.monotonic,
    sleep=time.sleep,
) -> None:
    """连续 idle_seconds 秒无请求则置 should_exit 让 uvicorn 优雅退出。"""
    while not server.should_exit:
        if now() - _LAST_ACTIVE["t"] >= idle_seconds:
            server.should_exit = True
            return
        sleep(2.0)


def _can_connect(host: str, port: int) -> bool:
    """端口是否已在监听（HTTP 实例是否在跑）。"""
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _ensure_http_server(
    config_path: str,
    host: str,
    port: int,
    token: str,
) -> None:
    """检查共享 HTTP 实例；没在跑则后台拉起（脱离会话常驻），等待就绪。"""
    if _can_connect(host, port):
        return
    log_dir = Path(tempfile.gettempdir())
    # 清理同端口旧日志（旧实例已死才会走到拉起），避免多进程 append 互相污染
    for old in log_dir.glob(f"dsv-mcp-http-{port}-*.log"):
        try:
            old.unlink()
        except OSError:
            pass
    log_path = log_dir / f"dsv-mcp-http-{port}-{os.getpid()}.log"
    cmd = [
        _child_interpreter(),
        "-m",
        "dsv_mcp",
        config_path,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if token:
        cmd += ["--token", token]
    with log_path.open("ab", buffering=0) as log_file:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    deadline = time.monotonic() + HTTP_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if _can_connect(host, port):
            return
        time.sleep(0.1)
    raise RuntimeError(f"HTTP 实例未能在 {host}:{port} 启动，日志: {log_path}")


def _resolve_token(token: str) -> str:
    """CLI 未给 token 时回退环境变量 DSV_MCP_TOKEN。"""
    return token or os.environ.get(TOKEN_ENV, "")


async def _call_http_tool(
    url: str,
    token: str,
    image_path: str,
    question: str,
    thinking_style: str,
    config_path: str = "",
    host: str = "",
    port: int = 0,
) -> str:
    """把工具调用转发给共享 HTTP 实例；实例已死则自动重新拉起（自愈）。"""
    if config_path:
        # 确保实例活着：端口已开则秒回，否则后台拉起
        _LAUNCH_FAILED.pop("err", None)
        await anyio.to_thread.run_sync(
            _ensure_http_server, config_path, host, port, token
        )
    launch_err = await _wait_http_ready(url, HTTP_STARTUP_TIMEOUT)
    if launch_err:
        if config_path:
            # 自愈重试：清掉旧错误再拉起一次，仍失败才报错
            _LAUNCH_FAILED.pop("err", None)
            try:
                await anyio.to_thread.run_sync(
                    _ensure_http_server, config_path, host, port, token
                )
            except Exception as exc:
                raise DeepSeekError(
                    "upstream_unavailable", f"{type(exc).__name__}: {exc}"
                ) from exc
            launch_err = await _wait_http_ready(url, HTTP_STARTUP_TIMEOUT)
        if launch_err:
            raise DeepSeekError("upstream_unavailable", launch_err)
    headers = {"Authorization": f"Bearer {token}"} if token else None
    http_client = httpx2.AsyncClient(headers=headers) if headers else None
    try:
        async with streamable_http_client(url, http_client=http_client) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=HTTP_TOOL_TIMEOUT) as session:
                await session.initialize()
                result = await session.call_tool(
                    "describe_image",
                    {
                        "image_path": image_path,
                        "question": question,
                        "thinking_style": thinking_style,
                    },
                    read_timeout_seconds=HTTP_TOOL_TIMEOUT,
                )
    finally:
        if http_client is not None:
            await http_client.aclose()
    if result.is_error:
        text = "\n".join(
            c.text for c in result.content if getattr(c, "type", None) == "text"
        )
        raise DeepSeekError("upstream_error", text or "工具调用失败")
    return "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")


async def _wait_http_ready(url: str, timeout: float) -> str | None:
    """等待共享 HTTP 实例就绪（后台拉起可能仍在启动），返回 None 或错误信息。"""
    parts = urlsplit(url)
    host = parts.hostname or "127.0.0.1"
    port = parts.port or AUTOSTART_PORT
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        launch_err = _LAUNCH_FAILED.get("err")
        if launch_err:
            return launch_err
        if _can_connect(host, port):
            return None
        await anyio.sleep(0.25)
    return _LAUNCH_FAILED.get("err") or f"HTTP 实例未能在 {host}:{port} 就绪"


def serve_stdio_autostart(
    config_path: str,
    host: str = AUTOSTART_HOST,
    port: int = AUTOSTART_PORT,
    token: str = "",
) -> None:
    """stdio 自动启动代理：后台拉起共享 HTTP 实例，握手不等待，工具调用时转发。"""
    token = _resolve_token(token)

    def _launch_background() -> None:
        try:
            _ensure_http_server(config_path, host, port, token)
        except Exception as exc:
            _LAUNCH_FAILED["err"] = f"{type(exc).__name__}: {exc}"

    threading.Thread(target=_launch_background, daemon=True).start()
    url = f"http://{host}:{port}{AUTOSTART_PATH}"
    mcp = MCPServer(name="dsv")

    @mcp.tool()
    async def describe_image(
        image_path: str,
        question: str = "请详细描述这张图片的内容。",
        thinking_style: str = "grounding",
    ) -> str:
        """Describe an image via DeepSeek vision mode.

        Args:
            image_path: local image file path.
            question: optional prompt; defaults to asking for a detailed description.
            thinking_style: "grounding" (default) anchors objects with bounding boxes in
                thinking, "pointing" anchors positions with point coordinates,
                "none" adds no mode prompt.

        Returns:
            Plain-text description, plus grounding boxes when available. Coordinates
            are normalized 0-999 values relative to the image, independent of its
            pixel size: scale them by image_width/999 and image_height/999 to draw.
            Grounding boxes are returned as "<|ref|>label<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|>"
            lines; pointing returns the thinking chain containing "<|point|>[[x,y]]<|/point|>".
        """
        return await _call_http_tool(
            url,
            token,
            image_path,
            question,
            thinking_style,
            config_path,
            host,
            port,
        )

    anyio.run(mcp.run_stdio_async)


def _parse_args(argv: list[str]) -> tuple[str, bool, str | None, int | None, str | None]:
    """解析 CLI 参数，返回 (config_path, autostart, host, port, token)；未指定的项为 None。"""
    root_config = PROJECT_ROOT / "config.json"
    config_path = str(root_config) if root_config.exists() else "config.json"
    autostart = False
    host: str | None = None
    port: int | None = None
    token: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--autostart":
            autostart = True
        elif arg == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
            i += 1
        elif arg == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
            i += 1
        elif arg == "--token" and i + 1 < len(argv):
            token = argv[i + 1]
            i += 1
        elif not arg.startswith("-"):
            config_path = arg
        i += 1
    return config_path, autostart, host, port, token


def main() -> None:
    config_path, autostart, host, port, token = _parse_args(sys.argv[1:])
    cfg = DsvConfig.load(config_path)
    host = host or cfg.server.host
    port = port or cfg.server.port
    token = token or cfg.server.token
    if autostart:
        serve_stdio_autostart(config_path, host=host, port=port, token=token)
        return
    server = DsvServer(config_path)
    serve_http(server, host=host, port=port, token=token)
