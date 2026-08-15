"""MCP 服务器装配：describe_image 工具（mcp>=2.0 MCPServer）。"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from dsv_mcp.client import DeepSeekClient, DeepSeekError
from dsv_mcp.config import DsvConfig
from dsv_mcp.http import HttpClient
from dsv_mcp.proxy import ProxyError, ProxyManager


MAX_IMAGE_EDGE = 1024
# 上传被风控（40301）后账号的冷却时长（秒）
UPLOAD_COOLDOWN = 300.0
# 验证码/风控挑战后账号的冷却时长（秒）
CAPTCHA_COOLDOWN = 1800.0
# 模式标题：提示词 = 标题 + 换行 + 用户问题
GROUNDING_TITLE = "[Think with Grounding]"
POINTING_TITLE = "[Think with Pointing]"
THINKING_STYLES = ("grounding", "pointing", "none")

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


def denormalize(coord: list[int], width: int, height: int) -> list[int]:
    """把 0-999 归一化坐标还原为像素坐标（bbox 4 元组 / point 2 元组）。"""
    if len(coord) == 4:
        x1, y1, x2, y2 = coord
        return [
            round(x1 / 999 * width),
            round(y1 / 999 * height),
            round(x2 / 999 * width),
            round(y2 / 999 * height),
        ]
    x, y = coord
    return [round(x / 999 * width), round(y / 999 * height)]


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
            if not self._busy[ident] and now >= self._cooldown.get(ident, 0.0):
                self._busy[ident] = True
                return ident, next(a for a in self.config.accounts if a.identifier() == ident)
        raise DeepSeekError("rate_limited", "所有账号忙或冷却中，请稍后重试")

    def _token_for(self, ident: str) -> str:
        token = self._tokens.get(ident, "")
        if token:
            return token
        account = next(a for a in self.config.accounts if a.identifier() == ident)
        token = self.client.login(account)
        self._tokens[ident] = token
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
                auto_delete=True,
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
            elif exc.code == "upload_rate_limited":
                self._cooldown[ident] = time.monotonic() + UPLOAD_COOLDOWN
            elif exc.code == "captcha_required":
                self._cooldown[ident] = time.monotonic() + CAPTCHA_COOLDOWN
            return f"识图失败: {exc.code} {exc}"
        except Exception as exc:
            return f"识图失败: {type(exc).__name__}: {exc}"
        finally:
            self._busy[ident] = False


def build_mcp(
    server: DsvServer,
    token: str = "",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> MCPServer:
    mcp_kwargs: dict = {"name": "dsv-mcp"}
    if token:
        from mcp.server.auth.provider import AccessToken
        from mcp.server.auth.settings import AuthSettings

        class StaticTokenVerifier:
            """SDK TokenVerifier 协议实现：固定 Bearer token 校验。"""

            async def verify_token(self, t: str) -> AccessToken | None:
                if t == token:
                    return AccessToken(token=t, client_id="dsv-mcp", scopes=[])
                return None

        mcp_kwargs["auth"] = AuthSettings(
            issuer_url=f"http://{host}:{port}",
            resource_server_url=f"http://{host}:{port}",
        )
        mcp_kwargs["token_verifier"] = StaticTokenVerifier()
    mcp = MCPServer(**mcp_kwargs)

    @mcp.tool()
    def dsv_describe_image(
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
            Plain-text description of the image.
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
    """以 streamable HTTP 常驻单实例（多客户端共享账号池）。"""
    mcp = build_mcp(server, token=token, host=host, port=port)
    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        streamable_http_path="/mcp",
    )


def _parse_args(argv: list[str]) -> tuple[str, str, int, str]:
    """解析 CLI 参数，返回 (config_path, host, port, token)。"""
    config_path = "config.json"
    host = "127.0.0.1"
    port = 8765
    token = ""
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--host" and i + 1 < len(argv):
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
    return config_path, host, port, token


def main() -> None:
    config_path, host, port, token = _parse_args(sys.argv[1:])
    server = DsvServer(config_path)
    serve_http(server, host=host, port=port, token=token)
