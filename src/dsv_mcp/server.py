"""MCP 服务器装配：describe_image 工具（mcp>=2.0 MCPServer）。"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import anyio
from mcp.server.mcpserver import MCPServer

from dsv_mcp.client import DeepSeekClient, DeepSeekError
from dsv_mcp.config import DsvConfig
from dsv_mcp.http import HttpClient
from dsv_mcp.proxy import ProxyError, ProxyManager


MAX_IMAGE_EDGE = 1024
# 上传被风控（40301）后账号的冷却时长（秒）
UPLOAD_COOLDOWN = 300.0


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
        include_thinking: bool = False,
    ) -> str:
        """识图主流程：取账号 → 登录态 → 压缩 → 单轮识图 → 语义化返回。"""
        image_bytes = Path(image_path).read_bytes()
        data, content_type = compress_image(image_bytes)
        try:
            ident, account = self._acquire()
        except DeepSeekError as exc:
            return f"账号不可用: {exc.code} {exc}"
        try:
            token = self._token_for(ident)
            result = self.client.describe_image(
                account,
                token,
                data,
                prompt=question,
                filename="image.jpg",
                content_type=content_type,
                thinking_enabled=include_thinking,
                auto_delete=True,
            )
            parts = [result["text"]]
            if include_thinking and result["thinking"]:
                parts.append(f"\n\n[思考过程]\n{result['thinking']}")
            return "\n".join(parts)
        except DeepSeekError as exc:
            if exc.code == "auth_failed":
                self._tokens.pop(ident, None)
            elif exc.code == "upload_rate_limited":
                self._cooldown[ident] = time.monotonic() + UPLOAD_COOLDOWN
            return f"识图失败: {exc.code} {exc}"
        except Exception as exc:
            return f"识图失败: {type(exc).__name__}: {exc}"
        finally:
            self._busy[ident] = False


def build_mcp(server: DsvServer) -> MCPServer:
    mcp = MCPServer(name="dsv-mcp")

    @mcp.tool()
    def describe_image(
        image_path: str,
        question: str = "请详细描述这张图片的内容。",
        include_thinking: bool = False,
    ) -> str:
        """使用 DeepSeek 识图模式描述图片，返回文字描述。"""
        return server.describe_image(image_path, question, include_thinking)

    return mcp


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    server = DsvServer(config_path)
    mcp = build_mcp(server)
    anyio.run(mcp.run_stdio_async)
