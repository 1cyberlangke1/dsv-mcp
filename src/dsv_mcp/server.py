"""MCP 服务器装配：describe_image 工具（mcp>=2.0 MCPServer）。"""

from __future__ import annotations

import io
import json
import re
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
                    return (
                        text
                        + "\n\n[Grounding]\n"
                        + json.dumps(groundings, ensure_ascii=False)
                    )
                return text
            if thinking_style == "pointing" and thinking and has_point_primitive(thinking):
                return thinking
            return text
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


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    server = DsvServer(config_path)
    mcp = build_mcp(server)
    anyio.run(mcp.run_stdio_async)
