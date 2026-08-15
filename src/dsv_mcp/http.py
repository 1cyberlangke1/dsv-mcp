"""HTTP 层：curl_cffi 浏览器伪装客户端。超时策略：普通 60s、流式不掐流。"""

from __future__ import annotations

import json
from typing import Any

from curl_cffi import CurlMime
from curl_cffi import requests as cffi_requests

from dsv_mcp.protocol import build_base_headers


class HttpClient:
    def __init__(self, proxy: str | None = None):
        kwargs = {"impersonate": "chrome", "timeout": 60}
        if proxy:
            kwargs["proxy"] = proxy
        self.session = cffi_requests.Session(**kwargs)

    @staticmethod
    def _parse(resp, url: str) -> dict[str, Any]:
        try:
            return resp.json()
        except ValueError:
            body = (resp.content or b"")[:200]
            raise RuntimeError(f"非 JSON 响应 {resp.status_code} {url}: {body!r}")

    def post_json(
        self, url: str, payload: dict, headers: dict[str, str] | None = None, timeout: float = 60
    ) -> dict[str, Any]:
        h = build_base_headers("zh_CN")
        if headers:
            h.update(headers)
        resp = self.session.post(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=h, timeout=timeout
        )
        return self._parse(resp, url)

    def get_json(
        self, url: str, headers: dict[str, str] | None = None, timeout: float = 60
    ) -> dict[str, Any]:
        h = build_base_headers("zh_CN")
        if headers:
            h.update(headers)
        resp = self.session.get(url, headers=h, timeout=timeout)
        return self._parse(resp, url)

    def post_multipart(
        self, url: str, files: dict, headers: dict[str, str] | None = None, timeout: float = 120
    ) -> dict[str, Any]:
        h = build_base_headers("zh_CN")
        h.pop("Content-Type", None)  # multipart 的 Content-Type（含 boundary）由 curl 自动生成
        if headers:
            h.update(headers)
        mime = CurlMime()
        for name, (filename, data, content_type) in files.items():
            mime.addpart(name=name, filename=filename, data=data, content_type=content_type)
        resp = self.session.post(url, multipart=mime, headers=h, timeout=timeout)
        return self._parse(resp, url)

    def stream(self, url: str, payload: dict, headers: dict[str, str] | None = None):
        """POST JSON 并逐行返回 SSE 文本（生成器）。流式不设低速率掐断（timeout=0）。"""
        h = build_base_headers("zh_CN")
        h["Content-Type"] = "application/json"
        if headers:
            h.update(headers)
        with self.session.stream(
            "POST",
            url,
            data=json.dumps(payload, ensure_ascii=False),
            headers=h,
            timeout=(30, 7200),  # 连接 30s；低速率 2h（流式 24h 不掐流）
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    yield line.decode("utf-8", errors="replace")

    def close(self) -> None:
        self.session.close()
