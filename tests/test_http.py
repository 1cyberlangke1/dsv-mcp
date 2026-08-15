"""HTTP 层测试（响应解析，不联网）。"""

from __future__ import annotations

import pytest

from dsv_mcp.http import HttpClient


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def json(self):
        import json as _json

        return _json.loads(self.content)


def test_parse_json_ok():
    resp = FakeResponse(b'{"code": 0}')
    assert HttpClient._parse(resp, "https://x") == {"code": 0}


def test_parse_non_json_raises():
    resp = FakeResponse(b"<html>blocked</html>", status_code=403)
    with pytest.raises(RuntimeError, match="非 JSON"):
        HttpClient._parse(resp, "https://x")
