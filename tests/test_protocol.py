"""协议常量与请求头测试。"""

from __future__ import annotations

from dsv_mcp.protocol import (
    CHROME_UA,
    build_base_headers,
    chat_session_referer,
    login_headers,
    timezone_offset_for,
)


def test_chrome_ua_version():
    assert "Chrome/150.0.0.0" in CHROME_UA


def test_login_headers_shape():
    h = login_headers("zh_CN")
    assert h["User-Agent"] == "DeepSeek/2.2.0"
    assert h["x-client-platform"] == "web"
    assert h["x-client-version"] == "2.2.0"
    assert h["x-client-bundle-id"] == "com.deepseek.chat"
    # 登录头不带浏览器专属头
    assert "sec-ch-ua" not in h
    assert "Origin" not in h
    assert "Referer" not in h


def test_base_headers_has_browser_headers():
    h = build_base_headers("zh_CN")
    assert "Chrome/150" in h["User-Agent"]
    assert h["sec-ch-ua-mobile"] == "?0"
    assert h["Origin"] == "https://chat.deepseek.com"
    assert h["Accept"] == "*/*"
    assert "zstd" in h["Accept-Encoding"]
    assert h["priority"] == "u=1, i"
    assert h["x-client-platform"] == "web"


def test_timezone_offset_is_digits():
    assert timezone_offset_for("zh_CN").isdigit()
    assert timezone_offset_for("unknown_locale").isdigit()


def test_chat_session_referer():
    assert chat_session_referer("sess-1") == "https://chat.deepseek.com/a/chat/s/sess-1"
