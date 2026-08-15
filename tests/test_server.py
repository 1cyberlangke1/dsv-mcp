"""服务器层测试：账号轮询、token 缓存、上传风控冷却（离线 mock）。"""

from __future__ import annotations

import json
import time

import pytest

from dsv_mcp.client import DeepSeekError
from dsv_mcp.server import UPLOAD_COOLDOWN, DsvServer


def _make_server(tmp_path, emails=("a@b.c",)):
    cfg = tmp_path / "config.json"
    accounts = [{"email": e, "password": "p"} for e in emails]
    cfg.write_text(
        json.dumps({"accounts": accounts, "proxy": {"mode": "none"}}),
        encoding="utf-8",
    )
    server = DsvServer(cfg)
    img = tmp_path / "img.jpg"
    from PIL import Image

    Image.new("RGB", (64, 64), (30, 80, 220)).save(img, format="JPEG", quality=85)
    return server, str(img)


def test_token_cached_between_calls(tmp_path, monkeypatch):
    server, _ = _make_server(tmp_path)
    calls = {"login": 0}

    def fake_login(account, device_id=None):
        calls["login"] += 1
        return "tok-1"

    monkeypatch.setattr(server.client, "login", fake_login)
    ident = server._order[0]
    assert server._token_for(ident) == "tok-1"
    assert server._token_for(ident) == "tok-1"
    assert calls["login"] == 1
    server.close()


def test_token_dropped_on_auth_failure(tmp_path, monkeypatch):
    server, img = _make_server(tmp_path)
    server._tokens["a@b.c"] = "tok-old"

    def fake_login(account, device_id=None):
        return "tok-new"

    monkeypatch.setattr(server.client, "login", fake_login)

    def boom(*args, **kwargs):
        raise DeepSeekError("auth_failed", "bad token")

    monkeypatch.setattr(server.client, "describe_image", boom)
    result = server.describe_image(img, "q")
    assert "auth_failed" in result
    assert "a@b.c" not in server._tokens
    server.close()


def test_upload_rate_limited_sets_cooldown(tmp_path, monkeypatch):
    server, img = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")

    def boom(*args, **kwargs):
        raise DeepSeekError("upload_rate_limited", "上传过于频繁")

    monkeypatch.setattr(server.client, "describe_image", boom)
    result = server.describe_image(img, "q")
    assert "upload_rate_limited" in result
    assert server._cooldown["a@b.c"] > time.monotonic()
    server.close()


def test_acquire_skips_cooldown_account(tmp_path):
    server, _ = _make_server(tmp_path, emails=("a@b.c", "c@d.e"))
    server._cooldown["a@b.c"] = time.monotonic() + 9999
    ident, _ = server._acquire()
    assert ident == "c@d.e"
    server.close()


def test_acquire_round_robin(tmp_path):
    server, _ = _make_server(tmp_path, emails=("a@b.c", "c@d.e", "e@f.g"))
    first, _ = server._acquire()
    second, _ = server._acquire()
    third, _ = server._acquire()
    assert (first, second, third) == ("a@b.c", "c@d.e", "e@f.g")
    server.close()


def test_acquire_all_cooldown_raises(tmp_path):
    server, _ = _make_server(tmp_path, emails=("a@b.c",))
    server._cooldown["a@b.c"] = time.monotonic() + UPLOAD_COOLDOWN
    with pytest.raises(DeepSeekError):
        server._acquire()
    server.close()


def test_thinking_style_inserts_prompt(tmp_path, monkeypatch):
    from dsv_mcp.server import GROUNDING_TITLE, POINTING_TITLE

    server, img = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")
    captured = {}

    def fake_describe(account, token, image_bytes, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["thinking_enabled"] = kwargs.get("thinking_enabled")
        return {"text": "ok", "thinking": "think", "message_id": 1}

    monkeypatch.setattr(server.client, "describe_image", fake_describe)
    result = server.describe_image(img, "问题", thinking_style="grounding")
    assert captured["prompt"] == f"{GROUNDING_TITLE}\n问题"
    assert captured["thinking_enabled"] is True
    assert "[思考过程]" in result
    server.describe_image(img, "问题", thinking_style="pointing")
    assert captured["prompt"] == f"{POINTING_TITLE}\n问题"
    server.describe_image(img, "问题", thinking_style="none")
    assert captured["prompt"] == "问题"
    server.close()


def test_thinking_style_invalid(tmp_path):
    server, img = _make_server(tmp_path)
    result = server.describe_image(img, "问题", thinking_style="bogus")
    assert "无效 thinking_style" in result
    server.close()
