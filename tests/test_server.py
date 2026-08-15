"""服务器层测试：账号轮询、token 缓存、上传风控冷却（离线 mock）。"""

from __future__ import annotations

import json
import sys
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


def test_captcha_required_sets_cooldown(tmp_path, monkeypatch):
    from dsv_mcp.server import CAPTCHA_COOLDOWN

    server, img = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")

    def boom(*args, **kwargs):
        raise DeepSeekError("captcha_required", "触发验证码/风控: captcha")

    monkeypatch.setattr(server.client, "describe_image", boom)
    result = server.describe_image(img, "q")
    assert "captcha_required" in result
    assert server._cooldown["a@b.c"] > time.monotonic() + CAPTCHA_COOLDOWN - 60
    server.close()


def test_acquire_skips_cooldown_account(tmp_path):
    server, _ = _make_server(tmp_path, emails=("a@b.c", "c@d.e"))
    server._cooldown["a@b.c"] = time.monotonic() + 9999
    ident, _ = server._acquire()
    assert ident == "c@d.e"
    server.close()


def test_acquire_skips_banned_account(tmp_path):
    server, _ = _make_server(tmp_path, emails=("a@b.c", "c@d.e"))
    next(a for a in server.config.accounts if a.identifier() == "a@b.c").banned = True
    ident, _ = server._acquire()
    assert ident == "c@d.e"
    server.close()


def test_acquire_all_banned_raises(tmp_path):
    server, _ = _make_server(tmp_path, emails=("a@b.c",))
    next(a for a in server.config.accounts if a.identifier() == "a@b.c").banned = True
    with pytest.raises(DeepSeekError):
        server._acquire()
    server.close()


def test_banned_persisted_to_config_file(tmp_path, monkeypatch):
    server, img = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")

    def boom(*args, **kwargs):
        raise DeepSeekError("account_banned", "账号已被停用")

    monkeypatch.setattr(server.client, "describe_image", boom)
    result = server.describe_image(img, "q")
    assert "account_banned" in result
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["accounts"][0]["banned"] is True
    assert server.config.accounts[0].banned is True
    with pytest.raises(DeepSeekError):
        server._acquire()
    server.close()


def test_muted_sets_cooldown_until(tmp_path, monkeypatch):
    server, img = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")
    until = time.time() + 3600

    def boom(*args, **kwargs):
        raise DeepSeekError("account_muted", "账号禁言中", until=until)

    monkeypatch.setattr(server.client, "describe_image", boom)
    result = server.describe_image(img, "q")
    assert "account_muted" in result
    assert server._cooldown["a@b.c"] > time.monotonic() + 3500
    server.close()


def test_banned_cleared_on_successful_login(tmp_path, monkeypatch):
    server, _ = _make_server(tmp_path)
    next(a for a in server.config.accounts if a.identifier() == "a@b.c").banned = True
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")
    assert server._token_for("a@b.c") == "tok"
    assert server.config.accounts[0].banned is False
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["accounts"][0]["banned"] is False
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
    assert result == "ok"
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


def test_extract_groundings_recorded_cases():
    from pathlib import Path

    from dsv_mcp.server import extract_groundings

    cases = Path("tests/cases")
    for name in ("奇怪的手_grounding", "棋子_grounding"):
        meta = json.loads((cases / f"{name}.json").read_text(encoding="utf-8"))
        groundings = extract_groundings(meta["thinking"])
        assert len(groundings) >= 3
        for g in groundings:
            assert isinstance(g["ref"], str) and g["ref"]
            assert isinstance(g["boxes"], list) and g["boxes"]
            assert len(g["boxes"][0]) == 4


def test_extract_groundings_none_on_pointing_case():
    from pathlib import Path

    from dsv_mcp.server import extract_groundings

    meta = json.loads(
        (Path("tests/cases") / "迷宫_pointing.json").read_text(encoding="utf-8")
    )
    assert extract_groundings(meta["thinking"]) == []


def test_has_point_primitive_recorded_cases():
    from pathlib import Path

    from dsv_mcp.server import has_point_primitive

    pointing = json.loads(
        (Path("tests/cases") / "迷宫_pointing.json").read_text(encoding="utf-8")
    )
    grounding = json.loads(
        (Path("tests/cases") / "棋子_grounding.json").read_text(encoding="utf-8")
    )
    assert has_point_primitive(pointing["thinking"]) is True
    assert has_point_primitive(grounding["thinking"]) is False


def _server_with_fake_describe(tmp_path, monkeypatch, text, thinking):
    server, img = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")

    def fake_describe(account, token, image_bytes, prompt, **kwargs):
        return {"text": text, "thinking": thinking, "message_id": 1}

    monkeypatch.setattr(server.client, "describe_image", fake_describe)
    return server, img


def test_return_modes_none_plain_text(tmp_path, monkeypatch):
    from pathlib import Path

    meta = json.loads(
        (Path("tests/cases") / "棋子_grounding.json").read_text(encoding="utf-8")
    )
    server, img = _server_with_fake_describe(
        tmp_path, monkeypatch, meta["text"], meta["thinking"]
    )
    result = server.describe_image(img, "q", thinking_style="none")
    assert result == meta["text"]
    server.close()


def test_return_modes_grounding_appends_ref_box_lines(tmp_path, monkeypatch):
    import json as _json
    from pathlib import Path

    meta = json.loads(
        (Path("tests/cases") / "棋子_grounding.json").read_text(encoding="utf-8")
    )
    from dsv_mcp.server import extract_groundings

    groundings = extract_groundings(meta["thinking"])
    assert groundings
    pipe = "\x7c"
    tag = lambda name: f"<{pipe}{name}{pipe}>"
    lines = [
        f"{tag('ref')}{g['ref']}{tag('/ref')}{tag('box')}"
        f"{_json.dumps(g['boxes'], ensure_ascii=False, separators=(',', ':'))}"
        f"{tag('/box')}"
        for g in groundings
    ]
    expected = meta["text"] + "\n\n" + "\n".join(lines)
    server, img = _server_with_fake_describe(
        tmp_path, monkeypatch, meta["text"], meta["thinking"]
    )
    result = server.describe_image(img, "q", thinking_style="grounding")
    assert result == expected
    server.close()


def test_return_modes_grounding_no_markers_falls_back(tmp_path, monkeypatch):
    server, img = _server_with_fake_describe(tmp_path, monkeypatch, "最终回答", "没有标记")
    result = server.describe_image(img, "q", thinking_style="grounding")
    assert result == "最终回答"
    server.close()


def test_return_modes_pointing_with_points_returns_thinking(tmp_path, monkeypatch):
    from pathlib import Path

    meta = json.loads(
        (Path("tests/cases") / "迷宫_pointing.json").read_text(encoding="utf-8")
    )
    from dsv_mcp.server import _normalize_primitives

    expected = _normalize_primitives(meta["thinking"])
    pipe = "\x7c"
    assert f"<{pipe}point{pipe}>" in expected
    server, img = _server_with_fake_describe(
        tmp_path, monkeypatch, meta["text"], meta["thinking"]
    )
    result = server.describe_image(img, "q", thinking_style="pointing")
    assert result == expected
    server.close()


def test_return_modes_pointing_no_points_falls_back(tmp_path, monkeypatch):
    server, img = _server_with_fake_describe(tmp_path, monkeypatch, "最终回答", "没有点标记")
    result = server.describe_image(img, "q", thinking_style="pointing")
    assert result == "最终回答"
    server.close()


def test_denormalize_bbox():
    from dsv_mcp.server import denormalize

    assert denormalize([22, 324, 74, 449], 1200, 800) == [26, 259, 89, 360]


def test_denormalize_point():
    from dsv_mcp.server import denormalize

    assert denormalize([261, 548], 1200, 800) == [314, 439]


def test_denormalize_edges():
    from dsv_mcp.server import denormalize

    assert denormalize([0, 0, 999, 999], 1000, 500) == [0, 0, 1000, 500]


def test_parse_args_defaults():
    from dsv_mcp.server import PROJECT_ROOT, _parse_args

    config, autostart, host, port, token = _parse_args([])
    assert config == str(PROJECT_ROOT / "config.json")
    assert autostart is False
    assert host == "127.0.0.1"
    assert port == 8765
    assert token == ""


def test_parse_args_full():
    from dsv_mcp.server import _parse_args

    config, autostart, host, port, token = _parse_args(
        ["--autostart", "--host", "0.0.0.0", "--port", "9000", "--token", "secret", "my.json"]
    )
    assert config == "my.json"
    assert autostart is True
    assert host == "0.0.0.0"
    assert port == 9000
    assert token == "secret"


def test_http_mode_serves_tool(tmp_path, monkeypatch):
    import asyncio
    import socket
    import threading

    import uvicorn
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from dsv_mcp.server import build_http_app

    server, img = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")

    def fake_describe(account, token, image_bytes, prompt, **kwargs):
        return {"text": "蓝色背景", "thinking": "", "message_id": 1}

    monkeypatch.setattr(server.client, "describe_image", fake_describe)
    app = build_http_app(server)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    config.install_signal_handlers = False
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()

    async def client_roundtrip():
        async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = [t.name for t in listed.tools]
                assert "dsv_describe_image" in names
                result = await session.call_tool(
                    "dsv_describe_image",
                    {"image_path": img, "question": "什么颜色", "thinking_style": "none"},
                )
                text = result.content[0].text
                assert text == "蓝色背景"

    try:
        import time

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                asyncio.run(client_roundtrip())
                break
            except (ConnectionError, OSError):
                time.sleep(0.2)
        else:
            raise AssertionError("HTTP server 未就绪")
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)
        server.close()


def test_ensure_http_server_skips_when_port_open(tmp_path, monkeypatch):
    import dsv_mcp.server as server_mod

    calls = {"popen": 0}
    monkeypatch.setattr(server_mod, "_can_connect", lambda host, port: True)
    monkeypatch.setattr(
        server_mod.subprocess, "Popen", lambda *a, **k: calls.__setitem__("popen", calls["popen"] + 1)
    )
    server_mod._ensure_http_server(str(tmp_path / "config.json"), "127.0.0.1", 9999, "")
    assert calls["popen"] == 0


def test_ensure_http_server_starts_when_down(tmp_path, monkeypatch):
    import dsv_mcp.server as server_mod

    probe = {"n": 0}
    calls = {"cmd": None}

    def fake_can_connect(host, port):
        probe["n"] += 1
        return probe["n"] > 1

    def fake_popen(cmd, **kwargs):
        calls["cmd"] = cmd
        return object()

    monkeypatch.setattr(server_mod, "_can_connect", fake_can_connect)
    monkeypatch.setattr(server_mod.subprocess, "Popen", fake_popen)
    cfg = str(tmp_path / "config.json")
    server_mod._ensure_http_server(cfg, "127.0.0.1", 9999, "sec")
    assert calls["cmd"] == [
        sys.executable,
        "-m",
        "dsv_mcp",
        cfg,
        "--host",
        "127.0.0.1",
        "--port",
        "9999",
        "--token",
        "sec",
    ]


def test_resolve_token_env_fallback(monkeypatch):
    from dsv_mcp.server import _resolve_token

    assert _resolve_token("") == ""
    monkeypatch.setenv("DSV_MCP_TOKEN", "from-env")
    assert _resolve_token("") == "from-env"
    assert _resolve_token("explicit") == "explicit"


def test_call_http_tool_forwards_to_http_instance(tmp_path, monkeypatch):
    import asyncio
    import socket
    import threading

    import uvicorn

    from dsv_mcp.server import _call_http_tool, build_http_app

    server, img = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")

    def fake_describe(account, token, image_bytes, prompt, **kwargs):
        return {"text": "转发成功", "thinking": "", "message_id": 1}

    monkeypatch.setattr(server.client, "describe_image", fake_describe)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = build_http_app(server)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    config.install_signal_handlers = False
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()

    async def run():
        return await _call_http_tool(
            f"http://127.0.0.1:{port}/mcp",
            "",
            img,
            "什么颜色",
            "none",
        )

    try:
        deadline = time.monotonic() + 10
        last_err = None
        while time.monotonic() < deadline:
            try:
                text = asyncio.run(run())
                assert text == "转发成功"
                break
            except (ConnectionError, OSError) as exc:
                last_err = exc
                time.sleep(0.2)
        else:
            raise AssertionError(f"HTTP server 未就绪: {last_err}")
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)
        server.close()


def test_http_mode_token_auth(tmp_path, monkeypatch):
    import asyncio
    import socket
    import threading
    import time
    import urllib.request

    import uvicorn

    from dsv_mcp.server import build_http_app

    server, img = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")
    monkeypatch.setattr(
        server.client,
        "describe_image",
        lambda *a, **k: {"text": "ok", "thinking": "", "message_id": 1},
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = build_http_app(server, token="secret", host="127.0.0.1", port=port)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    config.install_signal_handlers = False
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()

    try:
        deadline = time.monotonic() + 10
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                body = b'{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}'
                accept = "application/json, text/event-stream"
                headers = {"Content-Type": "application/json", "Accept": accept}
                url = f"http://127.0.0.1:{port}/mcp"

                authed_headers = {**headers, "Authorization": "Bearer secret"}
                req = urllib.request.Request(url, data=body, headers=authed_headers)
                with urllib.request.urlopen(req, timeout=3) as resp:
                    assert resp.status == 200

                req_noauth = urllib.request.Request(url, data=body, headers=headers)
                try:
                    with urllib.request.urlopen(req_noauth, timeout=3) as resp:
                        assert resp.status == 401
                except urllib.error.HTTPError as exc:
                    assert exc.code == 401
                break
            except (AssertionError, ConnectionError, OSError, urllib.error.URLError) as exc:
                last_err = exc
                time.sleep(0.2)
        else:
            raise AssertionError(f"HTTP server 未就绪: {last_err}")
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)
        server.close()
