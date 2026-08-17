"""服务器层测试：账号轮询、token 缓存、上传风控冷却（离线 mock）。"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time

import pytest

from dsv_mcp.client import DeepSeekError
from dsv_mcp.server import UPLOAD_COOLDOWN, DsvServer, _ensure_http_server, _stdio_log


def _make_server(tmp_path, emails=("a@b.c",), auto_delete_mode=None):
    cfg = tmp_path / "config.json"
    accounts = [{"email": e, "password": "p"} for e in emails]
    config = {"accounts": accounts, "proxy": {"mode": "none"}}
    if auto_delete_mode is not None:
        config["auto_delete"] = {"mode": auto_delete_mode}
    cfg.write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    server = DsvServer(cfg)
    img = tmp_path / "img.jpg"
    from PIL import Image

    Image.new("RGB", (64, 64), (30, 80, 220)).save(img, format="JPEG", quality=85)
    return server, str(img)


def test_auto_delete_mode_passed_to_client(tmp_path, monkeypatch):
    for mode in ("none", "single", "all"):
        server, img = _make_server(tmp_path, auto_delete_mode=mode)
        received = {}
        monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok")

        def fake_describe(
            account,
            token,
            data,
            prompt=None,
            filename=None,
            content_type=None,
            thinking_enabled=False,
            auto_delete="single",
        ):
            received["auto_delete"] = auto_delete
            return {"text": "ok", "thinking": "", "message_id": 1}

        monkeypatch.setattr(server.client, "describe_image", fake_describe)
        result = server.describe_image(img, "q")
        assert result == "ok"
        assert received["auto_delete"] == mode
        server.close()


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


def test_token_persisted_to_config_after_login(tmp_path, monkeypatch):
    """登录成功后 token 写入 config.json 的 tokens 字段（供实例重启复用）。"""
    server, _ = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok-1")
    ident = server._order[0]
    assert server._token_for(ident) == "tok-1"
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved.get("tokens", {}).get(ident) == "tok-1"
    server.close()


def test_token_loaded_from_config_after_restart(tmp_path, monkeypatch):
    """实例重启后从 config.json 读到磁盘 token，不重新登录。"""
    server, _ = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok-1")
    ident = server._order[0]
    assert server._token_for(ident) == "tok-1"
    server.close()

    calls = {"login": 0}

    def fake_login(account, device_id=None):
        calls["login"] += 1
        return "tok-2"

    server2 = DsvServer(tmp_path / "config.json")
    monkeypatch.setattr(server2.client, "login", fake_login)
    assert server2._token_for(ident) == "tok-1"
    assert calls["login"] == 0
    server2.close()


def test_token_removed_from_config_on_auth_failure(tmp_path, monkeypatch):
    """auth_failed 时磁盘 token 一并清除。"""
    server, img = _make_server(tmp_path)
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "tok-old")
    ident = server._order[0]
    assert server._token_for(ident) == "tok-old"

    def boom(*args, **kwargs):
        raise DeepSeekError("auth_failed", "bad token")

    monkeypatch.setattr(server.client, "describe_image", boom)
    result = server.describe_image(img, "q")
    assert "auth_failed" in result
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert ident not in saved.get("tokens", {})
    server.close()


def test_stale_disk_token_auto_relogin(tmp_path, monkeypatch):
    """磁盘 token 失效：自动清缓存重新登录并重试成功，用户无感。"""
    server, img = _make_server(tmp_path)
    ident = server._order[0]
    server.config.tokens[ident] = "stale-token"
    server.config.save(tmp_path / "config.json")
    calls = {"login": 0, "describe": 0}

    def fake_login(account, device_id=None):
        calls["login"] += 1
        return "fresh-token"

    def fake_describe(account, token, image_bytes, prompt, **kwargs):
        calls["describe"] += 1
        if token == "stale-token":
            raise DeepSeekError("auth_failed", "token 失效")
        return {"text": "重试成功", "thinking": "", "message_id": 1}

    monkeypatch.setattr(server.client, "login", fake_login)
    monkeypatch.setattr(server.client, "describe_image", fake_describe)
    result = server.describe_image(img, "q")
    assert result == "重试成功"
    assert calls["describe"] == 2
    assert calls["login"] == 1
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["tokens"][ident] == "fresh-token"
    server.close()


def test_auth_failed_retry_exhausted_returns_error(tmp_path, monkeypatch):
    """重试后仍 auth_failed（密码已错）：返回错误不无限重试。"""
    server, img = _make_server(tmp_path)
    ident = server._order[0]
    server.config.tokens[ident] = "stale"
    server.config.save(tmp_path / "config.json")
    monkeypatch.setattr(server.client, "login", lambda account, device_id=None: "bad")

    def fake_describe(account, token, image_bytes, prompt, **kwargs):
        raise DeepSeekError("auth_failed", "token 失效")

    monkeypatch.setattr(server.client, "describe_image", fake_describe)
    result = server.describe_image(img, "q")
    assert "auth_failed" in result
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


def test_parse_args_defaults():
    from dsv_mcp.server import PROJECT_ROOT, _parse_args

    config, autostart, host, port, token = _parse_args([])
    assert config == str(PROJECT_ROOT / "config.json")
    assert autostart is False
    assert host is None
    assert port is None
    assert token is None


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
                assert "describe_image" in names
                result = await session.call_tool(
                    "describe_image",
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


def test_third_party_loggers_quieted():
    """第三方库日志降到 WARNING：避免 Codex Windows stderr 管道 bug（issue #7155）。"""
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
    assert logging.getLogger("mcp").level == logging.WARNING
    assert logging.getLogger("client").level == logging.WARNING


def test_stdio_log_writes_and_rotates(tmp_path, monkeypatch):
    """stdio 日志写 %TEMP%，超过 512KB 重建防烧硬盘。"""
    monkeypatch.setattr("dsv_mcp.server.tempfile.gettempdir", lambda: str(tmp_path))
    _stdio_log("第一条")
    _stdio_log("第二条")
    log = next(tmp_path.glob("dsv-mcp-stdio-*.log"))
    content = log.read_text(encoding="utf-8")
    assert "第一条" in content and "第二条" in content

    log.write_text("x" * (513 * 1024), encoding="utf-8")
    _stdio_log("旋转后")
    rotated = next(tmp_path.glob("dsv-mcp-stdio-*.log"))
    content = rotated.read_text(encoding="utf-8")
    assert "旋转后" in content and len(content) < 512 * 1024


def test_ensure_http_server_single_launch_under_lock(tmp_path, monkeypatch):
    """并发调用 _ensure_http_server 只拉起一次（进程内锁防重复拉起抢端口）。"""
    monkeypatch.setattr("dsv_mcp.server.tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr("dsv_mcp.server._stdio_log", lambda msg: None)
    monkeypatch.setattr("dsv_mcp.server._child_interpreter", lambda: "python")

    probe = {"n": 0}

    def fake_can_connect(host, port):
        probe["n"] += 1
        return probe["n"] > 1  # 第一次探测失败触发拉起，之后视为已就绪

    monkeypatch.setattr("dsv_mcp.server._can_connect", fake_can_connect)
    pops: list = []

    def fake_popen(cmd, **kwargs):
        pops.append(cmd)
        return None

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    errors: list[Exception] = []

    def launch():
        try:
            _ensure_http_server("config.json", "127.0.0.1", 8765, "")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=launch) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert len(pops) == 1


def test_wait_http_ready_waits_then_ready(monkeypatch):
    import asyncio

    import dsv_mcp.server as server_mod

    probe = {"n": 0}

    def fake_can_connect(host, port):
        probe["n"] += 1
        return probe["n"] >= 3

    monkeypatch.setattr(server_mod, "_can_connect", fake_can_connect)
    server_mod._LAUNCH_FAILED.clear()

    async def run():
        return await server_mod._wait_http_ready("http://127.0.0.1:9999/mcp", 5.0)

    assert asyncio.run(run()) is None
    assert probe["n"] >= 3


def test_wait_http_ready_times_out(monkeypatch):
    import asyncio

    import dsv_mcp.server as server_mod

    monkeypatch.setattr(server_mod, "_can_connect", lambda host, port: False)
    server_mod._LAUNCH_FAILED.clear()

    async def run():
        return await server_mod._wait_http_ready("http://127.0.0.1:9999/mcp", 0.3)

    err = asyncio.run(run())
    assert err and "9999" in err


def test_wait_http_ready_reports_launch_failure(monkeypatch):
    import asyncio

    import dsv_mcp.server as server_mod

    server_mod._LAUNCH_FAILED["err"] = "boom: 启动失败"

    async def run():
        return await server_mod._wait_http_ready("http://127.0.0.1:9999/mcp", 1.0)

    try:
        assert asyncio.run(run()) == "boom: 启动失败"
    finally:
        server_mod._LAUNCH_FAILED.clear()


def test_call_http_tool_raises_when_upstream_not_ready(tmp_path, monkeypatch):
    import asyncio

    import dsv_mcp.server as server_mod

    async def fake_wait(url, timeout):
        return "启动失败"

    monkeypatch.setattr(server_mod, "_wait_http_ready", fake_wait)

    async def run():
        return await server_mod._call_http_tool(
            "http://127.0.0.1:9999/mcp", "", str(tmp_path / "x.jpg"), "q", "none"
        )

    with pytest.raises(DeepSeekError) as exc:
        asyncio.run(run())
    assert exc.value.code == "upstream_unavailable"


def test_call_http_tool_self_heals_retries_ensure(tmp_path, monkeypatch):
    """提供 config_path 时：实例未就绪会先确保拉起，失败后再重试一次才报错。"""
    import asyncio

    import dsv_mcp.server as server_mod

    calls = {"ensure": 0, "wait": 0}

    async def fake_wait(url, timeout):
        calls["wait"] += 1
        return "still down"

    def fake_ensure(config_path, host, port, token):
        calls["ensure"] += 1

    monkeypatch.setattr(server_mod, "_wait_http_ready", fake_wait)
    monkeypatch.setattr(server_mod, "_ensure_http_server", fake_ensure)
    server_mod._LAUNCH_FAILED.clear()

    async def run():
        return await server_mod._call_http_tool(
            "http://127.0.0.1:9999/mcp",
            "",
            str(tmp_path / "x.jpg"),
            "q",
            "none",
            str(tmp_path / "config.json"),
            "127.0.0.1",
            9999,
        )

    with pytest.raises(DeepSeekError) as exc:
        asyncio.run(run())
    assert exc.value.code == "upstream_unavailable"
    assert calls["ensure"] == 2  # 开头确保一次 + 失败后重试一次
    assert calls["wait"] == 2


def test_serve_stdio_autostart_does_not_block_on_http(tmp_path, monkeypatch):
    import threading

    import dsv_mcp.server as server_mod

    started = threading.Event()
    called = {"anyio": False}

    def slow_ensure(config_path, host, port, token):
        started.set()
        time.sleep(0.5)

    def fake_anyio_run(fn):
        called["anyio"] = True
        return None

    monkeypatch.setattr(server_mod, "_ensure_http_server", slow_ensure)
    monkeypatch.setattr(server_mod.anyio, "run", fake_anyio_run)
    server_mod._LAUNCH_FAILED.clear()

    begin = time.monotonic()
    server_mod.serve_stdio_autostart(str(tmp_path / "config.json"))
    elapsed = time.monotonic() - begin
    assert called["anyio"] is True
    assert elapsed < 0.4
    assert started.wait(timeout=2)


def test_serve_stdio_autostart_records_launch_failure(tmp_path, monkeypatch):
    import dsv_mcp.server as server_mod

    def failing_ensure(config_path, host, port, token):
        raise RuntimeError("启动失败: boom")

    def fake_anyio_run(fn):
        return None

    monkeypatch.setattr(server_mod, "_ensure_http_server", failing_ensure)
    monkeypatch.setattr(server_mod.anyio, "run", fake_anyio_run)
    server_mod._LAUNCH_FAILED.clear()

    server_mod.serve_stdio_autostart(str(tmp_path / "config.json"))
    deadline = time.monotonic() + 2
    while "err" not in server_mod._LAUNCH_FAILED and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        assert "启动失败" in server_mod._LAUNCH_FAILED.get("err", "")
    finally:
        server_mod._LAUNCH_FAILED.clear()


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
        server_mod._child_interpreter(),
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

    monkeypatch.delenv("DSV_MCP_TOKEN", raising=False)
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


def test_touch_middleware_refreshes_active(monkeypatch):
    import dsv_mcp.server as server_mod

    class FakeClock:
        now = 1000.0

    monkeypatch.setattr(server_mod.time, "monotonic", lambda: FakeClock.now)
    server_mod._LAST_ACTIVE["t"] = 0.0
    called = {"n": 0}

    async def app(scope, receive, send):
        called["n"] += 1

    import asyncio

    asyncio.run(server_mod._touch_middleware(app)({"type": "http"}, None, None))
    assert server_mod._LAST_ACTIVE["t"] == 1000.0
    assert called["n"] == 1


def test_idle_watchdog_exits_after_timeout(monkeypatch):
    import dsv_mcp.server as server_mod

    class FakeServer:
        should_exit = False

    fake = FakeServer()
    clock = {"t": 5000.0}
    server_mod._LAST_ACTIVE["t"] = 1000.0
    server_mod._idle_watchdog(
        fake,
        600.0,
        now=lambda: clock["t"],
        sleep=lambda s: None,
    )
    assert fake.should_exit is True


def test_idle_watchdog_keeps_running_while_active(monkeypatch):
    import dsv_mcp.server as server_mod

    class FakeServer:
        should_exit = False

    clock = {"t": 1500.0}
    sleeps = {"n": 0}
    server_mod._LAST_ACTIVE["t"] = 1000.0

    def fake_sleep(sec):
        sleeps["n"] += 1
        if sleeps["n"] >= 3:
            fake.should_exit = True  # 模拟外部请求同时发生

    fake = FakeServer()
    server_mod._idle_watchdog(
        fake,
        600.0,
        now=lambda: clock["t"],
        sleep=fake_sleep,
    )
    assert fake.should_exit is True
    assert sleeps["n"] == 3


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
