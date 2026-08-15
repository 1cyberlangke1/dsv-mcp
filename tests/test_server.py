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
    server, img = _server_with_fake_describe(
        tmp_path, monkeypatch, "最终回答", "思考内容 <|point|>[[1,2]]<|/point|>"
    )
    result = server.describe_image(img, "q", thinking_style="none")
    assert result == "最终回答"
    server.close()


def test_return_modes_grounding_appends_json(tmp_path, monkeypatch):
    server, img = _server_with_fake_describe(
        tmp_path,
        monkeypatch,
        "最终回答",
        "看到 <｜｜ref｜｜>手掌<｜｜/ref｜｜><｜｜box｜｜>[[1,2,3,4]]<｜｜/box｜｜>",
    )
    result = server.describe_image(img, "q", thinking_style="grounding")
    assert result.startswith("最终回答\n\n[Grounding]\n")
    payload = json.loads(result.rsplit("\n", 1)[1])
    assert payload == [{"ref": "手掌", "boxes": [[1, 2, 3, 4]]}]
    server.close()


def test_return_modes_grounding_no_markers_falls_back(tmp_path, monkeypatch):
    server, img = _server_with_fake_describe(tmp_path, monkeypatch, "最终回答", "没有标记")
    result = server.describe_image(img, "q", thinking_style="grounding")
    assert result == "最终回答"
    server.close()


def test_return_modes_pointing_with_points_returns_thinking(tmp_path, monkeypatch):
    server, img = _server_with_fake_describe(
        tmp_path, monkeypatch, "最终回答", "路径 <｜｜point｜｜>[[1,2],[3,4]]<｜｜/point｜｜>"
    )
    result = server.describe_image(img, "q", thinking_style="pointing")
    assert result == "路径 <｜｜point｜｜>[[1,2],[3,4]]<｜｜/point｜｜>"
    server.close()


def test_return_modes_pointing_no_points_falls_back(tmp_path, monkeypatch):
    server, img = _server_with_fake_describe(tmp_path, monkeypatch, "最终回答", "没有点标记")
    result = server.describe_image(img, "q", thinking_style="pointing")
    assert result == "最终回答"
    server.close()
