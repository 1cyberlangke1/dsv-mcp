"""客户端纯函数测试（响应解析、手机号规范化）。"""

from __future__ import annotations

import pytest

from dsv_mcp.client import (
    DeepSeekClient,
    DeepSeekError,
    detect_captcha_challenge,
    extract_create_session_id,
    extract_mute_until,
    extract_response_status,
    extract_upload_file_result,
    int_from,
    is_user_banned_response,
    normalize_mobile_for_login,
)
from dsv_mcp.config import Account


def test_normalize_mobile_china_full():
    assert normalize_mobile_for_login("+8613800138000") == ("13800138000", "+86")
    assert normalize_mobile_for_login("13800138000") == ("13800138000", "+86")


def test_normalize_mobile_with_dashes():
    assert normalize_mobile_for_login("+86 138-0013-8000") == ("13800138000", "+86")


def test_int_from_variants():
    assert int_from(42) == 42
    assert int_from(42.9) == 42
    assert int_from("42") == 42
    assert int_from("abc") == 0
    assert int_from(None) == 0


def test_extract_response_status():
    resp = {
        "code": 1,
        "msg": "err",
        "data": {"biz_code": 2, "biz_msg": "biz err"},
    }
    code, biz_code, msg, biz_msg = extract_response_status(resp)
    assert (code, biz_code, msg, biz_msg) == (1, 2, "err", "biz err")


def test_extract_response_status_falls_back_to_biz_data_msg():
    resp = {"code": 0, "data": {"biz_code": 0, "biz_data": {"msg": "nested"}}}
    code, biz_code, msg, biz_msg = extract_response_status(resp)
    assert biz_msg == "nested"


def test_is_user_banned_response():
    assert is_user_banned_response(10, "anything")
    assert is_user_banned_response(0, "USER_IS_BANNED")
    assert is_user_banned_response(7, "wrong password") is False


def test_extract_mute_until_nested():
    resp = {
        "code": 0,
        "data": {"biz_code": 0, "biz_data": {"is_muted": 1, "mute_until": 1799999999}},
    }
    assert extract_mute_until(resp) == 1799999999.0


def _login_client(resp):
    class FakeHttp:
        def post_json(self, url, payload, headers=None, timeout=60):
            return resp

    return DeepSeekClient(FakeHttp())


def _account():
    return Account(email="a@b.c", password="p")


def test_login_biz_code_10_raises_banned():
    resp = {"code": 0, "data": {"biz_code": 10, "biz_msg": "account suspended"}}
    with pytest.raises(DeepSeekError) as exc:
        _login_client(resp).login(_account())
    assert exc.value.code == "account_banned"


def test_login_user_is_banned_msg_raises_banned():
    resp = {"code": 0, "data": {"biz_code": 3, "biz_msg": "USER_IS_BANNED"}}
    with pytest.raises(DeepSeekError) as exc:
        _login_client(resp).login(_account())
    assert exc.value.code == "account_banned"


def test_login_muted_future_until_raises_muted():
    future = 4102444800  # 2100 年，测试用
    resp = {
        "code": 0,
        "data": {
            "biz_code": 0,
            "biz_data": {
                "user": {
                    "token": "tok",
                    "chat": {"is_muted": 1, "mute_until": future},
                }
            },
        },
    }
    with pytest.raises(DeepSeekError) as exc:
        _login_client(resp).login(_account())
    assert exc.value.code == "account_muted"
    assert exc.value.until == future


def test_login_muted_expired_returns_token():
    resp = {
        "code": 0,
        "data": {
            "biz_code": 0,
            "biz_data": {
                "user": {
                    "token": "tok",
                    "chat": {"is_muted": 1, "mute_until": 1},
                }
            },
        },
    }
    assert _login_client(resp).login(_account()) == "tok"


def test_extract_create_session_id_direct():
    resp = {"data": {"biz_data": {"id": "sess-direct"}}}
    assert extract_create_session_id(resp) == "sess-direct"


def test_extract_create_session_id_nested():
    resp = {"data": {"biz_data": {"chat_session": {"id": "sess-nested"}}}}
    assert extract_create_session_id(resp) == "sess-nested"


def test_extract_upload_file_result_nested():
    resp = {"data": {"biz_data": {"file": {"id": "file-1", "status": "PARSED"}}}}
    result = extract_upload_file_result(resp)
    assert result["id"] == "file-1"
    assert result["status"] == "PARSED"


def test_extract_upload_file_result_top_level():
    resp = {"code": 0, "data": {"biz_data": {"id": "file-2"}}}
    result = extract_upload_file_result(resp)
    assert result["id"] == "file-2"


def test_upload_40301_maps_to_rate_limited(monkeypatch):
    class FakeHttp:
        def post_multipart(self, url, files, headers=None, timeout=120):
            return {"code": 0, "data": {"biz_code": 40301, "biz_msg": ""}}

    client = DeepSeekClient(FakeHttp())
    monkeypatch.setattr(client, "get_pow_for_target", lambda *a, **k: "pow")
    with pytest.raises(DeepSeekError) as exc:
        client.upload_file(None, "tok", "a.jpg", "image/jpeg", b"x")
    assert exc.value.code == "upload_rate_limited"


def test_upload_other_error_keeps_code(monkeypatch):
    class FakeHttp:
        def post_multipart(self, url, files, headers=None, timeout=120):
            return {"code": 0, "data": {"biz_code": 50000, "biz_msg": "boom"}}

    client = DeepSeekClient(FakeHttp())
    monkeypatch.setattr(client, "get_pow_for_target", lambda *a, **k: "pow")
    with pytest.raises(DeepSeekError) as exc:
        client.upload_file(None, "tok", "a.jpg", "image/jpeg", b"x")
    assert exc.value.code == "upload_failed"


def test_pow_recreated_each_call(monkeypatch):
    import dsv_mcp.client as client_mod

    calls = {"n": 0}

    class FakeHttp:
        def post_json(self, url, payload, headers=None, timeout=60):
            calls["n"] += 1
            return {
                "code": 0,
                "data": {
                    "biz_code": 0,
                    "biz_data": {
                        "challenge": {
                            "algorithm": "DeepSeekHashV1",
                            "challenge": "c",
                            "salt": "s",
                            "expire_at": 1786807428328,
                            "difficulty": 1000,
                            "signature": "sig",
                            "target_path": "/api/v0/file/upload_file",
                        }
                    },
                },
            }

    client = client_mod.DeepSeekClient(FakeHttp())

    class FakeSolver:
        def solve_and_build_header(self, challenge):
            return f"header-{calls['n']}"

    monkeypatch.setattr(client_mod, "wasm_solver", lambda: FakeSolver())
    h1 = client.get_pow_for_target(None, "tok", "/x")
    h2 = client.get_pow_for_target(None, "tok", "/x")
    assert h1 == "header-1"
    assert h2 == "header-2"
    assert calls["n"] == 2


def test_detect_captcha_image_signal():
    resp = {
        "code": 0,
        "data": {
            "biz_code": 0,
            "biz_data": {
                "detail": {"bg": "https://captcha.example/bg.png", "rid": "r1"},
            },
        },
    }
    ch = detect_captcha_challenge(resp)
    assert ch is not None
    assert ch["image_url"] == "https://captcha.example/bg.png"


def test_detect_captcha_instruction_signal():
    resp = {"code": 0, "data": {"biz_code": 0, "biz_data": {"instruction": "请点击包含飞机的图片"}}}
    ch = detect_captcha_challenge(resp)
    assert ch is not None
    assert ch["instruction"] == "请点击包含飞机的图片"


def test_detect_captcha_keyword_with_failure():
    resp = {"code": 0, "data": {"biz_code": 40029, "biz_msg": "risk control triggered"}}
    ch = detect_captcha_challenge(resp)
    assert ch is not None


def test_detect_captcha_none_on_normal():
    resp = {"code": 0, "data": {"biz_code": 0, "biz_data": {"id": "ok"}}}
    assert detect_captcha_challenge(resp) is None


def test_create_session_raises_captcha(monkeypatch):
    class FakeHttp:
        def post_json(self, url, payload, headers=None, timeout=60):
            return {
                "code": 0,
                "data": {
                    "biz_code": 0,
                    "biz_data": {
                        "detail": {
                            "bg": "https://captcha.example/x.png",
                            "instruction": "选择所有的红绿灯",
                        }
                    },
                },
            }

    client = DeepSeekClient(FakeHttp())
    with pytest.raises(DeepSeekError) as exc:
        client.create_session(None, "tok")
    assert exc.value.code == "captcha_required"


def test_delete_failure_queued_and_retried(monkeypatch):
    from dsv_mcp import client as client_mod

    class _NoopHttp:
        pass

    c = client_mod.DeepSeekClient(_NoopHttp())
    deletes = {"n": 0}

    def fake_create(account, token):
        return f"sess-{deletes['n'] + 1}"

    def fake_delete(account, token, session_id):
        deletes["n"] += 1
        if deletes["n"] == 1:
            raise DeepSeekError("session_delete_failed", "boom")

    monkeypatch.setattr(c, "create_session", fake_create)
    monkeypatch.setattr(c, "delete_session", fake_delete)
    monkeypatch.setattr(c, "upload_file", lambda *a, **k: "f1")
    monkeypatch.setattr(c, "wait_for_uploaded_file", lambda *a, **k: None)
    monkeypatch.setattr(
        c,
        "call_completion",
        lambda *a, **k: {"text": "ok", "thinking": "", "message_id": 1},
    )
    # 第一次：删除失败 → 入队
    c.describe_image(None, "tok", b"x", "q")
    assert len(c._pending_deletes) == 1
    assert c._pending_deletes[0][2] == "sess-1"
    # 第二次：先补删成功 → 队列清空，本次会话也删掉
    c.describe_image(None, "tok", b"x", "q")
    assert c._pending_deletes == []
    assert deletes["n"] == 3


def test_delete_all_sessions_success():
    calls = {}

    class FakeHttp:
        def post_json(self, url, payload, headers=None, timeout=60):
            calls["url"] = url
            calls["payload"] = payload
            return {"code": 0, "data": {"biz_code": 0, "biz_data": {}}}

    client = DeepSeekClient(FakeHttp())
    client.delete_all_sessions(None, "tok")
    assert calls["url"].endswith("/chat_session/delete_all")
    assert calls["payload"] == {}


def test_delete_all_sessions_failure_raises():
    class FakeHttp:
        def post_json(self, url, payload, headers=None, timeout=60):
            return {"code": 0, "data": {"biz_code": 500, "biz_msg": "boom"}}

    client = DeepSeekClient(FakeHttp())
    with pytest.raises(DeepSeekError) as exc:
        client.delete_all_sessions(None, "tok")
    assert exc.value.code == "session_delete_failed"


def _client_with_fake_flow(monkeypatch, delete, delete_all):
    from dsv_mcp import client as client_mod

    c = client_mod.DeepSeekClient(type("_NoopHttp", (), {})())
    monkeypatch.setattr(c, "create_session", lambda a, t: "s1")
    monkeypatch.setattr(c, "upload_file", lambda *a, **k: "f1")
    monkeypatch.setattr(c, "wait_for_uploaded_file", lambda *a, **k: None)
    monkeypatch.setattr(
        c, "call_completion", lambda *a, **k: {"text": "ok", "thinking": "", "message_id": 1}
    )
    monkeypatch.setattr(c, "delete_session", delete)
    monkeypatch.setattr(c, "delete_all_sessions", delete_all)
    return c


def test_auto_delete_none_skips_delete(monkeypatch):
    calls = {"del": 0, "delall": 0}
    c = _client_with_fake_flow(
        monkeypatch,
        lambda *a, **k: calls.__setitem__("del", calls["del"] + 1),
        lambda *a, **k: calls.__setitem__("delall", calls["delall"] + 1),
    )
    c.describe_image(None, "tok", b"x", "q", auto_delete="none")
    assert calls == {"del": 0, "delall": 0}


def test_auto_delete_single_deletes_current_session(monkeypatch):
    calls = {"del": 0, "delall": 0}
    c = _client_with_fake_flow(
        monkeypatch,
        lambda *a, **k: calls.__setitem__("del", calls["del"] + 1),
        lambda *a, **k: calls.__setitem__("delall", calls["delall"] + 1),
    )
    c.describe_image(None, "tok", b"x", "q", auto_delete="single")
    assert calls == {"del": 1, "delall": 0}
    assert c._pending_deletes == []


def test_auto_delete_all_clears_account_sessions(monkeypatch):
    calls = {"del": 0, "delall": 0}
    c = _client_with_fake_flow(
        monkeypatch,
        lambda *a, **k: calls.__setitem__("del", calls["del"] + 1),
        lambda *a, **k: calls.__setitem__("delall", calls["delall"] + 1),
    )
    c.describe_image(None, "tok", b"x", "q", auto_delete="all")
    assert calls == {"del": 0, "delall": 1}
    assert c._pending_delete_alls == []


def test_delete_all_failure_queued_and_retried(monkeypatch):
    from dsv_mcp import client as client_mod

    c = client_mod.DeepSeekClient(type("_NoopHttp", (), {})())
    monkeypatch.setattr(c, "create_session", lambda a, t: "s1")
    monkeypatch.setattr(c, "upload_file", lambda *a, **k: "f1")
    monkeypatch.setattr(c, "wait_for_uploaded_file", lambda *a, **k: None)
    monkeypatch.setattr(
        c, "call_completion", lambda *a, **k: {"text": "ok", "thinking": "", "message_id": 1}
    )
    deletes = {"n": 0}

    def fake_delete_all(account, token):
        deletes["n"] += 1
        if deletes["n"] == 1:
            raise DeepSeekError("session_delete_failed", "boom")

    monkeypatch.setattr(c, "delete_all_sessions", fake_delete_all)
    # 第一次：清空失败 → 入队
    c.describe_image(None, "tok", b"x", "q", auto_delete="all")
    assert len(c._pending_delete_alls) == 1
    # 第二次：先补删成功 → 队列清空，本次也清空
    c.describe_image(None, "tok", b"x", "q", auto_delete="all")
    assert c._pending_delete_alls == []
    assert deletes["n"] == 3
