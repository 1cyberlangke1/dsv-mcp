"""客户端纯函数测试（响应解析、手机号规范化）。"""

from __future__ import annotations

from dsv_mcp.client import (
    extract_create_session_id,
    extract_response_status,
    extract_upload_file_result,
    int_from,
    normalize_mobile_for_login,
)


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
