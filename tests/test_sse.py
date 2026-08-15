"""SSE 解析测试（离线构造 chunk，不联网）。"""

from __future__ import annotations

from dsv_mcp.sse import (
    collect_stream,
    parse_deepseek_content_line,
    parse_deepseek_sse_line,
    trim_continuation_overlap,
)


def test_parse_done():
    chunk, done, parsed = parse_deepseek_sse_line("data: [DONE]")
    assert done and parsed


def test_parse_json_chunk():
    chunk, done, parsed = parse_deepseek_sse_line('data: {"p":"response/status","v":"WIP"}')
    assert parsed and not done
    assert chunk["p"] == "response/status"


def test_collect_text_and_finished():
    lines = [
        'data: {"response_message_id": 7, "p": "response/fragments", "o": "APPEND", '
        '"v": [{"type": "RESPONSE", "content": "你好"}]}',
        'data: {"p": "response/status", "v": "FINISHED"}',
    ]
    result = collect_stream(lines, thinking_enabled=False)
    assert result.text == "你好"
    assert result.thinking == ""
    assert result.response_message_id == 7


def test_collect_thinking_separated():
    lines = [
        'data: {"p": "response/fragments", "o": "APPEND", '
        '"v": [{"type": "THINK", "content": "思考中"}]}',
        'data: {"p": "response/fragments", "o": "APPEND", '
        '"v": [{"type": "RESPONSE", "content": "答案"}]}',
        'data: {"p": "response/status", "v": "FINISHED"}',
    ]
    result = collect_stream(lines, thinking_enabled=True)
    assert result.text == "答案"
    assert result.thinking == "思考中"


def test_collect_thinking_dropped_when_disabled():
    lines = [
        'data: {"p": "response/fragments", "o": "APPEND", '
        '"v": [{"type": "THINK", "content": "思考中"}]}',
        'data: {"p": "response/status", "v": "FINISHED"}',
    ]
    result = collect_stream(lines, thinking_enabled=False)
    assert result.text == ""
    assert result.thinking == ""


def test_collect_content_filter():
    lines = [
        'data: {"code": "content_filter"}',
    ]
    result = collect_stream(lines, thinking_enabled=False)
    assert result.content_filter


def test_collect_done_stops():
    lines = [
        'data: {"p": "response/fragments", "o": "APPEND", '
        '"v": [{"type": "RESPONSE", "content": "a"}]}',
        "data: [DONE]",
        'data: {"p": "response/fragments", "o": "APPEND", '
        '"v": [{"type": "RESPONSE", "content": "不应出现"}]}',
    ]
    result = collect_stream(lines, thinking_enabled=False)
    assert result.text == "a"


def test_split_think_tag_transitions_to_text():
    lines = [
        'data: {"p": "response/fragments", "o": "APPEND", '
        '"v": [{"type": "THINK", "content": "前半</think>后半"}]}',
        'data: {"p": "response/status", "v": "FINISHED"}',
    ]
    result = collect_stream(lines, thinking_enabled=True)
    assert result.thinking == "前半"
    assert result.text == "后半"


def test_trim_continuation_overlap():
    assert trim_continuation_overlap("", "abc") == "abc"
    assert trim_continuation_overlap("x" * 40, ("x" * 40) + "more") == "more"
    # 短片段（<32 字符）不做裁剪，原样返回
    assert trim_continuation_overlap("x" * 40, "x" * 20) == "x" * 20


def test_parse_content_line_error():
    result = parse_deepseek_content_line('data: {"type": "error", "content": "内容超长"}', False, "text")
    assert result["stop"]
    assert result["error_message"] == "内容超长"
