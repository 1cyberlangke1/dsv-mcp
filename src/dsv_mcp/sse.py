"""DeepSeek SSE 流解析：正文/思考分离、终态跟踪、续写信号。"""

from __future__ import annotations

import json
import re
from typing import Any

from dsv_mcp.protocol import DEFAULT_SKIP_CONTAINS_PATTERNS, DEFAULT_SKIP_EXACT_PATHS


MIN_CONTINUATION_SNAPSHOT_LEN = 32


def trim_continuation_overlap(existing: str, incoming: str) -> str:
    """TrimContinuationOverlap：前缀快照式去重。"""
    if incoming == "":
        return ""
    if existing == "":
        return incoming
    if len(incoming) < MIN_CONTINUATION_SNAPSHOT_LEN:
        return incoming
    if len(incoming) > len(existing):
        if incoming.startswith(existing):
            return incoming[len(existing) :]
        return incoming
    if len(incoming) < len(existing) and existing.startswith(incoming):
        return ""
    return incoming


def parse_deepseek_sse_line(raw: str) -> tuple[dict[str, Any] | None, bool, bool]:
    """ParseDeepSeekSSELine：返回 (chunk, done, parsed)。"""
    line = raw.strip()
    if line == "" or not line.startswith("data:"):
        return None, False, False
    data_str = line[len("data:") :].strip()
    if data_str == "[DONE]":
        return None, True, True
    try:
        chunk = json.loads(data_str)
    except json.JSONDecodeError:
        return None, False, False
    if not isinstance(chunk, dict):
        return None, False, False
    return chunk, False, True


def is_fragment_status_path(path: str) -> bool:
    """isFragmentStatusPath。"""
    if path == "" or path == "response/status":
        return False
    if not path.startswith("response/fragments/") or not path.endswith("/status"):
        return False
    mid = path[len("response/fragments/") : -len("/status")]
    if mid == "":
        return False
    mid = mid.lstrip("-")
    if mid == "":
        return False
    return mid.isdigit()


def should_skip_path(path: str) -> bool:
    """shouldSkipPath。"""
    if is_fragment_status_path(path):
        return True
    if path in DEFAULT_SKIP_EXACT_PATHS:
        return True
    return any(p in path for p in DEFAULT_SKIP_CONTAINS_PATTERNS)


def is_status_path(path: str) -> bool:
    return path == "response/status" or path == "status"


def parse_fragment_type_content(m: dict[str, Any]) -> tuple[str, str]:
    """parseFragmentTypeContent：返回 (upper_type, content)。"""
    return str(m.get("type") or "").upper(), str(m.get("content") or "")


def append_content_part(parts: list, content: str, kind: str) -> None:
    if content == "":
        return
    parts.append({"text": content, "type": kind})


def collect_direct_fragments(
    path: str, chunk: dict[str, Any], v: Any, new_type: list[str], parts: list
) -> None:
    """collectDirectFragments。new_type 是单元素 list（模拟指针）。"""
    if path != "response/fragments":
        return
    op = str(chunk.get("o") or "")
    if op.upper() != "APPEND":
        return
    if not isinstance(v, list):
        return
    for frag in v:
        if not isinstance(frag, dict):
            continue
        type_name, content = parse_fragment_type_content(frag)
        if type_name == "THINK" or type_name == "THINKING":
            new_type[0] = "thinking"
            append_content_part(parts, content, "thinking")
        elif type_name == "RESPONSE":
            new_type[0] = "text"
            append_content_part(parts, content, "text")
        else:
            append_content_part(parts, content, "text")


def update_type_from_explicit_path(path: str, thinking_enabled: bool, new_type: list[str]) -> None:
    """updateTypeFromExplicitPath。"""
    if path == "response/content":
        new_type[0] = "text"
    elif path == "response/thinking_content":
        if not thinking_enabled or new_type[0] != "text":
            new_type[0] = "thinking"


def update_type_from_nested_response(path: str, v: Any, new_type: list[str]) -> None:
    """updateTypeFromNestedResponse。"""
    if path != "response" or not isinstance(v, list):
        return
    for it in v:
        if not isinstance(it, dict) or it.get("p") != "fragments" or it.get("o") != "APPEND":
            continue
        frags = it.get("v")
        if not isinstance(frags, list):
            continue
        for fm in frags:
            if not isinstance(fm, dict):
                continue
            type_name, _ = parse_fragment_type_content(fm)
            if type_name == "THINK" or type_name == "THINKING":
                new_type[0] = "thinking"
            elif type_name == "RESPONSE":
                new_type[0] = "text"


def resolve_part_type(path: str, thinking_enabled: bool, new_type: str) -> str:
    """resolvePartType。"""
    if path == "response/thinking_content":
        if not thinking_enabled:
            return "thinking"
        if new_type == "text":
            return "text"
        return "thinking"
    if path == "response/content":
        return "text"
    if "response/fragments" in path and "/content" in path:
        return new_type
    if path == "":
        if new_type != "":
            return new_type
        return "text"
    return "text"


def append_object_content_by_path(
    path: str, val: dict[str, Any], part_type: str, parts: list
) -> bool:
    """appendObjectContentByPath。"""
    if path not in ("response/content", "response/thinking_content", ""):
        return False
    text = str(val.get("text") or "")
    if text == "":
        text = str(val.get("content") or "")
    if text == "":
        return False
    append_content_part(parts, text, part_type)
    return True


def append_wrapped_fragments(
    val: dict[str, Any], part_type: str, new_type: list[str], parts: list
) -> None:
    """appendWrappedFragments。"""
    resp = val
    if isinstance(val.get("response"), dict):
        resp = val["response"]
    frags = resp.get("fragments")
    if not isinstance(frags, list):
        return
    for item in frags:
        if not isinstance(item, dict):
            continue
        type_name, content = parse_fragment_type_content(item)
        if type_name == "THINK" or type_name == "THINKING":
            new_type[0] = "thinking"
            append_content_part(parts, content, "thinking")
        elif type_name == "RESPONSE":
            new_type[0] = "text"
            append_content_part(parts, content, "text")
        else:
            append_content_part(parts, content, part_type)


def append_chunk_value_content(
    v: Any, part_type: str, new_type: list[str], parts: list, path: str
) -> bool:
    """appendChunkValueContent：返回 finished。"""
    if isinstance(v, str):
        if v == "FINISHED" and (path == "" or path == "status"):
            return True
        if is_status_path(path):
            return False
        append_content_part(parts, v, part_type)
    elif isinstance(v, list):
        pp, finished = extract_content_recursive(v, part_type)
        if finished:
            return True
        parts.extend(pp)
    elif isinstance(v, dict):
        if append_object_content_by_path(path, v, part_type, parts):
            return False
        append_wrapped_fragments(v, part_type, new_type, parts)
    return False


THINK_CLOSE_PATTERN = re.compile(r"(?i)</\s*think\s*>")
THINK_OPEN_PATTERN = re.compile(r"(?i)<\s*think\s*>")


def strip_think_tags(s: str) -> str:
    """stripThinkTags。"""
    s = THINK_CLOSE_PATTERN.sub("", s)
    s = THINK_OPEN_PATTERN.sub("", s)
    return s


def split_thinking_parts(parts: list) -> tuple[list, bool]:
    """splitThinkingParts：</think> 后自动转为 text。"""
    out: list = []
    thinking_done = False
    for p in parts:
        if thinking_done and p["type"] == "thinking":
            cleaned = strip_think_tags(p["text"])
            if cleaned != "":
                out.append({"text": cleaned, "type": "text"})
            continue
        if p["type"] != "thinking":
            cleaned = strip_think_tags(p["text"])
            if cleaned != "":
                out.append({"text": cleaned, "type": p["type"]})
            continue
        loc = THINK_CLOSE_PATTERN.search(p["text"])
        if loc is None:
            out.append(p)
            continue
        thinking_done = True
        before = p["text"][: loc.start()]
        after = p["text"][loc.end() :]
        if before != "":
            out.append({"text": before, "type": "thinking"})
        after = strip_think_tags(after)
        if after != "":
            out.append({"text": after, "type": "text"})
    return out, thinking_done


def extract_content_recursive(items: list, default_type: str) -> tuple[list, bool]:
    """extractContentRecursive：返回 (parts, finished)。"""
    parts: list = []
    for it in items:
        if not isinstance(it, dict):
            continue
        item_path = str(it.get("p") or "")
        has_v = "v" in it
        if not has_v:
            continue
        item_v = it["v"]
        if is_status_path(item_path):
            if isinstance(item_v, str) and item_v.strip().upper() == "FINISHED":
                return [], True
            continue
        if should_skip_path(item_path):
            continue
        content = it.get("content")
        if isinstance(content, str) and content != "":
            type_name = str(it.get("type") or "").upper()
            if type_name == "THINK" or type_name == "THINKING":
                parts.append({"text": content, "type": "thinking"})
            elif type_name == "RESPONSE":
                parts.append({"text": content, "type": "text"})
            else:
                parts.append({"text": content, "type": default_type})
            continue
        part_type = default_type
        if "thinking" in item_path:
            part_type = "thinking"
        elif "content" in item_path or item_path == "response" or item_path == "fragments":
            part_type = "text"
        if isinstance(item_v, str):
            if is_status_path(item_path):
                continue
            if item_v != "" and item_v != "FINISHED":
                parts.append({"text": item_v, "type": part_type})
        elif isinstance(item_v, list):
            for inner in item_v:
                if isinstance(inner, dict):
                    ct = inner.get("content")
                    if not isinstance(ct, str) or ct == "":
                        continue
                    type_name = str(inner.get("type") or "").upper()
                    if type_name == "THINK" or type_name == "THINKING":
                        parts.append({"text": ct, "type": "thinking"})
                    elif type_name == "RESPONSE":
                        parts.append({"text": ct, "type": "text"})
                    else:
                        parts.append({"text": ct, "type": part_type})
                elif isinstance(inner, str):
                    if inner != "":
                        parts.append({"text": inner, "type": part_type})
    return parts, False


def parse_sse_chunk_for_content_detailed(
    chunk: dict[str, Any], thinking_enabled: bool, current_fragment_type: str
) -> tuple[list, list, bool, str]:
    """ParseSSEChunkForContentDetailed。"""
    if "v" not in chunk:
        return [], [], False, current_fragment_type
    v = chunk["v"]
    path = str(chunk.get("p") or "")
    if should_skip_path(path):
        return [], [], False, current_fragment_type
    if is_status_path(path):
        if isinstance(v, str) and v.strip().upper() == "FINISHED":
            return [], [], True, current_fragment_type
        return [], [], False, current_fragment_type
    new_type = [current_fragment_type]
    parts: list = []
    update_type_from_explicit_path(path, thinking_enabled, new_type)
    collect_direct_fragments(path, chunk, v, new_type, parts)
    update_type_from_nested_response(path, v, new_type)
    part_type = resolve_part_type(path, thinking_enabled, new_type[0])
    finished = append_chunk_value_content(v, part_type, new_type, parts, path)
    if finished:
        return [], [], True, new_type[0]
    parts, transitioned = split_thinking_parts(parts)
    if transitioned:
        new_type[0] = "text"
    detection_parts = [p for p in parts if p["type"] == "thinking"]
    if not thinking_enabled:
        parts = [p for p in parts if p["type"] != "thinking"]
    return parts, detection_parts, False, new_type[0]


def strip_leaked_content_filter_suffix(text: str) -> tuple[str, bool]:
    """stripLeakedContentFilterSuffix。"""
    if text == "":
        return text, False
    idx = text.upper().find("CONTENT_FILTER")
    if idx < 0:
        return text, False
    return text[:idx].rstrip(" \t\r"), True


def should_drop_cleaned_leaked_chunk(cleaned: str) -> bool:
    """shouldDropCleanedLeakedChunk。"""
    if cleaned == "":
        return True
    if "\n" in cleaned:
        return False
    return cleaned.strip() == ""


def filter_leaked_content_filter_parts(parts: list) -> list:
    """filterLeakedContentFilterParts。"""
    if len(parts) == 0:
        return parts
    out: list = []
    for p in parts:
        cleaned, stripped = strip_leaked_content_filter_suffix(p["text"])
        if stripped and should_drop_cleaned_leaked_chunk(cleaned):
            continue
        if stripped:
            p["text"] = cleaned
        out.append(p)
    return out


def has_content_filter_status_value(v: Any) -> bool:
    """hasContentFilterStatusValue。"""
    if isinstance(v, list):
        for item in v:
            if has_content_filter_status_value(item):
                return True
    elif isinstance(v, dict):
        if "status" in str(v.get("p") or "").lower():
            if str(v.get("v") or "").strip().upper() == "CONTENT_FILTER":
                return True
        if str(v.get("code") or "").strip().upper() == "CONTENT_FILTER":
            return True
        for vv in v.values():
            if has_content_filter_status_value(vv):
                return True
    return False


def has_content_filter_status(chunk: dict[str, Any]) -> bool:
    """hasContentFilterStatus。"""
    if str(chunk.get("code") or "").strip().upper() == "CONTENT_FILTER":
        return True
    return has_content_filter_status_value(chunk)


def extract_hint_error(chunk: dict[str, Any]) -> tuple[str, bool]:
    """extractHintError：type=error 顶层错误载荷。"""
    type_val = str(chunk.get("type") or "")
    if type_val.strip().upper() != "ERROR":
        return "", False
    content = str(chunk.get("content") or "").strip()
    if content == "":
        return "", False
    return content, True


def observe_response_message_id(chunk: dict[str, Any]) -> int:
    """observeResponseMessageID。"""
    out = 0
    try:
        mid = int(chunk.get("response_message_id") or 0)
        if mid > 0:
            out = mid
    except (TypeError, ValueError):
        pass
    v = chunk.get("v")
    if isinstance(v, dict) and isinstance(v.get("response"), dict):
        try:
            mid = int(v["response"].get("message_id") or 0)
            if mid > 0:
                out = mid
        except (TypeError, ValueError):
            pass
    message = chunk.get("message")
    if isinstance(message, dict) and isinstance(message.get("response"), dict):
        try:
            mid = int(message["response"].get("message_id") or 0)
            if mid > 0:
                out = mid
        except (TypeError, ValueError):
            pass
    return out


def parse_deepseek_content_line(
    raw: str, thinking_enabled: bool, current_type: str
) -> dict[str, Any]:
    """ParseDeepSeekContentLine：单行解析归一化结果。"""
    chunk, done, parsed = parse_deepseek_sse_line(raw)
    if not parsed:
        return {"parsed": False, "stop": False, "next_type": current_type}
    if done:
        return {"parsed": True, "stop": True, "next_type": current_type}
    if "error" in chunk:
        return {"parsed": True, "stop": True, "error_message": str(chunk["error"]), "next_type": current_type}
    if str(chunk.get("code") or "") == "content_filter":
        return {"parsed": True, "stop": True, "content_filter": True, "next_type": current_type}
    if has_content_filter_status(chunk):
        return {"parsed": True, "stop": True, "content_filter": True, "next_type": current_type}
    msg, ok = extract_hint_error(chunk)
    if ok:
        return {"parsed": True, "stop": True, "error_message": msg, "next_type": current_type}
    parts, detection_parts, finished, next_type = parse_sse_chunk_for_content_detailed(
        chunk, thinking_enabled, current_type
    )
    parts = filter_leaked_content_filter_parts(parts)
    detection_parts = filter_leaked_content_filter_parts(detection_parts)
    resp_msg_id = observe_response_message_id(chunk)
    return {
        "parsed": True,
        "stop": finished,
        "parts": parts,
        "tool_detection_parts": detection_parts,
        "next_type": next_type,
        "response_message_id": resp_msg_id,
    }


class CollectResult:
    """CollectResult。"""

    def __init__(self) -> None:
        self.text = ""
        self.thinking = ""
        self.content_filter = False
        self.upstream_error = ""
        self.response_message_id = 0


def collect_stream(lines, thinking_enabled: bool) -> CollectResult:
    """CollectStream：完整消费 SSE 流并分离思考/正文。"""
    result = CollectResult()
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    content_filter = False
    upstream_error = ""
    stopped = False
    response_message_id = 0
    current_type = "thinking" if thinking_enabled else "text"
    for line in lines:
        chunk, done, parsed = parse_deepseek_sse_line(line)
        if parsed and not done:
            mid = observe_response_message_id(chunk)
            if mid > 0:
                response_message_id = mid
        if done:
            break
        if stopped:
            continue
        result_line = parse_deepseek_content_line(line, thinking_enabled, current_type)
        current_type = result_line["next_type"]
        if not result_line["parsed"]:
            continue
        if result_line["stop"]:
            if result_line.get("content_filter"):
                content_filter = True
            if result_line.get("error_message") and upstream_error == "":
                upstream_error = result_line["error_message"]
            stopped = True
            continue
        for p in result_line.get("parts", []):
            if p["type"] == "thinking":
                thinking_parts.append(trim_continuation_overlap("".join(thinking_parts), p["text"]))
            else:
                text_parts.append(trim_continuation_overlap("".join(text_parts), p["text"]))
    result.text = "".join(text_parts)
    result.thinking = "".join(thinking_parts)
    result.content_filter = content_filter
    result.upstream_error = upstream_error
    result.response_message_id = response_message_id
    return result
