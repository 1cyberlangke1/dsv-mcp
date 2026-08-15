"""DeepSeek 网页版 API 客户端。"""

from __future__ import annotations

import time
import uuid
from typing import Any

from dsv_mcp.config import Account
from dsv_mcp.http import HttpClient
from dsv_mcp.pow import wasm_solver
from dsv_mcp.protocol import (
    COMPLETION_TARGET_PATH,
    COMPLETION_URL,
    CONTINUE_URL,
    CREATE_POW_URL,
    CREATE_SESSION_URL,
    DELETE_SESSION_URL,
    FETCH_FILES_URL,
    LOGIN_URL,
    STOP_STREAM_URL,
    UPLOAD_TARGET_PATH,
    UPLOAD_URL,
    chat_session_referer,
    login_headers,
)
from dsv_mcp.sse import collect_stream


class DeepSeekError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class LoginError(DeepSeekError):
    def __init__(self, message: str):
        super().__init__("auth_failed", message)


# defaultAutoContinueLimit
DEFAULT_AUTO_CONTINUE_LIMIT = 32
# fileReadyPoll*
FILE_READY_POLL_ATTEMPTS = 60
FILE_READY_POLL_INTERVAL = 1.0
FILE_READY_POLL_TIMEOUT = 65.0


def int_from(v: Any) -> int:
    """util.IntFrom。"""
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        try:
            return int(float(v))
        except ValueError:
            return 0
    return 0


def normalize_mobile_for_login(raw: str) -> tuple[str, str]:
    """normalizeMobileForLogin：返回 (mobile, area_code)。"""
    s = raw.strip()
    if s == "":
        return "", ""
    has_plus = s.startswith("+")
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return "", ""
    if (has_plus or digits.startswith("86")) and digits.startswith("86") and len(digits) == 13:
        return digits[2:], "+86"
    return digits, "+86"


def extract_response_status(resp: dict[str, Any]) -> tuple[int, int, str, str]:
    """extractResponseStatus。"""
    code = int_from(resp.get("code"))
    msg = str(resp.get("msg") or "")
    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
    biz_code = int_from(data.get("biz_code"))
    biz_msg = str(data.get("biz_msg") or "")
    if biz_msg.strip() == "":
        biz_data = data.get("biz_data") if isinstance(data.get("biz_data"), dict) else {}
        biz_msg = str(biz_data.get("msg") or "")
    return code, biz_code, msg, biz_msg


def extract_create_session_id(resp: dict[str, Any]) -> str:
    """extractCreateSessionID：biz_data.id 或 biz_data.chat_session.id。"""
    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
    biz_data = data.get("biz_data") if isinstance(data.get("biz_data"), dict) else {}
    session_id = str(biz_data.get("id") or "").strip()
    if session_id != "":
        return session_id
    chat_session = biz_data.get("chat_session")
    if isinstance(chat_session, dict):
        session_id = str(chat_session.get("id") or "").strip()
        if session_id != "":
            return session_id
    return ""


def extract_upload_file_result(resp: dict[str, Any]) -> dict[str, Any]:
    """extractUploadFileResult：从 resp/data/biz_data 嵌套查找。"""
    result: dict[str, Any] = {"id": "", "status": "uploaded"}
    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
    biz_data = data.get("biz_data") if isinstance(data.get("biz_data"), dict) else {}
    search_maps: list[dict[str, Any]] = [resp, data, biz_data]
    for parent in (resp, data, biz_data):
        for key in ("file", "biz_data", "data"):
            nested = parent.get(key)
            if isinstance(nested, dict):
                search_maps.append(nested)
    for m in search_maps:
        if result["id"] == "":
            for key in ("id", "file_id"):
                v = str(m.get(key) or "").strip()
                if v != "":
                    result["id"] = v
                    break
        if result["status"] == "uploaded":
            for key in ("status", "file_status"):
                v = str(m.get(key) or "").strip()
                if v != "":
                    result["status"] = v
                    break
    return result


class DeepSeekClient:
    def __init__(self, http: HttpClient):
        self.http = http
        self._pow_cache: dict[tuple[str, str], tuple[float, str]] = {}

    # ---------- 认证 ----------

    def login(self, account: Account, device_id: str | None = None) -> str:
        """登录并返回 token。payload client_auth.go Login。"""
        device_id = device_id or uuid.uuid4().hex
        payload: dict[str, Any] = {
            "email": "",
            "mobile": "",
            "password": (account.password or "").strip(),
            "area_code": "",
            "device_id": device_id,
            "os": "web",
        }
        email = (account.email or "").strip()
        mobile = (account.mobile or "").strip()
        if email != "":
            payload["email"] = email
        elif mobile != "":
            login_mobile, area_code = normalize_mobile_for_login(mobile)
            payload["mobile"] = login_mobile
            payload["area_code"] = area_code
        else:
            raise LoginError("账号缺少邮箱或手机号")
        resp = self.http.post_json(LOGIN_URL, payload, headers=login_headers(account.locale))
        code = int_from(resp.get("code"))
        if code != 0:
            raise LoginError(f"login failed: {resp.get('msg') or code}")
        data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
        biz_code = int_from(data.get("biz_code"))
        if biz_code != 0:
            raise LoginError(f"login failed: {data.get('biz_msg')}")
        biz_data = data.get("biz_data") if isinstance(data.get("biz_data"), dict) else {}
        user = biz_data.get("user") if isinstance(biz_data.get("user"), dict) else {}
        token = str(user.get("token") or "").strip()
        if token == "":
            raise LoginError("login succeeded without token")
        return token

    # ---------- PoW ----------

    def get_pow_for_target(self, account: Account, token: str, target_path: str) -> str:
        """获取并求解 PoW（challenge 一次性，取走即删）。"""
        cache_key = (account.identifier(), target_path)
        cached = self._pow_cache.get(cache_key)
        if cached:
            expire_at, header = cached
            self._pow_cache.pop(cache_key, None)
            if expire_at > time.time() + 30:
                return header
        resp = self.http.post_json(
            CREATE_POW_URL, {"target_path": target_path}, headers=self._auth_headers(token)
        )
        code, biz_code, _, biz_msg = extract_response_status(resp)
        if code != 0 or biz_code != 0:
            raise DeepSeekError("pow_failed", f"获取 PoW 失败: {biz_msg or code}")
        data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
        biz_data = data.get("biz_data") if isinstance(data.get("biz_data"), dict) else {}
        challenge = biz_data.get("challenge") if isinstance(biz_data.get("challenge"), dict) else {}
        if not challenge.get("challenge"):
            raise DeepSeekError("pow_failed", "PoW 响应缺少 challenge")
        header = wasm_solver().solve_and_build_header(challenge)
        self._pow_cache[cache_key] = (float(challenge.get("expire_at") or 0), header)
        return header

    # ---------- 会话 ----------

    def create_session(self, account: Account, token: str) -> str:
        """CreateSession：空 payload。"""
        resp = self.http.post_json(CREATE_SESSION_URL, {}, headers=self._auth_headers(token))
        code, biz_code, _, biz_msg = extract_response_status(resp)
        if code != 0 or biz_code != 0:
            raise DeepSeekError("session_failed", f"创建会话失败: {biz_msg or code}")
        session_id = extract_create_session_id(resp)
        if session_id == "":
            raise DeepSeekError("session_failed", "创建会话缺少 id")
        return session_id

    def delete_session(self, account: Account, token: str, session_id: str) -> None:
        resp = self.http.post_json(
            DELETE_SESSION_URL,
            {"chat_session_id": session_id},
            headers=self._session_headers(token, session_id),
        )
        code, biz_code, _, biz_msg = extract_response_status(resp)
        if code != 0 or biz_code != 0:
            raise DeepSeekError("session_delete_failed", f"删除会话失败: {biz_msg or code}")

    def stop_stream(self, account: Account, token: str, session_id: str, message_id: int) -> None:
        """StopStream。"""
        if session_id.strip() == "" or message_id <= 0:
            return
        resp = self.http.post_json(
            STOP_STREAM_URL,
            {"chat_session_id": session_id, "message_id": message_id},
            headers=self._session_headers(token, session_id),
        )
        code, biz_code, _, biz_msg = extract_response_status(resp)
        if code != 0 or biz_code != 0:
            raise DeepSeekError("stop_failed", f"停止生成失败: {biz_msg or code}")

    # ---------- 文件上传 ----------

    def upload_file(
        self, account: Account, token: str, filename: str, content_type: str, data: bytes
    ) -> str:
        """UploadFile：multipart 上传，返回 file id。"""
        pow_header = self.get_pow_for_target(account, token, UPLOAD_TARGET_PATH)
        headers = self._auth_headers(token)
        headers["x-ds-pow-response"] = pow_header
        headers["x-file-size"] = str(len(data))
        headers["x-thinking-enabled"] = "1"
        headers["x-model-type"] = "vision"
        resp = self.http.post_multipart(
            UPLOAD_URL, files={"file": (filename, data, content_type)}, headers=headers
        )
        code, biz_code, _, biz_msg = extract_response_status(resp)
        if code != 0 or biz_code != 0:
            if biz_code == 40301:
                raise DeepSeekError("upload_rate_limited", "上传过于频繁，请稍后再试")
            raise DeepSeekError("upload_failed", f"上传失败: {biz_msg or code}")
        result = extract_upload_file_result(resp)
        if result["id"] == "":
            raise DeepSeekError("upload_failed", "上传成功但缺少 file id")
        return result["id"]

    def wait_for_uploaded_file(self, account: Account, token: str, file_id: str) -> None:
        """waitForUploadedFile：轮询 fetch_files 直到 READY。"""
        deadline = time.monotonic() + FILE_READY_POLL_TIMEOUT
        attempt = 0
        while attempt < FILE_READY_POLL_ATTEMPTS and time.monotonic() < deadline:
            resp = self.http.get_json(
                FETCH_FILES_URL + "?file_ids=" + file_id, headers=self._auth_headers(token)
            )
            code, biz_code, _, _ = extract_response_status(resp)
            if code == 0 and biz_code == 0:
                status = self._find_file_status(resp, file_id)
                if status is not None and status.upper() in ("READY", "PARSED", "SUCCESS"):
                    return
            attempt += 1
            if attempt < FILE_READY_POLL_ATTEMPTS:
                time.sleep(FILE_READY_POLL_INTERVAL)
        raise DeepSeekError("upload_failed", f"文件 {file_id} 未在超时内就绪")

    @staticmethod
    def _find_file_status(obj: Any, target_id: str) -> str | None:
        """递归查找目标文件状态。"""
        if isinstance(obj, dict):
            if obj.get("id") == target_id and "status" in obj:
                return str(obj["status"])
            for v in obj.values():
                found = DeepSeekClient._find_file_status(v, target_id)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = DeepSeekClient._find_file_status(item, target_id)
                if found is not None:
                    return found
        return None

    # ---------- 识图对话 ----------

    def describe_image(
        self,
        account: Account,
        token: str,
        image_bytes: bytes,
        prompt: str,
        filename: str = "image.jpg",
        content_type: str = "image/jpeg",
        thinking_enabled: bool = False,
        auto_delete: bool = True,
    ) -> dict[str, Any]:
        """单轮识图：建会话 → 上传 → 就绪 → vision 对话 →（可选）删会话。"""
        session_id = self.create_session(account, token)
        try:
            file_id = self.upload_file(account, token, filename, content_type, image_bytes)
            self.wait_for_uploaded_file(account, token, file_id)
            result = self.call_completion(
                account,
                token,
                session_id,
                prompt=prompt,
                ref_file_ids=[file_id],
                thinking_enabled=thinking_enabled,
            )
            return result
        finally:
            if auto_delete:
                try:
                    self.delete_session(account, token, session_id)
                except DeepSeekError:
                    pass

    def call_completion(
        self,
        account: Account,
        token: str,
        session_id: str,
        prompt: str,
        ref_file_ids: list[str] | None = None,
        thinking_enabled: bool = False,
    ) -> dict[str, Any]:
        """CallCompletion + wrapCompletionWithAutoContinue。"""
        # 标准请求 payload（）
        payload: dict[str, Any] = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "model_type": "vision",
            "prompt": prompt,
            "ref_file_ids": ref_file_ids or [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": False,
            "action": None,
            "preempt": False,
        }
        result = self._completion_once(
            account, token, session_id, COMPLETION_URL, payload, thinking_enabled
        )
        # 自动续写：INCOMPLETE / 流中断 → continue（defaultAutoContinueLimit=32）
        message_id = result["response_message_id"]
        rounds = 0
        while result["need_continue"] and rounds < DEFAULT_AUTO_CONTINUE_LIMIT:
            rounds += 1
            cont_payload = {
                "chat_session_id": session_id,
                "message_id": message_id,
                "fallback_to_resume": True,
            }
            result = self._completion_once(
                account, token, session_id, CONTINUE_URL, cont_payload, thinking_enabled
            )
            message_id = result["response_message_id"]
        if result["content_filter"]:
            raise DeepSeekError("content_filter", "上游内容过滤")
        if result["upstream_error"]:
            raise DeepSeekError("upstream_error", result["upstream_error"])
        return {
            "text": result["text"],
            "thinking": result["thinking"],
            "message_id": message_id,
        }

    def _completion_once(
        self,
        account: Account,
        token: str,
        session_id: str,
        url: str,
        payload: dict[str, Any],
        thinking_enabled: bool,
    ) -> dict[str, Any]:
        """发起一次 completion/continue 流请求，返回解析结果。"""
        pow_header = self.get_pow_for_target(account, token, COMPLETION_TARGET_PATH)
        headers = self._auth_headers(token)
        headers["x-ds-pow-response"] = pow_header
        headers["Referer"] = chat_session_referer(session_id)
        collected = collect_stream(self.http.stream(url, payload, headers=headers), thinking_enabled)
        return {
            "text": collected.text,
            "thinking": collected.thinking,
            "content_filter": collected.content_filter,
            "upstream_error": collected.upstream_error,
            "response_message_id": collected.response_message_id,
            "need_continue": (
                collected.response_message_id > 0
                and not collected.content_filter
                and not collected.upstream_error
                and collected.text == ""
                and collected.thinking == ""
            ),
        }

    # ---------- 工具 ----------

    def _auth_headers(self, token: str) -> dict[str, str]:
        """authHeaders。"""
        return {"authorization": f"Bearer {token}"}

    def _session_headers(self, token: str, session_id: str) -> dict[str, str]:
        return {
            "authorization": f"Bearer {token}",
            "Referer": chat_session_referer(session_id),
        }
