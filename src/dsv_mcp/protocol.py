"""协议常量与请求头。.go。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


BASE_URL = "https://chat.deepseek.com"

LOGIN_URL = BASE_URL + "/api/v0/users/login"
CREATE_SESSION_URL = BASE_URL + "/api/v0/chat_session/create"
CREATE_POW_URL = BASE_URL + "/api/v0/chat/create_pow_challenge"
COMPLETION_URL = BASE_URL + "/api/v0/chat/completion"
CONTINUE_URL = BASE_URL + "/api/v0/chat/continue"
STOP_STREAM_URL = BASE_URL + "/api/v0/chat/stop_stream"
UPLOAD_URL = BASE_URL + "/api/v0/file/upload_file"
FETCH_FILES_URL = BASE_URL + "/api/v0/file/fetch_files"
DELETE_SESSION_URL = BASE_URL + "/api/v0/chat_session/delete"
DELETE_ALL_SESSIONS_URL = BASE_URL + "/api/v0/chat_session/delete_all"

COMPLETION_TARGET_PATH = "/api/v0/chat/completion"
UPLOAD_TARGET_PATH = "/api/v0/file/upload_file"

# transport.ChromeMajorVersion = "150"
CHROME_MAJOR_VERSION = "150"
CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/" + CHROME_MAJOR_VERSION + ".0.0.0 Safari/537.36"
)
CHROME_SEC_CH_UA = (
    '"Not;A=Brand";v="8", "Chromium";v="' + CHROME_MAJOR_VERSION
    + '", "Google Chrome";v="' + CHROME_MAJOR_VERSION + '"'
)

# defaultStaticBaseHeaders
STATIC_BASE_HEADERS = {
    "Host": "chat.deepseek.com",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "x-client-bundle-id": "com.deepseek.chat",
}

# webBrowserHeaders
WEB_BROWSER_HEADERS = {
    "User-Agent": CHROME_UA,
    "sec-ch-ua": CHROME_SEC_CH_UA,
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Origin": BASE_URL,
    "Referer": BASE_URL + "/",
    "sec-fetch-site": "same-origin",
    "sec-fetch-mode": "cors",
    "sec-fetch-dest": "empty",
    # 浏览器 fetch 发 */*，不是 application/json
    "Accept": "*/*",
    # 必须显式声明，否则只接受 gzip 是明显异常
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "priority": "u=1, i",
}

# localeTimezones
LOCALE_TIMEZONES = {
    "zh_CN": "Asia/Shanghai",
    "zh_TW": "Asia/Taipei",
    "en_US": "America/Los_Angeles",
    "en_GB": "Europe/London",
    "ja_JP": "Asia/Tokyo",
    "ko_KR": "Asia/Seoul",
}
DEFAULT_TIMEZONE_OFFSET = "28800"


def timezone_offset_for(locale: str) -> str:
    """按 locale 实时计算时区偏移（秒，含夏令时）。"""
    zone = LOCALE_TIMEZONES.get(locale, "Asia/Shanghai")
    try:
        return str(int(datetime.now(ZoneInfo(zone)).utcoffset().total_seconds()))
    except Exception:
        return DEFAULT_TIMEZONE_OFFSET


def accept_language_for(locale: str) -> str:
    """按 locale 给出 Accept-Language（默认安装形态的短形式）。"""
    if locale.startswith("zh"):
        return "zh-CN,zh;q=0.9"
    return "en-US,en;q=0.9"


def build_base_headers(locale: str) -> dict[str, str]:
    """BuildBaseHeaders：platform=web 时补齐浏览器头。"""
    out = dict(STATIC_BASE_HEADERS)
    out["x-client-timezone-offset"] = timezone_offset_for(locale)
    out.update(WEB_BROWSER_HEADERS)
    out["Accept-Language"] = accept_language_for(locale)
    out["x-client-platform"] = "web"
    out["x-client-version"] = "2.2.0"
    out["x-client-locale"] = locale
    return out


def login_headers(locale: str) -> dict[str, str]:
    """LoginHeaders：登录/刷新 token 的保守头（无 Origin/Referer/sec-*）。"""
    out = dict(STATIC_BASE_HEADERS)
    out["x-client-timezone-offset"] = timezone_offset_for(locale)
    out["User-Agent"] = "DeepSeek/2.2.0"
    out["x-client-platform"] = "web"
    out["x-client-version"] = "2.2.0"
    out["x-client-locale"] = locale
    return out


def chat_session_referer(session_id: str) -> str:
    """ChatSessionReferer：发消息时 Referer 指向会话页。"""
    return BASE_URL + "/a/chat/s/" + session_id


# defaultSkipContainsPatterns / defaultSkipExactPaths
DEFAULT_SKIP_CONTAINS_PATTERNS = (
    "quasi_status",
    "elapsed_secs",
    "token_usage",
    "pending_fragment",
    "conversation_mode",
    "fragments/-1/status",
    "fragments/-2/status",
    "fragments/-3/status",
)
DEFAULT_SKIP_EXACT_PATHS = {"response/search_status"}
