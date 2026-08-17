"""dsv-mcp 包入口。"""

import os
import sys


def _repair_windows_env() -> None:
    """补回 MCP 客户端不给子进程传的核心环境变量。

    实测：Codex 在 Windows 上启动 stdio MCP 子进程时只传 PATH/PATHEXT/
    USERNAME/USERDOMAIN/USERPROFILE/TEMP/TMP，缺 SYSTEMROOT 会让 Winsock
    服务提供程序初始化失败（WinError 10106），缺 PROCESSOR_ARCHITECTURE
    会让 wasmtime 拒绝加载（platform.machine() 返回空）。这里从注册表和
    WinAPI 读回缺失值，必须在任何第三方库 import 前执行。
    """
    if sys.platform != "win32":
        return
    if not os.environ.get("SYSTEMROOT"):
        root = r"C:\Windows"
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            ) as key:
                root = winreg.QueryValueEx(key, "SystemRoot")[0]
        except OSError:
            pass
        os.environ["SYSTEMROOT"] = root
        os.environ.setdefault("SYSTEMDRIVE", root[:2] + "\\")
    if not os.environ.get("PROCESSOR_ARCHITECTURE"):
        try:
            import ctypes

            class _SystemInfo(ctypes.Structure):
                _fields_ = [
                    ("wProcessorArchitecture", ctypes.c_ushort),
                    ("wReserved", ctypes.c_ushort),
                    ("dwPageSize", ctypes.c_ulong),
                    ("lpMinimumApplicationAddress", ctypes.c_void_p),
                    ("lpMaximumApplicationAddress", ctypes.c_void_p),
                    ("dwActiveProcessorMask", ctypes.c_size_t),
                    ("dwNumberOfProcessors", ctypes.c_ulong),
                    ("dwProcessorType", ctypes.c_ulong),
                    ("dwAllocationGranularity", ctypes.c_ulong),
                    ("wProcessorLevel", ctypes.c_ushort),
                    ("wProcessorRevision", ctypes.c_ushort),
                ]

            info = _SystemInfo()
            ctypes.windll.kernel32.GetNativeSystemInfo(ctypes.byref(info))
            arch = {0: "x86", 5: "ARM", 9: "AMD64", 12: "ARM64"}.get(
                info.wProcessorArchitecture
            )
            if arch:
                os.environ["PROCESSOR_ARCHITECTURE"] = arch
        except Exception:
            pass


def _quiet_third_party_loggers() -> None:
    """把第三方库的 INFO 日志降到 WARNING。

    实测（Codex issue #7155）：Windows 上 stdio MCP 服务器在工具执行期间
    往 stderr 大量输出（httpx 每次请求都打一行、mcp 库打 session 行）会让
    Codex 的 stderr 管道处理出错，最终整个工具调用报 Transport closed。
    这些日志对用户无价值，直接静音；自己写的日志走文件（见 server 的
    _stdio_log / HTTP 实例日志），不占 stderr。
    """
    import logging

    for name in ("httpx", "httpcore", "mcp", "client"):
        logging.getLogger(name).setLevel(logging.WARNING)


_repair_windows_env()
_quiet_third_party_loggers()


def main() -> None:
    """CLI 入口：启动 MCP streamable HTTP 服务器。"""
    from dsv_mcp.server import main as _server_main

    _server_main()
