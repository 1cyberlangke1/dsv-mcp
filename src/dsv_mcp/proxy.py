"""代理管理：直连 / 显式代理 / 托管 sing-box 子进程（Job Object 防孤儿）。"""

from __future__ import annotations

import ctypes
import json
import socket
import subprocess
import threading
import time
from pathlib import Path

from dsv_mcp.config import ProxyConfig


class ProxyError(RuntimeError):
    pass


class JobObject:
    """Windows Job Object：进程被杀时自动结束关联的子进程（防孤儿）。"""

    def __init__(self) -> None:
        if not hasattr(ctypes, "windll"):
            raise ProxyError("Job Object 仅支持 Windows")
        from ctypes import wintypes

        self._wintypes = wintypes
        self._job = ctypes.windll.kernel32.CreateJobObjectW(None, None)
        if not self._job:
            raise ProxyError("CreateJobObject 失败")

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        self._info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ctypes.windll.kernel32.SetInformationJobObject(
            self._job, 9, ctypes.byref(self._info), ctypes.sizeof(self._info)
        )

    def assign(self, pid: int) -> None:
        process = ctypes.windll.kernel32.OpenProcess(
            0x1000, False, pid  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        )
        if not process:
            raise ProxyError(f"OpenProcess 失败 pid={pid}")
        try:
            if not ctypes.windll.kernel32.AssignProcessToJobObject(self._job, process):
                raise ProxyError(f"AssignProcessToJobObject 失败 pid={pid}")
        finally:
            ctypes.windll.kernel32.CloseHandle(process)

    def close(self) -> None:
        if self._job:
            ctypes.windll.kernel32.CloseHandle(self._job)
            self._job = None


class ProxyManager:
    """按配置创建代理客户端；managed 模式负责 sing-box 生命周期。"""

    def __init__(self, config: ProxyConfig):
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._job: JobObject | None = None
        self._lock = threading.Lock()
        self._managed_url = ""

    def proxy_url(self) -> str | None:
        """返回给 HttpClient 的代理地址；none 返回 None（直连）。"""
        if self.config.mode == "none":
            return None
        if self.config.mode == "manual":
            if not self.config.url:
                raise ProxyError("manual 模式需要配置 proxy.url")
            return self.config.url
        return self._start_managed()

    def _start_managed(self) -> str:
        """启动 sing-box 子进程并等待本地端口就绪，返回本地代理地址。"""
        with self._lock:
            if self._proc is not None:
                return self._managed_url
            if not self.config.singbox_bin or not self.config.singbox_config:
                raise ProxyError("managed 模式需要配置 singbox_bin 与 singbox_config")
            config_path = Path(self.config.singbox_config)
            if not config_path.exists():
                raise ProxyError(f"sing-box 配置文件不存在: {config_path}")
            listen = self._read_listen(config_path)
            try:
                self._proc = subprocess.Popen(
                    [self.config.singbox_bin, "run", "-c", str(config_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                raise ProxyError(f"sing-box 启动失败: {exc}") from exc
            try:
                self._job = JobObject()
                self._job.assign(self._proc.pid)
            except ProxyError:
                self._proc.terminate()
                raise
            self._wait_ready(listen)
            self._managed_url = "socks5://" + listen
            return self._managed_url

    @staticmethod
    def _read_listen(config_path: Path) -> str:
        """从 sing-box 配置提取 inbound listen 地址（如 127.0.0.1:10808）。"""
        data = json.loads(config_path.read_text(encoding="utf-8"))
        inbounds = data.get("inbounds") or []
        for inbound in inbounds:
            listen = inbound.get("listen")
            port = inbound.get("port")
            if listen and port:
                return f"{listen}:{port}"
        raise ProxyError("sing-box 配置中未找到 inbound listen/port")

    def _wait_ready(self, addr: str, timeout: float = 30.0) -> None:
        host, port = addr.rsplit(":", 1)
        deadline = time.monotonic() + timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise ProxyError(f"sing-box 提前退出: code={self._proc.returncode}")
            try:
                with socket.create_connection((host, int(port)), timeout=1):
                    return
            except OSError as exc:
                last_err = exc
                time.sleep(0.5)
        raise ProxyError(f"sing-box 端口 {addr} 未就绪: {last_err}")

    def close(self) -> None:
        """停止托管子进程（若存在）。"""
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None
            if self._job is not None:
                self._job.close()
                self._job = None
