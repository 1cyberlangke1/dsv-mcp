"""代理管理：直连 / 显式代理 / 托管 mihomo 子进程（Job Object 防孤儿）。"""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from dsv_mcp.config import ProxyConfig
from dsv_mcp.errors import ProxyError
from dsv_mcp.mihomo import build_config, ensure_core, fetch_subscription


class JobObject:
    """Windows Job Object：进程被杀时自动结束关联的子进程（防孤儿）。"""

    def __init__(self) -> None:
        if not hasattr(ctypes, "windll"):
            raise ProxyError("Job Object 仅支持 Windows")
        from ctypes import wintypes

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
            0x0101, False, pid  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
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
    """按配置创建代理客户端；managed 模式负责 mihomo 生命周期。"""

    def __init__(self, config: ProxyConfig):
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._job: JobObject | None = None
        self._lock = threading.Lock()
        self._managed_url: str | None = None

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
        """启动 mihomo 子进程并等待本地端口就绪，返回本地代理地址。"""
        with self._lock:
            if self._proc is not None:
                return self._managed_url or ""
            if not self.config.managed_subscription:
                raise ProxyError("managed 模式需要配置 proxy.managed_subscription")
            cache_dir = Path(
                self.config.cache_dir
                or os.path.join(
                    os.environ.get("LOCALAPPDATA", str(Path.home())), "dsv-mcp", "managed"
                )
            )
            core = ensure_core(cache_dir)
            nodes = fetch_subscription(self.config.managed_subscription)
            listen = self._pick_port()
            config_path = cache_dir / "config.yaml"
            config_path.write_text(
                build_config(
                    nodes,
                    port=int(listen.rsplit(":", 1)[1]),
                    index=self.config.managed_node,
                ),
                encoding="utf-8",
            )
            try:
                self._proc = subprocess.Popen(
                    [str(core), "-d", str(cache_dir), "-f", str(config_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                raise ProxyError(f"mihomo 启动失败: {exc}") from exc
            try:
                self._job = JobObject()
                self._job.assign(self._proc.pid)
            except ProxyError:
                self._proc.terminate()
                raise
            self._wait_ready(listen)
            # socks5h：域名由代理解析（本地 DNS 可能污染/失败）
            self._managed_url = "socks5h://" + listen
            return self._managed_url

    @staticmethod
    def _pick_port(start: int = 10808) -> str:
        """选一个空闲端口（从 start 起向上探测），返回 127.0.0.1:port。"""
        for port in range(start, start + 100):
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                    return f"127.0.0.1:{port}"
                except OSError:
                    continue
        raise ProxyError("未找到可用端口")

    def _wait_ready(self, addr: str, timeout: float = 30.0) -> None:
        host, port = addr.rsplit(":", 1)
        deadline = time.monotonic() + timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise ProxyError(f"mihomo 提前退出: code={self._proc.returncode}")
            try:
                with socket.create_connection((host, int(port)), timeout=1):
                    return
            except OSError as exc:
                last_err = exc
                time.sleep(0.5)
        raise ProxyError(f"mihomo 端口 {addr} 未就绪: {last_err}")

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
