"""代理配置与管理测试（离线，不联网）。"""

from __future__ import annotations

import json

import pytest

from dsv_mcp.config import DsvConfig, ProxyConfig
from dsv_mcp.proxy import JobObject, ProxyError, ProxyManager


def test_config_parses_all_modes(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "accounts": [{"email": "a@b.c", "password": "p"}],
                "proxy": {
                    "mode": "managed",
                    "managed_subscription": "https://example.com/sub",
                    "managed_node": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    config = DsvConfig.load(cfg)
    assert config.proxy.mode == "managed"
    assert config.proxy.managed_subscription == "https://example.com/sub"
    assert config.proxy.managed_node == 3
    assert config.proxy.mode in ("none", "manual", "managed")


def test_proxy_none_returns_none():
    manager = ProxyManager(ProxyConfig(mode="none"))
    assert manager.proxy_url() is None
    manager.close()


def test_proxy_manual_returns_url():
    manager = ProxyManager(ProxyConfig(mode="manual", url="socks5://127.0.0.1:1080"))
    assert manager.proxy_url() == "socks5://127.0.0.1:1080"
    manager.close()


def test_proxy_manual_requires_url():
    manager = ProxyManager(ProxyConfig(mode="manual"))
    with pytest.raises(ProxyError):
        manager.proxy_url()
    manager.close()


def test_proxy_managed_requires_subscription():
    manager = ProxyManager(ProxyConfig(mode="managed"))
    with pytest.raises(ProxyError):
        manager.proxy_url()
    manager.close()


def test_pick_port_returns_local_addr():
    addr = ProxyManager._pick_port()
    host, port = addr.rsplit(":", 1)
    assert host == "127.0.0.1"
    assert port.isdigit()


def test_job_object_roundtrip():
    if not hasattr(__import__("ctypes"), "windll"):
        pytest.skip("仅 Windows 支持 Job Object")
    job = JobObject()
    job.close()
