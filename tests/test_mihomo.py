"""mihomo 配置生成与订阅解析测试（离线）。"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pytest

from dsv_mcp.errors import ProxyError
from dsv_mcp.mihomo import build_config, ensure_core, fetch_subscription, parse_vless

VLESS_ECH = (
    "vless://11111111-2222-4333-8444-555555555555@203.0.113.10:2053"
    "?security=tls&type=ws&ech=cloudflare-ech.com%2Bhttps%3A%2F%2Fdns.alidns.com%2Fdns-query"
    "&host=worker.example.com&sni=worker.example.com&fp=chrome&path=%2F#n0"
)
VLESS_PLAIN = (
    "vless://11111111-2222-4333-8444-555555555555@203.0.113.11:8443"
    "?security=tls&type=ws&host=worker.example.com&fp=chrome&path=%2F#n1"
)


def test_parse_vless_with_ech():
    n = parse_vless(VLESS_ECH)
    assert n["uuid"] == "11111111-2222-4333-8444-555555555555"
    assert n["server"] == "203.0.113.10"
    assert n["server_port"] == 2053
    assert n["host"] == "worker.example.com"
    assert n["fp"] == "chrome"
    assert n["ech"] == "cloudflare-ech.com+https://dns.alidns.com/dns-query"


def test_parse_vless_plain():
    n = parse_vless(VLESS_PLAIN)
    assert n["ech"] == ""
    assert n["path"] == "/"


def test_build_config_ech_policy():
    nodes = [parse_vless(VLESS_ECH), parse_vless(VLESS_PLAIN)]
    yaml = build_config(nodes, port=10808)
    assert "mixed-port: 10808" in yaml
    assert "ech-opts:" in yaml
    assert "query-server-name: \"cloudflare-ech.com\"" in yaml
    assert 'nameserver-policy:' in yaml
    assert '"cloudflare-ech.com": "https://dns.alidns.com/dns-query"' in yaml
    assert "type: url-test" in yaml
    assert "MATCH,auto" in yaml
    assert "ws-opts:" in yaml
    assert '      path: "/"' in yaml
    assert '        Host: "worker.example.com"' in yaml


def test_build_config_limit():
    nodes = [parse_vless(VLESS_PLAIN) for _ in range(5)]
    yaml = build_config(nodes, port=10808, limit=2)
    assert yaml.count("type: vless") == 2
    assert yaml.count('name: "n0"') == 1
    assert yaml.count('name: "n1"') == 1
    assert "n2" not in yaml


def test_build_config_empty_raises():
    with pytest.raises(ProxyError):
        build_config([], port=10808)


def test_fetch_subscription_parses_lines(monkeypatch):
    class FakeResp:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data.encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        return FakeResp(f"{VLESS_ECH}\n{VLESS_PLAIN}\nnot-a-vless\n")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    nodes = fetch_subscription("https://example.com/sub")
    assert len(nodes) == 2
    assert nodes[0]["ech"].startswith("cloudflare-ech.com")


def test_fetch_subscription_empty_raises(monkeypatch):
    class FakeResp:
        def read(self):
            return b"# hello\n"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProxyError):
        fetch_subscription("https://example.com/sub")


def test_ensure_core_cached(tmp_path):
    (tmp_path / "mihomo.exe").write_bytes(b"core")
    assert ensure_core(tmp_path) == tmp_path / "mihomo.exe"


def test_ensure_core_downloads(monkeypatch, tmp_path):
    import shutil

    def fake_urlopen(request, timeout):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("mihomo-windows-amd64-v1.exe", b"binary")
        buf.seek(0)

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, n=-1):
                return buf.read(n)

        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(shutil, "copyfileobj", lambda src, dst: dst.write(src.read()))
    core = ensure_core(tmp_path)
    assert core == tmp_path / "mihomo.exe"
    assert core.read_bytes() == b"binary"
