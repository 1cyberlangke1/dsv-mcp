"""mihomo 内核管理：订阅解析、内核下载、clash yaml 配置生成。"""

from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from dsv_mcp.errors import ProxyError

# 固定版本，避免运行时动态查 release API 的额外请求
MIHOMO_VERSION = "v1.19.29"
MIHOMO_URL = (
    "https://github.com/MetaCubeX/mihomo/releases/download/"
    f"{MIHOMO_VERSION}/mihomo-windows-amd64-v1-{MIHOMO_VERSION}.zip"
)


def parse_vless(line: str) -> dict:
    """解析单行 vless:// 链接为节点字典。"""
    body = line[len("vless://") :]
    if "#" in body:
        body, _ = body.split("#", 1)
    userinfo, rest = body.split("@", 1)
    hostport, query = rest.split("?", 1)
    host, port = hostport.rsplit(":", 1)
    params = dict(urllib.parse.parse_qsl(query))
    return {
        "uuid": userinfo,
        "server": host,
        "server_port": int(port),
        "sni": params.get("sni", ""),
        "host": params.get("host", ""),
        "path": params.get("path", "/"),
        "fp": params.get("fp", "chrome"),
        "ech": params.get("ech", ""),
    }


def fetch_subscription(url: str, timeout: float = 30.0) -> list[dict]:
    """拉取订阅文本，返回 vless 节点列表（非 vless 行忽略）。"""
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise ProxyError(f"订阅拉取失败: {exc}") from exc
    nodes = [
        parse_vless(line.strip())
        for line in text.splitlines()
        if line.strip().startswith("vless://")
    ]
    if not nodes:
        raise ProxyError("订阅中未找到 vless 节点")
    return nodes


def _yaml_str(value: str) -> str:
    """JSON 字符串是合法 YAML 标量，避免手工转义。"""
    return json.dumps(str(value), ensure_ascii=False)


def build_config(nodes: list[dict], port: int, limit: int = 0) -> str:
    """生成 mihomo clash yaml：前 limit 个节点进 url-test 组，全走代理。"""
    picked = nodes[:limit] if limit > 0 else nodes
    if not picked:
        raise ProxyError("没有可用节点")
    lines = [
        f"mixed-port: {port}",
        "allow-lan: false",
        "mode: rule",
        "log-level: warning",
        "ipv6: false",
        "",
        "dns:",
        "  enable: true",
        "  default-nameserver:",
        "    - 223.5.5.5",
        "    - 119.29.29.29",
    ]
    # ECH 需要把 query-server 域名走 DoH，否则拿不到 ECHConfigs
    policies: dict[str, str] = {}
    for n in picked:
        if not n.get("ech"):
            continue
        parts = n["ech"].split("+", 1)
        if len(parts) == 2:
            policies.setdefault(parts[0], parts[1])
    doh_servers = sorted(set(policies.values()))
    if doh_servers:
        lines.append("  nameserver:")
        for server in doh_servers:
            lines.append(f"    - {_yaml_str(server)}")
        lines.append("  nameserver-policy:")
        for qs, server in sorted(policies.items()):
            lines.append(f"    {_yaml_str(qs)}: {_yaml_str(server)}")
    lines += ["", "proxies:"]
    for i, n in enumerate(picked):
        name = f"n{i}"
        lines.append(f"  - name: {_yaml_str(name)}")
        lines.append(f"    server: {_yaml_str(n['server'])}")
        lines.append(f"    port: {n['server_port']}")
        lines.append("    type: vless")
        lines.append(f"    uuid: {_yaml_str(n['uuid'])}")
        lines.append("    tls: true")
        lines.append(f"    servername: {_yaml_str(n['sni'] or n['host'])}")
        lines.append(f"    client-fingerprint: {_yaml_str(n['fp'])}")
        lines.append("    network: ws")
        lines.append("    ws-opts:")
        lines.append(f"      path: {_yaml_str(n['path'])}")
        lines.append("      headers:")
        lines.append(f"        Host: {_yaml_str(n['host'])}")
        if n.get("ech") and n["ech"].split("+", 1)[0] in policies:
            lines.append("    ech-opts:")
            lines.append("      enable: true")
            lines.append(f"      query-server-name: {_yaml_str(n['ech'].split('+', 1)[0])}")
    lines += [
        "",
        "proxy-groups:",
        "  - name: auto",
        "    type: url-test",
        "    url: https://www.gstatic.com/generate_204",
        "    interval: 300",
        "    proxies:",
    ]
    for i in range(len(picked)):
        lines.append(f"      - {_yaml_str(f'n{i}')}")
    lines += ["", "rules:", "  - MATCH,auto"]
    return "\n".join(lines) + "\n"


def ensure_core(cache_dir: Path) -> Path:
    """确保 mihomo 内核存在（缺失时下载 zip 解压），返回 exe 路径。"""
    core = cache_dir / "mihomo.exe"
    if core.exists():
        return core
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "mihomo.zip"
    request = urllib.request.Request(
        MIHOMO_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp, open(zip_path, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except Exception as exc:
        raise ProxyError(f"mihomo 内核下载失败: {exc}") from exc
    try:
        with zipfile.ZipFile(zip_path) as zf:
            exe_name = next(n for n in zf.namelist() if n.endswith(".exe"))
            with zf.open(exe_name) as src, open(core, "wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as exc:
        raise ProxyError(f"mihomo 内核解压失败: {exc}") from exc
    zip_path.unlink(missing_ok=True)
    return core
