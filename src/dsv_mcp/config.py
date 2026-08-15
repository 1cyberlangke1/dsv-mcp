"""配置模型：账号与代理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Account(BaseModel):
    """DeepSeek 托管账号。email / mobile 至少一个。"""

    email: str | None = None
    mobile: str | None = None
    password: str = ""
    locale: str = "zh_CN"

    def identifier(self) -> str:
        return (self.email or self.mobile or "").strip()


class ProxyConfig(BaseModel):
    """代理配置：none 直连 / manual 显式代理 / managed 托管子进程。"""

    mode: Literal["none", "manual", "managed"] = "none"
    url: str | None = None  # manual 模式的代理地址，如 socks5://127.0.0.1:1080
    managed_subscription: str | None = None  # managed 模式的 vless 订阅 URL
    managed_node: int = 0  # managed 模式使用的节点（订阅中的序号，从 0 开始）
    cache_dir: str | None = None  # 内核与生成配置的缓存目录（默认 %LOCALAPPDATA%/dsv-mcp）


class DsvConfig(BaseModel):
    """总配置。"""

    accounts: list[Account] = Field(default_factory=list)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)

    @classmethod
    def load(cls, path: str | Path) -> "DsvConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        return cls.model_validate(json.loads(p.read_text(encoding="utf-8")))
