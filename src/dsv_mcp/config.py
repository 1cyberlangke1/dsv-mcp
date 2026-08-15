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
    singbox_bin: str | None = None  # managed 模式的 sing-box 可执行文件路径
    singbox_config: str | None = None  # managed 模式的 sing-box 配置文件路径


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
