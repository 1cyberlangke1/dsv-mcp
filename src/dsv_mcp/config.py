"""配置模型：账号与调度参数。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class Account(BaseModel):
    """DeepSeek 托管账号。email / mobile 至少一个。"""

    email: str | None = None
    mobile: str | None = None
    password: str = ""
    locale: str = "zh_CN"

    def identifier(self) -> str:
        return (self.email or self.mobile or "").strip()


class DsvConfig(BaseModel):
    """总配置。"""

    accounts: list[Account] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "DsvConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        return cls.model_validate(json.loads(p.read_text(encoding="utf-8")))
