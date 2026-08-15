"""配置模型测试。"""

from __future__ import annotations

import json

import pytest

from dsv_mcp.config import Account, DsvConfig


def test_account_identifier_prefers_email():
    assert Account(email="a@b.c", password="p").identifier() == "a@b.c"
    assert Account(mobile="+8613800138000", password="p").identifier() == "+8613800138000"


def test_config_load(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "accounts": [
                    {"email": "a@b.c", "password": "p"},
                    {"mobile": "+8613800138000", "password": "p"},
                ],
                "proxy": {"mode": "manual", "url": "socks5://127.0.0.1:1080"},
            }
        ),
        encoding="utf-8",
    )
    config = DsvConfig.load(cfg)
    assert len(config.accounts) == 2
    assert config.accounts[0].email == "a@b.c"
    assert config.accounts[1].mobile == "+8613800138000"
    assert config.proxy.mode == "manual"
    assert config.proxy.url == "socks5://127.0.0.1:1080"


def test_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DsvConfig.load(tmp_path / "nope.json")


def test_config_invalid_proxy_mode(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"accounts": [{"email": "a@b.c", "password": "p"}], "proxy": {"mode": "auto"}}),
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        DsvConfig.load(cfg)


def test_banned_flag_roundtrip(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {"accounts": [{"email": "a@b.c", "password": "p", "banned": True}]}
        ),
        encoding="utf-8",
    )
    config = DsvConfig.load(cfg)
    assert config.accounts[0].banned is True
    config.save(cfg)
    reloaded = DsvConfig.load(cfg)
    assert reloaded.accounts[0].banned is True


def test_banned_flag_defaults_false():
    assert Account(email="a@b.c", password="p").banned is False
