"""PoW 求解测试（wasm 官方向量）。"""

from __future__ import annotations

import base64
import json

import pytest

from dsv_mcp.pow import PowError, wasm_solver


pytestmark = pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("wasmtime"),
    reason="wasmtime 未安装",
)


@pytest.mark.parametrize(
    "salt,expire,answer,diff,challenge",
    [
        ("testsalt", 1700000000, 42, 1000, "d4a2ea58c89e40887c933484868380c6f803eaa8dc53a3b9df8e431b921a4f09"),
        ("abc123salt", 1700000000, 12345, 20000, "74b3b7452745b70e85eb32ee7f0a9ec0381d42dd5137b695da915e104fc390e1"),
    ],
)
def test_wasm_solve_matches_vectors(salt, expire, answer, diff, challenge):
    prefix = f"{salt}_{expire}_"
    got = wasm_solver().solve(challenge, prefix, diff)
    assert got == answer


def test_wasm_header_roundtrip():
    header = wasm_solver().solve_and_build_header(
        {
            "algorithm": "DeepSeekHashV1",
            "challenge": "d4a2ea58c89e40887c933484868380c6f803eaa8dc53a3b9df8e431b921a4f09",
            "salt": "testsalt",
            "expire_at": 1700000000,
            "difficulty": 1000,
            "signature": "sig",
            "target_path": "/api/v0/chat/completion",
        }
    )
    payload = json.loads(base64.b64decode(header))
    assert payload["answer"] == 42
    assert payload["target_path"] == "/api/v0/chat/completion"


def test_wasm_unsupported_algorithm():
    with pytest.raises(PowError):
        wasm_solver().solve_and_build_header({"algorithm": "Other"})
