"""PoW（DeepSeekHashV1）求解：wasmtime 跑官方 wasm。"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any

import wasmtime


WASM_PATH = Path(__file__).resolve().parent / "sha3_wasm_bg.wasm"


class PowError(RuntimeError):
    pass


class WasmPowSolver:
    def __init__(self, wasm_path: Path = WASM_PATH):
        self._store = wasmtime.Store()
        module = wasmtime.Module.from_file(self._store.engine, str(wasm_path))
        self._inst = wasmtime.Instance(self._store, module, [])
        exp = self._inst.exports(self._store)
        self._memory = exp["memory"]
        self._solve = exp["wasm_solve"]
        self._malloc = exp["__wbindgen_export_0"]
        self._add_to_stack = exp["__wbindgen_add_to_stack_pointer"]

    def _write_str(self, text: str) -> tuple[int, int]:
        data = text.encode("utf-8")
        ptr = self._malloc(self._store, len(data), 1)
        base = self._memory.data_ptr(self._store)
        for i, b in enumerate(data):
            base[ptr + i] = b
        return ptr, len(data)

    def solve(self, challenge: str, prefix: str, difficulty: float) -> int | None:
        retptr = self._add_to_stack(self._store, -16)
        try:
            c_ptr, c_len = self._write_str(challenge)
            p_ptr, p_len = self._write_str(prefix)
            self._solve(self._store, retptr, c_ptr, c_len, p_ptr, p_len, float(difficulty))
            mem = self._memory.data_ptr(self._store)
            status = struct.unpack("<i", bytes(mem[retptr : retptr + 4]))[0]
            value = struct.unpack("<d", bytes(mem[retptr + 8 : retptr + 16]))[0]
        finally:
            self._add_to_stack(self._store, 16)
        if status == 0:
            return None
        return int(value)

    def solve_and_build_header(self, challenge: dict[str, Any]) -> str:
        if challenge.get("algorithm") != "DeepSeekHashV1":
            raise PowError(f"pow: unsupported algorithm: {challenge.get('algorithm')}")
        prefix = f"{challenge['salt']}_{challenge['expire_at']}_"
        answer = self.solve(challenge["challenge"], prefix, challenge.get("difficulty", 144000))
        if answer is None:
            raise PowError("pow: wasm solver returned no answer")
        payload = {
            "algorithm": challenge["algorithm"],
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": answer,
            "signature": challenge["signature"],
            "target_path": challenge["target_path"],
        }
        return base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


_solver: WasmPowSolver | None = None


def wasm_solver() -> WasmPowSolver:
    global _solver
    if _solver is None:
        _solver = WasmPowSolver()
    return _solver
