"""录制 case 重放测试：真实 SSE 流 → 解析结果与录制一致（离线）。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dsv_mcp.sse import collect_stream


CASES_DIR = Path(__file__).parent / "cases"


def _cases() -> list[Path]:
    return sorted(CASES_DIR.glob("*.sse"))


@pytest.mark.parametrize("sse_path", _cases(), ids=lambda p: p.stem)
def test_recorded_case_replays(sse_path):
    meta_path = sse_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lines = sse_path.read_text(encoding="utf-8").splitlines()
    result = collect_stream(lines, thinking_enabled=True)
    assert result.text == meta["text"]
    assert result.thinking == meta["thinking"]
    assert result.response_message_id == meta["message_id"]


@pytest.mark.parametrize("sse_path", _cases(), ids=lambda p: p.stem)
def test_recorded_case_image_exists(sse_path):
    meta = json.loads(sse_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert Path(meta["image"]).exists()


def test_grounding_case_contains_bbox_primitives():
    meta = json.loads((CASES_DIR / "棋子_grounding.json").read_text(encoding="utf-8"))
    # bbox 是 4 元组归一化坐标 [x1,y1,x2,y2]
    boxes = re.findall(r"\[\[\d+,\d+,\d+,\d+\]\]", meta["thinking"])
    assert len(boxes) >= 3
    assert all(len(re.findall(r"\d+", b)) == 4 for b in boxes)


def test_pointing_case_contains_point_primitives():
    meta = json.loads((CASES_DIR / "迷宫_pointing.json").read_text(encoding="utf-8"))
    # point 是 2 元组归一化坐标 [x1,y1]
    points = re.findall(r"\[\[\d+,\d+\]\]", meta["thinking"])
    assert len(points) >= 3
    assert all(len(re.findall(r"\d+", p)) == 2 for p in points)
