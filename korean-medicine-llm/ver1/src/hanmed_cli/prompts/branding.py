"""§10.11 CLI Visual Identity v3 — turtle apothecary + chat box.

단일 책임: 상수만. 렌더링은 render.py.

v3 (2026-04-20) 변경:
    - DONGUI figlet BANNER → 삭제 (v1)
    - 약장 MASCOT → 삭제 (v2)
    - 거북 약사 MASCOT_LINES (24 col × 12 row, 24-bit truecolor half-block) 도입
    - CLI_TITLE / CLI_SUBTITLE / CLI_WELCOME 3-line header
    - DIVIDER_UNICODE / DIVIDER_ASCII 구분선
    - STATUSLINE_TAG

디자인 근거: docs/10_cli_visual_identity/03_claude_code_style.md
"""

from __future__ import annotations

from pathlib import Path


COMMAND_NAME = "hanmed"
DISPLAY_NAME = "DONGUI"

# Prompt labels
LABEL_USER = "[you]"
LABEL_ASSISTANT = f"[{DISPLAY_NAME.lower()}]"
LABEL_ERROR = f"[{DISPLAY_NAME.lower()}:error]"
LABEL_SAFE = f"[{DISPLAY_NAME.lower()}:safe]"


# ── Header text ───────────────────────────────────────────────
CLI_TITLE = "HanMed · 동의보감 v3.1"
CLI_SUBTITLE = "Dongui Bogam assistant · Bllossom-8B + LoRA SFT"
CLI_WELCOME = "동의보감 전용 · 진단·처방 불가  ·  /help /exit"


# ── Mascot (24-bit truecolor ANSI, alpha-aware) ───────────────
# docs/10_cli_visual_identity/png2block_tool.py 로 hammed_icon/HanMed_2.png 에서
# 24 col × 12 row 로 변환. 투명 영역은 " " 로 렌더돼 터미널 배경 그대로 비침.
_MASCOT_FILE = Path(__file__).parent / "turtle_24col.ansi"
try:
    _raw = _MASCOT_FILE.read_text(encoding="utf-8").rstrip("\n")
    MASCOT_LINES: tuple[str, ...] = tuple(_raw.split("\n")) if _raw else tuple()
except FileNotFoundError:
    MASCOT_LINES = tuple()


# ── Dividers ──────────────────────────────────────────────────
_DIVIDER_WIDTH = 70
DIVIDER_UNICODE = "─" * _DIVIDER_WIDTH
DIVIDER_ASCII = "-" * _DIVIDER_WIDTH


# ── Statusline ────────────────────────────────────────────────
STATUSLINE_TAG = "┌─ hanmed ─────"
