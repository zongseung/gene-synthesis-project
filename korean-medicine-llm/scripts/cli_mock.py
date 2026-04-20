"""HanMed CLI v3 mock renderer.

디자인 검증용 — branding.py 미수정 상태에서 03_claude_code_style.md 레이아웃을
그대로 터미널에 한 번 그려본다.

사용:
  .venv/bin/python scripts/cli_mock.py
  # 또는
  /usr/bin/python3 scripts/cli_mock.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASCOT_PATH = ROOT / "docs/10_cli_visual_identity/turtle_24col.ansi"

# ── ANSI helpers ────────────────────────────────────────────────────────
R = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

def fg(r, g, b): return f"\x1b[38;2;{r};{g};{b}m"
def bg(r, g, b): return f"\x1b[48;2;{r};{g};{b}m"

# Palette (03_claude_code_style.md §4 / v2 §2)
AMBER       = fg(230, 160, 70)    # welcome · soft amber
WHITE_BOLD  = BOLD + fg(240, 240, 240)
GREY70      = fg(178, 178, 178)
GREY58      = fg(148, 148, 148)
CYAN        = fg(90, 190, 220)
MAGENTA     = fg(200, 110, 200)

# ── content ─────────────────────────────────────────────────────────────
TITLE      = "HanMed CLI v0.1"
SUBTITLE   = "DONGUI AI"
WELCOME    = "Welcome to DONGUI · /help, /save, /reset, /exit"

def cwd_short() -> str:
    try:
        cwd = os.getcwd()
        home = os.path.expanduser("~")
        if cwd.startswith(home):
            cwd = "~" + cwd[len(home):]
        return cwd
    except Exception:
        return "~"

def git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=1,
        )
        return out.stdout.strip() or "—"
    except Exception:
        return "—"

def load_mascot_lines() -> list[str]:
    if not MASCOT_PATH.exists():
        return [""] * 6
    return MASCOT_PATH.read_text().rstrip("\n").split("\n")

# ── layout ──────────────────────────────────────────────────────────────
WIDTH = 72
GAP = "   "   # mascot 과 text 사이 공백 3-col

def divider() -> str:
    return f"  {GREY58}{'─' * (WIDTH - 2)}{R}"

def header_block(mascot: list[str]) -> list[str]:
    """mascot N line 옆에 text block 배치. 텍스트는 mascot 중앙에 수직 정렬."""
    text_items = [
        f"{WHITE_BOLD}{TITLE}{R}",
        f"{GREY70}{SUBTITLE}{R}",
        f"{AMBER}{WELCOME}{R}",
    ]
    m_rows = len(mascot)
    # 텍스트를 mascot 중앙에 위치. top padding 으로 맞춤.
    top_pad = max(0, (m_rows - len(text_items)) // 2)
    text = [""] * top_pad + text_items + [""] * (m_rows - top_pad - len(text_items))
    out = []
    for i in range(m_rows):
        m = mascot[i] if i < len(mascot) else ""
        out.append(f"  {m}{R}{GAP}{text[i]}")
    return out

def statusline() -> list[str]:
    # v3.2: P-A+ · uptime · branch · CTX pill 제거. 맨 위 tag 만 유지.
    return [
        f"  {GREY58}┌─ hanmed ─────{R}",
    ]

def render() -> str:
    mascot = load_mascot_lines()
    lines = []
    lines.append("")
    lines.extend(header_block(mascot))
    lines.append("")
    lines.append(divider())
    lines.append("")
    lines.append(f"  {WHITE_BOLD}>{R} ")
    lines.append("")
    lines.append(divider())
    lines.extend(statusline())
    lines.append("")
    return "\n".join(lines)

if __name__ == "__main__":
    sys.stdout.write(render())
    sys.stdout.write("\n")
    sys.stdout.flush()
