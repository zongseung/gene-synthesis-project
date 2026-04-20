"""§10.2 + §10.11 Render — v3 turtle mascot + chat-box layout + streaming.

단일 책임:
    - splash (mascot + header + divider)
    - prompt label / 상태 문구 ([you], [dongui], [dongui:error], [dongui:safe])
    - streaming output (rich 없이 sys.stdout.flush)
    - 한문 block 감지 (contains_hanja)

색감 (v3):
    - title        → bold white
    - subtitle     → dim grey (grey70)
    - welcome      → soft amber (dark_orange3)
    - divider      → grey58
    - statusline   → grey58
    - mascot       → 파일의 24-bit ANSI 그대로 (자체 컬러)
    - prompt user  → muted sage (dark_sea_green4)
    - assistant    → soft amber (dark_orange3)
    - refusal      → burnt orange
    - error        → red
"""

from __future__ import annotations

import sys
from typing import Iterable

from rich.console import Console
from rich.text import Text

from hanmed_cli.prompts import branding as B


_console = Console()

# CJK Unified Ideographs
_HANJA_LO, _HANJA_HI = "\u4e00", "\u9fff"

# mascot 의 embedded ANSI 와 충돌 없이 reset 하려고 raw 상수.
_R = "\x1b[0m"


# --- Splash -----------------------------------------------------------

def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def print_banner(
    version: str,
    adapter_label: str,
    *,
    plain: bool = False,
) -> None:
    """v3 splash: mascot 좌 + 3-line header 우 + 구분선 + status tag.

    plain=True or non-TTY: 한 줄 요약만.
    """
    if plain or not _is_tty():
        # rich markup 해석 방지 — raw stdout
        sys.stdout.write(
            f"{B.DISPLAY_NAME} v{version} — {B.CLI_SUBTITLE}  "
            f"[adapter: {adapter_label}]\n"
        )
        sys.stdout.flush()
        return

    mascot = B.MASCOT_LINES
    m_rows = len(mascot)

    text_items = [B.CLI_TITLE, B.CLI_SUBTITLE, B.CLI_WELCOME]
    text_styles = ["bold white", "grey70", "dark_orange3"]
    top_pad = max(0, (m_rows - len(text_items)) // 2)

    gap = "   "
    sys.stdout.write("\n")
    for i in range(m_rows):
        m = mascot[i] if i < m_rows else ""
        idx = i - top_pad
        if 0 <= idx < len(text_items):
            # mascot raw ANSI + reset + gap, 그 뒤에 rich styled text + newline
            sys.stdout.write(f"  {m}{_R}{gap}")
            sys.stdout.flush()
            _console.print(Text(text_items[idx], style=text_styles[idx]))
        else:
            sys.stdout.write(f"  {m}{_R}\n")
    sys.stdout.write("\n")
    sys.stdout.flush()
    _console.print(Text(f"  {B.DIVIDER_UNICODE}", style="grey58"))
    _console.print(Text(f"  {B.STATUSLINE_TAG}", style="grey58"))
    _console.print()


# --- Prompt labels ----------------------------------------------------

def print_user_prefix() -> None:
    _console.print(Text(f"\n{B.LABEL_USER} ", style="dark_sea_green4 bold"), end="")


def print_assistant_prefix() -> None:
    _console.print(Text(f"\n{B.LABEL_ASSISTANT}", style="dark_orange3 bold"), end=" ")


def print_refusal(text: str) -> None:
    _console.print(Text(f"\n{B.LABEL_SAFE}", style="dark_orange bold"), end=" ")
    _console.print(Text(text, style="dark_orange"))


def print_info(msg: str) -> None:
    _console.print(Text(msg, style="grey70"))


def print_error(msg: str) -> None:
    _console.print(
        Text(f"{B.LABEL_ERROR} ", style="red bold"),
        Text(msg, style="red"),
        sep="",
    )


# --- Streaming --------------------------------------------------------

def stream_tokens(iterator: Iterable[str]) -> str:
    """토큰 chunk 를 flush 출력하면서 누적 문자열 반환.

    rich 의 color tag 렌더링은 streaming 중에는 애매하므로 plain print 사용.
    """
    chunks: list[str] = []
    for tok in iterator:
        chunks.append(tok)
        sys.stdout.write(tok)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(chunks)


# --- Helpers ----------------------------------------------------------

def contains_hanja(s: str) -> bool:
    return any(_HANJA_LO <= ch <= _HANJA_HI for ch in s)
