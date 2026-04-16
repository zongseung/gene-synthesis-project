"""§10.2 + §10.11 Render — DONGUI visual identity + streaming.

단일 책임:
    - splash banner (Banner A + Mascot B + intro + slash hint)
    - prompt label / 상태 문구 ([you], [dongui], [dongui:error], [dongui:safe])
    - streaming output (rich text 없이 sys.stdout.flush)
    - 한문 block 감지 (contains_hanja — 향후 구문 강조용)

색감 규칙 (§10.11.10):
    - title        → soft amber   ("dark_orange3")
    - mascot       → muted sage   ("dark_sea_green4")
    - metadata     → dim parchment("grey70")
    - refusal      → burnt orange ("dark_orange")
    - 형광·네온 금지, TTY/non-TTY 모두 읽히게 절제.
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
    """§10.11.8 최종 조합안 — banner + mascot + intro + slash hint.

    Args:
        plain: True 면 ASCII art 생략 (pipe/redirect, --plain 옵션).
    """
    if plain or not _is_tty():
        _console.print(
            f"{B.DISPLAY_NAME} v{version} — {B.SUBTITLE}  [adapter: {adapter_label}]"
        )
        return

    # Banner (soft amber)
    _console.print(Text(B.BANNER, style="dark_orange3"), end="")
    _console.print(Text(B.SUBTITLE_LINE, style="dark_orange3 bold"))
    _console.print()

    # Mascot (muted sage)
    _console.print(Text(B.MASCOT, style="dark_sea_green4"), end="")
    _console.print()

    # Intro + adapter info + slash hint (dim parchment)
    for line in B.INTRO_LINES:
        _console.print(Text(f"  {line}", style="grey70"))
    _console.print(Text(f"  adapter: {adapter_label}", style="grey70"))
    _console.print(Text(f"  version: v{version}", style="grey70"))
    _console.print()
    _console.print(Text(f"  {B.SLASH_HINT}", style="grey58"))
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
    (배너·라벨·색감은 한 번만 렌더 — 답변 본문 색감까지 rich 로 처리 안 함.)
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
