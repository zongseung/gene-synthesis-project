"""§10.2 Render — Rich 기반 스트리밍 출력 + 한문 block 강조.

단일 책임: 토큰 스트림을 받아 터미널에 출력, 시작/종료 decoration.
"""

from __future__ import annotations

from typing import Iterable

from rich.console import Console
from rich.panel import Panel


_console = Console()

_HANJA_RANGE = ("\u4e00", "\u9fff")  # CJK Unified Ideographs


def print_banner(version: str, adapter_label: str) -> None:
    banner = (
        f"[bold cyan]HanMed-CLI v{version}[/]  |  "
        f"Bllossom-8B + {adapter_label}  |  bf16\n"
        "KIOM mediclassics.kr 기반 학습. 임상 결정 도구 아님.\n"
        "[dim]/help, /exit, /save, /load, /reset, /temp, /max[/]"
    )
    _console.print(Panel(banner, border_style="cyan"))


def print_user_prefix() -> None:
    _console.print("[bold green]\n[you][/] ", end="")


def print_assistant_prefix() -> None:
    _console.print("[bold magenta]\n[hanmed][/]", end=" ")


def stream_tokens(iterator: Iterable[str]) -> str:
    """토큰 스트림을 그대로 flush 출력하면서 전체 응답을 모아 반환."""
    chunks: list[str] = []
    for tok in iterator:
        chunks.append(tok)
        print(tok, end="", flush=True)
    print()  # trailing newline
    return "".join(chunks)


def print_refusal(text: str) -> None:
    print_assistant_prefix()
    _console.print(f"[yellow]{text}[/]")


def print_info(msg: str) -> None:
    _console.print(f"[dim]{msg}[/]")


def print_error(msg: str) -> None:
    _console.print(f"[bold red]ERROR:[/] {msg}")


def contains_hanja(s: str) -> bool:
    return any(_HANJA_RANGE[0] <= ch <= _HANJA_RANGE[1] for ch in s)
