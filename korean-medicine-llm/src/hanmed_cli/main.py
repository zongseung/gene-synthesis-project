"""Click CLI entry — §10.10.6 + ver4 §03.02.

Bare `hanmed` → splash + remote REPL (docker vLLM 기동 중이라 가정).
`hanmed --splash-only` → splash 출력 후 exit (backend 접속 X).
`hanmed chat [...]` → 기존 local backend 경로 (학습·디버그용).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from hanmed_cli import __version__ as PKG_VERSION
from hanmed_cli.config import DEFAULTS
from hanmed_cli.inference.base import SamplingConfig
from hanmed_cli.render import print_banner, print_error, print_info
from hanmed_cli.session import Session, ModelPin, SamplingState, list_sessions, load_session, remove_session

# ver4 §03.02 default — vLLM OpenAI-compatible 엔드포인트
_DEFAULT_ENDPOINT = os.environ.get("HANMED_ENDPOINT", "http://localhost:8000/v1")
_DEFAULT_MODEL_NAME = os.environ.get("HANMED_MODEL", "hanmed-p-a-plus")


@click.group(invoke_without_command=True)
@click.version_option(PKG_VERSION, prog_name="hanmed")
@click.option("--verbose", "-v", is_flag=True, help="verbose 로그")
@click.option("--splash-only", is_flag=True, help="splash 만 출력 후 exit (backend 로드 X)")
@click.option("--plain", is_flag=True, help="splash 배너·마스코트 생략 (pipe/redirect 용)")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, splash_only: bool, plain: bool) -> None:
    """HanMed-LLM 인터랙티브 CLI (§10 ver4)."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if ctx.invoked_subcommand is not None:
        return

    # bare `hanmed` — splash + (optional REPL via remote vLLM)
    print_banner(PKG_VERSION, f"{_DEFAULT_MODEL_NAME} (remote)", plain=plain)

    if splash_only:
        return

    from hanmed_cli.inference import get_backend
    from hanmed_cli.chat import run_repl

    be = get_backend("remote_openai")
    try:
        be.load(
            base_model=_DEFAULT_MODEL_NAME,
            tokenizer_dir="",
            adapter_path=None,
            endpoint=_DEFAULT_ENDPOINT,
        )
    except Exception as exc:
        print_error(
            f"backend 접속 실패 ({_DEFAULT_ENDPOINT}): {exc!r}\n"
            f"  vLLM 서버 기동 확인: docker compose ps\n"
            f"  또는: hanmed chat --help 로 local backend 사용"
        )
        sys.exit(2)

    session = Session()
    session.model = ModelPin(
        base=_DEFAULT_MODEL_NAME, adapter=None, adapter_mode="P-CPT"
    )
    sampling = SamplingConfig()
    try:
        rc = run_repl(
            be, session, sampling,
            adapter_label=f"{_DEFAULT_MODEL_NAME} (remote)",
            plain=plain,
        )
    finally:
        try:
            session.save(DEFAULTS.session_autosave_name)
        except Exception:
            pass
        be.close()
    sys.exit(rc)


@cli.command("chat")
@click.option("--adapter", type=click.Path(exists=False, path_type=Path), default=None,
              help="LoRA adapter 경로 (P-CPT 또는 P-SFT). 없으면 base 만으로 동작.")
@click.option("--mode", type=click.Choice(["cpt", "sft"]), default="cpt",
              help="adapter 경로 타입")
@click.option("--session", "session_name", type=str, default=DEFAULTS.session_autosave_name,
              help="세션 name (autosave)")
@click.option("--base-model", type=str, default=DEFAULTS.base_model)
@click.option("--tokenizer", "tokenizer_dir", type=str, default=DEFAULTS.tokenizer_ext_dir,
              help="extended tokenizer 경로")
@click.option("--backend", type=click.Choice(["transformers", "vllm", "remote_openai", "auto"]), default="transformers",
              help="v0 권장: transformers. remote_openai 은 vLLM OpenAI API 를 HANMED_ENDPOINT 로 호출.")
@click.option("--remote", type=str, default=None,
              help="v1 reserved — v0 에서는 사용 불가")
@click.option("--system-prompt-version", type=str, default=DEFAULTS.system_prompt_version)
@click.option("--temperature", type=float, default=DEFAULTS.temperature)
@click.option("--top-p", type=float, default=DEFAULTS.top_p)
@click.option("--max-new-tokens", type=int, default=DEFAULTS.max_new_tokens)
@click.option("--repetition-penalty", type=float, default=DEFAULTS.repetition_penalty)
@click.option("--load-existing", is_flag=True, help="기존 session 을 이어서 시작")
@click.option("--plain", is_flag=True, help="splash 배너·마스코트 생략 (pipe/redirect 용)")
def chat_cmd(
    adapter: Path | None,
    mode: str,
    session_name: str,
    base_model: str,
    tokenizer_dir: str,
    backend: str,
    remote: str | None,
    system_prompt_version: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float,
    load_existing: bool,
    plain: bool,
) -> None:
    """REPL 시작."""
    if remote:
        print_error("--remote 는 v1 reserved. v0 에서는 로컬 backend 만 사용 가능 (§10.7).")
        sys.exit(2)

    # session 준비
    if load_existing:
        try:
            session = load_session(session_name)
            print_info(f"기존 세션 로드: {session_name} ({len(session.messages)} msgs)")
        except FileNotFoundError:
            print_info(f"세션 {session_name} 없음 — 새로 생성.")
            session = Session()
    else:
        session = Session()

    # model pin 업데이트
    session.model = ModelPin(
        base=base_model,
        adapter=str(adapter) if adapter else None,
        adapter_mode="P-SFT" if mode == "sft" else "P-CPT",
    )
    session.system_prompt_version = system_prompt_version
    session.sampling = SamplingState(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    sampling = SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    # splash — bare `hanmed` 와 동일한 위치(backend load 전)에서 한 번만 출력.
    # run_repl 은 더 이상 splash 를 찍지 않는다 (중복 방지).
    adapter_label_splash = (
        f"HanMed-{mode.upper()} ({adapter.name})" if adapter else "(base only)"
    )
    print_banner(PKG_VERSION, adapter_label_splash, plain=plain)

    # backend 로드 (지연 import — CLI --help 빠르게 유지)
    from hanmed_cli.inference import get_backend

    print_info(f"[hanmed] loading backend={backend} base={base_model}…")
    be = get_backend(backend)
    try:
        be.load(
            base_model=base_model,
            tokenizer_dir=tokenizer_dir,
            adapter_path=str(adapter) if adapter else None,
        )
    except Exception as exc:
        print_error(f"backend load 실패: {exc!r}")
        sys.exit(1)

    from hanmed_cli.chat import run_repl

    try:
        rc = run_repl(be, session, sampling, adapter_label=adapter_label_splash, plain=plain)
    finally:
        # autosave
        try:
            session.save(session_name)
        except Exception:
            pass
        be.close()
    sys.exit(rc)


@cli.group("sessions")
def sessions_cmd() -> None:
    """세션 관리."""


@sessions_cmd.command("list")
def sessions_list() -> None:
    for name in list_sessions():
        click.echo(name)


@sessions_cmd.command("rm")
@click.argument("name")
def sessions_rm(name: str) -> None:
    if remove_session(name):
        click.echo(f"removed: {name}")
    else:
        click.echo(f"not found: {name}", err=True)
        sys.exit(1)


@sessions_cmd.command("export")
@click.argument("name")
@click.option("--output", type=click.Path(path_type=Path), required=True)
def sessions_export(name: str, output: Path) -> None:
    sess = load_session(name)
    lines = [f"# Session `{name}` ({sess.created})", ""]
    for m in sess.messages:
        if m.role == "system":
            continue
        prefix = {"user": "### 사용자", "assistant": "### HanMed"}.get(m.role, m.role)
        lines.append(f"{prefix} ({m.ts})")
        lines.append("")
        lines.append(m.content)
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    click.echo(f"exported: {output}")


if __name__ == "__main__":
    cli()
