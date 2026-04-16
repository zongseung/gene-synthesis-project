"""§10.2 데이터 플로우 — REPL loop (pre-safety → conversation → backend → post-safety).

단일 책임: 한 세션의 대화 주도. 모듈 조립만 하고 비즈니스 로직 최소.
"""

from __future__ import annotations

import time
from dataclasses import asdict

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from hanmed_cli import __version__ as PKG_VERSION
from hanmed_cli.config import DEFAULTS
from hanmed_cli.conversation import Conversation
from hanmed_cli.inference.base import Backend, SamplingConfig
from hanmed_cli.render import (
    print_assistant_prefix,
    print_banner,
    print_error,
    print_info,
    print_refusal,
    print_user_prefix,
    stream_tokens,
)
from hanmed_cli.safety import post_check, pre_check
from hanmed_cli.session import Session, load_session


_SLASH_HELP = """
명령어:
  /help            이 도움말
  /exit | /quit    종료
  /reset           대화 초기화 (system 유지)
  /save <name>     현재 세션 저장
  /load <name>     세션 불러오기
  /temp <float>    sampling temperature 변경
  /max <int>       max_new_tokens 변경
  /tokens          현재 context token 수 보기
"""


def _parse_slash(line: str) -> tuple[str, str] | None:
    if not line.startswith("/"):
        return None
    parts = line[1:].strip().split(maxsplit=1)
    if not parts:
        return None
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    return cmd, arg


def _handle_slash(
    cmd: str,
    arg: str,
    conv: Conversation,
    session: Session,
    cfg: SamplingConfig,
) -> bool:
    """True = REPL 종료 요청. False = 계속."""
    if cmd in ("exit", "quit"):
        return True
    if cmd == "help":
        print_info(_SLASH_HELP)
        return False
    if cmd == "reset":
        conv.reset()
        print_info("대화 초기화 완료 (system prompt 유지).")
        return False
    if cmd == "save":
        if not arg:
            print_error("/save <name> 형식으로 입력하세요.")
            return False
        session.sampling.temperature = cfg.temperature
        session.sampling.max_new_tokens = cfg.max_new_tokens
        path = session.save(arg)
        print_info(f"세션 저장: {path}")
        return False
    if cmd == "load":
        if not arg:
            print_error("/load <name>")
            return False
        try:
            new_sess = load_session(arg)
        except FileNotFoundError as exc:
            print_error(str(exc))
            return False
        # in-place 교체: session 객체 필드 복사
        session.messages = new_sess.messages
        session.sampling = new_sess.sampling
        session.model = new_sess.model
        session.system_prompt_version = new_sess.system_prompt_version
        session.context_drops = new_sess.context_drops
        cfg.temperature = session.sampling.temperature
        cfg.max_new_tokens = session.sampling.max_new_tokens
        cfg.top_p = session.sampling.top_p
        cfg.repetition_penalty = session.sampling.repetition_penalty
        print_info(f"세션 불러옴: {arg} ({len(session.messages)} msgs)")
        return False
    if cmd == "temp":
        try:
            cfg.temperature = float(arg)
            print_info(f"temperature = {cfg.temperature}")
        except ValueError:
            print_error("/temp <float>")
        return False
    if cmd == "max":
        try:
            cfg.max_new_tokens = int(arg)
            print_info(f"max_new_tokens = {cfg.max_new_tokens}")
        except ValueError:
            print_error("/max <int>")
        return False
    if cmd == "tokens":
        n = conv.current_token_count()
        print_info(
            f"context tokens = {n}/{DEFAULTS.context_window_tokens} "
            f"(messages={len(session.messages)}, drops={len(session.context_drops)})"
        )
        return False

    print_error(f"unknown command: /{cmd}. /help 참조.")
    return False


def run_repl(
    backend: Backend,
    session: Session,
    sampling: SamplingConfig,
    *,
    adapter_label: str,
    plain: bool = False,
) -> int:
    """REPL main loop. 정상 종료 0, 에러 1."""
    print_banner(PKG_VERSION, adapter_label, plain=plain)

    conv = Conversation(backend.get_tokenizer(), session)
    pt_session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            print_user_prefix()
            line = pt_session.prompt("").strip()
        except (EOFError, KeyboardInterrupt):
            print_info("\n종료합니다.")
            return 0

        if not line:
            continue

        parsed = _parse_slash(line)
        if parsed is not None:
            if _handle_slash(parsed[0], parsed[1], conv, session, sampling):
                return 0
            continue

        # Layer 1 — pre-safety
        pre = pre_check(line)
        if pre.refused:
            print_refusal(pre.refusal_text or "")
            conv.append_user(line)
            conv.append_assistant(
                pre.refusal_text or "",
                safety={"pre_match": True, "post_disclaimer_appended": False},
            )
            continue

        # prompt 빌드 + 스트리밍 생성
        conv.append_user(line)
        prompt = conv.render_prompt()

        print_assistant_prefix()
        t0 = time.time()
        try:
            text = stream_tokens(backend.stream_generate(prompt, sampling))
        except Exception as exc:  # 런타임 에러 (OOM, dtype 등) 모델 내려앉지 않게
            print_error(f"generation failed: {exc!r}")
            # 이미 append 된 user 제거하지 않고 continue (재시도 가능)
            continue
        elapsed_ms = int((time.time() - t0) * 1000)

        # Layer 2 — post-safety + footer
        final = post_check(text)
        if final != text:
            # 후처리로 추가된 텍스트 출력 (disclaimer + footer)
            print(final[len(text):], flush=True)

        conv.append_assistant(
            final,
            gen_stats={"latency_ms": elapsed_ms, "raw_len": len(text)},
            safety={
                "pre_match": False,
                "post_disclaimer_appended": final != text,
                "footer_appended": DEFAULTS.footer in final,
            },
        )
