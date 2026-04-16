"""§10.5.1 ChatML + §10.5.3 Sliding Window.

Bllossom-8B 는 Llama-3 chat template 을 사용한다. tokenizer `apply_chat_template`
로 prompt 를 빌드하고, 8K context 초과 시 가장 오래된 user/assistant pair 부터 drop.
system prompt 는 항상 유지.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hanmed_cli.config import DEFAULTS
from hanmed_cli.session import Message, Session, utc_now_iso

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


_SYS_PROMPT_FILE = Path(__file__).parent / "prompts" / "system_v0.1.md"


def load_system_prompt(version: str = DEFAULTS.system_prompt_version) -> str:
    """system_vX.Y.md 로드. (R3.4: version 관리 §10.5.2)."""
    path = Path(__file__).parent / "prompts" / f"system_{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"system prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


class Conversation:
    """§10.5.1~§10.5.3 책임:
    - 메시지 누적
    - ChatML prompt 렌더 (tokenizer.apply_chat_template)
    - 8K 초과 시 oldest pair drop (system 유지)
    - session 에 붙이기/불러오기
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        session: Session,
        *,
        context_window: int = DEFAULTS.context_window_tokens,
    ) -> None:
        self.tokenizer = tokenizer
        self.session = session
        self.context_window = context_window
        # system prompt 를 session 에 보장
        self._ensure_system()

    # --- internal ------------------------------------------------------

    def _ensure_system(self) -> None:
        if self.session.messages and self.session.messages[0].role == "system":
            return
        sys_text = load_system_prompt(self.session.system_prompt_version)
        self.session.messages.insert(
            0,
            Message(role="system", content=sys_text, ts=utc_now_iso()),
        )

    def _chat_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.session.messages]

    def _count_tokens(self) -> int:
        ids = self.tokenizer.apply_chat_template(
            self._chat_messages(),
            tokenize=True,
            add_generation_prompt=False,
        )
        return len(ids)

    # --- public --------------------------------------------------------

    def append_user(self, text: str) -> None:
        self.session.messages.append(Message(role="user", content=text))
        self._apply_sliding_window()

    def append_assistant(
        self,
        text: str,
        *,
        gen_stats: dict | None = None,
        safety: dict | None = None,
    ) -> None:
        self.session.messages.append(
            Message(role="assistant", content=text, gen_stats=gen_stats, safety=safety)
        )

    def render_prompt(self) -> str:
        """현재 대화 → ChatML prompt (모델 입력용)."""
        return self.tokenizer.apply_chat_template(
            self._chat_messages(),
            tokenize=False,
            add_generation_prompt=True,
        )

    def _apply_sliding_window(self) -> int:
        """8K 초과 시 oldest user+assistant pair 부터 drop. system 은 유지.

        Returns:
            dropped pair 개수.
        """
        dropped_pairs = 0
        before = self._count_tokens()
        while self._count_tokens() > self.context_window:
            # idx 0 = system, idx 1 = 가장 오래된 user, idx 2 = 그 assistant
            if len(self.session.messages) <= 3:
                break  # system + current user + current assistant 만 남으면 중단
            # 가장 오래된 user 부터 assistant 까지 drop
            # 안전: idx 1 이 user 가 아니면 한 턴만 drop
            del self.session.messages[1]
            if len(self.session.messages) > 1 and self.session.messages[1].role == "assistant":
                del self.session.messages[1]
            dropped_pairs += 1

        after = self._count_tokens()
        if dropped_pairs:
            self.session.context_drops.append(
                {
                    "ts": utc_now_iso(),
                    "dropped_pairs": dropped_pairs,
                    "tokens_before": before,
                    "tokens_after": after,
                    "context_window": self.context_window,
                }
            )
        return dropped_pairs

    def current_token_count(self) -> int:
        return self._count_tokens()

    def reset(self) -> None:
        """대화 초기화 (system 만 유지, sampling/model pin 유지)."""
        sys_msg = self.session.messages[0] if self.session.messages else None
        self.session.messages = [sys_msg] if sys_msg else []
        self._ensure_system()
        self.session.context_drops = []
