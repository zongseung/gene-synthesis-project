"""§10.6 Session Management — JSON schema v0.2, UTC storage, atomic write.

저장 위치:
    - 기본: $XDG_DATA_HOME/hanmed/sessions/
    - fallback: ~/.local/share/hanmed/sessions/

Schema v0.2 (§10.6.2):
    - base_revision, base_manifest_sha256, system_prompt_sha256 권장 필수
    - ts 는 UTC (+00:00) 저장, UI 에서만 로컬 변환 (§10.10.5)
    - autosave 는 `current.json.tmp` → rename() 으로 atomic
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hanmed_cli.config import DEFAULTS


def sessions_dir() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    d = base / "hanmed" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ModelPin:
    base: str = DEFAULTS.base_model
    base_revision: str | None = None
    base_manifest_sha256: str | None = None
    adapter: str | None = None
    adapter_sha256: str | None = None
    adapter_mode: str = "P-CPT"  # or "P-SFT"


@dataclass
class SamplingState:
    temperature: float = DEFAULTS.temperature
    top_p: float = DEFAULTS.top_p
    max_new_tokens: int = DEFAULTS.max_new_tokens
    repetition_penalty: float = DEFAULTS.repetition_penalty


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str
    ts: str = field(default_factory=utc_now_iso)
    gen_stats: dict[str, Any] | None = None
    safety: dict[str, Any] | None = None


@dataclass
class Session:
    schema_version: str = DEFAULTS.session_schema_version
    created: str = field(default_factory=utc_now_iso)
    updated: str = field(default_factory=utc_now_iso)
    model: ModelPin = field(default_factory=ModelPin)
    system_prompt_version: str = DEFAULTS.system_prompt_version
    system_prompt_sha256: str | None = None
    sampling: SamplingState = field(default_factory=SamplingState)
    messages: list[Message] = field(default_factory=list)
    context_drops: list[dict[str, Any]] = field(default_factory=list)

    # --- serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "created": self.created,
            "updated": self.updated,
            "model": {
                "base": self.model.base,
                "base_revision": self.model.base_revision,
                "base_manifest_sha256": self.model.base_manifest_sha256,
                "adapter": self.model.adapter,
                "adapter_sha256": self.model.adapter_sha256,
                "adapter_mode": self.model.adapter_mode,
            },
            "system_prompt_version": self.system_prompt_version,
            "system_prompt_sha256": self.system_prompt_sha256,
            "sampling": {
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                "max_new_tokens": self.sampling.max_new_tokens,
                "repetition_penalty": self.sampling.repetition_penalty,
            },
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "ts": m.ts,
                    **({"gen_stats": m.gen_stats} if m.gen_stats else {}),
                    **({"safety": m.safety} if m.safety else {}),
                }
                for m in self.messages
            ],
            "context_drops": list(self.context_drops),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        model_d = d.get("model", {})
        s = cls(
            schema_version=d.get("schema_version", "v0.1"),
            created=d.get("created", utc_now_iso()),
            updated=d.get("updated", utc_now_iso()),
            model=ModelPin(
                base=model_d.get("base", DEFAULTS.base_model),
                base_revision=model_d.get("base_revision"),
                base_manifest_sha256=model_d.get("base_manifest_sha256"),
                adapter=model_d.get("adapter"),
                adapter_sha256=model_d.get("adapter_sha256"),
                adapter_mode=model_d.get("adapter_mode", "P-CPT"),
            ),
            system_prompt_version=d.get("system_prompt_version", DEFAULTS.system_prompt_version),
            system_prompt_sha256=d.get("system_prompt_sha256"),
        )
        samp = d.get("sampling", {})
        s.sampling = SamplingState(
            temperature=samp.get("temperature", DEFAULTS.temperature),
            top_p=samp.get("top_p", DEFAULTS.top_p),
            max_new_tokens=samp.get("max_new_tokens", DEFAULTS.max_new_tokens),
            repetition_penalty=samp.get("repetition_penalty", DEFAULTS.repetition_penalty),
        )
        s.messages = [
            Message(
                role=m["role"],
                content=m["content"],
                ts=m.get("ts", utc_now_iso()),
                gen_stats=m.get("gen_stats"),
                safety=m.get("safety"),
            )
            for m in d.get("messages", [])
        ]
        s.context_drops = list(d.get("context_drops", []))
        return s

    # --- persistence --------------------------------------------------

    def save(self, name: str) -> Path:
        """§10.6.1 atomic write — `.tmp` → rename."""
        self.updated = utc_now_iso()
        target = sessions_dir() / f"{name}.json"
        # NamedTemporaryFile in same dir for atomic rename on same filesystem
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            suffix=".tmp",
            delete=False,
        ) as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            tmp_path = Path(f.name)
        os.replace(tmp_path, target)
        return target


def load_session(name: str) -> Session:
    path = sessions_dir() / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"session not found: {path}")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return Session.from_dict(d)


def list_sessions() -> list[str]:
    return sorted(p.stem for p in sessions_dir().glob("*.json"))


def remove_session(name: str) -> bool:
    path = sessions_dir() / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False
