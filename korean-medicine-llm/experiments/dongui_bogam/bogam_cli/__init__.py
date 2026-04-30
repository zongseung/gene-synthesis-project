"""HanMed ver8.1 — Bogam (동의보감) RAG CLI.

별도 entry point 패키지. root 의 ``hanmed`` CLI 와 충돌 없이 ``hanmed-bogam``
명령어로 RAG sidecar (8080) 통합 채팅 REPL 을 띄운다.

사용:
  pip install -e experiments/dongui_bogam
  hanmed-bogam                       # REPL
  hanmed-bogam -q "사물탕의 구성"    # one-shot
  hanmed-bogam --show-retrieved      # REPL + 발췌 path
"""

__version__ = "0.1.0"
