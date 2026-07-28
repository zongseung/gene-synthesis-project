# 10.6 Session Management

## 10.6.1 저장 위치

- 기본: `$XDG_DATA_HOME/hanmed/sessions/`
- `XDG_DATA_HOME` 미설정 fallback: `~/.local/share/hanmed/sessions/`
- autosave: 매 turn 후 `current.json.tmp` 에 먼저 저장한 뒤 `rename()` 으로 원자 교체
- 명시 저장: `/save {name}` → `{name}.json`

## 10.6.2 세션 JSON 스키마

```json
{
  "schema_version": "v0.2",
  "created": "2026-04-16T03:34:56+00:00",
  "updated": "2026-04-16T03:45:22+00:00",
  "model": {
    "base": "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    "base_revision": "3c9b6f7...  # immutable HF snapshot revision",
    "base_manifest_sha256": "sha256 of config/tokenizer/generation manifest",
    "adapter": "outputs/cpt_bllossom/adapter",
    "adapter_sha256": "...",
    "adapter_mode": "P-CPT"
  },
  "system_prompt_version": "v0.1",
  "system_prompt_sha256": "...",
  "sampling": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.1
  },
  "messages": [
    {
      "role": "user",
      "content": "인삼의 성미와 귀경",
      "ts": "2026-04-16T03:34:56+00:00"
    },
    {
      "role": "assistant",
      "content": "인삼은 맛이 달고...",
      "ts": "2026-04-16T03:35:01+00:00",
      "gen_stats": {
        "prompt_tokens": 340,
        "completion_tokens": 128,
        "latency_ms": 2300,
        "throughput_tok_per_s": 55.7
      },
      "safety": {
        "pre_match": false,
        "post_disclaimer_appended": false,
        "footer_appended": true
      }
    }
  ],
  "context_drops": [
    {"ts": "...", "dropped_turns": 2, "remaining_tokens": 6140}
  ]
}
```

## 10.6.3 불러오기

```bash
hanmed chat --session ginseng_session
```

1. JSON 파싱 후 `schema_version` 체크 (`v0.1` 은 `base_revision`/`system_prompt_sha256` 누락 허용, `v0.2` 부터 권장 필수)
2. `base_revision` / `base_manifest_sha256` / `adapter_sha256` 불일치 시 경고 (모델 재현성 깨짐)
3. `system_prompt_version` 또는 `system_prompt_sha256` 불일치 시 경고 후 기본은 **세션 저장본 기준** 으로 replay
4. Conversation 초기화 + system prompt + 기존 messages append
5. REPL 진입

## 10.6.4 삭제 / 정리

```bash
hanmed sessions list
hanmed sessions rm {name}
hanmed sessions export {name} --output conversation.md  # markdown 변환
```

## 10.6.5 Privacy

- v0 기본 모드(로컬/SSH REPL) 에서는 세션을 **로컬 파일** 에만 저장. 별도 세션 업로드는 없다.
- `hanmed serve` (v1) 는 기본적으로 **세션 저장 비활성**. opt-in.
- 클라우드 배포 (v1 C 경로) 는 **세션 저장 금지** 정책 명시 (§10.7 deployment).

주의: 위 "네트워크 전송 없음" 은 **v0 로컬/SSH REPL** 기준이다. v1 `--remote` 경로에서는 prompt 자체가 서버로 전송되므로, 이 문구를 원격 모드에 그대로 적용하면 오해를 부른다.

## 10.6.6 Round-trip test (E4)

`tests/hanmed_cli/test_session_roundtrip.py`:

```python
def test_save_load_roundtrip():
    conv1 = Conversation.new(system="...")
    conv1.append_user("q1")
    conv1.append_assistant("a1")
    conv1.append_user("q2")
    conv1.append_assistant("a2")

    Session.save(conv1, "test_session")
    conv2 = Session.load("test_session")

    assert conv1.messages == conv2.messages
    assert conv1.sampling == conv2.sampling
    assert conv1.system_prompt_version == conv2.system_prompt_version
```

## 10.6.7 열린 결정

1. **SQLite backend** (v1): Simon Willison `llm` 스타일 — 검색·통계 가능. 현재 JSON 은 단순하지만 scale 시 느림
2. **Multi-user 기관 서버**: 세션 파일 namespace 를 user 별 분리 필요 — v1 `hanmed serve` 에서
3. **Session 암호화**: 환자 식별 정보가 우연히 포함될 가능성 — v1 opt-in AES-GCM
