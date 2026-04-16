# 10.9 Milestones and Exit Criteria

## 10.9.1 마일스톤 (v0)

| M | 작업 | 의존 | 결과물 |
|---|---|---|---|
| **D1** | vLLM backend 로드 + Bllossom + CPT adapter 단발 generate | §04a §D gates green | `inference/vllm_backend.py` + 수동 smoke |
| **D2** | REPL loop + ChatML history + streaming | D1 | `chat.py` + `conversation.py` |
| **D3** | Safety refusal 2-layer + T4 redteam **20개** 평가 + 30 paraphrase held-out 추가 | D2 + `eval/hanmed_eval_v0/T4.jsonl` | `safety.py` + refusal rate ≥ 99% |
| **D4** | 세션 save/load + `rich` 렌더 + slash commands | D2 | `session.py` + `render.py` |
| **D5** | transformers fallback backend | D1 | `inference/transformers_backend.py` |
| **D6** | pyproject.toml 패키징 + 로컬 테스트 + docs + immutable revision pinning | 전체 | PyPI 준비 상태 (KIOM 승인 대기) |

## 10.9.2 Exit Criteria (v0)

| # | 조건 | 측정 |
|---|---|---|
| **E1** | REPL prompt < 5 s, cold first-token < 30 s, warm first-token < 5 s | startup + latency timer |
| **E2** | §05 safety gate: core 20개 ≥ 99%, paraphrase held-out 30개 ≥ 95%, 한문 jailbreak 10개 ≥ 90% | `hanmed eval` |
| **E3** | 8K context sliding window 후에도 system prompt 보존 | `test_conversation_sliding.py` |
| **E4** | 세션 save/load round-trip 무결성 | `test_session_roundtrip.py` |
| **E5** | Peak GPU mem < 30 GB (8K context) | `nvidia-smi --loop` |
| **E6** *(R3.4 강화)* | P-CPT 경로가 Llama-3 chat template 준수 — **H1 실측 gate** | `test_chatml_template.py` + **200 generic multi-turn prompt (KoAlpaca 류) 자동 평가 + ΔEOT-rate < 2%p vs base + 3 seed variance 보고**. 수동 10개 prompt 는 sanity 용으로만 |
| **E7** *(R3.3 신규)* | 한문 출력 렌더링 (UTF-8 터미널) 에서 깨짐 없음 | macOS Terminal / iTerm2 / Ubuntu GNOME 수동 |

## 10.9.3 v0 → v1 승격 조건

| 조건 | 내용 |
|---|---|
| KIOM 서면 승인 | adapter 공개 + PyPI 배포 |
| `hanmed serve` 개발 | vLLM OpenAI server wrap + auth + rate limit |
| Stage 2 SFT adapter (P-SFT) | §04.6 SFT dataset + 평가 green |
| llama.cpp (GGUF) backend | CPU/Apple Silicon 지원 |

## 10.9.4 v1 → v2 승격 조건

| 조건 | 내용 |
|---|---|
| 상업 라이선스 협상 | KIOM 상업 계약 |
| 의료기기 규제 자문 | 보건복지부 / 식약처 검토 |
| 클라우드 public deployment | AWS/RunPod + 로깅 + 감사 |
| 웹 UI | React/Next.js 클라이언트 (서버 = `hanmed serve`) |

## 10.9.5 일정 (초안, ver2 논문 기준)

| 분기 | 작업 |
|---|---|
| 2026 Q2 | D1~D3 (데모 코어) |
| 2026 Q3 | D4~D6 (UX + 패키징) + ver2 논문 submit |
| 2026 Q4 | v1 — `hanmed serve` + KIOM 승인 절차 + P-SFT 검토 |
| 2027 Q1 | v1 공개 배포 (PyPI + HF Hub) |

데이터 (Core 25) 수집 재개 + §04a §D gates 순차 해제가 D1 착수 전제.

## 10.9.6 실패 시 대응

| 실패 상황 | 대응 |
|---|---|
| E2 safety gate 실패 | regex 보강 + false positive 재측정. paraphrase/hanmun 만 실패하면 classifier 도입 우선 검토 |
| E6 chat template 깨짐 (CPT 후 `<|eot_id|>` 누락) | CPT loss masking / hyperparameter 재검토 (§04a §C.5), 심하면 P-SFT 경로 필수화 |
| vLLM 설치 실패 | transformers backend 단독 운용 (throughput 낮음) |
| KIOM 승인 지연 | v0 는 로컬·기관 내부 배포로 종료, v1 보류 |
| §05 T1 chrF degradation > baseline | CPT 자체 전략 재검토 (preprocessing_and_cpt_spec.md §C.4.3 null-result 대응) |
