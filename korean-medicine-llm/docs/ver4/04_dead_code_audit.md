# ver4 · 04. Dead Code Audit — 구현 불필요 코드 조사

**작성**: 2026-04-21
**범위**: `/home/user/gene-synthesis-project/korean-medicine-llm` 전체
**방법**: grep 외부 참조 수 + 역할 중복성 + ver2/ver3 → ver4 전환 시 폐기된 경로 교차 검증
**상태**: 후보 목록 (삭제 실행은 사용자 확인 후 별도 라운드)

---

## 0. 판정 기준

| 범주 | 정의 | 판정 |
|---|---|---|
| **A. Dead code** | 외부 호출자 0, import 안 됨 | 즉시 삭제 안전 |
| **B. Legacy/Superseded** | ver2/ver3 잔재, ver4 가 대체한 경로 | 정책 결정 후 삭제 |
| **C. One-shot scripts** | 1회 실행 완료 + 산출물 확보 + 재실행 불요 | 아카이브/삭제 선택 |
| **D. Planned but unused** | ver2 에서 "planned" 표기, 실제 미호출 | 로드맵 vs 제거 판단 |

---

## 1. 🔴 High Confidence Dead Code (외부 참조 0)

즉시 삭제해도 파이프라인 영향 없음. 회수량은 작지만 혼란 요소 제거.

| 파일 | 크기 | grep 결과 | 이유 |
|---|---|---|---|
| `src/data/crawler/mediclassics_crawler.py` | ~20 KB | 외부 0 | v1 single-book 크롤러, `mediclassics_orchestrator.py` 로 완전 대체 |
| `src/data/crawler/mediclassics_multi_crawler.py` | ~15 KB | 외부 0 | v2 다중 크롤러, orchestrator 로 완전 대체 |
| `scripts/tokenizer_probe_quick.py` | ~3 KB | 외부 0 | 1회성 quick 조사 완료 |
| `scripts/cli_oneshot_smoke.py` | ~3 KB | 외부 0 | CLI 초기 스모크, 역할 종료 |
| `scripts/cli_mock.py` | ~3 KB | 외부 1 (본 doc) | v3 디자인이 `branding.py` + `render.py` 로 통합 — 역할 종료 |
| `tea_debug.log` | 0 bytes | 빈 파일 | 4/16 이후 미사용 |

**합계**: ~45 KB + 1 빈 파일
**위험도**: ⚠ 없음

---

## 2. 🟡 Medium — Superseded Data Artifacts

데이터/바이너리 아티팩트. 회수량 큼.

### 2.1 Raw 구 스냅샷

| 경로 | 크기 | 근거 |
|---|---|---|
| `data/raw/mediclassics/` | 4.5 MB | v1 crawl (book_008 등 단건) — `unified/` 에 재수집됨 |
| `data/raw/mediclassics_multi/` | 4.3 MB | v2 crawl — 동일하게 `unified/` 가 대체 |

**선결**: 삭제 전 `unified/` 에 동일 책/레코드 다 포함됐는지 교차 검증 1회.

### 2.2 Legacy outputs (ver2 실패 adapter 연관)

| 경로 | 크기 | 근거 |
|---|---|---|
| `outputs/cpt_bllossom/adapter` | 2.3 GB | ver2 실패 adapter. ver4 README "보관" 명기 |
| `outputs/cpt_bllossom/checkpoint-156` | 2.9 GB | ver2 legacy, `adapter/` 와 본질 동일 |
| `outputs/cpt_bllossom/train_v4_FAILED_22-46.log` | ~16 KB | zero-step crash #1 post-mortem |
| `outputs/cpt_bllossom/train_v4_FAILED2_22-50.log` | ~4 KB | zero-step crash #2 post-mortem |

**메모리 반영**: `feedback_torchrun_venv.md` 에 교훈 이미 저장됨 → FAILED log 삭제 안전.

### 2.3 ver4 전환 백업

| 경로 | 크기 | 정책 |
|---|---|---|
| `data/cpt.bak_pre_v4/` | 32 MB | rollback 용 — EXP-V4-03 성공 확정까지 **keep** |
| `data/cpt_processed.bak_pre_v4/` | 66 MB | 동일 — **keep** |
| `data/facts/core_factsheet.yaml.bak_n3` | 소용량 | 3권 샘플 시절. 26권 검증 완료 → 삭제 가능 |

**즉시 삭제 시 회수**: ~5.2 GB (checkpoint-156 + adapter/, 단 `adapter/` 는 보관 정책 위임)
**보관 유지 시 회수**: ~2.9 GB (checkpoint-156 만)

---

## 3. 🟠 Low Confidence — 정책 결정 필요

ver2/ver3 에서 "planned" 였으나 ver4 가 다른 경로 채택한 건들. 향후 재활성 여부에 따라 판단.

### 3.1 ver2 스모크/매니페스트 파이프라인 (미완 · 대체됨)

| 파일 | ver2 상태 | ver4 대체 | 권장 |
|---|---|---|---|
| `src/training/smoke_cpt.py` | ❌ "Qwen 하드코딩, DDP 충돌" (ver2/11_implementation/current_state.md:69) | `cpt_trainer.py --dry-run` 이 동일 역할 | **제거** (README.md:227 언급 같이 정리) |
| `src/data/builder/build_corpus_manifest.py` | "planned" (ver2 §F.1) + 파일 존재 but 호출자 0 | ver4 는 `preprocess_stats.json` 으로 충분 | **제거** (ver2 README 언급 3건 정리) |
| `src/hanmed_cli/inference/vllm_backend.py` | "v1 reserved" stub | ver4 는 `remote_openai` 로 vLLM 서빙 해결 | **제거 권장** (stub 유지 명분 약함) |

### 3.2 One-shot 토크나이저 조사 스크립트

ver2 시절 토크나이저 확장 설계 시 1회 실행. 재실행 예정 없음.

| 파일 | 외부 참조 | 성격 |
|---|---|---|
| `scripts/tokenizer_probe_bllossom.py` | 3 (docs) | 조사 완료 |
| `scripts/tokenizer_compare.py` | 8 (ver2 docs) | 조사 완료 |
| `scripts/tokenizer_verify.py` | 4 | 조사 완료 |
| `scripts/verify_packed_content.py` | 1 (README) | 재실행 예정? |
| `scripts/classify_books.py` | 1 | 1회 분류 완료 |

**패턴**: 전부 C 범주 (one-shot 완료). 삭제 or `scripts/archive/` 이동 선택.

### 3.3 유지 권장 (산출물 재생성 경로에 필요)

| 파일 | 이유 |
|---|---|
| `scripts/build_factsheet_draft.py` | factsheet 재생성 경로 |
| `scripts/fetch_book_metadata.py` | `build_factsheet_draft.py` 가 호출 |

---

## 4. 🟢 Keep — 명확히 활용 중

- `src/hanmed_cli/*` 전체 (v3 CLI 통합 완료)
- `src/hanmed_cli/inference/{base, transformers_backend, remote_openai}.py`
- `src/training/cpt_trainer.py` (학습 중)
- `src/data/builder/{extract_corpora, preprocess, tokenizer_extend}.py`
- `src/data/synth/expand_facts.py`
- `src/data/crawler/mediclassics_orchestrator.py`
- `src/utils/seed.py` · `src/hanmed_cli/safety.py`
- `scripts/{build_merged_model, entity_delta, probe_factual, verify_synth_facts}.py`
- `docker/Dockerfile.vllm`, `docker/docker-compose.yml`
- `data/facts/core_factsheet.yaml` · `data/tokenizer/*` · `data/cpt/*` · `data/cpt_processed/*`
- `data/raw/mediclassics_unified/`
- `pyproject.toml`

---

## 5. 의사결정 질문 (사용자 확인 필요)

사용자가 아래 체크박스에 결정을 표시한 뒤 실제 삭제 라운드 실행.

- [ ] **Q1.** ver2 legacy `outputs/cpt_bllossom/adapter` (2.3 GB) 삭제? (README 에 "보관" 명기됨)
- [ ] **Q2.** `outputs/cpt_bllossom/checkpoint-156` (2.9 GB) 삭제?
- [ ] **Q3.** `data/raw/mediclassics*` 2 구 디렉토리 (8.8 MB) 삭제? (`unified/` 교차검증 후)
- [ ] **Q4.** `smoke_cpt.py` + `build_corpus_manifest.py` + `vllm_backend.py` 3건 제거? (향후 계획 없음 확정 시)
- [ ] **Q5.** tokenizer probe 4종 (`tokenizer_compare/verify/probe_bllossom/probe_quick` + `verify_packed_content`) 삭제 or `scripts/archive/` 이동?
- [ ] **Q6.** `cli_oneshot_smoke.py`, `cli_mock.py`, `classify_books.py` 삭제?
- [ ] **Q7.** `train_v4_FAILED*.log` 2건 삭제? (메모리에 교훈 저장됨)
- [ ] **Q8.** `data/facts/core_factsheet.yaml.bak_n3` 삭제?
- [ ] **Q9.** `tea_debug.log` 빈 파일 삭제?
- [ ] **Q10.** `data/cpt.bak_pre_v4/` + `data/cpt_processed.bak_pre_v4/` — EXP-V4-03 성공 판정 후 삭제? (지금은 보류)

---

## 6. 권장 실행 순서 (답변 후)

| Tier | 대상 | 조건 | 회수 |
|---|---|---|---|
| **T1** | 🔴 High dead code 6건 + Q9 (tea_debug) | 답변 불요 | ~45 KB |
| **T2** | Q7 (FAILED logs) + Q8 (bak_n3) + Q5 (tokenizer probes) | Y 답변 시 | ~20 KB |
| **T3** | Q4 (ver2 planned 3건) + Q6 (one-shot 3건) | Y 답변 시 | ~30 KB |
| **T4** | Q1 (adapter) + Q2 (checkpoint-156) + Q3 (raw legacy) | Y 답변 시 | ~5.2 GB |
| **T5** | Q10 (bak_pre_v4) | EXP-V4-03 성공 판정 후 | ~98 MB |

**T1~T3**: 바이너리 영향 0 (코드/로그 소거)
**T4~T5**: 디스크 공간 큰 회수 대상 (신중 확인)

---

## 7. 비고

### grep 기반 판단 한계
- 동적 import / 문자열 경로 참조는 미감지 가능. 삭제 전 `rg --pcre2 '"filename"' .` 추가 검증 권장.
- `wandb/` · `outputs/` 는 런타임 생성물이라 본 audit 대상 외.

### docs/ver2, docs/ver3 는 유지
- 역사 기록·라운드 추적 가치 있음 (총 428 KB 소량)
- ver4 문서에서 ver2 경로 인용이 많음 (근거 추적)

### CLAUDE.md 지침 재확인
- "Avoid backwards-compatibility hacks — If you are certain that something is unused, you can delete it completely."
- 본 audit 는 "certain" 판정을 위한 근거 수집 단계.
