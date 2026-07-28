# ver4 · 02. 기획서 — Expository Knowledge-Injection CPT (단일 paradigm)

**버전**: ver4 r2 (2026-04-20, 데이터 수집 감사 반영 — r1: SFT 제거, r0: 3-paradigm 초안)
**선행 문서**: [`01_validation_report.md`](01_validation_report.md)
**폐기 대상**:
- ver2의 "CPT + eval_loss 중심" 방향
- ver4 r0의 3-paradigm 비교(P-A / P-B / P-C)
- 1차 진단의 "block prefix 주입"

---

## 0. 한 줄 요약

**(1) 크롤 중단된 핵심 5권(동의보감·향약집성방 포함)의 서문·발문을 resume 크롤로 먼저 확보하고, (2) 사람-검증 fact sheet에서 권당 long-form 평서문을 합성해 CPT 코퍼스를 재구성한다.** SFT를 배제하여 답변 길이 분포를 보존하고, RAG는 deployment가 아니라 **코퍼스 fact-coverage 상한 측정 용도로만** 남긴다.

### 0.0 왜 데이터 수집을 먼저 다시 하는가 (ver4 r2, r2.1 업데이트 2026-04-20 03:00~04:00 KST 크롤 관측)

2026-04-20 실측 결과 ver2 README의 "Core 14 수집 완료 ✅"는 **사실이 아님**. 단순 "마지막 seq 끊김"이 아니라 **책에 따라 원본 전체의 5~30%만 수집됐던 상태**:

| 미완료 권 | 책 | resume 전 vol / 원본 규모 | 관찰된 누락 |
|---|---|---|---|
| book_008 | **동의보감** | 8 / 25권 | 🔴 32%만 수집. 본문 다수 누락 |
| book_024 | 본초정화 | 2 / 3권 → resume 후 완주 | 🟡 마지막 권 + 판권 |
| book_056 | 의방유취 | 12 / 266권 | 🔴 5% 미만. 사실상 기여 없음 |
| book_093 | **향약집성방** | 26 / 85권 | 🔴 ~30%만 수집 |
| book_139 | 경악전서 | 60 / 64권 → resume 후 완주 | 🟡 마지막 4권 + 판권 |

**환각과의 인과 (업데이트)**:
- 단순 발문·간기 누락이 아니라 **본문의 대다수가 누락된 상태**에서 LLM을 학습했음이 구조적으로 확인됨
- Q1 "동의보감 저자=이시진" 환각은 book_008이 25권 중 8권만 있던 탓에 "許浚" 언급 기회 자체가 3배 축소된 결과
- discriminator 실측 "허준 7회" 같은 희소성이 수집 단계 원인에서 상당 부분 설명됨

따라서 **데이터 재크롤이 P-A+ synthetic augmentation보다 논리적으로 선행**한다. 원문에서 fact가 직접 확인되면 fact sheet 수기 curation 공수 30~50% 감소. resume 크롤 완료 후 **raw corpus 규모가 5.5M → 10~50M tokens 수준으로 증가할 가능성** (특히 book_056 의방유취 기여가 큼).

### 0.1 왜 SFT를 버리는가

- Instruction-tuning QA 쌍은 "짧은 정답" 분포를 학습시켜 모델 답변을 단문으로 collapse시킨다. 사용자 프로젝트 목표(한의학 고전 해제·해설체 출력)와 어긋남.
- chat template + answer masking은 CPT adapter 단일 경로를 강제로 2-stage pipeline으로 만든다 (배포 복잡도 증가).
- → ver4는 **CPT 단일 paradigm** 유지.

### 0.2 왜 RAG를 deployment에서 빼는가

- 추가 infra(인덱스·retriever·chunker) 요구, 본 프로젝트 범위 이탈.
- 그러나 "현재 raw corpus로 달성 가능한 T1 상한"을 알아야 EXP-V4-03의 실패를 "모델 한계"와 "데이터 한계"로 구분 가능 → **upper-bound measurement로는 유지**.

## 1. 목표 지표

| 기호 | 정의 | 목표 (phase 1) | 측정 |
|---|---|---|---|
| `T1_acc` | 한의학 entity factual Q&A 정답률 (closed-book) | ≥ 70% | 사람-라벨, 단답+근거 |
| `T1_hallu` | 오답 중 confabulation 비율 | ≤ 15% | "모름" vs "가공" 분류 |
| `T1_paraphrase` | T1 재표현 holdout 30문항 정답률 | ≥ 50% | over-fit 검출 |
| `answer_length_ratio` | adapter 답변 평균 토큰 / base 답변 평균 토큰 | 0.8 ~ 1.2 | **SFT collapse 방지 지표** (측정 프로토콜 §1.1) |
| `bind_density` | "X는 Y가 편찬" 패턴 코퍼스 내 빈도 (grep) | baseline 대비 × ≥ 14 | entity imbalance 14:1 역전용 |
| `retrieval_recall@3` | (RAG 측정용) 정답 청크 top-3 포함률 | ≥ 70% | upper bound 판정 |
| `forgetting_rate` | KLUE-YNAT 100 정답률 drop | ≤ 5%p | general Korean |

**주 지표 추가**: `answer_length_ratio` — 본 프로젝트가 SFT를 피하는 이유를 숫자로 검증하는 역할. adapter가 base 대비 너무 짧아지면 fail.

### 1.1 `answer_length_ratio` 측정 프로토콜

- **프롬프트 집합**: T1_factual 30문항 + `eval/hanmed_eval_v0/T1_openended.jsonl` 10문항 (해설체 요구 질문). 총 40 프롬프트.
- **decoding**: `do_sample=False`, `temperature=0`, `max_new_tokens=512`, `repetition_penalty=1.1`, `eos_token_id=[eos, eot]`.
- **샘플**: 동일 프롬프트 집합을 base와 adapter 두 모델에 통과, per-response token 수(special token 제외, BPE 기준)를 취합.
- **산식**: `answer_length_ratio = mean(adapter_tokens) / mean(base_tokens)`.
- **보조 지표**: `answer_length_ratio_median`, `pct_short_responses` (≤ 20 tok 응답 비율 — SFT collapse 정밀 탐지).
- **측정 스크립트**: `scripts/probe_answer_length.py` (EXP-01과 EXP-03 양쪽에서 호출).

## 2. 단일 paradigm: P-A+ (Expository Knowledge-Injection CPT)

### 2.1 핵심 설계

Scope 변수 `N` = EXP-V4-00 이후 `post_resume_complete_books` 수 (14 ≤ N ≤ 26). 아래 산식은 모두 `N`의 함수.

1. **수기 fact sheet** (`data/facts/core_factsheet.yaml`) — 완주 검증된 `N` 권 각각에 대해:
   - 저자, 왕대, 편찬 연도, 간행 연도, 장르, 주요 주제 3~5개, 연관 서적 2~3개, 대표 처방/구절 3~5개
   - 10~15개 factual triple / 권 → 총 `10N ~ 15N` triple
   - **원본 오류 검수 포함** (§01의 "향약집성방 태조" 같은 코퍼스 오류는 fact sheet 단계에서 차단)

2. **long-form expository paragraph 합성** — 각 fact를 seed로 200~400 token 서술문 생성:
   - 카테고리 (권당 ~50개 target):
     - 저자 전기 narrative (10)
     - 시대·왕대 배경 (10)
     - 서문·해제 스타일 요약 (15)
     - 주요 처방·본초 narrative (10)
     - 다른 책과의 관계·영향 (5)
   - **평서문만**, Q&A·대화체 금지 → 답변 길이 분포 보존
   - paraphrase ×3~5 (동일 fact를 문체·어순·어휘 다양화)

3. **생성 방식**:
   - Template-based expansion을 **기본**으로 하고, LLM-rewrite는 fact sheet 값을 placeholder로 주입해 자유 생성 금지
   - 생성 후 **entity validation**: fact sheet에 없는 인명/연도가 본문에 등장하면 자동 reject
   - SHA-256으로 중복 차단

4. **규모 (N의 함수)**:

   | N (완주 권) | paragraph 총 | paraphrase ×4 후 | token (×300) | raw 코퍼스와 합 |
   |---|---|---|---|---|
   | 14 | 1,050 | 4,200 | ~1.26M | ~6.8M |
   | 21 | 1,575 | 6,300 | ~1.89M | ~7.4M |
   | 25 | 1,875 | 7,500 | ~2.25M | ~7.8M |

   (책당 75 paragraph 기준. paraphrase ×4, 평균 300 token.)

   **주의**: ver4 r1 draft의 "12.6M tokens"는 책당 300 token × 4 paraphrase = 1,200 token을 seed 당으로 계산한 실수. 정확한 산식은 `paragraph × paraphrase × 300`. 위 표가 정본.

   **cap_tokens 재산정 (EXP-00 완료 후 결정)**: 이전 raw 5.5M은 resume 전 수치. resume 완주 후 **10~50M tokens 수준 예상** (book_056 의방유취 vol=46+ 진행 기준 추정). synth_facts는 1~2M 규모라 비중이 오히려 낮아져 `mix synth_facts 15~25%`로 재조정 가능. cap_tokens는 `raw_post_resume × 2 ~ 3 epoch`으로 재산정.

5. **혼합 전략**:
   - `data/cpt/hanmed_synth_facts.jsonl` 신규 코퍼스로 분리
   - mix 재배분 (N=21 기준, synth_facts 비중이 raw 대비 적정이 되도록):
     synth_facts **25%** / bilingual 35% / zh_only 15% / ko_only 25%
   - synth 25%는 knowledge injection density 확보 + raw corpus 다양성 보존 균형점 (N에 따라 ±5% 조정)

### 2.2 SFT 대비 차별점

| 축 | SFT | P-A+ (expository CPT) |
|---|---|---|
| Loss | answer-only masking | full-text next-token |
| 포맷 | Q/A 쌍 | 평서문 문단 |
| 답변 길이 | 짧아짐 (collapse) | 보존 |
| chat template | 필수 | 선택 |
| 배포 | 2-stage | 1-stage adapter |
| Data 규모 | ~1M tok | ~12M tok |
| 사람 공수 | 저 (QA 50~200쌍) | 고 (fact sheet 14권 × 2~3h) |

### 2.3 "1차 block prefix 처방"과의 차이

| 항목 | 1차 진단 (기각) | P-A+ |
|---|---|---|
| 위치 | 모든 block 앞 | book당 1회 prolog + ~25% synth 코퍼스 |
| 길이 | 짧은 메타줄 | 200~400 token long-form |
| 반복 | sequence 내 중복 | book 경계 존중, paraphrase로 분산 |
| 실패 모드 | shortcut learning | paraphrase 다양성이 shortcut 완화 |
| 기대 `T1_acc Δ` | +5~10%p (추정) | +30~50%p (Allen-Zhu 계열 참고) |

## 3. 실험 설계 (EXP-V4-00 ~ 06)

**배치 원칙**: 아래 상세 블록은 **논리 전제 순**(선행 실험이 먼저 등장: 00 → 01 → 02 → 06 → 03 → 05 → (기각) 04)으로 정렬됨. 실제 실행 순서는 §6 우선순위 표를 따른다. EXP-06이 EXP-03의 전제이므로 06 블록이 03보다 앞에 위치.

### EXP-V4-00 — 데이터 수집 감사 + resume 크롤 (선결 0번)
- **동기**: 2026-04-20 실측 결과 ver2 README의 "Core 14 수집 완료 ✅"는 사실이 아님. **5권(book_008 동의보감 / 024 본초정화 / 056 의방유취 / 093 향약집성방 / 139 경악전서)이 각 책 마지막 vol의 끝 seq에서 로그 끊긴 상태로 저장됨**. manifest.json도 누락. 판권·발문(간기·후서)이 누락됐을 가능성이 Q1/Q3 환각의 근본 원인일 수 있음.
- **가설**: 미완료 5권을 resume 크롤로 완주시키면 (a) raw corpus에 "허준"·"이제마"·"세종" 절대 빈도가 증가하고, (b) EXP-V4-06의 fact sheet curation 공수가 감소(원문에 직접 명시됨) 한다.
- **실행 명령** (2026-04-20 02:53 KST 백그라운드 가동 중, PID 1504679):
  ```bash
  PYTHONHASHSEED=0 nohup .venv/bin/python src/data/crawler/mediclassics_orchestrator.py \
    --output data/raw/mediclassics_unified \
    --books 8,24,56,93,139 \
    --pause 120 \
    --concurrency 2 \
    > data/raw/mediclassics_unified/resume_v4_00.log 2>&1 &
  ```
  재실행 시 content_seq resume (`max(content_seq)+1`부터 이어감)은 orchestrator 기본 동작.
- **신규 스크립트 규격**:
  - `scripts/rebuild_manifests.py`
    - **입력**: `data/raw/mediclassics_unified/book_*/vol_*.jsonl`, `data/stats/mediclassics_book_list.json`
    - **출력**: 각 book 디렉토리의 `manifest.json` (책별로 `book_id`, `book_name`, `crawl_date`, `volumes_meta[{volume_id, volume_nm, content_total}]` 재생성)
    - **로직**: `vol_*.jsonl`의 `max(content_seq)`를 `content_total`로 간주, book_list.json에서 `book_name`·`up_path_nm` 조회
  - `scripts/audit_collection.py`
    - **입력**: `data/raw/mediclassics_unified/`
    - **출력**: `data/stats/book_completeness.json` 스키마
      ```json
      {"book_id": 8, "book_name": "東醫寶鑑",
       "volumes_expected": 25, "volumes_present": 8,
       "records_fetched": 12322, "records_expected_from_orchestrator_log": null,
       "last_seq_per_volume": {"1": 1876, "2": 1543, ...},
       "done_count_from_log": 7, "status": "incomplete|complete",
       "manifest_exists": true, "orchestrator_returned": true}
      ```
  - `scripts/entity_delta.py`
    - **입력**: resume 전 snapshot (이번 감사 시점의 grep 결과) vs 재크롤 후 grep 결과
    - **출력**: `data/stats/entity_delta_v4_00.json` — 허준/이제마/세종/선조/광해군/고종 등 핵심 entity의 빈도 변화
- **고정**: 나머지 21권은 건드리지 않음.
- **지표**:
  - `resume_records_added` (5권 재크롤 후 추가된 record 수)
  - `post_resume_complete_books` / 26 (manifest + DONE 메시지 모두 있는 권 수)
  - `entity_delta`: checkpoint_01(midway) → post_resume 간 "허준"/"이제마"/"세종" 빈도 변화 (raw 기준)
- **측정 스냅샷**:
  - `data/stats/entity_snapshots/checkpoint_01.json` — 2026-04-20 09:35 midway 캡처 완료
    - raw 기준: 허준 43 / 이제마 4 / 세종 9 / 이시진 932 / 레코드 167,713
  - `data/stats/entity_snapshots/post_resume.json` — 크롤 DONE 후 재캡처 (TODO)
  - `data/stats/entity_delta_v4_00.json` — diff 리포트
- **성공 기준** (raw 기준, midway→post_resume delta):
  - `post_resume_complete_books ≥ 25 / 26` (book_139 경악전서는 중국 서적이라 선택적)
  - `delta[허준] ≥ 30` AND `delta[이제마] ≥ 20` AND `delta[세종] ≥ 10` — book_008/182/093 완주 시 서문·본문 누적 기대치
  - `delta[이시진] / delta[허준] < 5` — Chinese prior 상대 약화 (현재 ratio 22:1, post_resume 5:1 이하로 내려가야 P-A+ synth 부담 현실적)
- **실패 해석**:
  - resume 후에도 record 추가 ≤ 1,000 → mediclassics 원본 자체에 해당 seq 없음. 다른 경로 보조 수집 필요.
  - `delta[이제마] < 10` → 동의수세보원(book_182) 서문·본문이 mediclassics에 충분히 없음. Core 확장 또는 외부 출처(규장각 원문 이미지) 필수.
  - `delta[이시진]/delta[허준] ≥ 10` → Chinese prior 악화. bilingual mix 비율 추가 하향 필요.
- **소요**: 크롤 8~16h (rate limit 의존) + 검증 스크립트 2h.
- **실행 명령** (스냅샷/diff):
  ```bash
  # 크롤 완료 후 post snapshot
  .venv/bin/python scripts/entity_delta.py snapshot \
    --label post_resume --output data/stats/entity_snapshots/post_resume.json

  # diff 리포트
  .venv/bin/python scripts/entity_delta.py diff \
    --before data/stats/entity_snapshots/checkpoint_01.json \
    --after  data/stats/entity_snapshots/post_resume.json \
    --output data/stats/entity_delta_v4_00.json
  ```

### EXP-V4-01 — Base Bllossom baseline probe
- **가설**: 환각 중 일부는 base 자체 오답. adapter의 marginal이 독립 측정됨.
- **변경점**: adapter 미로드, base 원형으로 T1 30문항 평가.
- **고정**: 프롬프트, decoding (temp 0, max_new 300), tokenizer.
- **지표**: `T1_acc_base`, 정답/환각/모름 3분류, `answer_length_base`.
- **성공 기준**: 측정 완료 + adapter delta·길이 분포 baseline 확정.
- **실패 해석**: base ≈ adapter → CPT 중립. base > adapter → catastrophic forgetting. base ≫ adapter → adapter rollback.
- **소요**: 1h.

### EXP-V4-02 — T1 eval set 구축 (30~50문항)
- **가설**: 30+ 문항이면 분산 p ≤ 0.08. n=4는 유의성 없음.
- **변경점**: eval set 생성만.
- **고정**: 정답 소스 = 교과서·위키 확정 사실. contamination hash 사전 등록.
- **지표**: 문항 수 ≥ 30, Cohen's κ (이중 라벨링) ≥ 0.9, leak rate = 0.
- **카테고리 (min cover)**: 저자·왕대·연도 (10), 처방·본초 (10), 한문 번역 (10).
- **저장**: `eval/hanmed_eval_v0/T1_factual.jsonl`.
- **소요**: 4h.

### EXP-V4-06 — Fact sheet + synthetic corpus pipeline (P-A+ 선결, **EXP-00 이후**)
- **선결 조건**: EXP-V4-00 성공 (resume 크롤 완료 후 entity_delta 측정).
- **가설**: 책당 fact 10~15개 + long-form paraphrase로 12M token 합성 시 `bind_density × ≥ 14` 달성 (Chinese prior 14:1 imbalance 역전).
- **변경점**: `data/facts/core_factsheet.yaml` 수기 작성 + `src/data/synth/expand_facts.py` 신규 (template expansion + entity validation).
  - Fact sheet scope: EXP-00 후 `post_resume_complete_books` 기준 21~26권 (경악전서는 중국서이므로 scope 여부는 별도 판정).
  - Seed 우선순위: 한국 한의학 핵심 10권 (book 1, 4, 8, 9, 38, 59, 69, 86, 93, 100, 182, 291) 중 EXP-00 이후 완주 검증된 권.
- **고정**: raw corpus 불변, 합성만 별도 파일.
- **지표**: triple 수, entity validation pass rate, paraphrase 다양성 (trigram jaccard), bind_density.
- **성공 기준**: ≥ 150 triple, entity validation ≥ 98%, jaccard 중앙값 ≤ 0.3, bind_density × ≥ 14.
- **실패 해석**:
  - entity validation 저 → template 재설계
  - jaccard 고 → paraphrase 생성 prompt 다변화
  - triple 부족 → Core 25 확장 크롤 or 외부 보조 수집 필요
- **Fact 출처 우선순위** (환각 방지):
  1. **1순위**: 원문 서문·발문 (EXP-00 resume 후 확보되면 직접 인용)
  2. **2순위**: `data/stats/mediclassics_book_list.json` 메타 (KIOM 자체 제공)
  3. **3순위**: 한국민족문화대백과사전·규장각 해제 (교차 검증 2출처 이상)
  4. **금지**: LLM 자유 생성
- **소요**: 2~3일 (fact sheet 수기 curation 12~18h + 합성·검증 2h). EXP-00에서 서문·발문 확보 시 curation 시간 30~50% 감소 가능.

### EXP-V4-03 — P-A+ 재학습 (SFT 없이)
- **가설**: 합성 코퍼스 ~25% 혼합 + 학습 버그 전수 수정 시 `T1_acc ≥ 70%` AND `answer_length_ratio ∈ [0.8, 1.2]`.
- **변경점**:
  - `extract_corpora.py`: book_meta_prolog 1회 + 합성 코퍼스 통합
  - `preprocess.py`: book 경계 BOS/EOS 고정, contamination hash 정규화
  - `cpt_trainer.py`: 5개 버그 수정
    - `modules_to_save=["embed_tokens","lm_head"]`
    - `num_train_epochs=3`
    - `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False`
    - `save_steps = eval_steps = 50`
    - `lr_scheduler_type="cosine_with_min_lr"`, `min_lr_rate=0.1`
- **고정**: LoRA r=32, 7 target modules, seq_len 2048.
- **지표**: `T1_acc_v4a`, `T1_paraphrase`, `answer_length_ratio`, `bind_density`, `forgetting_rate`.
- **성공 기준**: `T1_acc ≥ 70%` AND `T1_paraphrase ≥ 50%` AND `answer_length_ratio ∈ [0.8, 1.2]` AND `forgetting ≤ 5%p`.
- **실패 해석**:
  - `T1_acc` 저 AND `bind_density` 저 → EXP-06 triple 수 불충분, 확장 필요
  - `T1_acc` 저 AND `bind_density` 고 → paraphrase 다양성 부족 → rewrite 재생성
  - `answer_length_ratio` < 0.8 → 합성 문단 길이 분포 조정
- **소요**: 12~16h (데이터 재처리 2h + 학습 ~8h × 2 GPU + 평가 2h).

### EXP-V4-05 — RAG upper-bound measurement (deployment 후보 아님)
- **가설**: bge-m3 top-3 context 제공 시 `T1_acc`가 현재 raw corpus(EXP-00 후 N권)로 도달 가능한 **상한**을 정의.
- **변경점**: retrieval layer만 (가중치 불변). **본 실험은 측정 도구이지 배포 후보가 아님**.
- **고정**: base Bllossom (no adapter), top_k=3, chunk=512.
- **지표**: `T1_acc_rag`, `retrieval_recall@3`.
- **성공 기준**: 측정 완료 (임계값 판정 아님).
- **실패 해석**:
  - `retrieval_recall@3 ≥ 70%` AND `T1_acc_rag ≥ 70%` → 현재 raw corpus만으로 달성 가능 증명
  - `retrieval_recall@3 < 50%` → **raw corpus에 fact 부재 확정. Core 확장 크롤 선결 필수, EXP-03 skip**
  - `recall` 고 `acc` 저 → reader 한계, prompt 개선 필요 (EXP-03 성공 가능성 시사)
- **소요**: 5h.

### (기각) EXP-V4-04 — Synthetic QA SFT
- **기각 사유**: SFT 사용 시 `answer_length_ratio` collapse 위험 + 사용자 요건 이탈.
- **기록만 유지**: 추후 SFT가 필요해지면 별도 round에서 재검토.

## 4. 중단 / 전환 / 종료 기준

| 트리거 | 조건 | 판정 |
|---|---|---|
| **EXP-00 재시도** | resume 크롤 후에도 5권 미완료 ≥ 3권 | rate limit 90→180s, `--concurrency 1`로 완화 후 재시도 |
| **Core 확장 불가피** | EXP-00 `entity_delta[허준+이제마+세종] < 3` | 원문에 서문·발문 부재 확정. 외부 출처 (한국민족문화대백과 등) 보조 수집 필요 |
| P-A+ 채택 | EXP-03 `T1_acc ≥ 70%` AND `paraphrase ≥ 50%` AND `answer_length_ratio ∈ [0.8, 1.2]` | 경로 유지, phase 2 착수 |
| EXP-06 재설계 | EXP-03 `T1_acc < 50%` AND `bind_density × < 14` | 합성 코퍼스 규모 2배 증강, paraphrase ×4 → ×8 |
| paraphrase 품질 문제 | EXP-03 `T1_acc ≥ 70%` AND `paraphrase < 30%` | triple 동일/paraphrase prompt 다변화 후 재학습 |
| 데이터 확장 선결 | EXP-05 `retrieval_recall@3 < 50%` | raw corpus fact 부족 확정, Core 확장 크롤 후 재진입 |
| 하네스 재설계 | EXP-02 κ < 0.7 OR EXP-01 base ≥ 70% | probe 설계 재검토 |
| 종료 | `T1_acc ≥ 80%` AND `paraphrase ≥ 70%` AND RAG gap ≤ 10%p AND `answer_length_ratio ∈ [0.8, 1.2]` | ver4 1차 완료 |
| SFT 재검토 | phase 2에서도 `paraphrase < 40%` 지속 | round_2에서 SFT 재평가 여부 결정 (긴 답변 유지 제약 포함) |

## 5. 전처리 파이프라인 재설계 (ver4 필수)

### 5.1 `src/data/builder/extract_corpora.py`
- 각 book 첫 block 앞 1회 **book_meta_prolog** 삽입: 200~400 token long-form 서문 (짧은 메타줄 아님).
  - 데이터: `data/facts/core_factsheet.yaml` (fact sheet에 근거 없으면 prolog 삽입 금지).
  - 목적: book identity의 문맥적 앵커 제공. 합성 코퍼스(~25%)와 시너지.
- `book_id`별 manifest에 `has_meta_prolog` 플래그 기록.
- **EXP-00 resume 크롤 후 재처리 정책**:
  - **전체 재처리** 기본. `data/cpt/hanmed_{bilingual,zh_only,ko_only}.jsonl`을 삭제 후 재생성. 이유: book_meta_prolog 삽입 + book 경계 fix 변경분이 기존 block에도 적용되어야 함.
  - resume 전 스냅샷을 `data/cpt.prev_v3/`로 백업 (롤백용).
  - dedupe 기준: `(book_id, volume_id, content_seq)` 튜플 유일성. 중복 제거는 `max(content_seq)` 우선 (resume 후 늦게 수집된 record 우선 채택).
- `data/stats/corpus_diff_v3_to_v4.json` 출력: record 추가/삭제/변경 카운트 리포트.

### 5.2 `src/data/builder/preprocess.py`
- Stage 2 pack에 **book 경계 고정**: book 마지막 block 뒤 EOS, 다음 book 시작 전 BOS. greedy pack이 book boundary 침범 금지 (assertion).
- contamination hash: `<ZH>...</ZH>` 순수 원문만 정규화 후 hash. `eval/hashes/heldout_T1.txt`에 EXP-02 구축물 최소 20개 사전 등록.

### 5.3 `src/training/cpt_trainer.py`

**5 클러스터 버그 (총 8개 인자 수정)**:

| 클러스터 | 인자 | 수정값 | 사유 |
|---|---|---|---|
| (1) embed/lm_head 학습 | `modules_to_save` | `["embed_tokens","lm_head"]` | 신규 4 special token + vocab resize 반영, adapter 크기 2.4GB→~150MB로 축소 |
| (2) 실효 epoch | `num_train_epochs` | `3` (명시) | 기존 HF default 1 → 스케줄 완결 |
| (3) best model 선택 | `load_best_model_at_end` / `metric_for_best_model` / `greater_is_better` | `True` / `"eval_loss"` / `False` | step 150 ≈ 156 중 낮은 eval_loss 채택 |
| (4) save-eval 정렬 | `save_steps` / `eval_steps` | 둘 다 `50` | 현재 39 vs 50 교차하여 best save 불가 |
| (5) LR floor | `lr_scheduler_type` / `min_lr_rate` | `"cosine_with_min_lr"` / `0.1` | step 150에서 LR=5.4e-7 바닥 → floor 1e-5 수준 유지 |

### 5.4 신규: `src/data/synth/expand_facts.py`
- 입력: `data/facts/core_factsheet.yaml`
- 처리: template × paraphrase × entity validation
- 출력: `data/cpt/hanmed_synth_facts.jsonl` (기존 bilingual/zh/ko와 동일 스키마)
- 검증 스크립트: `scripts/verify_synth_facts.py` (entity match rate, jaccard, bind_density 리포트)

### 5.5 재현성
- `src/utils/seed.py`: `torch.use_deterministic_algorithms(True, warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG`.
- `extract_corpora.py`, `mediclassics_orchestrator.py`, `expand_facts.py` 모두 `set_global_seed` 호출.

## 6. 우선순위 & 타임라인

| 순서 | 작업 | 소요 | 의사결정 게이트 |
|---|---|---|---|
| **0** | **EXP-V4-00 (데이터 수집 감사 + resume 크롤)** | **8~16h + 2h 검증** | **`post_resume_complete_books ≥ 25/26` AND `entity_delta ≥ 5`. 모든 이후 단계의 전제** |
| 1 | EXP-V4-02 (T1 구축) | 4h | ≥30문항, κ≥0.9 |
| 2 | EXP-V4-01 (base baseline) | 1h | adapter delta·길이 baseline |
| 3 | EXP-V4-05 (RAG upper bound) | 5h | raw corpus 충분성 판정 — skip/continue 결정 |
| 4 | **EXP-V4-06 (fact sheet + 합성)** | 2~3d | **P-A+의 선결 조건**. triple ≥ 150, entity validation ≥ 98% |
| 5 | EXP-V4-03 (P-A+ 재학습) | 12~16h | **최종 판정 실험** |

**핵심 결정 노드**:
- **EXP-00의 `entity_delta`** → 서문·발문이 코퍼스에 실제 추가되는지, fact sheet 공수 규모 결정
- EXP-05의 `retrieval_recall@3` → raw corpus 충분성
- EXP-03의 `T1_acc` × `answer_length_ratio` 2D 결과 → P-A+ 채택/재설계

**병렬화 여지 및 의존 관계**:

| 실험 | EXP-00 의존? | 근거 |
|---|---|---|
| EXP-00 | — | self |
| EXP-01 (base probe) | **독립** | base Bllossom만 사용, corpus 무관. EXP-00과 병렬 가능 |
| EXP-02 (T1 구축) | **독립** | 사람 라벨, corpus 무관. EXP-00과 병렬 가능 |
| EXP-05 (RAG) | **의존** | raw corpus에서 retrieval → EXP-00 완료 후 실행 |
| EXP-06 (fact sheet) | **의존** | 원문 서문·발문 확보 후 curation 공수 단축 |
| EXP-03 (재학습) | **의존** | EXP-06 산출물 + 수정된 파이프라인 요구 |

EXP-00 크롤이 8~16h 소요되는 동안 EXP-01 (1h, base Bllossom probe)과 EXP-02 (4h, 사람 라벨)는 백그라운드 대기 없이 즉시 착수 가능.

## 7. 폐기 결정 정리

- ❌ ver2의 `eval_loss` 중심 평가
- ❌ 1차 진단의 "모든 block 앞 prefix 주입" (shortcut learning)
- ❌ ver4 r0의 SFT 경로 (P-B) — `answer_length_ratio` collapse 위험
- ❌ ver4 r0의 "RAG를 deployment 후보로" 표현 — RAG는 측정 도구만
- ❌ `epoch_variant` 같은 라벨-only manifest — 실제 `num_train_epochs` 인자와 일치 강제
- ❌ contamination hash gate 전체 bilingual 해시 — 원문 정규화 해시로 교체

## 8. 미해결 · 추후 라운드

- **Fact sheet 저작권·출처 표기**: KIOM 비상업 조항(§07) 하에 fact sheet가 "가공물"인지 해제 필요. phase 2 법무 검토.
- **Core 25 확장 완료 후 재처리**: EXP-05에서 `retrieval_recall` 부족 판정 시 선결 필수.
- **Synth 코퍼스의 자기참조 위험**: 모델이 자기가 생성한 문장에 과적합하는 "synthetic collapse" 방지 위해 raw : synth 비율 모니터링.
- **KLUE-YNAT 100**: `forgetting_rate` 측정 스크립트 `scripts/build_t5_klue_subset.py`는 ver4 출하 후 M2.
- **Phase 2 SFT 재평가 조건**: EXP-03에서도 `paraphrase < 40%` 지속 시 **긴 답변 보존 제약을 명시한 instruction SFT** (e.g. 답변 평균 250 token target) 재도입 검토. 현재 ver4에서는 금지.

---

**참조 산출물**:
- 검증: [`01_validation_report.md`](01_validation_report.md)
- 하네스 round_1 원문: `../../.claude/harness-evals/hanmed_cpt/round_1/`
- 폐기된 ver2 근거: `../ver2/04_model_strategy/preprocessing_and_cpt_spec.md`
- 폐기된 ver4 r0 3-paradigm: 본 파일 이전 revision (git history)
