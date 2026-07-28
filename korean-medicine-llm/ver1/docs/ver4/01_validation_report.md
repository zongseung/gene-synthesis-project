# ver4 · 01. Validation Report — HanMed-LLM CPT round_1

**작성일**: 2026-04-20
**대상**: `outputs/cpt_bllossom/adapter` (checkpoint-156, Bllossom-8B + LoRA r=32 CPT)
**하네스**: `.claude/skills/harness-engineering-loop` round_1 (generator / discriminator / reviewer / iteration-planner 병렬 실행)
**근거 원문**:
- `.claude/harness-evals/hanmed_cpt/round_1/generator.md`
- `.claude/harness-evals/hanmed_cpt/round_1/discriminator.md`
- `.claude/harness-evals/hanmed-cpt-spec/round_3/reviewer.md`
- `.claude/harness-evals/hanmed_cpt/round_1/iteration_plan.md`

---

## 1. 검증의 동기

ver2.2 R3.5 스펙으로 학습된 adapter를 실제 REPL 수준에서 질의한 결과, **4문항 factual probe 중 3문항에서 심각한 환각**이 확인됨. 사용자 가설은 두 갈래:

1. 전처리 파이프라인에 구조적 결함이 있다
2. 1차 진단에서 제안된 "서지 메타 prefix 주입" 처방 역시 틀렸을 수 있다

본 보고서는 두 가설을 코드·코퍼스 실측으로 검증하고, ver4에서 무엇을 바꿔야 하는지 정량 근거를 제공한다.

## 2. 관측된 환각 증거

### 2.1 probe 스크립트
`/tmp/hanmed_probe.py` — 동일 sampling config (do_sample=False, repetition_penalty=1.1, max_new_tokens=300, chat_template) 으로 checkpoint-156 adapter를 로드해 4문항 질의.

### 2.2 정답 대조

| # | 질문 | 모델 답 | 정답 | 판정 |
|---|---|---|---|---|
| Q1 | 동의보감 저자 / 왕 / 연도 | 이시진 / 세종 / 1593 | **허준 / 선조 / 1610 (간행 1613)** | ❌ 전부 오답 |
| Q2 | 사상의학 창시자 / 저서 | 張邯 / 『사상론』 | **이제마 / 『동의수세보원』** | ❌ 인물·저서 모두 가공 |
| Q3 | 향약집성방 왕 / 편찬자 | 인종 / 이시진 | **세종 / 유효통·노중례·박윤덕** | ❌ 전부 오답 |
| Q4 | 오장 | 신·비·간·심·폐 | 간·심·비·폐·신 | ✅ (순서만 상이) |

**환각률 3/4 (75%)**. Q4는 base Bllossom이 이미 갖고 있는 교과서 지식(한·중 공통)이므로 adapter 기여분이 아님.

### 2.3 환각 패턴

- **중국 prior 누출**: "이시진"(중국 명대『본초강목』저자)이 한국 조선 저자 자리에 두 번 등장 (Q1, Q3).
- **confabulation**: "張邯(장원)"은 실존하지 않는 인물. 근거 없는 생성.
- **왕대 오배치**: 세종→인종, 선조→세종 등 dynasty mismatch.

## 3. 학습 실행 상태 실측

### 3.1 학습 지표

| 항목 | 값 | 의미 |
|---|---|---|
| total_steps | 156 | cap 20.4M ÷ 131,072 tok/step |
| epoch 실측 | 0.931 | `trainer_state.json:5` |
| num_train_epochs | **1** | manifest의 `epoch_variant=3`과 불일치 — cap만 늘었을 뿐 실질 1-epoch |
| train_loss (10→150) | 2.81 → 1.86 | 안정적 하강 |
| eval_loss (50/100/150/156) | 2.07 / 1.91 / **1.887 / 1.887** | step 150 이후 완전 plateau |
| LR at step 150 | 5.4e-7 | cosine 바닥 — plateau는 수렴이 아니라 LR 소진 |
| best_model_checkpoint | **null** | `load_best_model_at_end` 미지정, best 선택 없음 |
| modules_to_save | **null** | vocab resize(128256→128260)했으나 새 embed/lm_head 미학습 |
| adapter 파일 크기 | **2.4 GB** | r=32 치고 비정상. 미학습 embedding이 통째로 저장된 것 |

### 3.2 학습된 LoRA 타겟
`adapter_config.json` 실측: `q/k/v/o + gate/up/down` 7개 projection, r=32, α=64, dropout 0.05. MLP FFN의 key-value memory (Geva 2020) 경로는 열려 있음. 그러나 embedding은 닫혀 있어 신규 4 special token (`<ZH>/</ZH>/<KO>/</KO>`)은 영구 random init.

## 4. 코퍼스 실측 — 환각의 진짜 원인

### 4.1 저자·왕대 entity 빈도 (discriminator 실측)

| entity | 코퍼스 내 카운트 | 비고 |
|---|---|---|
| 이시진 (중국, 본초강목) | **101** | 학습 데이터에 가장 빈번 |
| 허준 | 7 | 동의보감 저자 |
| 이제마 | 2 | 사상의학 창시자 |
| 세종 | 2 | 향약집성방·의방유취 편찬기 왕 |

**14:1 비대칭**. next-token CE는 window 공존 토큰쌍 빈도에 비례해 prior를 만들기 때문에, 아무리 오래 학습해도 모델은 "저자" 슬롯에서 "이시진"이 이길 확률이 높다. 이것이 Q1·Q3에서 "이시진" 누출의 직접 원인.

### 4.2 "허준 → 동의보감" 명시 binding 문장의 희소성

`data/cpt/hanmed_ko_only.jsonl`에 명시적 저자 binding은 3~4건에 불과 (book_id=4 content_seq=80, book_id=8 content_seq=5,6, book_id=59 content_seq=116). 존재는 하나 14:1 imbalance를 뚫기에 절대 부족.

### 4.3 원본 오류

`hanmed_ko_only.jsonl` 내부에 "**향약집성방 태조 때에 편찬**" 류의 **원문 자체 오류**가 존재. 즉 코퍼스 자체에 잘못된 fact가 섞여 있어 정답조차 오염됨. 이것이 Q3 "인종" 환각의 보조 요인일 수 있음.

### 4.4 실제 mix 비율 재측정

1차 진단에서 "zh 62.5%" 라고 쓴 것은 **manifest의 target weight**였다. `corpus_v1.json`을 실제 byte 비중으로 측정하면:

- bilingual 48.7% / zh_only 23.8% / ko_only 27.5%

한자 노출 총량은 72.5%로 여전히 높지만, zh 편중 자체가 환각의 주원인이라는 주장은 **부분 철회**. 주원인은 **entity frequency imbalance**.

## 5. 전처리 파이프라인 구조적 결함

### 5.1 서지 메타 탈락 (extract_corpora.py)

`src/data/builder/extract_corpora.py:114-149` — raw record의 `book_id`·`volume_id`·`up_path_nm`은 있으나, **저자·왕대·편찬연도는 raw에도 학습 block에도 삽입되지 않음**. `text` 필드 = `<ZH>본문</ZH>\n<KO>본문</KO>\n\n` 만.

즉 "동의보감 = 허준" 같은 triple은 **코퍼스 어디에도 명시되지 않은 사실**이다. CPT next-token objective는 코퍼스에 없는 사실을 만들어낼 수 없다 (generator 결론과 일치).

### 5.2 book 경계 무시 pack (preprocess.py)

`src/data/builder/preprocess.py:304-328` — Stage 2 greedy packing이 **book 경계를 넘어 2048-token sequence를 채움**. 결과적으로 한 sequence에 서로 다른 책의 본문이 섞여 "book identity"가 soft-blend 됨.

이것이 **"메타 prefix 주입" 제안이 실패할 주된 이유**다 (discriminator 논거). prefix를 매 block 앞에 붙여도 packed sequence 안에서 prefix들이 중복·겹침 → LM이 prefix 재생산에 loss를 쓰게 되고, "prefix↔내용" binding이 shortcut으로 학습되지 entity-level knowledge로 내재화되지 않는다.

### 5.3 contamination hash gate 미작동

`src/data/builder/preprocess.py:217-223` — Stage 1 contamination gate는 `<ZH>...</ZH>\n<KO>...</KO>\n\n` 전체 block의 SHA-256으로 계산된다. 그러나 `eval/hashes/heldout_T1.txt`는 "한문 원문"만의 해시를 담도록 설계됨. **포맷 mismatch로 실제 bilingual 파일에서는 매치가 영영 불가능**. zh_only만 부분 가드 유효.

게다가 `eval/hashes/heldout_T1.txt` 현재 empty → 사실상 contamination 검증 없이 학습됨.

### 5.4 `corpus_v1.json` 의 drop_ratio nulls

Stage 1 quality/dedup drop ratio가 전부 null 상태로 기록됨 (reviewer 지적). Stage 1이 정상 실행됐는지, 아니면 `--allow-missing-eval` 우회 경로로 스킵됐는지 불분명.

### 5.5 seed / determinism 누락

- `src/data/crawler/mediclassics_orchestrator.py`, `src/data/builder/extract_corpora.py`에 `set_global_seed` 호출 없음.
- `src/utils/seed.py` 자체가 `torch.use_deterministic_algorithms` / cudnn 설정 미포함.

## 6. 학습 코드의 실행 버그

| 파일:라인 | 문제 | 영향 |
|---|---|---|
| `cpt_trainer.py:419-435` | `modules_to_save=null` + vocab resize | 신규 4 special token embedding 영구 random init, 저장됨에도 학습 안 됨 |
| `cpt_trainer.py` (TrainingArguments) | `num_train_epochs` 미지정 → HF default 1.0 | manifest의 `epoch_variant=3`은 라벨일 뿐, 실제 epoch=0.93 |
| `cpt_trainer.py` | `load_best_model_at_end` / `metric_for_best_model` 미설정 | best_model 미선택, 마지막 step adapter만 export |
| `cpt_trainer.py` | `save_steps=39` ∌ `eval_steps=50` | save 시점과 eval 시점이 교차 안 해 어차피 best 선택 불가 |
| `cpt_trainer.py` (cosine) | `min_lr_rate` 미지정 | step 150에서 LR=5.4e-7, 실질 학습 중단 |

## 7. "메타 prefix 주입" 처방의 결함

1차 진단이 제안한 "block 앞에 `저자: 허준 · 조선 · 1613` 삽입"의 실패 모드:

1. **shortcut learning**: packed sequence 안에서 prefix가 본문과 독립적인 템플릿처럼 취급되어, LM은 "prefix를 복제하는 법"만 학습하고 "사실을 recall하는 법"은 배우지 못한다 (discriminator 논거).
2. **book boundary violation**: 5.2에서 지적한 대로 book 경계를 넘어 pack되므로 한 sequence에 여러 책의 prefix가 섞여 binding이 흐려진다.
3. **QA format generalization 실패**: prefix는 평서문 템플릿인데 실제 QA는 질문-응답 형식 → 분포 불일치로 transfer 저조.
4. **원본 오류 증폭**: 5.3에서 본 "향약집성방 태조 때" 같은 원문 오류가 prefix로 승격되면 calibration이 악화된다.
5. **CPT paradigm의 한계 미해결**: next-token CE는 QA 양식의 recall에 최적이 아니다. SFT 또는 RAG가 구조적으로 더 적합.

**결론**: 1차 진단의 "메타 prefix CPT"는 방향 자체가 틀렸다. ver4는 paradigm 전환을 비교 측정하는 체계로 설계해야 한다.

## 8. 하네스 간 합의 / 쟁점

### 합의
- **CPT alone으로 factual entity recall 불가**: generator·discriminator·reviewer 독립 확인.
- **현재 adapter는 "스타일 적응"만 기여**: loss 2.81→1.86 하강은 fluency, fact 내재화와 무관.
- **전처리 파이프라인에 최소 3개의 구조적 결함 존재**: 서지 메타 탈락 / book 경계 무시 pack / contamination gate 미작동.

### 쟁점
- **환각 기전**: generator = "binding 문장 부재", discriminator = "문장은 있으나 14:1 imbalance". 실은 **둘 다 맞음** — 절대 빈도도 절대 부족, 상대 분포도 비대칭. 두 원인이 곱해짐.
- **개선 기대치**: generator는 서지 메타 재학습으로 환각률 75→60%(-15%p, 추정), discriminator는 synthetic QA SFT 50~200쌍으로 +40%p 가능하다고 예측. 쟁점 해소 방법 = **EXP-V4-03과 EXP-V4-04 직접 대조**.
- **RAG의 upper bound**: 미측정. EXP-V4-05 필요.

## 8.5. 데이터 수집 감사 (2026-04-20 실측, ver4 r2 추가)

### 8.5.1 ver2 README와 실제 상태 불일치

ver2 README `## 1. 현재 상태`는 Core 14를 "✅ 수집 완료"로 표기하고 있으나, `data/raw/mediclassics_unified/`를 실측한 결과 **핵심 5권이 vol 마지막 seq에서 로그가 끊긴 채 manifest.json 없이 저장**된 상태다.

| book | 이름 | resume 전 vol | records | 실측 원본 규모 | 누락 심각도 |
|---|---|---|---|---|---|
| 008 | **동의보감** | 8 | 12,322 | **25권** (크롤 재개 시 vol=20+ 확인) | 🔴 전체 권수의 ~30%만 수집됐던 상태 |
| 024 | 본초정화 | 2 | 5,767 | 3권 (resume 후 전체 완주, DONE) | 🟡 1권 + 말미 누락 |
| 056 | 의방유취 | 12 | 7,190 | 266권 (의방유취 원본 규모, resume 후 vol=46+ 확인) | 🔴 5% 미만만 수집됐던 상태 |
| 093 | **향약집성방** | 26 | 7,032 | 85권 (resume 후 vol=81+ 확인) | 🔴 ~30%만 수집됐던 상태 |
| 139 | 경악전서 | 60 | 14,234 | 64권 (resume 후 64권 전체 완주, DONE) | 🟡 마지막 4권 + 판권 누락 |

**2026-04-20 02:53 KST resume 크롤 중간 관측 (03:25~04:01)**:
- book_024 / book_139: DONE (총 +3,774 records 추가 확보)
- book_008 / 056 / 093: 진행 중. 특히 **의방유취 266권 중 12권만 수집된 상태였음이 크롤 재개로 판명** — ver2 README의 "Core 14 완료 ✅"가 사실이 아니었음이 구조적으로 확인

**ver2 README `current state` 테이블과의 괴리**:
- ver2는 "Core 14 14권 chars_zh 1.20M / chars_ko 1.97M" 로 기록 → 이는 **각 책의 극히 일부만 포함한 수치**
- 실제로 의방유취 한 권만도 266배 가량 더 큰 원본을 갖는 대형 서적
- `orchestrator.log`와 per-book 로그 교차검증으로 확인
- resume 전 총 records: 109,288. resume 후 얼마나 증가할지는 EXP-00 완료 시점에 기록.

### 8.5.2.pre. entity 빈도 실측 정정 (2026-04-20 09:35 checkpoint, raw 기준)

1차 discriminator 실측 "허준 7 / 이제마 2 / 세종 2 / 이시진 101"은 `data/cpt/hanmed_ko_only.jsonl` (한글 국역 전용 파일) 기준이었다. raw corpus (`data/raw/mediclassics_unified/book_*/vol_*.jsonl`의 `original` hanja + `trans_ko` + `trans_en` 연결)에서 재측정한 결과, resume 크롤 **진행 중 midway checkpoint**에서 다음과 같이 수치가 상향된다:

| entity | ko_only (초기) | raw midway checkpoint_01 | ratio |
|---|---|---|---|
| 이시진 (중국, 본초강목) | 101 | **932** | ×9.2 |
| 허준 | 7 | **43** | ×6.1 |
| 이제마 | 2 | **4** | ×2.0 |
| 세종 | 2 | **9** | ×4.5 |
| 허임 (침구경험방) | — | 61 | — |
| 강명길 (제중신편) | — | 30 | — |

**핵심 해석**:
- 절대 binding density는 discriminator 추정보다 큼 (허준 43). 그러나 **이시진 : 허준 비율은 14:1 → 22:1로 악화** — Chinese prior가 raw 전체로 보면 더 지배적.
- 이제마 4회는 여전히 극심히 부족. 동의수세보원(book_182) 볼륨이 일부만 수집된 한계.
- 간기 연도(1433, 1610, 1613, 1894) 모두 1~2회 — fact sheet에서 의도적 보강 필수.

snapshot: `data/stats/entity_snapshots/checkpoint_01.json` (306 files, 167,713 records 스캔).

### 8.5.2 환각과의 인과 가설 (신규)

한의서의 저자·편찬연도·왕대는 주로 다음 3곳에 기록된다:

| 위치 | 권 내 배치 | 내용 |
|---|---|---|
| **서문(序)·인(引)** | 맨 앞 (작은 seq) | 저술 경위, 저자 서명, 왕명·연도 |
| **범례(凡例)** | 서문 뒤, 본문 앞 | 편찬 원칙, 편자 명기 |
| **발문(跋)·후서(後序)·간기(刊記)** | 맨 뒤 (큰 seq) | 간행 주관자, 간행 연도, 판목 정보 |

**실제 실측 (2026-04-20 resume 크롤 관측)**:

- **book_008 동의보감**: 25권 원본 중 resume 전 8 vol 만 수집 → **권 중반 이후 전체 누락**. 서문·범례·발문·간기 모두 가능성 없음 단정 불가하나 **본문 다수 누락이 주 문제**.
- **book_056 의방유취**: 266권 원본 중 resume 전 12 vol 만 수집 (5% 미만). 사실상 **코퍼스 기여가 거의 없던 상태**.
- **book_093 향약집성방**: 85권 원본 중 26 vol 만 수집.
- **book_024 / 139**: 원본 거의 완주했으나 마지막 vol seq 끝에서 끊김 → 주로 발문·간기만 누락.

Q1·Q3 환각과의 인과 재정리:

- Q1 "동의보감 저자=이시진" 환각 → book_008은 **전체 25권 중 8권만** 수집됐던 상태였으므로, 권 전반부에 등장할 법한 저자 언급 빈도조차 낮을 수밖에 없음. "許浚" 빈도 7회는 이 누락을 직접 반영.
- Q3 "향약집성방 왕=인종" 환각 → book_093은 85권 중 26권만 수집 → 세종·유효통·노중례 등의 명시 기회가 원본 대비 1/3로 줄어듦. 게다가 그 중 발문·범례도 누락.

→ **환각의 "수집 단계 원인"이 ver4 r0/r1 설계 가정보다 훨씬 크다**는 것이 크롤 재개로 정량 확인되는 중. EXP-00 완료 후 raw corpus 규모가 10~50배 증가할 가능성 있음 (특히 book_056 의방유취).

Q1·Q3 환각과의 인과:

- Q1 "동의보감 저자=이시진" 환각 → book_008의 발문·간기 누락 + 서문에 "許浚" 언급 빈도 부족 가설. resume 후 vol=1~2 범위 "許浚"·"許氏" 빈도 재측정 필요.
- Q3 "향약집성방 왕=인종" 환각 → book_093의 발문 누락 + 세종 원년(1433년) 편찬 관련 범례 누락 가설.

discriminator가 실측한 "허준 7회 / 이제마 2회 / 세종 2회"의 극단적 희소성은 위 누락 중 **최소 발문·간기 계열**을 반영하는 수치로 해석할 수 있다. 서문 누락 여부는 resume 후 확정된다.

**결과적으로 환각의 원인은 3중 구조**:
1. **수집 단계**: 서문·발문이 애초에 누락 (8.5.1)
2. **전처리 단계**: 남은 본문에서도 저자/왕대 메타 탈락 (§5.1)
3. **학습 단계**: 짧은 1 epoch + LR 소진 + embed 미학습 (§6)

→ ver4 r2는 (1)을 EXP-V4-00, (2)를 §5 전처리 재설계, (3)을 §6 학습 코드 수정으로 분리 대응한다.

## 9. 결론 — 무엇이 검증되었고 무엇이 남았나

**검증됨**:
1. ✅ 환각률 3/4 실증 (n=4, 통계 유의성은 약함)
2. ✅ 전처리 파이프라인의 구조적 결함 3건 (서지 메타 / book 경계 / contamination gate)
3. ✅ 학습 코드의 실행 버그 5건 (modules_to_save / epochs / best_model / save-eval 정렬 / min_lr)
4. ✅ 코퍼스의 entity imbalance 14:1
5. ✅ "메타 prefix CPT" 방향 오류 (5가지 실패 모드 식별)
6. ✅ **데이터 수집 미완료 5권 확인** (ver2 README 표기와 불일치; §8.5)

**남은 미해결** (ver4 r2 기준):
- 🔲 **미완료 5권의 resume 크롤** (EXP-V4-00) — 서문·발문 확보로 환각 원인 1차 제거
- 🔲 base Bllossom baseline (adapter marginal contribution 및 답변 길이 분포)
- 🔲 T1 factual eval set 30+문항 구축 (현재 n=4로 약함)
- 🔲 사람 검증 fact sheet + long-form 합성 코퍼스 (N 권 × 1.26M~2.25M tok, EXP-00 후 N 확정) 생성 파이프라인
- 🔲 P-A+ (Expository KI-CPT) 재학습 시 `T1_acc` × `answer_length_ratio` 동시 달성 가능성
- 🔲 RAG upper bound — **deployment가 아닌 측정 용도**로만 (raw corpus fact 충분성 판정)

→ ver4 기획서 `02_plan_v4.md`에서 이 6건을 **EXP-V4-00 (선결) / 01 / 02 / 03 / 05 / 06**으로 설계한다. (EXP-V4-04 SFT는 길이 collapse 우려로 r1에서 기각.)
