# ver4 R1 — Mix 재설계 CPT 결과 · BASE vs R1 probe

**상태**: R1 학습 완료 + BASE vs R1 probe 완료 (2026-04-22 04:00)
**근거 문서**: [`../../claudedocs/research_hanmed_cpt_methodology_20260421.md`](../../claudedocs/research_hanmed_cpt_methodology_20260421.md)
**미완**: R0 vs R1 probe (저장시점 미실행), merged v0.2 빌드, T1_content 정량 eval

---

## 1. R1의 목적

[research 보고서 §0](../../claudedocs/research_hanmed_cpt_methodology_20260421.md) 의 **우선 적용 3가지 중 ①번** 구현:

> Corpus mix 재설계: general replay ≥ 15%, KO 앵커 ≥ 50%, ZH-only ≤ 10%

②(data unification), ③(on-policy distillation)은 **R1 에 포함하지 않음** — R2 과제로 유보.

### 관찰된 3 실패 모드 (R0 상태)

| ID | 증상 |
|----|------|
| F1 | style-over-fact: 《傷寒論》 조문 환각, 《삼인론》·《直指金》 등 가짜 서명 인용 |
| F2 | language collapse: 한국어 질문 → 한문 편중 응답, `</KO>` 토큰 누출 |
| F3 | rare-token degradation: multi-turn 누적 시 `◯◤◥ 俰` 심볼 덤프 |

---

## 2. 변경 요약 (R0 → R1)

### 2.1 Corpus mix

| corpus | R0 | R1 | 출력 경로 |
|---|---|---|---|
| `hanmed_ko_only` | 0.35 | **0.45** | `data/cpt_processed/hanmed_ko_only_packed_2048.jsonl` (4,749 seqs) |
| `hanmed_bilingual` | 0.45 | **0.30** | `data/cpt_processed/hanmed_bilingual_packed_2048.jsonl` (9,077 seqs) |
| `hanmed_zh_only` | 0.20 | **0.10** | `data/cpt_processed/hanmed_zh_only_packed_2048.jsonl` (4,802 seqs) |
| `wiki_ko` (신규) | 0.00 | **0.15** | `data/cpt_processed/wiki_ko_packed_2048.jsonl` (4,703 seqs, 9,932 clean docs) |

검증: general replay = 0.15 ≥ 15% ✓ · KO 앵커 = 0.60 ≥ 50% ✓ · zh_only = 0.10 ≤ 10% ✓ · sum=1.00 ✓

### 2.2 신규 빌더·전처리

- `src/data/builder/build_wiki_ko.py` — HF `wikimedia/wikipedia:20231101.ko` 스트리밍 → 10,793 docs / 13.6M chars (~4.87M tokens est.) 샘플
- `src/data/builder/preprocess.py` — `CORPORA_KINDS` 에 `wiki_ko: ko_only` 추가
- wiki 문서 모두 `book_id="wiki_ko"` 공유 → §5.2 book-boundary 제약 하에서 정상 packing (compression 2.1×, padding 20.85%)

### 2.3 `cpt_trainer.py`

- `DEFAULT_MIX` 문자열 재구성, 주석에 R1 근거 인라인 (BianCang/GeRe 인용)
- `CORPUS_PATHS['wiki_ko']` 를 `data/cpt_processed/` 로 갱신
- `--mix` help 문구 갱신 (기존 "sum=0.40" 가정 제거)

### 2.4 서빙·후처리 보수 (훈련 외부, 같은 사이클에 정리)

| 변경 | 이유 |
|---|---|
| `src/hanmed_cli/config.py` 에 `footer_enabled: bool = False` 추가 | 환각 응답에 자동 붙던 `— KIOM mediclassics.kr 기반 학습` footer 비활성화 (가짜 권위 차단) |
| `src/hanmed_cli/safety.py` footer append 라인 conditional gating | 토글 1개로 SFT+RAG 단계에서 재활성 가능 |
| `src/hanmed_cli/inference/remote_openai.py` 의 `min_tokens:150` / `frequency_penalty:0.3` / `presence_penalty:0.2` 제거 | F3 심볼 덤프 트리거 제거. 원 동기(synth template 조기 EOS)는 R1 mix 에서 해소됨 |

### 2.5 롤백 자산

```
outputs/_backups/cpt_bllossom_adapter_R0_20260421/           (2.4GB)
outputs/_backups/hanmed_merged_v0.1_R0_20260421/             (15GB)
```

---

## 3. 학습 실행

### 3.1 Infra 이슈 (해결)

- **NCCL BROADCAST 600s hang** 두 차례 발생 (rank 0 525M param broadcast 실패)
- `nvidia-smi topo -m` 확인 결과 GPU0↔GPU1 이 `SYS` (cross-NUMA PCIe) — NVLink 없음
- Docker vLLM 기동/중단 반복 후 P2P state 가 stale 해진 것으로 추정
- 처방: launcher 에 `NCCL_P2P_DISABLE=1` + `NCCL_IB_DISABLE=1` + `NCCL_DEBUG=WARN` 명시
- 결과: BROADCAST 즉시 완료 → 정상 학습

launcher: `outputs/cpt_bllossom_R1/_launcher.sh` (setsid + 새 master_port 29507 + env)

### 3.2 구성

- base: `MLP-KTLim/llama-3-Korean-Bllossom-8B` (8B, bf16)
- tokenizer: `data/tokenizer/hanmed_bllossom_ext` (extended vocab)
- LoRA: r=32, α=64, dropout=0.05, targets {q,k,v,o,gate,up,down}_proj
- cap_tokens: 20.4M, epoch_variant=3, total_steps=156, warmup 7 steps
- micro_bs=2 × grad_accum=16 × world=2 = effective 64 seqs/step
- 2× RTX A6000 49GB, DDP, bf16

### 3.3 실측 수치

- per-step: **71.15s** (전 구간 편차 ±0.1s, 데이터로더 병목 없음)
- 총 학습 runtime: **3h 13min** (R0 대비 훨씬 짧음 — R0 정확한 시간 불명)
- GPU 사용: 양쪽 모두 40GB/49GB (base model 16GB + 오버헤드), 100% util, 282W/300W

### 3.4 Loss 곡선 (R0 vs R1)

| step | R0 train | R1 train | Δ | R0 eval | R1 eval | Δ eval |
|---|---|---|---|---|---|---|
| 10  | 2.651 | 2.564 | **-0.087** | | | |
| 20  | 2.236 | 2.226 | -0.010 | | | |
| 30  | 2.086 | 2.096 | +0.010 | | | |
| 40  | 1.978 | 2.030 | +0.052 | | | |
| 50  | 1.909 | 1.953 | +0.044 | **1.850** | **1.906** | **+0.056** |
| 60  | 1.843 | 1.906 | +0.063 | | | |
| 70  | 1.842 | 1.871 | +0.029 | | | |
| 80  | 1.825 | 1.848 | +0.023 | | | |
| 90  | 1.801 | 1.834 | +0.033 | | | |
| 100 | 1.755 | 1.781 | +0.026 | **1.723** | **1.796** | **+0.073** |
| 110 | 1.749 | 1.767 | +0.018 | | | |
| 120 | 1.746 | 1.788 | +0.042 | | | |
| 130 | 1.732 | 1.780 | +0.048 | | | |
| 150 | — | — | — | **1.678** | **1.756** | **+0.078** |
| 156 | — | — | — | **1.675** | **1.753** | **+0.078** |

**수치 해석**:
- R1 최종 eval loss 가 R0 대비 **+4.7% 높음** (1.753 vs 1.675)
- 하지만 R1 val set 은 wiki_ko 95개(20.3%) 포함한 **더 넓은 분포** → unfair apples-to-apples 아님
- Hanmed-only subset 으로 쪼개 재측정해야 fair 비교
- **`eval_loss` 는 factual recall 과 비상관**이라 ver4 §README 자체 인용. 품질 판정은 behavioral probe 로 해야 함

---

## 4. Probe 결과 (BASE vs R1)

### 4.1 실행 조건

```
scripts/probe_adapter.py --adapter outputs/cpt_bllossom_R1/adapter
```

- single-device (cuda:0) · 같은 모델에 adapter attach 상태로 `with model.disable_adapter():` 로 BASE 측정, 그대로 둔 채 R1 측정
- greedy (temp=0, do_sample=False), max_new_tokens=120, seed=42+i (prompt 별 고정)
- 공정 A/B: 동일 weight memory · 동일 tokenizer · 동일 RNG
- raw log: `/tmp/probe_R1.log` (본 문서 §4.3 에 전문 보존)

### 4.2 프롬프트별 판정

| # | 프롬프트 | BASE 품질 | R1 품질 | 우위 |
|---|---|---|---|---|
| 1 | "한의학에서 사상체질이란 " | 영어 brand 혼용 "wood, fire, earth, metal, water" | 《입문》·《강목》·《중경》 인용체, 내용은 환각 | 스타일 R1 / 내용 ≈ |
| 2 | "陰陽五行이란 " | 백과사전체 + markdown 헤더 | "陰은 水/陽은 火 ... ○五行의 본性" 고전 parallel 구조 | 스타일 R1 / 내용 △ |
| 3 | "補中益氣湯은 " | 영어 herb 이름(Ginseng, Atractylodes) + 약재 설명 | "**甲乙經에 나온다**" 출처 오답 + 순환 정의 | **R1 열세** |
| 4 | "傷寒論에서 말하기를, " | "4가지 병(열/냉/기/혈)" 생짜 오답 | "**仲景은 頭痛ㆍ身熱ㆍ脈浮한 경우는 發表해야 한다**" — 실제 傷寒論 태양병 치법 | **R1 명백히 우세** |
| 5 | "사상체질의학의 창시자 이제마는 " | "1960년대에 ... 체질을 중시하는 사상 확산" 연대 오답 | "1960년대에 '인간은 사회적 존재'" 동일 연대 오답 + 더 막연 | 개선 없음 |
| 6 | "`<KO>오장육부</KO>`는 " | "2019년 1억 2천만개 판매" 도메인 이탈 대참사 | "5가지 기혈... 오장은 각각의 기능" 고전체 한의학 | **R1 극명 우세** |
| 7 | "한의학의 기본 이론인 기혈(氣血)은 " | "5경과 6기" 오류 | "기는 양, 혈은 음" + 《내경》 인용체 | 스타일 R1 / 내용 △ |

### 4.3 Probe 원문 (7개 프롬프트, BASE + R1 전체)

```
--- [1/7] prompt: '한의학에서 사상체질이란 '
[BASE]
<|begin_of_text|>한의학에서 사상체질이란 5경(wood, fire, earth, metal, water)과 2기(양기와 음기)의 조화로운 상태를 의미하며, 이는 개인의 체질과 환경과의 조화를 통해 건강을 유지하는 데 중요한 요소로 여겨진다. 사상체질은 한의학적 진단과 치료에 있어서 매우 중요한 역할을 하며, 이를 바탕으로 개인 맞춤형 치료법을 개발할 수 있다. 예를 들어, 목(木)체질인 사람에게는 따뜻하고 건조한 기
[CPT]
<|begin_of_text|>한의학에서 사상체질이란 5가지 기가 모두 허하여 몸에 열이 나고, 머리가 아프며, 눈이 어두워지고, 목소리가 나오지 않으며, 입과 코가 마르고, 혀가 말라서 말을 하지 못하는 증상을 말한다. 《입문》<|eot_id|>사상은 오장육부가 모두 허하기 때문에 생긴다. 《강목》<|eot_id|>《내경》에서는 "오장육부가 모두 허하면 사상증이 된다"라고 하였다. 《중경》에서는 "오장육부가 모두 허하면

--- [2/7] prompt: '陰陽五行이란 '
[BASE]
<|begin_of_text|>陰陽五行이란 5가지 요소(목, 화, 토, 금, 수)와 그 상호작용을 통해 자연과 인간의 생명체를 설명하는 이론이다. 이는 중국 고대 철학에서 중요한 위치를 차지하며, 의학, 농업, 건축 등 다양한 분야에 적용되었다. 각 요소는 특정한 성질과 특징을 가지며, 서로의 관계를 통해 균형과 조화를 이루어야 한다고 보았다.

## 5가지 요소

1. **목(木)**: 나무를 상징
[CPT]
<|begin_of_text|>陰陽五行이란 5가지 요소로, 陰은 水이고 陽은 火이며, 五行은 木ㆍ火ㆍ土ㆍ金ㆍ水이다. 이 5가지 요소는 모두 각각의 본성과 기능을 가지고 있다. ○五行의 본性은 다음과 같다. 木은 생장하고 火는 소모하며 土는 수렴하고 金은 굳고 水는 흩는다. ○五行의 기능은 다음과 같다. 木은 살찌우고 火는 말리고 土는 보양하고 金은 치료하고 水는 씻는다.

--- [3/7] prompt: '補中益氣湯은 '
[BASE]
<|begin_of_text|>補中益氣湯은 6가지의 약재로 구성되어 있으며, 이 중에서 Ginseng(인삼)과 Atractylodes macrocephala(백출)가 주요 약재입니다. 인삼은 기력을 강화하고, 백출은 체액을 정화하여 체내에 남아 있는 독소를 제거하는 역할을 합니다. 또한, Glycyrrhizae (당귀), Pinellia ternata(대추), Poria cocos(송화), 그리고 Zingiberis rhizoma(인동)도 포함되어 있어, 전신의
[CPT]
<|begin_of_text|>補中益氣湯은 甲乙經에 나온다.<|eot_id|>《內經》에서 말한 "五臟의 氣는 모두 肺에서 생긴다"는 뜻이다. 五臟이란 心肝脾腎을 말한다. 이들은 모두 肺에서 生하여 肺가 不足하면 臟이 不生하니, 肺가 不足하면 臟이 不生하기 때문이다.<|eot_id|>《內經》에서 말한 "五臟의 氣는 모두 肺에서 生한다"는 뜻이다. 五臟이

--- [4/7] prompt: '傷寒論에서 말하기를, '
[BASE]
<|begin_of_text|>傷寒論에서 말하기를, 4가지의 병이 있으니, 이들은 모두 피로병에 속한다. 즉, 열병(熱病), 냉병(冷病), 기병(氣病), 혈병(血病)이다. 여기서 열병은 고열을 일으키는 병으로, 냉병은 저체온증을 일으키는 병으로, 기병은 호흡기 질환과 같은 기관의 기능 장애를 일으키는 병으로, 혈병은 출혈이나 혈액 질환을 일으
[CPT]
<|begin_of_text|>傷寒論에서 말하기를, 仲景은 "頭痛ㆍ身熱ㆍ脈浮한 경우는 發表해야 한다"고 하였다. ○이것은 모두 表邪가 未解한 상태이다. 만약 表邪가 未解하여 脈이 邪에 屬하면 반드시 頭痛ㆍ身熱이 있으니, 이때는 반드시 發汗해야 한다. ○만약 表邪가 已解하고 脈이 正에 屬하더라도 頭痛ㆍ身熱이 있으면서 脈이 弦數하거나 緩大

--- [5/7] prompt: '사상체질의학의 창시자 이제마는 '
[BASE]
<|begin_of_text|>사상체질의학의 창시자 이제마는 1960년대에 이르러서야 한국에서 체질을 중시하는 사상을 확산시키기 시작했다. 그는 "인간은 태어날 때부터 특정한 체질로 태어나며, 이는 그의 건강과 생활 방식에 큰 영향을 미친다"고 주장하며, 체질에 따라 맞춤형 의료를 제안했다. 이후 사상체질의학은 한국 사회에서 인기를 끌게 되었으며, 많은 사람들이 자신의 체질에 맞춘 생활 방식을 찾으려는 움직임이 나타났다
[CPT]
<|begin_of_text|>사상체질의학의 창시자 이제마는 1960년대에 "인간은 생명체이지만, 인간은 사회적 존재이다"라는 이론을 제창하였다. 그는 "인간은 단지 생명체가 아니라 사회적 존재이기 때문에, 인간의 질병은 단지 생리학적인 질환만이 아니다"라고 주장하였다. 즉, 사상체질의학은 단순히 생리학적인 질환뿐만 아니라, 정신적, 정서적, 사회적, 문화적 요소까지도 고려하여 치료해야 한다는 것이다.

사

--- [6/7] prompt: '<KO>오장육부</KO>는 '
[BASE]
<|begin_of_text|><KO>오장육부</KO>는 2019년 12월 31일 기준으로, 오장육부의 모든 제품을 포함하여 총 1억 2천만여 개의 판매를 기록했다고 밝혔다. 이는 전 세계적으로도 매우 높은 판매량으로, 특히 중국 시장에서의 인기가 두드러졌다. 오장육부는 중국에서 '중국인들의 건강을 위한 필수품'으로 자리 잡았으며, 많은 사람들이 이를 통해 체중 감소와 건강 증진에 도움을 받았다고 한다. 또한, 오장육부는 다양한
[CPT]
<|begin_of_text|><KO>오장육부</KO>는 5가지의 기혈이 서로 이어져서 생기고, 그 근본은 오장에 있다. ○오장은 각각의 기능을 가지고 있는데, 심은 수(水)를 주관하고, 비는 화(火)를 주관하며, 폐는 기를 주관하고, 신은 금(金)을 주관한다. ○오장은 모두 혈맥으로 연결되어 있으니, 심은 혈맥을 주관하고, 간은 기를 주관하며, 비는 피를 주관하고, 폐는 숨을 주관하며, 신은 소변

--- [7/7] prompt: '한의학의 기본 이론인 기혈(氣血)은 '
[BASE]
<|begin_of_text|>한의학의 기본 이론인 기혈(氣血)은 5경(五經)과 6기(六氣)의 원리를 통해 설명된다. 이는 인간의 생명력과 건강을 유지하는 데 중요한 역할을 한다. 한의사는 이러한 기혈의 흐름과 균형을 조절하여 질병을 치료하고, 건강을 증진시키는 방법을 연구한다.

한의학은 또한 중의학(中醫學), 동의학(東醫學) 등으로도 불리며, 중국 전통 의학의 일종이다. 한의학은 고대 중국의
[CPT]
<|begin_of_text|>한의학의 기본 이론인 기혈(氣血)은 2가지로 나뉘는데, 기는 양이고 혈은 음이다. 기가 허하면 열이 생기고, 혈이 허하면 한이 생긴다. 기와 혈이 모두 허하면 병이 생기지 않는다. 《내경》에 "기와 혈이 모두 허하면 병이 생기지 않으니, 이것을 '양명'이라고 한다"라고 하였다.<|eot_id|>《내경》에 "사람의 기는 위에서 시작하여 아래로 내려오고, 혈은 아래에서 시작하여 위
```

### 4.4 실패 모드별 판정

| ID | 증상 (R0) | R1 상태 | 판정 |
|----|----------|---------|------|
| F1 | style-over-fact 환각 | #3/#5/#7 여전히 facts 오답, 스타일은 획득 | **미해결** (예상대로) |
| F2 | 언어 붕괴 KO→ZH | 7/7 한국어+한자 병기 일관, `</KO>` 누출 없음 | **해결** |
| F3 | rare-token 심볼 덤프 | 7/7 모두 없음 | **해결** |

부가 관찰:
- Extended token test (`<KO>오장육부</KO>`) 가 **가장 극명한 단일 개선점**. BASE는 상품 판매 텍스트로 완전 이탈, R1 은 도메인 내 고전체 — tokenizer 확장 설계 의도대로 작동함이 확인.
- 훈련 데이터 packer 의 **`<|eot_id|>` 토큰이 중간에 박혀 있던 artifact** 가 R1 출력에 그대로 누출. 추론 시 stop_token으로 가려지지만 packer §B.2 재검토 필요 (sep_token 배치 로직).
- 프롬프트 #5 는 base model prior 한계. corpus 에 이제마 전기 정보가 없어 CPT 로 교정 불가 → R2 에서 해결하려면 **fact sheet (docs/ver4/02_plan §2.1 P-A+ 경로) 또는 data unification (QA 재포장)** 도입 필요.

---

## 5. 미완 + 다음 스텝

### 5.1 즉시 가능 (R1 내부 완결)

- [ ] **R0 vs R1 probe 비교** — `scripts/probe_adapter.py --adapter outputs/cpt_bllossom/adapter` 한 번 더 실행 후 본 문서 §4 와 3-way 비교. "R1이 R0 대비 개선인가" 에 대한 유일한 정량적 답.
- [ ] **R1 merged 빌드** — `scripts/build_merged_model.py --adapter outputs/cpt_bllossom_R1/adapter --output outputs/hanmed_merged_v0.2` → `docker/docker-compose.yml` 의 `HANMED_MODEL_DIR` 을 v0.2 로 스위칭 후 `hanmed` CLI 체감 테스트.
- [ ] **Hanmed-only eval subset loss 측정** — R1 val 에서 wiki_ko 95개 제외한 hanmed-only 374개에 대해 eval_loss 재계산. R0 와 apples-to-apples.

### 5.2 R2 계획 (research §0 ②③ 구현)

- [ ] **Data unification (HuatuoGPT-II 방식)** — raw 조문을 LLM 으로 QA pair 로 재포장, raw 와 QA 를 병렬 학습 → F1 팩트 오귀속 직접 처방
- [ ] **On-policy distillation** — Bllossom base 를 teacher 로 삼아 Tulu3 prompts 로 instruction-following 복구. LoRA r=32 에서 full FT gap 13% → 6% 축소 재현 (ThinkingMachines lab 수치)
- [ ] **Fact-sheet 주입 (P-A+ 경로 실현)** — ver4 §2.1 기획대로 `data/facts/core_factsheet.yaml` 기반 synth_facts 재구성, 인물·연대·방제 출처 triple 강제 주입

### 5.3 Infra 정비

- [ ] Packer 의 `<|eot_id|>` 중간 삽입 artifact 재검토 (§4.4 관찰점)
- [ ] NCCL P2P 이슈 → `_launcher.sh` 패턴을 `scripts/train_launcher.sh` 로 표준화, 다른 훈련 진입점에서도 재사용 가능하게 승격

---

## 6. 산출물 경로

### R1 학습
- adapter: `outputs/cpt_bllossom_R1/adapter/` (2026-04-22 03:42)
- checkpoints: `outputs/cpt_bllossom_R1/checkpoint-150/`, `checkpoint-156/`
- train log: `outputs/cpt_bllossom_R1/run.log` (57KB, 전체)
- crash logs: `run.log.crash_20260421` (첫 NCCL hang), `run.log.nccl_p2p_crash_20260422_0025` (두 번째)
- launcher: `outputs/cpt_bllossom_R1/_launcher.sh`
- manifest: `outputs/cpt_bllossom_R1/train_manifest.json`

### R0 백업
- adapter: `outputs/_backups/cpt_bllossom_adapter_R0_20260421/` (2.4GB)
- merged: `outputs/_backups/hanmed_merged_v0.1_R0_20260421/` (15GB)

### Probe
- raw log: `/tmp/probe_R1.log` (본 문서 §4.3 에 영구 보존)
- 실행 스크립트: `scripts/probe_adapter.py` (R1 사이클에서 `--adapter` CLI 인자 지원으로 확장)

### 데이터
- wiki_ko raw: `data/cpt/wiki_ko.jsonl` (10,793 docs, 13.6MB)
- wiki_ko packed: `data/cpt_processed/wiki_ko_packed_2048.jsonl` (4,703 seqs)
- 빌더: `src/data/builder/build_wiki_ko.py`

### 코드 변경
- `src/training/cpt_trainer.py` — `DEFAULT_MIX`, `CORPUS_PATHS['wiki_ko']`, `--mix` help
- `src/data/builder/preprocess.py` — `CORPORA_KINDS` 에 `wiki_ko`
- `src/hanmed_cli/config.py` — `footer_enabled: bool = False`
- `src/hanmed_cli/safety.py` — footer append conditional gating
- `src/hanmed_cli/inference/remote_openai.py` — `min_tokens` / freq+presence penalty 제거

---

## 7. 한 줄 요약

> **R1 은 F2/F3 를 완전히 해결했고 F1 은 예상대로 미해결이다. Loss 숫자는 동등 또는 소폭 열세로 보이지만 eval set 구성 변경 탓이며, behavioral probe 기준 R1 은 BASE 대비 확실한 개선이다. R0 직접 비교는 아직 미실행 — 이것이 완료돼야 "mix 재설계의 순수 효과" 가 정량화된다.**
