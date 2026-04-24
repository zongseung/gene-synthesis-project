# Research · 한의학 고전 CPT 실패 모드 해결 방법론 (bilingual/CJK 도메인 유사 사례 조사)

- **작성일**: 2026-04-21
- **대상 프로젝트**: HanMed CPT (Llama-3-Korean-Bllossom-8B + LoRA, 20M tokens, KIOM mediclassics 26권)
- **전제**: 이 문서는 research report이며 구현/변경은 포함하지 않는다. 각 주장에는 출처를 명시했고, 우리가 검증해야 할 가정은 "(검증 필요)"로 표시했다.

---

## 0. Executive Summary — 다음 CPT 사이클 우선 적용 3가지

증거가 가장 강하고 구현 비용 대비 기대 효과가 큰 순서:

### ① Corpus mix 재설계: general replay ≥ 15%, KO 앵커 ≥ 50%, ZH-only ≤ 10%

- **근거**: BianCang(460M tokens) 코퍼스의 약 **16%(72M)** 가 general domain(COIG, math, code) replay였고, 이 구성에서 TCM 벤치마크 +60pp(TCMSD 21→82)를 유지하면서도 원능력 붕괴 없이 학습을 마쳤다 [BianCang §Table 1]. 우리 현재 mix(bilingual 0.45 / zh_only 0.20 / ko_only 0.35)는 general replay가 **0%** 이고 실질 중문 편중 65% — language collapse의 직접 원인이다.
- **참고 하한**: GeRe 실험은 "1K 고정 general 샘플"만으로 LoRA 조건에서 +6.5pp F1 회복(63.37→69.86)을 보고하므로 [GeRe §Table III], 20M 토큰 기준 **200K~1M 토큰의 일반 코퍼스 replay** 만으로도 "학습은 되면서 기본 언어 앵커는 유지"가 가능할 것으로 추정한다.

### ② Data unification: 원문을 그대로 쓰지 말고 "QA 형식으로 재포장" 후 CPT

- **근거**: HuatuoGPT-II의 핵심 기여 — 사전훈련 corpus를 그대로 쓰지 않고 **LLM으로 질문·답변 쌍으로 재생성해 SFT와 동일한 포맷으로 통일**. 이 unification만으로도 2단계 분리 학습 대비 CMExam +14%, 약사시험 +10% [HuatuoGPT-II §4]. 우리 증상(style-over-fact)은 raw continuation LM의 고질적 실패 모드이고, format unification이 직접적 처방이다.
- **추가 근거**: REALM의 **salient span masking**은 uniform 마스킹 대비 factual task에서 큰 폭 개선을 보고했으며, 이는 "팩트에 더 많은 학습 신호를 주는" 방향성과 같다 [REALM §3.2, Guu et al. 2020].

### ③ Base checkpoint 재검토 + instruction-following 복구 distillation 경로 마련

- **근거**: "Examining Forgetting" 연구는 Llama-2-7B-chat에 전통중국어 CPT를 돌린 결과 **reliability(사실성·지시 따름)가 체계적으로 저하**됨을 보고했다 [Kuan-Hao Chao et al., 2401.03129]. Thinking Machines Lab은 내부 문서로 mid-train 후 **on-policy distillation 150 steps + Tulu3 prompts**로 IF-eval 거의 전량 회복을 시연했으며, **LoRA rank=32 에서도 full FT 대비 gap이 SFT 후 13%에서 distillation 후 6%로 축소**됐다 [ThinkingMachines "On-Policy Distillation"]. 우리가 SFT로 바로 가기 전에 이 루프를 준비하면 SFT 자체의 "환각에 권위 붙이기" 리스크를 낮출 수 있다.

비용 추정: ①은 corpus 재구성 작업(1~2일). ②는 LLM 호출 비용 + 재학습(어댑터 재훈련 + API 비용). ③는 SFT 파이프라인에 teacher 호출 루프 추가(수일). 세 가지 모두 **LoRA 경로를 유지한 채 적용 가능**하다 (근거: LoRA Learns Less and Forgets Less §Tables).

---

## 1. 현상 재정의와 조사 범위

### 1.1 관측된 3가지 실패 모드 (인벤토리)

| ID | 증상 | 재현 증거 (본 레포 커밋 이전 단계 probe) |
|----|------|----------------------------------------|
| F1 | **Style-over-fact**: 《傷寒論》제1조를 틀린 내용으로 꾸며내고, 존재하지 않는 서명(《삼인론》·《直指金》)을 자신있게 인용 | 5/5 시드 전부 환각, 모두 다른 가짜 출처 |
| F2 | **Language collapse (KO→ZH)**: 도메인 트리거 시 응답이 한문 편중 | 단일 턴 한국어 질문에 `</KO>` 토큰 누출 + 한문 인용체 지배 |
| F3 | **Rare-token degradation**: multi-turn 누적 시 심볼 덤프 | `◯ ◤ ◥ 俰` 등 degeneracy spiral (엣지 케이스, 단일턴에선 미재현) |

### 1.2 조사 대상 메서드 카테고리

A. **Corpus mix / ratio tuning** — 도메인:일반 비율, 언어 비율, replay.
B. **Fact grounding during CPT** — entity/span masking, citation-aware, data unification.
C. **Language anchoring** — 2단계 학습, 언어별 loss weighting, instruction pre-injection.
D. **Tokenizer extension stability** — embedding 초기화, rare-token 학습.

---

## 2. 메서드별 조사 결과

### 2.1 Corpus Mix / Ratio Tuning

#### 2.1.1 HuatuoGPT-II (Baichuan2-7B, 의학, 1.1TB)
- **구성**: 중국어 58% / 영어 42%, 4 tier priority sampling (웹 β⁵ > 문헌 β⁴ > 백과 β³ > 교과서 β² > SFT β⁰). 도메인·언어·품질의 3축을 **단일 확률 텐서**로 통합.
- **단계**: 1단계 unified (CPT + SFT 동시). 2단계 분리 대비 **5.3~23% 개선**.
- **기여**: raw corpus → LLM으로 QA 형식 재생성 후 학습. 이게 본 연구에서 가장 이식 가치 큰 기법.
- **이식 비용(우리 조건)**: 중간. 26권 × 평균 수십만 토큰을 LLM으로 QA 재생성 시 수백만 건 API 호출 필요. raw + QA 2종을 병렬 학습하면 "원문 암기 + 사용 가능 형식" 둘 다 얻을 수 있음 (검증 필요).

#### 2.1.2 BianCang (Qwen2/2.5-7B/14B, TCM, 460M tokens)
- **구성**: 의료 교과서 76M + 백과/문헌 161M + 약전 3M + 임상/증후 47M + CMB 시험 22M + **일반 도메인 72M (COIG/math/code, ~16%)** [BianCang Table 1].
- **학습**: 2 epoch, full parameter, weight decay 0.01(7B)/0.1(14B), gradient norm 0.5, warmup 0.03/0.05.
- **결과**: TCMSD 21.29 → 82.10 (+60.8pp), TCMDD 43.88 → 82.65 (+38.8pp), MLEC-TCM 83.32 → 90.06 (+6.7pp).
- **관찰**: **약 16%를 일반 도메인 replay**로 쓰면서도 도메인 성능을 크게 끌어올림. 우리의 0% replay는 이 기준 대비 명백한 outlier.
- **한계**: 고전 원문 전처리(조문 단위·주석 분리)는 명시하지 않음. 우리 문제와 직접 일대일 대응은 아님.

#### 2.1.3 Me-LLaMA (Llama2-13B/70B, 129B tokens)
- **설계 원칙**: "biomedical literature + clinical notes + **general domain data** to mitigate catastrophic forgetting" [Me-LLaMA abstract].
- **시사점**: 규모는 다르지만 **general 혼합을 forgetting 방지 수단으로 명시**. 20M 토큰 규모에서도 같은 원리가 더 강하게 적용돼야 함 (토큰이 적을수록 한 번 잘못 학습하면 복구 어려움).

#### 2.1.4 GeRe (general sample replay)
- **핵심 발견**: LoRA + **1K 고정 general replay 샘플**로 MMLU +12%, 평균 F1 63.37 → 69.86 (+6.5pp) [GeRe §Table III].
- **시사점**: 우리 조건(LoRA, 20M 토큰)에 **가장 직접 이식 가능한 증거**. "일반 코퍼스 1K~10K 샘플을 모든 배치에 섞어라"가 실용 레시피.

#### 2.1.5 On-Policy Distillation (Qwen3-8B, ThinkingMachines)
- **관찰**: 내부 문서 70% + 채팅 30% mid-train에서도 IF-eval 완전 보존 불가. **distillation 복구가 필수**.
- **수치**: LoRA r=32 조건에서 full FT 대비 SFT gap 13% → distillation 후 6%.
- **시사점**: 우리가 SFT로 가기 전, CPT 단계에서 이미 chat 템플릿 데이터를 일정 비율 섞는 설계가 권장됨 (검증 필요: 30%는 Qwen3-8B 실측치, Llama-3-Bllossom 기반은 별도 실측 필요).

### 2.2 Fact Grounding in CPT

#### 2.2.1 REALM / Salient Span Masking (SSM) — Guu et al. 2020
- **메서드**: 랜덤 마스킹 대신 **named entity + date** 를 선별적으로 마스킹. 이 방식이 uniform mask 대비 factual task에서 "significantly outperforms" [REALM §3.2].
- **한의학 적용 제안(가설)**: 조문 원문에서 **방제명·인물·조문번호·본초명·병명**을 선별 마스킹하는 span 기반 objective를 LoRA CPT 중에 추가. autoregressive LM 목적과 병립 가능.
- **이식 비용**: 중간-높음. NER 태깅 필요(한의학 entity dict 자체 구축). 구현은 기존 loss에 term 추가.

#### 2.2.2 HuatuoGPT-II data unification (재서술)
- corpus → QA 형식 rewrite. 이 변환만으로 팩트 학습 신호가 강화된다는 간접 증거(2단계 분리 대비 전 벤치 +5~23%).
- **우리 적용 시나리오**: 《傷寒論》 조문마다 "Q: 太陽病의 정의는? / A: 太陽之為病, 脈浮, 頭項強痛而惡寒 (第1條)" 형식 pair 생성 → mix에 포함.

#### 2.2.3 Xunzi-Baichuan (고전 한문, Baichuan2-7B, 2B chars)
- 중국 고전 corpus CPT + SFT. **BLEU 38.56 (+1.16 vs base)** on 고서 번역.
- **시사점**: CJK 고전 도메인 CPT가 원리적으로 작동함을 확인. 다만 평가가 번역 BLEU 중심이라 우리의 "조문 팩트 recall" 과제와는 다름.

#### 2.2.4 Zhongjing (TCM, Llama-based, CPT→SFT→RLHF)
- 3단계 파이프라인 자체의 이득 강조. **RLHF가 "능동적 질문·다턴 이해"를 강화**한다고 보고 — 우리가 환각한 팩트를 억제하는 데는 간접적. SFT 이후 단계 계획 시 참고.

### 2.3 Bilingual / Language Anchoring

#### 2.3.1 Chinese-LLaMA-Alpaca (Cui et al. 2023)
- **2단계 학습**: Stage 1 = 임베딩만 학습(다른 파라미터 동결) → Stage 2 = LoRA로 전체 adapt. 새 토큰이 먼저 "원 임베딩 공간에 정착"한 다음에야 모델 파라미터가 움직이기 시작.
- **시사점**: 우리는 현재 tokenizer 확장 후 첫 epoch부터 LoRA가 전체 파라미터에 gradient를 흘리고 있음. Stage 1 = **embedding-only warmup** 을 앞에 넣으면 rare-token degradation(F3) 완화 여지.

#### 2.3.2 Examining Forgetting (Kuan-Hao Chao et al. 2401.03129)
- Llama-2-7B-chat + Traditional Chinese CPT → **반복 생성 증가, reliability 저하**.
- **결론 인용**: "more than straightforward methods are required". 즉 LoRA·freezing 단독으로는 해결 불충분 — replay + distillation 같은 2차 개입 필요.

#### 2.3.3 Taiyi (DUTIR BioNLP, Qwen-7B, SFT only)
- 영어 73% / 중국어 27% bilingual 비율로 SFT. CPT는 사용 안 함.
- **반례 시사**: 도메인 지식 주입에 반드시 CPT가 필요한 건 아님 — SFT만으로 bilingual biomedical 성능 달성. 우리가 **CPT 단계를 건너뛰고 고품질 SFT로만 갈 수 있는지** 를 separate question으로 제기할 수 있음 (단 고전 원문 스타일은 잃을 수 있음).

#### 2.3.4 언어 붕괴 완화 일반 기법
- **Curriculum learning for code-switching**: 난이도 3단계(least→most CS) 순서 학습이 bilingual model 성능 향상 [ACL Findings 2025]. 우리에 직접 이식 시: ko_only → bilingual → zh_only 순서 epoch 배치.
- **Language-ID 프리픽스**: 입력 앞에 `[KO]`/`[ZH]` 토큰을 명시. 일부 다국어 모델에서 code-switch 억제 효과 보고. 저비용 baseline.

### 2.4 Tokenizer Extension Stability

#### 2.4.1 Chinese-LLaMA 초기화 전략
- 새 토큰 임베딩을 원 매트릭스 끝에 **append** (기존 임베딩 불변). 초기값은 random.
- 연구 community 기본 권장: **mean init** (새 토큰 = 기존 임베딩 평균) 또는 **kNN/FVT** (서브토큰 평균).

#### 2.4.2 Empirical Comparison (Yamaguchi et al. 2024, arxiv 2407.05841)
- 이론: 새 임베딩이 기존 임베딩의 **convex hull** 안에 있으면 원 언어 생성이 손상되지 않음을 증명.
- 실험: **Mean / CW2V (convex init) ≈ OFA** (정교한 방법). "simpler methods within convex hull suffice".
- **시사점**: 복잡한 FOCUS/OFA 쓰지 않아도 **mean init 만으로 충분**. 현재 우리 tokenizer 확장이 어떤 초기화를 썼는지 확인 필요 (검증 필요: `data/tokenizer/hanmed_bllossom_ext` 생성 스크립트 추적).

#### 2.4.3 KM-BERT (한국어 의료 BERT)
- **교훈**: 한국어 의학 morphology가 복잡해 **형태소 단위 subword 전략이 성능을 좌우**. MLM+NSP 정확도 +14.7%, +14.8%.
- 우리 한정 적용: Bllossom extended tokenizer가 한자 토큰 추가 중심이라면, 한국어 한의학 용어의 morphological coverage도 같이 점검해야 함.

#### 2.4.4 rare-token degeneracy (F3 원인 가설)
- vLLM 문서: `min_tokens`는 **greedy 결과까지 변형하는 non-argmax-invariant processor**. EOS 억제로 모델이 자연 종료 지점을 지나 저확률 공간으로 밀림.
- LZ Penalty 연구: **낮은 temperature + frequency/presence penalty 조합이 degeneracy 유발** — "distribution mode collapses into degenerate repetitions". 우리가 관측한 심볼 덤프의 표면적 원인 가설로 유력.
- **처방**: `min_tokens` 대폭 완화(또는 제거), `frequency_penalty`/`presence_penalty` 0으로 낮추고, 대신 SFT 단계에서 "간결 답변"을 학습시키는 게 정도.

---

## 3. 우리 상황 맵핑 — 20M tokens · 8B-LoRA · KR/ZH 고전의서

### 3.1 corpus 규모 관점

| 참고 | 토큰 | 유형 | 비고 |
|------|------|-----|-----|
| HuatuoGPT-II | 1.1TB (≈수천억) | full FT | 참고만, 규모 다름 |
| BianCang | 460M | full FT, 2ep, 16% general replay | **가장 근접**한 규모·철학 |
| Me-LLaMA | 129B | full FT + LoRA IFT | |
| Xunzi-Baichuan | 2B chars | full FT 추정 | 고전 도메인 |
| **HanMed (우리)** | **20M** | LoRA r=?, 3ep, 0% general | **replay 0%가 outlier** |

20M은 BianCang의 ~4%, Xunzi의 ~1% 규모. 이 체급에선:
- full FT는 forgetting 위험과 compute 비용 모두 큼 → LoRA 유지가 타당.
- 다만 LoRA Learns Less 논문 관점에서 **rank가 낮으면 domain 학습 자체가 약해짐** (r=16 → 열세, r=256 → full FT 근접). 우리 현재 rank 확인 필요 (검증 필요).
- **"적은 토큰 × LoRA × 0% replay × style-heavy synth mix"** 는 모든 축에서 구조적으로 F1/F2/F3에 취약한 조합.

### 3.2 예상 ablation 매트릭스 (다음 사이클)

| 실험 | 예상 소요 | F1 영향 | F2 영향 | F3 영향 |
|------|----------|---------|---------|---------|
| E0: baseline 재학습 (현 설정) | 16h | — | — | — |
| E1: general replay 15% 추가 | +16h | 약 | **강** | 약 |
| E2: QA unification 적용 | LLM호출 + 16h | **강** | 약 | 중 |
| E3: corpus mix ko≥0.5, zh≤0.1 | +16h | 중 | **강** | 약 |
| E4: embedding-only warmup 1ep | +4h | 약 | 약 | **중** |
| E5: min_tokens=0, penalty=0 (추론만) | 0h | — | 약 | **중** |

E5는 학습 없이 즉시 가능 — 최우선으로 먼저 끊고 F3 재관측이 필요 (본 조사와 별개의 운영 조치).

### 3.3 이식 시 리스크

- **HuatuoGPT-II data unification**: raw 조문의 원문 표현이 QA 재생성 과정에서 약화될 수 있음. → raw + QA **병렬 학습**으로 완화, 원문 암기 유지.
- **general replay**: 의료 도메인 drift 가능. → replay 소스는 "한국어 일반 위키 + 뉴스"처럼 중립 도메인 선호.
- **embedding-only warmup**: CPT 총 compute 증가(~10~20%). 단 rare-token 안정성 개선 가치로 충분히 상쇄.
- **langauge anchor 변경(ko 0.5↑)**: zh_only 감소 → 한문 원문 암기 약화 우려. → 원문은 **bilingual pair 형태**로 유지(한문 + 한국어 직역 병기)해 양쪽에 카운트.

---

## 4. 출처 (References)

### 기본 논문
- [HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs (arXiv 2311.09774)](https://arxiv.org/abs/2311.09774) — Baichuan2 기반, priority sampling, data unification
- [Zhongjing: Enhancing Chinese Medical Capabilities (arXiv 2308.03549)](https://arxiv.org/abs/2308.03549) — CPT→SFT→RLHF 3단계
- [BianCang: A Traditional Chinese Medicine LLM (arXiv 2411.11027)](https://arxiv.org/abs/2411.11027) — Qwen2.5, 460M corpus, 16% general replay
- [Me-LLaMA: Medical Foundation LLM (arXiv 2402.12749)](https://arxiv.org/abs/2402.12749) — 129B tokens, general mix 명시
- [Taiyi: Bilingual Fine-Tuned LLM for Biomedical (arXiv 2311.11608)](https://arxiv.org/abs/2311.11608) — SFT only, 73:27 EN:ZH
- [Xunzi LLM for Chinese Classics (Nanjing Agricultural University)](https://github.com/Xunzi-LLM-of-Chinese-classics/XunziALLM) — Baichuan2-7B CPT on 2B chars
- [KM-BERT: Korean Medical BERT (Nature Sci. Reports)](https://www.nature.com/articles/s41598-022-17806-8) — Korean medical morphology

### 방법론 논문
- [REALM: Retrieval-Augmented Language Model Pre-Training (Guu et al. 2020)](https://arxiv.org/pdf/2002.08909) — salient span masking
- [Knowledgeable Salient Span Mask (arXiv 2204.07994)](https://arxiv.org/html/2204.07994) — SSM 확장
- [LoRA Learns Less and Forgets Less (arXiv 2405.09673)](https://arxiv.org/html/2405.09673v2) — LoRA vs full FT, rank 영향
- [GeRe: General Samples Replay for Anti-Forgetting (arXiv 2508.04676)](https://arxiv.org/html/2508.04676v1) — 1K 샘플 replay, LoRA 효과
- [Examining Forgetting in Continual Pre-training (arXiv 2401.03129)](https://arxiv.org/html/2401.03129v1) — aligned LLM CPT의 reliability decline
- [On-Policy Distillation (Thinking Machines Lab blog)](https://thinkingmachines.ai/blog/on-policy-distillation/) — 30% chat mix + Tulu3 distillation, LoRA r=32 결과
- [Chinese LLaMA and Alpaca (arXiv 2304.08177)](https://arxiv.org/pdf/2304.08177) — tokenizer extension + 2단계 학습
- [Empirical Comparison of Vocabulary Expansion and Initialization (arXiv 2407.05841)](https://arxiv.org/html/2407.05841v1) — convex hull, mean init 충분
- [FOCUS: Effective Embedding Init (EMNLP 2023, arXiv 2305.14481)](https://arxiv.org/abs/2305.14481)
- [LZ Penalty: information-theoretic repetition penalty (arXiv 2504.20131)](https://arxiv.org/html/2504.20131v2) — frequency/repetition penalty degeneracy
- [Replay to Remember (arXiv 2504.17780)](https://arxiv.org/html/2504.17780v1) — streaming LoRA + replay, medical domain
- [Code-Switching Curriculum Learning (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.407.pdf) — 난이도별 CS 학습 순서

### 추가 자료
- [Continual Learning of LLMs: A Comprehensive Survey (CSUR 2025)](https://github.com/Wang-ML-Lab/llm-continual-learning-survey) — continual learning 총설
- [vLLM Inference Parameters docs](https://docs.vllm.ai/en/v0.8.4/api/inference_params.html) — min_tokens non-argmax-invariant behavior
- [TongGu: Mastering Classical Chinese Understanding (arXiv 2407.03937)](https://arxiv.org/html/2407.03937v1) — 고전중국어 knowledge-grounded LLM

---

## 5. 결론 — 판정

현 어댑터가 보이는 F1/F2/F3는 **"CPT 레시피 구성의 복합 결함"** 으로 해석되며, 학술적으로 이미 풀린 문제의 조합이다. 구체적으로:

- F1(style-over-fact) ← **raw continuation LM objective + style-heavy synth 비중**. HuatuoGPT-II가 보인 QA-unification이 직접 처방.
- F2(language collapse) ← **general/ko 앵커 0% + zh 편중 65%**. BianCang/Me-LLaMA식 ≥15% general replay + ko 중심 재편이 정답.
- F3(rare-token degeneracy) ← **inference-time forced length + frequency penalty**의 서빙 설정 문제가 1차, embedding 초기화 품질이 2차. Chinese-LLaMA 2단계(embedding warmup) + min_tokens/penalty 재조정 조합.

**다음 CPT 사이클 결정**: 현 어댑터는 폐기하고 §0 ①②③을 동시 적용한 재학습을 권장한다. 비용은 현재 학습의 ~2배(16h → 32~40h) 수준으로 추정되며, 이는 SFT 단계에서 잘못 학습된 팩트를 교정하는 비용 대비 명백히 싸다.

검증이 필요한 가정들(문서 중 "(검증 필요)"로 표기):
- 현재 LoRA rank 값과 embedding 초기화 방식 (`scripts/build_merged_model.py` · tokenizer 확장 스크립트 확인)
- Bllossom-3-Korean 기반에서 "30% chat mix" 수치의 이식성
- 20M 토큰 조건에서 general replay 최소 유효 하한
