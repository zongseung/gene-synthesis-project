# 10.7 Deployment — 로컬 / 기관 서버 / 클라우드

> **사용자 요청**: 서버 + 클라우드 배포 검토. 본 섹션은 3-way 트레이드오프 를 명시하고 v0/v1/v2 단계별 진입 조건을 정의.

## 배포 매트릭스

| 경로 | 환경 | 사용자 | 접근 방식 | KIOM 라이선스 | 비용 | 레이턴시 | v0 포함 |
|---|---|---|---|---|---|---|---|
| **L — 로컬** | 개발자 A6000 | 1인 | `hanmed chat` 직접 | ✅ §07 범위 | 0 | warm FT < 1s | ✅ |
| **S1 — 기관 서버 SSH** | 가천대 GPU 서버 | 연구실 내부 | `ssh` 후 `hanmed chat` | ✅ §07 비상업 | 0 (기관 자원) | warm FT 1~2s | ✅ |
| **S2 — 기관 서버 HTTP** | 가천대 GPU 서버 | 연구실 내부 다수 | `hanmed serve` + `hanmed chat --remote` | ✅ §07 비상업 | 0 (기관 자원) | 1~3s FT | ❌ (v1) |
| **C — 클라우드** | AWS g5 / RunPod | 외부 사용자 | 웹 or CLI 클라이언트 | **❌ KIOM 승인 필수** | $1~2/h | 1~3s FT | ❌ (v1 조건부) |

FT = first-token.

## L — 로컬 (v0 primary)

### 환경
- RTX A6000 48GB × 1 (DDP 불필요)
- Ubuntu 22.04 / Python 3.10+
- CUDA 12.x

### 설치
```bash
uv sync
uv run hanmed chat --adapter outputs/cpt_bllossom/adapter
```

### 적합한 사용 상황
- ver2 논문 데모 녹화
- 개발자 자신의 실험 반복
- 단일 사용자 데모 (학회 발표 laptop 연결)

### 제약
- 1 사용자 동시. 여러 명이 쓰려면 S 경로.
- GPU 없는 laptop 데모 → 10.4 `llama.cpp` GGUF 옵션 (v1) 또는 원격 S 경로 접속

## S — 기관 서버 (v0 secondary)

### 환경
- 가천대 GPU 서버 (A6000 1~2대 가정)
- 로컬 네트워크 또는 학교 VPN
- 기관 내부 사용자만 접속

### 2가지 모드

#### S1. SSH + REPL
```bash
# 연구실 멤버가 직접 SSH
ssh user@gachon-gpu.lab
hanmed chat --adapter /srv/hanmed/outputs/cpt_bllossom/adapter
```

장점: 추가 서버 코드 불필요. v0 CLI 를 그대로 사용.
단점: 동시 사용자 수 = GPU 수. 여러 명 동시 쓰면 queue.

운영 원칙:
- **공용 Unix 계정 공유 금지**. 사용자별 계정/홈 디렉터리를 분리해야 세션 파일 충돌과 접근권한 문제가 없다.
- adapter 경로는 사용자 홈 상대경로보다 공용 read-only 절대경로가 낫다.

#### S2. HTTP 서버 모드 (`hanmed serve`)

**R3.3/R3.5 결정**: v0 에는 `hanmed serve` 포함하지 않음. 기관 서버의 v0 범위는 **S1 (SSH + REPL)** 까지다. S2 는 **v1 이후**.

v1 구조 (계획):
```bash
# 서버 측
hanmed serve --host 0.0.0.0 --port 8000 --adapter outputs/cpt_bllossom/adapter

# 클라이언트 측
hanmed chat --remote http://gachon-gpu.lab:8000
```

vLLM OpenAI-compatible API 를 wrap (vLLM 가 이미 지원). continuous batching 으로 2~4 동시 사용자 처리.

### 적합한 사용 상황
- 연구실 내부 평가 세션 (3~5명이 동일 시스템에 prompt 던져 비교)
- 논문 실험용 batch inference
- KIOM 와의 협업 데모

### 라이선스
KIOM 비상업 이용 범위 내 (§07). 기관 외부 사용자가 접속하려면 C 경로.

## C — 클라우드 (v1 이후, KIOM 승인 조건부)

### 시나리오

| 사용자 유형 | 동기 | 요구 라이선스 |
|---|---|---|
| ver2 논문 독자 | 재현 / 테스트 | KIOM 서면 승인 + adapter 공개 |
| 외부 연구자 | fine-tune 경험 공유 | 동일 |
| 일반 사용자 | 한의학 질의 | 상업 범위 가능성 → KIOM 정식 상업 계약 |

### 후보 제공자

| 제공자 | GPU | 시간당 비용 | 장점 | 단점 |
|---|---|---|---|---|
| **AWS g5.2xlarge** | A10G 24GB | ~$1.00 | 안정적, Terraform | 한국 리전 latency |
| **RunPod** | A6000 48GB / A100 | ~$0.79 / ~$1.89 | GPU 유연, 저가 | 가용성 변동 |
| **Lambda Labs** | A6000 48GB | ~$0.80 | GPU 전용 | 재고 부족 |
| **Paperspace** | A6000 | ~$1.10 | 노트북 IDE 통합 | — |
| **Hugging Face Inference Endpoints** | L4 / A10 | ~$0.60~ | HF 생태계 연동 | LoRA serving 제한 |
| **자체 on-prem** (가천대) | A6000 | 0 | 라이선스 안전 | scale 제한 |

**v1 권고**: RunPod A6000 (비용·사양 우수) 또는 AWS g5.2xlarge (기업 사용자 신뢰). HF Inference Endpoints 는 adapter 재업로드 필요 (§07 승인 조건부).

### 클라우드 배포 사전 조건

1. **KIOM 서면 승인** — adapter 외부 배포 허가 (`kiombook@kiom.re.kr`)
2. **출처 표기 의무** — API 응답 footer 에 "KIOM mediclassics.kr 기반 학습" 고정
3. **안전성 게이트 통과** — §05 T4 redteam ≥ 99% refusal 실측
4. **사용량 로깅** — 프롬프트 저장 금지 (privacy), 집계 통계만 KIOM 에 정기 보고
5. **모델 카드 공개** — HuggingFace model card + 학습 데이터 출처 + 한계 명시

### 클라우드 아키텍처 초안 (v1)

```
┌──────────────┐           ┌──────────────────────────────┐
│   User CLI   │  HTTPS    │  AWS g5 / RunPod             │
│ hanmed chat  │◄────────►│  ┌────────────────────────┐  │
│ --remote ... │           │  │ hanmed-cli 'serve'     │  │
└──────────────┘           │  │  ├─ auth (API key)     │  │
                           │  │  ├─ rate limit         │  │
                           │  │  ├─ safety filter      │  │
┌──────────────┐           │  │  └─ vLLM + LoRA        │  │
│  Web client  │  HTTPS    │  └────────────────────────┘  │
│  (v2 옵션)   │◄────────►│  ┌────────────────────────┐  │
└──────────────┘           │  │ CloudWatch / Grafana   │  │
                           │  │  (throughput, errors)   │  │
                           │  └────────────────────────┘  │
                           └──────────────────────────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │ HuggingFace Hub │
                                  │  (adapter 저장)  │
                                  │  KIOM 승인 후만  │
                                  └─────────────────┘
```

## 단계별 진입 조건 (gate)

| 단계 | Gate | 해제 조건 |
|---|---|---|
| **v0 L** | 즉시 | 현재 A6000 보유 + CPT adapter (§04a §D gate 전체 green 후) |
| **v0 S1** | 기관 서버 SSH 계정 + 사용자별 Unix 계정 분리 | IT 팀 요청 |
| **v1 S2** (`hanmed serve`) | vLLM OpenAI server wrap + auth | 2026년 후반기 검토 |
| **v1 C** (RunPod/AWS) | KIOM 서면 승인 + 안전성 검증 + 모델 카드 | §07 §09 M5 이후 |
| **v2 C** (공개 웹) | 상업 라이선스 협상 + 의료기기 규제 검토 | 법무 자문 |

## 비용 추정 (v1 C, RunPod A6000, 월 100 active hours)

- GPU: $0.79/h × 100h = **$79/월**
- 스토리지: HF Hub 무료 / 자체 S3 ~$5/월
- 로깅: CloudWatch ~$5/월
- **합계 ≈ $90/월** (외부 배포 + 안전성 검증 인력비 제외)

2026-04-16 시점 추정치. RunPod 요금 변동 가능.

## 결정 사항

- **v0 = L + S1 만**. `hanmed chat` 단일 CLI 로 로컬/기관 공용
- **v1 = S2 (`hanmed serve` + `--remote`) 추가**. 기관 외부 공개는 KIOM 승인 후
- **v2 = C 클라우드 + 웹 UI**. 상업 라이선스 + 규제 자문 후

## 열린 결정

1. HF Inference Endpoints 가 LoRA serving 을 정식 지원하면 C 경로 단순화 가능 — 2026 하반기 재검토
2. 가천대 GPU 서버 가용성 — 학과 사용률 확인 필요
3. 멀티 GPU (A6000 × 2) 필요 시 — v0 단일 사용자 기준 불필요, v1 동시 4~8 사용자 이상이면 vLLM tensor_parallel 활성화
