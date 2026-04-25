# AKMPP — 한의약 정밀의료 플랫폼 (Frontend Mockup)

가천대 MRC 연구과제 `mrc_0417`의 KH-MFM (Korean-medicine Multimodal Foundation Model) 기반 CDSS 콘솔 프론트엔드 목업.

## Stack

- Next.js 16 (App Router) + TypeScript
- Tailwind CSS v4 (`@theme inline` 토큰)
- 디자인 콘셉트: **INK & HANJI** (한지 #F0E9D8 / 먹 #1A1612 / 朱印 #B83025 / 갈색 #6E6353)
- Pretendard + Noto Serif KR 페어링

## Pages

- `/` — 랜딩 (히어로, 4단계 파이프라인, 모듈 4종)
- `/platform` — CDSS 콘솔 (3-패널 + 합성 시각화 + 진단 챗봇)

## 백엔드 연동 포인트

| 모듈 | 연결 대상 |
|---|---|
| HanMed-LLM 진단 챗봇 | `../../korean-medicine-llm` (vLLM `/v1/completions`) |
| HybridGenoDiT 합성 시각화 | `../../src/` (gene-synthesis-project) |

현재는 둘 다 mock 응답으로 동작합니다. 실제 백엔드 연동은 후속 작업.

---

## 🐳 Docker (권장)

운영/시연 환경에서는 Docker 단독으로 띄울 수 있습니다. Next.js standalone output 기반의 multi-stage build로, 최종 이미지에 빌드 도구는 포함되지 않습니다.

### docker compose

```bash
docker compose up -d --build
# → http://localhost:3000

# 종료
docker compose down
```

다른 포트로 띄우고 싶다면 `WEB_PORT`로 오버라이드:

```bash
WEB_PORT=8080 docker compose up -d --build
```

### docker run (compose 없이)

```bash
docker build -t akmpp-mockup:latest .
docker run -d --name akmpp-mockup -p 3000:3000 --restart unless-stopped akmpp-mockup:latest
```

### 헬스체크

컨테이너에 `wget` 기반 HEALTHCHECK가 있어 `docker ps` STATUS 컬럼에 `healthy` 표시:

```bash
docker ps --filter name=akmpp-mockup --format "table {{.Names}}\t{{.Status}}"
```

---

## 로컬 개발 (dev server)

Docker 없이 직접 돌릴 때:

```bash
npm install
npm run dev
# http://localhost:3000
```

> Mac/Windows에서 핫 리로드 성능을 고려하면 개발 중에는 Docker 대신 `npm run dev`를 권장합니다.

## 빌드 (no Docker)

```bash
npm run build
npm run start
```

## 디렉토리

```
web/akmpp-mockup/
├── app/                    # App Router (page.tsx, layout.tsx, platform/)
├── components/
│   ├── nav.tsx
│   ├── ui/                 # Button, Card, Badge, Slider, Progress, Input, Textarea, Select
│   └── platform/           # CDSS 콘솔 컴포넌트
├── lib/
│   ├── utils.ts            # cn helper
│   └── inference-types.ts  # PatientInput, INFERENCE_STEPS, mock 데이터
├── public/
├── Dockerfile              # multi-stage (deps → builder → runner, non-root)
├── docker-compose.yml      # 단일 서비스
├── .dockerignore
└── next.config.ts          # output: 'standalone'
```
