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

## Run

```bash
npm install
npm run dev
# http://localhost:3000
```
