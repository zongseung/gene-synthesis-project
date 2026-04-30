# ver8.1 vLLM + RAG sidecar — 배포·smoke test 가이드

## 0. 선결 조건

| 항목 | 확인 |
|---|---|
| GPU | 1 × 24 GB+ (Gemma3-12B bf16 + KV cache) |
| Disk | merged model ~24 GB + HF cache ~4 GB (BGE-M3) |
| Docker | `runtime: nvidia` 가능, compose v2 |
| Adapter | `experiments/dongui_bogam/outputs_ver8_1_gemma_v1/adapter/` 존재 |
| RAG corpus | `data/rag/book_008.index` + `book_008.meta.jsonl` 존재 |
| info prompts | `experiments/dongui_bogam/eval/info_mode_prompts.yaml` 존재 (Phase 6 후속) |

```bash
# 위 4개 한 번에 점검
ls models/gemma-3-12b-it/config.json \
   experiments/dongui_bogam/outputs_ver8_1_gemma_v1/adapter/adapter_config.json \
   data/rag/book_008.index \
   data/rag/book_008.meta.jsonl \
   experiments/dongui_bogam/eval/info_mode_prompts.yaml
```

## 1. Merged model 빌드 (한 번만, ~20-30분)

```bash
PYTHONHASHSEED=0 .venv/bin/python \
  experiments/dongui_bogam/scripts/build_merged_model_ver8_1.py
# → experiments/dongui_bogam/outputs_ver8_1_gemma_v1/merged/  (~24 GB)
```

옵션:
- `--device cuda` : GPU 에서 merge (빠름, 24GB+ VRAM 필요)
- `--device cpu`  : CPU + 64 GB RAM (안전)

## 2. 스택 기동

```bash
cd experiments/dongui_bogam
docker compose -f docker/compose.ver8_1.yml up -d --build
```

기동 시간:
- vLLM: weight load + cuda graph compile ~90-120s (HEALTHCHECK start_period=120s)
- RAG: BGE-M3 download (첫 실행 시) ~2분, 캐시되면 ~10s

## 3. Health check

```bash
# RAG sidecar (외부 8080)
curl -sf http://localhost:8080/health
# {"rag":"ok","vllm":"ok"} 면 정상

# 직접 vLLM 확인 (compose 에서 ports 주석 해제했을 때만)
docker exec hanmed_vllm_ver8_1 curl -sf http://localhost:8000/health
docker compose -f docker/compose.ver8_1.yml logs -f hanmed_vllm_ver8_1
```

## 4. Smoke test — probe_ver8_1_rag_v4 baseline 과 비교

### 4.1 retrieval-only (LLM 호출 없음)

```bash
curl -sG "http://localhost:8080/rag/retrieve" \
  --data-urlencode "query=인삼(人蔘)의 성미와 귀경에 대해 간단히 설명해줘." | jq
# 기대: extracted_names=["人參","인삼"], boost 매칭 2건 이상
```

### 4.2 풀 RAG (LLM 호출)

```bash
# Q1 — 한자 이형자 정규화 (蔘 → 參)
curl -sX POST http://localhost:8080/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"인삼(人蔘)의 성미와 귀경에 대해 간단히 설명해줘."}' | jq

# D2 — 변형 한자 (丹蔘膏 → 丹參膏)
curl -sX POST http://localhost:8080/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"유옹(乳癰) 에 쓰는 단삼고(丹蔘膏) 의 조성과 적응증을 알려주세요."}' | jq

# D4 — 정상 한자
curl -sX POST http://localhost:8080/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"동의보감 소아문의 진경환(鎭驚丸) 은 어떤 증에 쓰며 구성 약재는 무엇인가요?"}' | jq

# 진료성 — STRICT mode 가 거부하는지 확인 (pre_check)
curl -sX POST http://localhost:8080/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"제가 가슴이 답답하고 호흡 곤란이 있는데 어떤 처방이 좋을까요?"}' | jq
# 기대: {"mode":"REFUSED", "safety":{"pre_refused":true,...}}
```

## 5. Baseline 비교

| Query | probe_v4 (local) 기대 | RAG sidecar 기대 |
|---|---|---|
| Q1 인삼(人蔘) | extracted=`['人參','인삼']`, boost 2건 | 동일 |
| D2 丹蔘膏 | boost 1건 (丹參膏 leaf) | 동일 |
| D4 鎭驚丸 | boost 1건 | 동일 |
| 진료성 | (probe 는 pre_check 우회) | `mode=REFUSED` |

retrieval 결과가 일치하면 v4 → sidecar 이식 성공. 답변 텍스트는 vLLM 의 sampling/seed 차이로 약간 변동 가능 (greedy 라 사실상 동일해야 함).

## 6. 정지 / 재배포

```bash
# 중지
docker compose -f docker/compose.ver8_1.yml down

# 새 adapter 라운드 후 재배포:
#   1. build_merged_model_ver8_1.py --output outputs_ver8_1_gemma_v2/merged
#   2. export HANMED_MERGED_DIR=../outputs_ver8_1_gemma_v2/merged
#   3. docker compose -f docker/compose.ver8_1.yml up -d --force-recreate hanmed_vllm_ver8_1
```

## 7. Trouble-shoot

| 증상 | 진단 / 처치 |
|---|---|
| `vllm` health 가 `error` | `docker logs hanmed_vllm_ver8_1` — Gemma3 호환 vLLM 버전 (0.8.5+) 인지 |
| RAG `503 Service Unavailable` | depends_on healthy 가 안 풀림 — vLLM 기동 대기 |
| BGE-M3 download timeout | `HF_HOME_HOST` 마운트 캐시 확인, 또는 사전 `huggingface-cli download BAAI/bge-m3` |
| `boost` 매칭 0 (Q1 등) | `data/rag/` 마운트 경로 확인, `meta.jsonl` 의 path 표기가 표준 한자(參·鎭)인지 grep |
| 한국어 어미 noise (`'심하고'`) 가 추출 | 무해 (leaf 매칭 0건) — 답변엔 영향 없음 |
