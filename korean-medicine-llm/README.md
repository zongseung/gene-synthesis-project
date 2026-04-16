# HanMed-LLM

한의학 고전 + 한국어 도메인 특화 LLM (Solar-10.7B 위 bf16 LoRA CPT + SFT) 프로젝트.

데이터: 한국한의학연구원 [mediclassics.kr](https://mediclassics.kr) — 한문 원문 + 국역 + 영역 3중 병렬.

상세 기획서: `docs/ver2/README.md` 참조 (ver2.1 패치 완료).

---

## 1. 환경

```bash
# Python 3.10+, Linux/macOS
pip install httpx
# (학습 단계에서 추가: torch, transformers, peft, trl, llama-factory 등)
```

GPU: RTX A6000 48GB × 2 (bf16 LoRA 기준 1장 충분, DDP는 throughput 확장용).

## 2. 데이터 수집 (현재 구현됨)

### 2.1 빠른 시작 — Core 7 (최소)

```bash
python3 src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 8,56,69,86,93,182,291 \
  --delay 0.5 \
  --concurrency 2 \
  --pause 60 \
  --rl-max-retries 30 \
  --net-max-retries 5
```

7권 (동의보감, 의방유취, 제중신편, 침구경험방, 향약집성방, 동의수세보원_신축본, 방약합편).
완료 시간: **약 12~15시간** (서버 per-book rate limit 때문).

### 2.2 Core 14 (권장)

```bash
# Core 7 + 추가 7권 (사의경험방, 광제비급, 동의사상신편, 본초정화, 식료찬요, 의종손익, 외과심법요결)
python3 src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 8,56,69,86,93,182,291,1,4,9,24,38,59,100
```

또는 Core 7 진행 중에 별도 process로 추가 7권 동시 발사:

```bash
python3 src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 1,4,9,24,38,59,100 &
```

### 2.3 한국 한의학 고전 25권 전체 (Core 25)

```bash
python3 src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 1,4,7,8,9,24,38,44,46,47,49,54,56,59,60,69,70,71,86,93,94,100,139,182,183,291
```

### 2.4 Resume 모드 (자동)

기존 jsonl이 있으면 `max(content_seq) + 1`부터 자동 이어감. 따로 옵션 안 줘도 됨. 중단·재시작 안전.

## 3. 진행 모니터링

```bash
# 통합 로그
tail -f data/raw/mediclassics_unified/orchestrator.log

# 책별 상세 로그
tail -f data/raw/mediclassics_unified/logs/book_008.log

# 전체 record 수
find data/raw/mediclassics_unified -name "vol_*.jsonl" | xargs wc -l | tail -1

# 책별 진행
for d in data/raw/mediclassics_unified/book_*/; do
  bid=$(basename "$d" | sed 's/book_//')
  cnt=$(wc -l "$d"vol_*.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
  echo "  book_${bid}: ${cnt:-0} records"
done

# 살아있는 워커
ps -ef | grep mediclassics_orchestrator | grep -v grep | wc -l
```

## 4. 데이터 산출물

```
data/raw/mediclassics_unified/
├── orchestrator.log              # 전체 통합 로그
├── unified_manifest.json         # 모든 책 통합 통계 (총 record, KO/EN coverage)
├── logs/
│   └── book_{NNN}.log            # 책별 상세 로그
└── book_{NNN}/
    ├── manifest.json             # 책별 통계
    └── vol_{VV}.jsonl            # 권별 record (한 줄 = 한 record)
```

각 jsonl record:

```json
{
  "book_id": 8,
  "volume_id": 1,
  "content_seq": 138,
  "content_level": "ZZ",
  "up_path_nm": "內景篇卷之一 > 身形 > 形氣之始",
  "original": "乾鑿度云, …",
  "trans_ko": "《건착도》에, …",
  "trans_en": "In the Book of Changes …",
  "annotation": null,
  "index_num": 1
}
```

`content_level` 2자 prefix 의미는 `docs/ver2/02_data_source/mediclassics_parsing_spec.md` §4 참조 (실측: AA권 / BB문 / CC분류 / DD조 / DP처방표제 / SS처방본문 / ZZ본문 등).

## 5. 아키텍처 (크롤러)

```
ProcessPoolExecutor(max_workers = min(cpu_count, num_books))
   │
   ├─ Worker 1 (book_8)   ─── asyncio + httpx (concurrency=2)
   │                          ├─ resume from existing jsonl max_seq
   │                          ├─ fetch-until-empty (volume boundary)
   │                          └─ retry: rate-limit ≤ 30회, network ≤ 5회
   ├─ Worker 2 (book_56)  ─── (동일 구조)
   └─ ...
```

- **Multiprocess (책당 1 process)**: per-book rate limit 분리 활용 (실측 책당 quota ~240/window)
- **각 process 내부 asyncio**: I/O-bound이라 GIL 무관, concurrency=2면 sustained quota 거의 채움
- **CPU 코어 vs 책 수**: `min(cpu, books)`. 32 코어면 책 32권까지는 1:1 분배, 그 이상이면 큐
- **같은 책에 워커 여럿 = 무의미**: quota 더 빨리 소진해도 60s pause 같음

자세한 rate limit 실측·retry 정책: `docs/ver2/03_data_pipeline/acquisition.md` §3.2.5~§3.2.6.

## 6. API 출처 (검증 — 2026-04-16)

```
GET https://mediclassics.kr/books/{id}                            # HTML, 책 메타·권 목록 추출용
GET https://mediclassics.kr/books/{id}/volume/{v}/content/{seq}   # JSON, 본문 record (한문/국역/영역)
GET https://mediclassics.kr/api/books/8/volumes/                  # JSON, 동의보감 권 목록 (id=8만 가용)

Header: Authorization: 5fe23edf9dec4c718e188073e46274bd
        Content-Type: application/json
```

(인증 헤더는 배포 앱 JS에 hardcoded — 공개 정보)

## 7. 라이선스 메모

- **mediclassics 데이터**: KIOM 비상업 무료 이용. **상업 이용은 `kiombook@kiom.re.kr` 서면 문의 필수**. 출처 표기 의무 = "한의학고전DB (mediclassics.kr)".
- **본 코드**: 연구용으로만 사용. 데이터 가공물 공개 시 KIOM 사전 승인 권장.
- **모델 weights**: KIOM 데이터 학습한 LoRA adapter 공개는 KIOM 승인 후. 자세히는 `docs/ver2/07_license_ethics/`.

## 8. 디렉토리 (전체)

```
korean-medicine-llm/
├── README.md                       # 이 파일
├── docs/
│   ├── proposal_v0_draft.md        # v0 (rejected by discriminator)
│   ├── 01_overview/                # ver1 archive
│   ├── 02_data_source/
│   ├── ...
│   ├── 09_roadmap/
│   └── ver2/                       # ver2.1 (현재 canonical) + ver2.2 패치
│       ├── README.md
│       ├── 01_overview/overview.md
│       ├── 02_data_source/
│       │   ├── data_verification.md
│       │   └── mediclassics_parsing_spec.md   # ver2.2 실측 반영
│       ├── 03_data_pipeline/
│       │   └── acquisition.md                 # ver2.2 실측 반영
│       ├── 04_model_strategy/
│       ├── 05_evaluation/
│       ├── 06_infrastructure/
│       ├── 07_license_ethics/
│       ├── 08_risks/
│       └── 09_roadmap/
├── src/
│   └── data/
│       └── crawler/
│           ├── mediclassics_crawler.py        # v1 (single-book, 동의보감 전용)
│           ├── mediclassics_multi_crawler.py  # v2 (multi-book, 단일 process)
│           └── mediclassics_orchestrator.py   # v3 (multi-process, 현재 사용)
└── data/
    └── raw/
        └── mediclassics_unified/              # 크롤 출력
```

## 9. 다음 단계 (M0~M1)

수집 완료 후:
1. `build_bilingual_blocks.py` 작성 — 정렬 record → D2 포맷 (`<ZH>...</ZH>\n<KO>...</KO>`)
2. `corpus_v1.json` 동결 (총 token 수, KO/EN coverage, SHA256)
3. Solar tokenizer 실측 → tokens/char (Stage 0 결정)
4. HanMed-Eval v0 200문항 큐레이션 (T1 30 / T2 30 / T3 20 / T4 20 / T5 100)
5. KIOM `kiombook@kiom.re.kr` 이메일 발송 (라이선스 critical path)
6. Solar Apache-2.0 variant 라이선스 확증 (M0 24h)

자세한 마일스톤: `docs/ver2/09_roadmap/milestones.md`.

## 10. 트러블슈팅

| 증상 | 원인 | 대응 |
|---|---|---|
| 모든 book 동시 405 | 서버 일시 장애 | 로그 확인, 5분 후 재실행 (resume) |
| 특정 book 무한 405 | 해당 책 quota 소진 | `--pause` 90~120s로 상향 |
| `httpx.ReadTimeout` 빈발 | 네트워크 불안정 | `CONTENT_TIMEOUT`을 60으로 (코드 수정) |
| 권 boundary 오감지 | EMPTY_RUN_THRESHOLD 부족 | 코드의 `EMPTY_RUN_THRESHOLD = 3` → `5`로 |
| 프로세스 죽음 | 메타 fetch timeout | `META_TIMEOUT = 120` 이미 적용됨, 안 되면 재실행 (resume) |
| 일부 record 깨짐 | API 응답 일시 깨짐 | resume 모드가 다음 실행에서 자동 보충 |

진행 중 모든 프로세스 일괄 중지:
```bash
pkill -f mediclassics_orchestrator
```

다시 시작 시 resume 모드로 자동 이어감.
