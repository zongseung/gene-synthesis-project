# Mediclassics Markup & 동의보감 Parsing Specification (ver2.2 — 실측 반영)

- 작성일: 2026-04-16 (ver2.2 = API·실파일 검증 후 갱신)
- 상태: ✅ **실측 완료** (동의보감 내경편 권1 전수 분석 + API 응답 검증)
- 공식 markup 스펙: `info.mediclassics.kr/document/document/markup-syntax-for-classics` (2016-02-12)
- 실데이터 소스: 배포 txt 파일 + `https://mediclassics.kr/books/{id}/volume/{v}/content/{seq}` API
- 소비자: `src/data/parser/mediclassics_parser.py` (placeholder), `src/data/crawler/mediclassics_orchestrator.py` (구현 완료)

## 0. 검증 결과 요약

| 항목 | 상태 |
|---|---|
| 공식 markup 문법 문서 | ✅ 확인, §1~§5에 보존 (웹 뷰어 전용 reference) |
| **배포 파일·API 응답 inline 태그** | ✅ **0건** — `[xx/..]`, `{..}`, `#..#` 모두 제거 (배포 정책) |
| 권/편/문/조 구조 태그 | ✅ heading prefix(AA/BB/CC/DD/DP/SS) **실측 매핑 완료** (§4.4) |
| 원문/국역 정렬 단위 | ✅ **API record 단위 자동 정렬** (한문/한국어/영어 한 record에 동시 제공) |
| 동의보감 총 분량 | ✅ 23권 (목록 제외), API content_total 실측 — vol_01=1254 record |
| 이미지 파일 본체 | ❌ 배포 안 됨 (정책: "도상 제외, 이미지 내 텍스트만 포함") — PP record는 라벨 텍스트만 |

**핵심 발견 (이 문서 핵심)**:
- 배포 파일/API는 **평문화된 단순 형태**: `{2자 prefix}\t{text}` (배포 txt) 또는 단순 JSON record (API)
- API는 한문 원문 + `up_path_nm`(계층 경로) + `trans_2`(국역) + `trans_1`(영역)을 **이미 정렬된 record로 한번에** 제공 → 별도 파싱·정렬 불필요
- 따라서 **배포 파일 파서는 fallback** 용도이며, 실제 데이터 파이프라인은 API 응답을 직접 record-level로 사용

## 1. 데이터 형태 — API vs 배포 파일

### 1.1 API 응답 (권장 경로)

```json
{
  "book_id": 8,
  "volume_id": 1,
  "content_seq": 138,
  "content_level": "Z",            // 첫 글자
  "content_level_depth": "Z",      // 둘째 글자
  "up_path_nm": "內景篇卷之一 > 身形 > 形氣之始",
  "original": "乾鑿度云, …",
  "trans_2": "《건착도》에, …",     // 국역
  "trans_1": "In the Book of …",   // 영역
  "annotation": [],                 // 항상 빈 배열
  "index_num": 1
}
```

→ **파싱 거의 trivial**. `content_level + content_level_depth` 합치면 2자 prefix.

### 1.2 배포 txt 파일 (fallback / 단일 파일 다운로드 시)

```
// 이 파일은 한국한의학연구원에서 동의보감 원문을 디지타이즈하여 배포한 것입니다.
// File Info: { encoding: 'UTF-8', end_of_line: 'CRLF' }
// 다운로드 일시: 20260415

AA	內景篇卷之一
XX	御醫 忠勤貞亮扈聖功臣 崇祿大夫 陽平君 臣 許浚奉敎撰
OO	東醫寶鑑序
ZZ	醫者雅言軒岐, 軒岐上窮天紀, …
ZZ	我宣宗大王, 以理身之法, …
```

| 항목 | 값 |
|---|---|
| 인코딩 | UTF-8 |
| 줄바꿈 | CRLF (`\r\n`) |
| 헤더 | 3줄 `//` 주석 + 빈 줄 |
| 본문 라인 | `{prefix 2자}\t{text}` |
| **inline 태그** | **0건 (배포 정책상 제거)** |
| 이미지 | 캡션 텍스트만 PP 라인으로 (이미지 파일 자체는 미포함) |

→ 정규식 `^([A-Z0-9]{2})\t(.*)$` 한 줄로 파싱 완료.

## 2~3. (참고) 웹 뷰어용 inline 태그 — 배포 파일 미포함

> 아래는 공식 스펙 페이지에 나오는 inline 태그 명세 (웹 뷰어용). **배포 파일·API 응답에는 모두 제거되어 0건**. LLM 학습 입력에서는 무시. 향후 KIOM이 raw markup 포함 데이터를 제공하면 이 명세로 파싱 가능.

### 2. 형태기호 (Style, `[ ]`)

| 기호 | 의미 |
|---|---|
| `[sm/텍스트]` | 작은글자 |
| `[lg/텍스트]` | 큰글자 |
| `[ps/텍스트]` | 양각 |
| `[ng/텍스트]` | 음각 |

이중대괄호 `[[ ]]` 중첩 허용.

### 3. 주석기호 (Annotation, `{ }`)

| 기호 | 의미 |
|---|---|
| `{A=B@출전}` | 교감기 |
| `{표현:설명@출전}` | 역자주 |
| `{n}` | 강제 줄바꿈 |

세미콜론 `;`로 다중 항목 분리.

## 4. 문단 태그 — 라인 prefix 2자 (실측)

### 4.1 형식

- 첫 글자: 수준 (A→가장 상위, Z→본문, S→처방본문, X→meta, O→서두, P→이미지)
- 둘째 글자: 의미 (반복 = 일반, P/K/H = 처방/침구/본초, 0~3 = 들여쓰기)

### 4.2 도메인 특화 표제

| 둘째 글자 | 의미 |
|---|---|
| `P` | 처방 표제 (方劑) |
| `K` | 침구 표제 |
| `H` | 본초 표제 |

→ DP, SS는 **본초·처방·혈자리 entity 자동 추출의 1차 신호**. NER seed (`§03.4.1`)에 직접 활용.

### 4.3 본문 들여쓰기

- `Z0`/`Z1`/`Z2`/`Z3`: 본문 들여쓰기 (위 1칸 / 아래 1·2·3칸)
- `S0`/`S1`/`S2`/`S3`: 처방 본문 들여쓰기
- LLM 학습 시 동일 본문 문단으로 취급, `layout_indent` 필드에 보존만

### 4.4 실측 prefix 매핑 — 동의보감 내경편 권1 (1254 record 전수)

```
prefix  count  실제 용도
─────  ─────  ──────────
ZZ      495   일반 본문
SS      332   처방 본문 (DP 직속)
DP      269   ★ 처방 표제 (방제 이름) — 가장 많은 heading
CC      104   중간 수준 (脉法, 灸法, 治法 류 소분류)
Z2       25   본문 2칸 들여 (목차 다단 나열용)
DD       10   조(條) — 七情 등 작은 분류
XX        7   meta — 저자, "凡二十三種" 단원 종료 기록
OO        5   서두 섹션 — 序, 集例, 歷代醫方, 總目, 身形藏府圖
BB        5   ★ 문(門) — 身形, 附養老, 精, 氣, 神
AA        1   ★ 권 타이틀 — "內景篇卷之一"
PP        1   이미지 캡션 (身形藏府圖 부위 라벨)

발견 안 된 prefix: EE, FF, CP, CK, CH, EP, EK, EH (이 책에서)
```

### 4.5 실측 매핑이 가설과 다른 점 (ver2.0 → ver2.2)

| 레벨 | ver2.0 가설 | **실측 (ver2.2)** |
|---|---|---|
| AA | 편 (篇) | **권 타이틀** (편은 파일·디렉토리 분리로 표현) |
| BB | 권 (卷) | **문 (門)** |
| CC | 문 (門) | 중간 수준 (방법·소주제) |
| DD | 조 (條) | **조 (條)** ✓ |
| DP | (처방) | **처방 표제 — 가장 빈번** |
| SS | (처방 본문) | **처방 본문** ✓ |

→ ver2.0의 "AA=편 / BB=권 / CC=문 / DD=조"는 **틀렸다**. 실제는 편이 파일 단위로 분리되고 (volume_id 1~4 = 內景篇 권1~4), AA가 권 타이틀.

## 5. (참고) 도상·표 태그 — 배포 파일에서 단순화

### 5.1 PP 이미지
- 공식 스펙: `<파일명,제목>내용` 형식
- 실측: 파일명·제목 없이 **라벨 텍스트만 평문**으로 (이미지 본체 미배포)
- 예: `PP\t泥丸宮, 髓海腦, 玉枕關, 轆轤關, 尾閭關, 喉, 咽, 肺, 心, ...`

### 5.2 TT 표
- 공식 스펙: `#열1#열2#열3{n}#열4...`
- 실측: 권1에 0건. 다른 권에서 확인 필요. (이미지·표가 많은 외형편·탕액편 추정)

## 6. 동의보감 문헌 구조

### 6.1 전체 체제 (실측 — API)

API `volumes/` 응답 기준 23권 (목록 2권 제외):

| 편 | 권 | content_total (record 수) |
|---|---|---|
| 內景篇 (내경) | 1~4 | 1254, 1415, 1781, 1504 (합 5954) |
| 外形篇 (외형) | 5~8 | 1520, 1541, 1927, 1441 (합 6429) |
| 雜病篇 (잡병) | 9~19 | 11권, 합 14,746 |
| 湯液篇 (탕액) | 20~22 | 2028, 1536, 1588 (합 5152) |
| 鍼灸篇 (침구) | 23 | 1759 |
| **합계** | **23권** | **34,040 record** |

### 6.2 내경편 권1 세부 (첫 파싱 타겟, 실측)

내경편 권1에 포함된 문(門)은 **4개 + 부록 1개** (실측 BB 5건):
- 身形, 附養老, 精, 氣, 神

(나머지 6문 — 血, 夢, 聲音, 言語, 津液, 痰飮 — 은 권2에 있음)

### 6.3 heading 매핑 (실측)

```
파일별 분리:        內景篇 권1 = volume_id=1, 권2 = volume_id=2, ...
AA (1회/권)        = 권 타이틀          ("內景篇卷之一")
BB (4~10회/권)      = 문(門)             ("身形", "精", "氣", "神")
CC (~100회/권)      = 중간 분류           ("形氣之始", "脉法", "單方", "灸法")
DD (~10회/권)       = 조(條) — 작은 주제   ("喜", "怒", "憂", ...)
DP (~270회/권)      = 처방 표제           ("瓊玉膏", "三精丸")
SS (~330회/권)      = 처방 본문 (DP 직속)  ("塡精補髓 …")
ZZ (~500회/권)      = 일반 본문
XX (~7회/권)        = meta (저자·단원종료)
OO (~5회/권)        = 서두 (序, 總目, 集例)
PP (~1회/권)        = 이미지 캡션 라벨
```

→ 다른 책(향약집성방, 의방유취 등)에서 prefix 분포가 다를 수 있음. 책별 실측은 M2 검증 단계에서.

## 7. 파서 알고리즘

### 7.1 API 경로 (권장 — 현재 구현됨)

`mediclassics_orchestrator.py` 가 처리. 파싱 거의 trivial:

```python
def extract_record(raw_api_response: dict) -> dict:
    return {
        "book_id":       raw.get("book_id"),
        "volume_id":     raw.get("volume_id"),
        "content_seq":   raw.get("content_seq"),
        "content_level": (raw.get("content_level") or "") + (raw.get("content_level_depth") or ""),
        "up_path_nm":    raw.get("up_path_nm"),  # 이미 계층 경로 제공
        "original":      raw.get("original"),     # 한문
        "trans_ko":      raw.get("trans_2"),      # 국역
        "trans_en":      raw.get("trans_1"),      # 영역
        "annotation":    raw.get("annotation") or None,
        "index_num":     raw.get("index_num"),
    }
```

`up_path_nm`이 `"內景篇卷之一 > 身形 > 形氣之始"` 형태로 계층 경로를 직접 제공하므로 **SectionStack trie 재구성 불필요**.

### 7.2 배포 txt 파일 경로 (fallback)

배포 파일을 직접 다운로드한 경우 (수기 다운로드 / API 미가용 책):

```python
import re
LINE_RE = re.compile(r"^([A-Z0-9]{2})\t(.*)$")

def parse_dist_txt(path):
    out = []
    section_stack = []  # for hierarchy reconstruction
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("//"):
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            prefix, text = m.group(1), m.group(2)
            level_char = prefix[0]
            kind_char  = prefix[1]
            if level_char in "ABCDEF":
                # heading: pop stack until level
                level = ord(level_char) - ord('A') + 1
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, prefix, text))
                out.append({"type": "heading", "prefix": prefix, "text": text, "path": list(section_stack)})
            elif level_char == 'Z' or level_char == 'S':
                category = "prescription" if level_char == 'S' else "body"
                indent = 0 if kind_char == level_char else int(kind_char) if kind_char.isdigit() else 0
                out.append({"type": "para", "prefix": prefix, "text": text, "category": category, "indent": indent, "path": list(section_stack)})
            else:
                out.append({"type": "other", "prefix": prefix, "text": text, "path": list(section_stack)})
    return out
```

**구현 위치 (placeholder)**: `src/data/parser/mediclassics_dist_parser.py` (현재 미구현, 필요 시 추가)

### 7.3 인라인 태그 처리 — 배포 파일에는 0건이므로 사실상 미사용

만약 raw markup 포함 데이터를 향후 받게 되면 §2~§3 태그 명세대로 처리:

```python
INLINE_STYLE = re.compile(r"\[{1,2}(sm|lg|ps|ng)/([^\]]*)\]{1,2}")
INLINE_NOTE  = re.compile(r"\{([^{}]+)\}")
NEWLINE      = re.compile(r"\{n\}")
```

현재 파이프라인에서는 호출 안 함.

## 8. 출력 스키마 (실측 — API 기반)

크롤러가 저장하는 record (`data/raw/mediclassics_unified/book_{id}/vol_{v}.jsonl`):

```json
{
  "book_id": 8,
  "volume_id": 1,
  "content_seq": 138,
  "content_level": "ZZ",
  "up_path_nm": "內景篇卷之一 > 身形 > 形氣之始",
  "original": "乾鑿度云, 天形出乎乾, 有太易太初太始太素 …",
  "trans_ko": "《건착도》에, 하늘[天]의 형(形)은 건(乾)에서 나오니 …",
  "trans_en": "In the Book of Changes Heavenly-Qian Chisel Measurement (乾鑿度), it is said …",
  "annotation": null,
  "index_num": 1
}
```

스키마 버전: `schema_version: 2` (ver2.2 시점, ver2.0 가공 스키마는 폐기).

## 9. 원문 ↔ 국역 정렬

**자동 정렬됨** — API 응답 한 record에 `original` / `trans_2` / `trans_1` 동시 제공. 별도 alignment 단계 불필요.

ver2.0의 가설 A/B (파일 분리 / interleaved)는 **둘 다 해당 안 됨** — API record 단위가 정답.

## 10. 검증 전략

| 단계 | 방법 | 통과 기준 |
|---|---|---|
| API 응답 형식 일치 | 모든 record에 `original` 키 존재 | 100% |
| 국역 coverage | `trans_ko` 비율 | 책별 ≥ 90% (manifest.json `ko_coverage_pct`) |
| 영역 coverage | `trans_en` 비율 | 책별 측정만 (강제 아님) |
| content_seq 연속성 | 1~content_total 빠짐 없음 | gap ≤ 5% |
| 계층 일관성 | `up_path_nm` 단조 path | 위배 0건 |
| 단일 record 수기 검수 | 무작위 50개 | 정렬 정확도 ≥ 95% (실측 100% 예상) |

실패 시 ver2 `08_risks/risk_register.md` §8.2 M2 gate "Corpus parse 정확도 ≥ 95%" 호출.

## 11. 예상 실패 케이스 (실측 후 갱신)

| # | 케이스 | 대응 |
|---|---|---|
| F1 | API rate limit 405 | 60s pause + max 30 retry (`mediclassics_orchestrator.py` 구현됨) |
| F2 | API 일시 5xx | exponential backoff max 5 retry |
| F3 | content_total 미상 (volumes API 미가용 책) | seq=1부터 빈 응답까지 fetch (구현됨) |
| F4 | 국역 누락 record (`trans_2 == null`) | 한문 단독 corpus로 분리 (`hanmed_zh_only.jsonl`) |
| F5 | up_path_nm 빈 record | content_level 단독으로 처리, 상위 hierarchy 없음으로 표기 |
| F6 | 새 prefix 발견 (스펙 밖) | 로그 후 `category="unknown"` 으로 통과 |
| F7 | 이미지 본체 부재 | PP record는 캡션 텍스트만 보존 (정상) |

이미 ver2.0 가설(중첩 대괄호, 주석 안 태그, 표 #, 이미지 파일명 등)은 **배포 정책상 모두 제거**되었으므로 fallback 파서에서만 발생 가능.

## 12. 구현 마일스톤 (ver2.2 갱신)

| # | 산출물 | 상태 |
|---|---|---|
| ✅ | `mediclassics_orchestrator.py` (multi-process) | **완료** (446 라인, 운영 중) |
| ✅ | API record schema 검증 | **완료** (실측 100% 일치) |
| ✅ | Core 14 데이터 수집 | **진행 중** (book_8 등 14권, 백그라운드) |
| ☐ | `build_bilingual_blocks.py` (D2 포맷 변환) | M2 초 |
| ☐ | `mediclassics_dist_parser.py` (배포 txt fallback) | 필요 시 (현재 미사용) |
| ☐ | 책별 prefix 분포 통계 리포트 | M2 초 |
| ☐ | NER 사전 자동 추출 (DP·SS prefix 활용) | M2 |

## 13. 해소된 열린 질문

ver2.0의 7개 열린 질문 모두 실측으로 해소:

1. ~~이중 대괄호 `[[ ]]`~~ → **배포 파일에 0건**, 무관
2. ~~권/편 ↔ heading 매핑~~ → **§4.4·§6.3 실측 매핑 확정**
3. ~~원문↔국역 파일 분리~~ → **API record 단위 자동 정렬, 분리 없음**
4. ~~저자주·편자주·교감주 구분~~ → **배포 파일 모두 제거됨**, 무관
5. ~~이미지 파일 포함 여부~~ → **❌ 미포함 정책 확정**, 캡션 텍스트만
6. ~~xP/xK/xH 혼용 level ordering~~ → **DP가 권당 ~270회 최빈, P>K>H 빈도 추정**
7. ~~책별 heading 체계 동일 여부~~ → **책별로 다를 수 있음**, M2에 책별 prefix 통계 산출

## 14. 참고

- 공식 markup spec: `info.mediclassics.kr/document/document/markup-syntax-for-classics` (2016-02-12)
- 배포 서비스 프론트엔드: `info.mediclassics.kr/apps/dist-texts/` (AngularJS, Firebase 카운터)
- 배포 앱 main.js (auth header, chunk size 출처): `info.mediclassics.kr/apps/dist-texts/assets/js/main.js`
- 동의보감 편제: 한국민족문화대백과사전, 위키백과
- KIOM 동의보감 교감본 PDF (별도 경로): "내손안에 동의보감 원문강독편" 앱
- 본 문서는 ver2.0의 공식 스펙 기반 가설을 **실데이터 검증으로 갱신**한 ver2.2.
