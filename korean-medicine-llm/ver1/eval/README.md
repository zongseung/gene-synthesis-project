# eval/ — HanMed-Eval v0

Held-out 평가 세트 + contamination gate 용 SHA-256 해시.

## 현재 상태 (R3, M0/M1 진행 중)

- `hashes/heldout_T1.txt` — **R3 placeholder** (positive-control 1 샘플). M2 에서 29개 추가 예정.
- `hashes/heldout_{T2,T5}.txt` — M2 예정 (T2: 독해 QA 원문 30 / T5: KLUE-YNAT 100).
- `hanmed_eval_v0/{T1,T2,T5}.jsonl` — M2 예정 (전문가 curation 후).

## 용도 (§03.4.2, §04a §D Gate G1)

CPT 입력 빌드 (`src/data/builder/preprocess.py`) 시 각 **record `text` 전체** hash 대조 → 매치되는 record drop. drop 비율 > 0.5% 시 파이프라인 실패 처리 (§F B4 planned).

**Hash 단위 (R3.1 명시)**: `preprocess.py:204` 가 `hashlib.sha256(normalize(rec["text"]).encode("utf-8"))` 를 계산. bilingual 의 경우 `<ZH>...</ZH>\n<KO>...</KO>\n\n` 블록 전체가 hash 대상. **단독 문장 hash 를 heldout 파일에 넣으려면 해당 문장이 eval curation 시 단일 record 로 구성돼야 함** (bilingual block 에 embedding 된 문장 hash 는 match 불가).

## R3 placeholder 설명

T1 1 샘플만 commit 한 이유: `preprocess.py:321-325` silent skip 이 M2 에서 hard-fail 로 바뀌기 전까지는 `eval/hashes/` 디렉토리 "존재 자체" 만 positive-control 로 작동. 실측 drop 은 §F B4 코드 수정 후에만 측정 가능.
