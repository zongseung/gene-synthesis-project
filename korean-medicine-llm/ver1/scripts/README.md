# ver1/scripts — 역할별 구조

ver1(텍스트 CPT/SFT) 파이프라인 스크립트. **실행 기준 디렉토리는 `ver1/`** 이며, 경로 인자/상수는 `ver1/` 기준이다.

```
scripts/
├── sft/        # SFT 학습데이터 빌드 (ver8 최종)
│   ├── build_sft_full_corpus.py    # 전체 코퍼스 → SFT
│   ├── augment_sft_v7.py           # v7 증강(거부/다양성)
│   └── _v7_refusal_variants.py     # augment 헬퍼(거부문 변형)
├── corpus/     # 원문(동의보감 book_008) 준비
│   ├── build_book008_splits.py
│   ├── classify_books.py
│   ├── fetch_book_metadata.py
│   └── build_factsheet_draft.py
├── rag/        # RAG 인덱스/조회
│   ├── build_rag_index.py
│   ├── probe_ver8_1_rag_v4.py      # retrieval 본체(hybrid_search 등)
│   └── probe_ver8_1_rag_info.py    # v4 재사용(info 모드)
├── eval/       # 평가·검증·감사
│   ├── eval_phaseA.py
│   ├── verify_sft_against_raw.py
│   ├── verify_synth_facts.py
│   ├── verify_packed_content.py
│   ├── audit_sft_diversity.py
│   └── entity_delta.py
├── model/
│   └── build_merged_model.py       # LoRA adapter merge
└── deploy/
    ├── deploy_phaseA.sh            # vLLM 컨테이너 기동
    └── cli_phaseA.sh               # CLI 접속
```

## 실행 예 (반드시 `ver1/`에서)
```bash
cd ver1
.venv/bin/python scripts/sft/build_sft_full_corpus.py ...
.venv/bin/python scripts/model/build_merged_model.py ...
scripts/deploy/deploy_phaseA.sh direct
```

## 정리 이력 (2026-06-29)
- **삭제(−25파일)**: 일회성 진단 스크립트 21개(probe_ver6/7/8·tokenizer·gemma_zero·cli smoke·`_common`) + 구버전 SFT 빌더 4개(`build_sft_qa/complex/diverse/clinical`, ver8 `full_corpus`로 대체).
- **재배치**: 잔존 17개 + .sh 2개를 역할별 폴더로 분리. 폴더 깊이 +1에 맞춰 `Path(__file__)` ROOT 상수(+1 parent)와 `.sh`의 `cd`(`../..`)를 보정.
- 코드 동작(생성물·출력)은 보존. import 체인(`augment→_v7_refusal_variants`, `rag_info→rag_v4`)은 동일 폴더 배치로 유지.
