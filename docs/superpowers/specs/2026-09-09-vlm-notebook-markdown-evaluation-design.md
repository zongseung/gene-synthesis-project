# VLM 노트북 Markdown 평가 설계

## 목표

기존 `notebooks/vlm_initial_evaluation.ipynb`에서 완료된 text-transfer 최적 adapter를 불러오고, 모델 답변을 안전하게 Markdown으로 표시하며, 소규모 smoke 평가 결과를 바로 확인한다.

## 설계

- 기존 평가 헬퍼 `korean-medicine-llm/scripts/quick_eval_sft_varco.py`를 재사용한다.
- adapter는 최종 저장본을 사용한다. 이 파일은 best checkpoint-2100과 동일하다.
- 별도 Markdown 패키지 없이 `IPython.display.Markdown`을 사용한다.
- 모델이 답을 코드 펜스로 감싸면 바깥쪽 펜스만 제거하고, 질문·정답·지연시간·답변을 Markdown 블록으로 표시한다.
- 전체 벤치마크 대신 설진 1건과 안전성 텍스트 1건의 smoke 평가를 먼저 실행한다.

## 검증 기준

1. 노트북 JSON이 유효하다.
2. adapter, config, benchmark 파일이 존재한다.
3. Markdown 파서가 일반 텍스트와 바깥쪽 코드 펜스를 올바르게 처리한다.
4. smoke 평가가 오류 없이 두 결과를 생성하고 `report.json`을 저장한다.

