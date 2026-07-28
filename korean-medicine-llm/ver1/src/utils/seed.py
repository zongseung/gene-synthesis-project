"""전역 seed 유틸 — ver4 r2 §5.5 재현성 요건.

모든 진입 스크립트(preprocess, tokenizer_extend, cpt_trainer, eval,
extract_corpora, mediclassics_orchestrator, expand_facts 등)는
`set_global_seed(42)` 를 main() 최상단에서 호출한다.

PYTHONHASHSEED 는 환경변수이므로 **프로세스 기동 이전**에 설정되어야 한다.
set_global_seed 내부에서 값을 변경해도 이미 시작된 파이썬 해시 시드에
영향을 주지 못하므로, 0 이 아닐 경우 assert 로 중단시킨다.

CUBLAS_WORKSPACE_CONFIG 는 torch.use_deterministic_algorithms(True) 와 함께
cuBLAS 결정성을 보장하기 위해 CUDA context 초기화 이전에 설정되어야 한다.
따라서 이 모듈 import 시점에 한 번 os.environ 에 기본값(`:4096:8`)을 써 둔다
(이미 설정돼 있으면 기존 값을 존중).
"""

from __future__ import annotations

import os
import random
import warnings

# ---------------------------------------------------------------------------
# Module-level CUBLAS_WORKSPACE_CONFIG 세팅 (torch import 이전에 실행).
#   - CUDA context 가 잡힌 뒤에 바꾸면 무효이므로 **import 시점 1회** 만 수행.
#   - 사용자가 외부에서 이미 다른 값을 지정한 경우는 경고 후 그 값을 존중.
# ---------------------------------------------------------------------------
_CUBLAS_RECOMMENDED = ":4096:8"
_CUBLAS_ALLOWED = {":4096:8", ":16:8"}

_existing_cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _existing_cublas_cfg is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_RECOMMENDED
elif _existing_cublas_cfg not in _CUBLAS_ALLOWED:
    warnings.warn(
        f"CUBLAS_WORKSPACE_CONFIG={_existing_cublas_cfg!r} is not one of "
        f"{_CUBLAS_ALLOWED}; keeping existing value but deterministic cuBLAS "
        f"behavior is not guaranteed.",
        stacklevel=2,
    )
# else: 사용자가 이미 허용값을 지정 → 그대로 존중.


def set_global_seed(
    seed: int = 42,
    *,
    deterministic: bool = True,
    require_pythonhashseed: bool = True,
) -> None:
    """전역 seed 및 결정성 옵션 설정.

    Args:
        seed: random / numpy / torch 공통 시드.
        deterministic: True 면 torch.use_deterministic_algorithms(True, warn_only=True)
            와 cudnn.deterministic=True, cudnn.benchmark=False 를 강제.
            False 면 성능 우선 모드 (cudnn benchmark 허용, 알고리즘 비결정 허용).
            기본값 True.
        require_pythonhashseed: True 면 PYTHONHASHSEED=="0" 을 요구하고,
            아니면 RuntimeError.

    시그니처는 키워드 전용 인자만 추가되므로 기존 positional 호출
    (`set_global_seed(42)`) 은 그대로 하위호환.
    """
    if require_pythonhashseed:
        phs = os.environ.get("PYTHONHASHSEED")
        if phs != "0":
            raise RuntimeError(
                f"PYTHONHASHSEED must be '0' for reproducibility, got {phs!r}. "
                f"Rerun with: PYTHONHASHSEED=0 python ..."
            )

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # warn_only=True: 결정성 미지원 op 만난 경우 raise 대신 warning
            # (모델 전역 학습/추론 중 cuDNN fallback 등에서 안전).
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
