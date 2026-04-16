"""전역 seed 유틸 — §04a §A.5 재현성 요건.

모든 진입 스크립트(preprocess, tokenizer_extend, cpt_trainer, eval 등)는
`set_global_seed(42)` 를 main() 최상단에서 호출한다.

PYTHONHASHSEED 는 환경변수이므로 **프로세스 기동 이전**에 설정되어야 한다.
set_global_seed 내부에서 값을 변경해도 이미 시작된 파이썬 해시 시드에
영향을 주지 못하므로, 0 이 아닐 경우 assert 로 중단시킨다.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42, *, require_pythonhashseed: bool = True) -> None:
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
    except ImportError:
        pass
