"""stage1_llm·stage2_vlm 공용 헬퍼."""
from __future__ import annotations

import hashlib


def _pick(opts, key: str):
    """어투/템플릿 결정론적 회전. 내장 hash() 는 PYTHONHASHSEED 에 좌우돼
    실행마다 문구가 바뀌므로(재현 불가) 쓰지 않고 md5 로 고정한다."""
    return opts[hashlib.md5(key.encode("utf-8")).digest()[0] % len(opts)]
