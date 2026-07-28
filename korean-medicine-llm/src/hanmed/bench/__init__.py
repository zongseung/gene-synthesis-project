"""한의학 멀티모달 평가셋(hanmed_bench) — 빌더·채점 모듈.

- build_bench.py : 원천 데이터(설진 SFT val·book_008·species_annotation·tongue_rule_kb)에서
                   5개 트랙 평가 jsonl 을 결정론적으로 생성.
- run_eval.py    : 모델 예측 jsonl 을 트랙별로 채점(멀티라벨 P/R/F1, citation P/R,
                   abstain 적정성, 독초 per-class P/R).
설계 문서: claudedocs/hanmed_benchmark_design.md
"""

# 설진 sign → (카테고리, 한국어 라벨). tongue_rule_kb.json 어휘와 1:1.
SIGN_META = {
    "baitaishe":   ("coating_color",  "백태(白苔)"),
    "huangtaishe": ("coating_color",  "황태(黃苔)"),
    "heitaishe":   ("coating_color",  "흑태(黑苔)"),
    "huataishe":   ("coating_quality", "활태(滑苔)"),
    "botaishe":    ("coating_quality", "박태/무태(薄苔/無苔)"),
    "hongshe":     ("body_color",     "홍설(紅舌)"),
    "zishe":       ("body_color",     "자설(紫舌)"),
    "pangdashe":   ("shape",          "반대설(胖大/腫)"),
    "liewenshe":   ("shape",          "열문설(裂紋)"),
    "hongdianshe": ("shape",          "홍점/망자(紅點/芒刺)"),
    "chihenshe":   ("shape",          "치흔설(齒痕)"),
    "shoushe":     ("shape",          "수설(瘦舌)"),
    "jiankangshe": ("normal",         "정상(健康)"),
}

CATEGORY_KO = {
    "coating_color":   "설태색",
    "coating_quality": "설태질",
    "body_color":      "설질색",
    "shape":           "설형태",
    "normal":          "정상",
}

# 근거(동의보감 원문) 없이 설진 SFT 가 보류 프레이밍을 쓰는 변증 카테고리.
# tongue_rule_kb.json _meta.abstain_categories 와 동일.
ABSTAIN_SIGNS = {"zishe", "chihenshe", "shoushe"}
