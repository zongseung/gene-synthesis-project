from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import yaml

import os

# symlink 로부터 실행돼도 (experiments/dongui_bogam/scripts/build_sft_qa.py)
# 해당 경로 기준으로 EXP_ROOT 를 잡도록 resolve() 대신 원래 __file__ 을 사용.
_SCRIPT_FILE = Path(__file__)
EXP_ROOT_ENV = os.environ.get("HANMED_EXP_ROOT")
if EXP_ROOT_ENV:
    EXP_ROOT = Path(EXP_ROOT_ENV)
else:
    # 실제 파일(project-root/scripts) 경로이면 experiments/dongui_bogam 로 대체
    direct = _SCRIPT_FILE.parents[1]
    if direct.name == "dongui_bogam":
        EXP_ROOT = direct
    else:
        EXP_ROOT = _SCRIPT_FILE.resolve().parents[1].parent / "experiments" / "dongui_bogam"
ROOT = EXP_ROOT.parents[1]
RAW_DIR = EXP_ROOT / "raw"
FACTSHEET_PATH = EXP_ROOT / "core_factsheet.yaml"
DEFAULT_WHITELIST = EXP_ROOT / "data" / "sft" / "entity_whitelist.yaml"
FALLBACK_WHITELIST = ROOT / "data" / "sft" / "entity_whitelist.yaml"
DEFAULT_TOKENIZER = ROOT / "data" / "tokenizer" / "hanmed_bllossom_ext"
DEFAULT_SEEDS = EXP_ROOT / "data" / "sft" / "phaseB_qa_seeds.yaml"
DEFAULT_FULL_SYSTEM_PROMPT = (
    "당신은 동의보감(book_008) 문헌 연구 보조 AI 입니다. "
    "질문이 가리키는 record의 위치를 설명하고, 현대 한국어로 정리하며, "
    "문헌적 의미를 짧게 덧붙입니다. 개인 증상에 대한 진단·처방·복용 지시는 하지 않습니다."
)

YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
BOOK_RE = re.compile(r"[《『]([^》』]+)[》』]")
NAME_RE = re.compile(r"[가-힣]{2,5}(?:\([一-鿿]{1,6}\))?|[一-鿿]{2,6}")
LIST_LEVELS = {"Z2", "TT", "PP"}
TITLEISH_LEVELS = {"AA", "BB", "CC", "DD", "EE", "CH", "DH", "DK", "DP", "EP", "CP"}
DISEASE_PATH_KEYWORDS = (
    "病",
    "症",
    "風",
    "寒",
    "熱",
    "吐",
    "瀉",
    "痢",
    "痛",
    "咳",
    "嗽",
    "喘",
    "痰",
    "瘡",
    "瘍",
    "癰",
    "疽",
    "瘧",
    "勞",
    "霍亂",
    "痔",
    "淋",
    "帶下",
    "胎",
    "産",
    "眼",
    "耳",
    "鼻",
    "咽",
    "頭",
    "腹",
)
DISEASE_KO_KEYWORDS = (
    "병",
    "증",
    "통",
    "중풍",
    "두통",
    "해수",
    "기침",
    "허로",
    "창",
    "담음",
    "설사",
    "토혈",
    "토",
    "열",
    "한",
    "풍",
    "복통",
    "임신",
    "출산",
    "눈병",
    "귀병",
    "비병",
    "인후",
)

QUESTION_VARIATIONS = [
    ("누구인가요?", "누가 편찬했나요?"),
    ("무엇인가요?", "어떻게 볼 수 있나요?"),
    ("알려주세요.", "설명해 주세요."),
]

PARAPHRASE_REWRITES = [
    ("편찬한 의서입니다.", "편찬된 대표 의서입니다."),
    ("알 수 있습니다.", "확인할 수 있습니다."),
    ("반드시 전문의 진료를 받으시기 바랍니다.", "우선 전문의 진료를 받아 보시기 바랍니다."),
    ("문헌 해설이 필요하시면", "문헌 소개가 필요하시면"),
    ("설명합니다.", "해설합니다."),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ver5 SFT QA data for dongui_bogam experiment.")
    p.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    p.add_argument("--mode", choices=["template", "paraphrase", "full_corpus"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--validation-out", type=Path, default=None)
    p.add_argument("--rejection-log", type=Path, default=None)
    p.add_argument("--whitelist", type=Path, default=None)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--record-types", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-catalog-out", type=Path, default=None)
    p.add_argument("--stats-out", type=Path, default=None)
    p.add_argument("--auto-source-top-k", type=int, default=4)
    p.add_argument("--max-question-chars", type=int, default=220)
    p.add_argument("--max-answer-chars", type=int, default=900)
    p.add_argument("--min-answer-tokens", type=int, default=32)
    return p.parse_args()


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_book_facts(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    for book in data["books"]:
        if int(book["book_id"]) == 8:
            return book
    raise KeyError("book_id=8 not found in factsheet")


def load_record_index(raw_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for vol_path in sorted(raw_dir.glob("vol_*.jsonl")):
        volume = vol_path.stem
        with vol_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ref = f"book_008/{volume}/seq_{row['content_seq']}"
                index[ref] = row
    return index


def normalize_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip().lower()


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = text.replace("{n}", "\n").replace("#", " ")
    normalized = normalized.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def split_path_parts(up_path_nm: str | None) -> list[str]:
    return [clean_text(part) for part in (up_path_nm or "").split(">") if clean_text(part)]


def clip_text(text: str | None, max_chars: int) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rstrip()
    if " " in clipped[-30:]:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped + " ...(중략)"


def build_source_catalog(record_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for ref, row in sorted(record_index.items()):
        ko = (row.get("trans_ko") or "").strip()
        zh = (row.get("original") or "").strip()
        up = (row.get("up_path_nm") or "").strip()
        search_blob = normalize_search_text(" ".join(part for part in [ko, zh, up] if part))
        catalog.append(
            {
                "ref": ref,
                "volume_id": row.get("volume_id"),
                "content_seq": row.get("content_seq"),
                "content_level": row.get("content_level"),
                "up_path_nm": row.get("up_path_nm"),
                "text_ko": ko,
                "text_zh": zh,
                "search_blob": search_blob,
            }
        )
    return catalog


def seed_search_terms(seed: dict[str, Any], question: str, book: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    terms.extend(seed.get("key_entities", []))
    terms.extend(seed.get("factsheet_keys", []))
    terms.extend(re.findall(r"[가-힣A-Za-z0-9]{2,}", question))
    if seed.get("subcat") == "author_fact":
        terms.extend(["허준", "許浚", "양평군", "어의"])
    if seed.get("subcat") == "king_command":
        terms.extend(["선조", "宣祖", "1610", "1613", "하교", "편찬"])
    if seed.get("category") == "medical_literature":
        terms.extend(["동의보감", "문", "병", "기", "오장"])
    # factsheet field values도 검색어에 포함
    for key in seed.get("factsheet_keys", []):
        value = book.get(key)
        if isinstance(value, list):
            terms.extend(str(v) for v in value)
        elif value not in (None, ""):
            terms.append(str(value))
    deduped: list[str] = []
    seen = set()
    for term in terms:
        norm = normalize_search_text(str(term))
        if len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped


def auto_discover_source_records(
    seed: dict[str, Any],
    question: str,
    record_index: dict[str, dict[str, Any]],
    catalog: list[dict[str, Any]],
    book: dict[str, Any],
    top_k: int,
) -> list[str]:
    explicit = set(seed.get("source_records", []))
    terms = seed_search_terms(seed, question, book)
    scored: list[tuple[int, str]] = []
    for item in catalog:
        score = 0
        blob = item["search_blob"]
        if not blob:
            continue
        for term in terms:
            if term in blob:
                score += 3 if term in normalize_search_text(item["text_ko"]) else 1
        # 제목/서문 계열은 기본 가산점
        if item["content_level"] in {"AA", "OO", "XX"}:
            score += 1
        if score > 0:
            scored.append((score, item["ref"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    discovered: list[str] = []
    for _, ref in scored:
        if ref in explicit or ref in discovered:
            continue
        discovered.append(ref)
        if len(discovered) >= top_k:
            break
    return list(explicit) + discovered


def load_entity_policy(path: Path | None) -> tuple[set[str], set[str]]:
    chosen = path or (DEFAULT_WHITELIST if DEFAULT_WHITELIST.exists() else FALLBACK_WHITELIST)
    data = load_yaml(chosen)
    allow: set[str] = set()
    deny: set[str] = set()
    for group in data.get("allow", {}).values():
        for item in group:
            if isinstance(item, dict):
                allow.add(item["name"])
                if item.get("hanja"):
                    allow.add(item["hanja"])
    for group in data.get("deny", {}).values():
        for item in group:
            if isinstance(item, dict):
                deny.add(item["name"])
                if item.get("hanja"):
                    deny.add(item["hanja"])
    return allow, deny


def build_token_counter(tokenizer_path: Path):
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(tokenizer_path))

        def count(text: str) -> int:
            return len(tok(text, add_special_tokens=False).input_ids)

        return count
    except Exception:
        return lambda text: max(len(text.split()), len(text) // 3)


def extract_source_quotes(source_records: list[str], record_index: dict[str, dict[str, Any]]) -> list[str]:
    quotes: list[str] = []
    for ref in source_records:
        record = record_index.get(ref)
        if not record:
            continue
        quote = (record.get("trans_ko") or record.get("original") or "").strip()
        if quote:
            quotes.append(quote.replace("\r", " ").replace("\n", " ").strip())
    return quotes


def choose_question(seed: dict[str, Any], idx: int) -> str:
    templates = seed.get("question_templates") or [seed.get("user") or seed["id"]]
    return templates[idx % len(templates)]


def maybe_variation(question: str, idx: int) -> str:
    varied = question
    for src, dst in QUESTION_VARIATIONS:
        if idx % 2 and src in varied:
            varied = varied.replace(src, dst)
            break
    return varied


def compose_answer(category: str, subcat: str, question: str, seed: dict[str, Any], book: dict[str, Any], quotes: list[str]) -> str:
    quote = quotes[0] if quotes else "관련 원문 기록"
    second = quotes[1] if len(quotes) > 1 else quote
    author = book["author"]
    author_hanja = book["author_hanja"]
    reign = book["reign"]
    reign_hanja = book["reign_hanja"]
    compiled_year = book["compiled_year"]
    published_year = book["published_year"]
    pyeon_list = "내경편·외형편·잡병편·탕액편·침구편"
    topic = seed.get("key_entities", ["동의보감"])[0]

    if category.startswith("in_scope") and subcat == "author_fact":
        return (
            f"동의보감은 조선 중기의 어의 {author}({author_hanja})이 편찬한 의서입니다. "
            f"서문에는 \"{quote}\"라고 적혀 있어 {author}이 {reign}({reign_hanja})의 하교를 받들어 저술했음을 확인할 수 있습니다. "
            f"편찬은 {compiled_year}년의 왕명으로 본격화되었고, {published_year}년에 간행되었습니다. "
            f"동의보감은 {pyeon_list}의 5편 체제로 구성된 종합의서입니다. [출처: 동의보감 내경편 권1 서문]"
        )
    if category.startswith("in_scope") and subcat == "king_command":
        return (
            f"동의보감 편찬은 조선 {reign}({reign_hanja})의 명으로 추진되었습니다. "
            f"서문 기록 \"{quote}\"는 왕명이 분명히 내려졌음을 보여 줍니다. "
            f"{reign}은 전란 이후 백성을 구제할 수 있는 의서를 만들기 위해 {author}에게 편찬을 맡겼고, "
            f"그 결과 {compiled_year}년 완성, {published_year}년 간행이라는 흐름이 정리됩니다. [출처: 동의보감 내경편 권1 서문]"
        )
    if category == "out_of_scope":
        entity = topic
        return (
            f"{entity}는 본 실험 모델의 학습 범위인 동의보감(book_008) 단권에 포함되지 않습니다. "
            f"따라서 해당 저작이나 개념의 저자·연혁에 대해 정확한 답을 드릴 수 없습니다. "
            f"이 모델은 동의보감의 서지 정보와 본문 해설에 초점을 맞추고 있으므로, "
            f"{entity}는 한국민족문화대백과사전이나 원문 해제처럼 별도 신뢰 출처를 확인해 주시기 바랍니다."
        )
    if category == "safety_refusal":
        symptom = question.replace("제가", "").replace("저는", "").strip()
        safe_name = seed.get("subcat", "증상")
        return (
            "본 모델은 한의학 고전 문헌 연구 보조 AI로, 개인 증상에 대한 진단이나 구체적인 약물 처방을 제공할 수 없습니다. "
            f"{symptom}와 같은 상황은 여러 원인이 가능하므로 우선 전문의 진료를 받아야 합니다. "
            "증상이 급격하거나 심해지면 119 또는 응급실을 우선 고려해야 합니다. "
            f"동의보감에는 {safe_name}와 유사한 주제 설명이 있을 수 있으나, 이는 17세기 문헌 해설일 뿐 현대 진단과 처방을 대체하지 않습니다. "
            "문헌 소개가 필요하시면 개인 증상 대신 고전 텍스트 기준으로 다시 질문해 주세요. [전문의 상담 필수]"
        )
    if category == "medical_literature":
        # 안전 원칙 (모든 subcat 공통):
        #   - 처방명·용량 언급 금지: 원문 quote 는 literal 그대로 내보내되,
        #     해설 문장(우리가 composed 한 부분)에서는 처방·용량을 생성하지 않음
        #   - "17세기 문헌" 맥락 명시
        #   - 끝에 반드시 전문의 상담 고지
        disclaimer = (
            "위 인용은 17세기 조선(1610년 완성·1613년 간행)에 편찬된 동의보감 원문의 "
            "해당 대목을 소개한 것이며, 현대 의학의 진단이나 처방을 대체하지 않습니다. "
            "본 해설은 개인 증상에 대한 약물·용량·처방 제안을 포함하지 않으며, "
            "실제 치료가 필요하시면 반드시 전문의와 상담해 주십시오."
        )

        if subcat == "symptom_literature":
            # 병증의 원인·분류의 문헌적 해설
            return (
                f"{topic}에 대한 동의보감의 해설은 17세기 문헌 기준의 **문헌적 설명**입니다.\n\n"
                f"동의보감 원문에는 다음과 같이 기록되어 있습니다:\n"
                f"\"{quote}\"\n\n"
                f"이어지는 대목에서는 다음과 같이 덧붙입니다:\n"
                f"\"{second}\"\n\n"
                f"즉, 동의보감은 {topic}의 원인을 당시 장부(臟腑)·병인(病因) 체계 속에서 "
                f"해설하며, 이는 현대 의학의 진단 기준과 다른 17세기의 설명 틀입니다.\n\n"
                f"{disclaimer} [출처: 동의보감 관련 권·문]"
            )
        if subcat == "concept_explanation":
            # 한의학 추상 개념(정·기·신·혈·오장육부)의 문헌적 정의
            return (
                f"동의보감에서 \"{topic}\"은(는) 한의학 고전의 추상 개념으로 설명됩니다. "
                f"본 해설은 17세기 동의보감 원문이 이 개념을 어떻게 기술했는지 소개하는 것이며, "
                f"현대 생의학 개념과는 다릅니다.\n\n"
                f"동의보감은 다음과 같이 기술합니다:\n"
                f"\"{quote}\"\n\n"
                f"또 다른 대목에서는 다음과 같이 말합니다:\n"
                f"\"{second}\"\n\n"
                f"이 인용들은 {topic}이(가) 동의보감 체계 안에서 어떻게 자리매김하는지를 "
                f"보여 주는 문헌적 기술입니다.\n\n"
                f"{disclaimer} [출처: 동의보감 내경편]"
            )
        if subcat == "pyeon_content":
            # 편별 총목 구성 해설
            return (
                f"동의보감 {topic}은(는) 하위 항목이 다음과 같이 배열되어 있습니다:\n"
                f"\"{quote}\"\n"
                f"\"{second}\"\n\n"
                f"즉, {topic}은(는) 위와 같이 세부 병증·부위·개념을 체계적으로 분류하여 "
                f"하나의 편으로 묶은 것입니다.\n\n"
                f"{disclaimer} [출처: 동의보감 내경편 권1 동의보감 총목]"
            )
        # 기본 — 이전 행동 유지 (pyeon_content / book_structure / classical_reference fallback)
        return (
            f"동의보감에는 {topic}와(과) 관련된 설명이 실려 있으며, 관련 기록에는 "
            f"\"{quote}\"와 같은 표현이 보입니다. 또 다른 문맥 \"{second}\"를 함께 보면, "
            f"해당 주제가 단순 나열이 아니라 장부와 병인 체계 속에서 해설되고 있음을 "
            f"알 수 있습니다.\n\n"
            f"{disclaimer} [출처: 동의보감 관련 권차]"
        )
    return (
        f"동의보감 관련 질문에 대해 핵심 사실만 정리하면 다음과 같습니다. \"{quote}\"라는 기록을 중심으로 보면 "
        f"{author}·{reign}·{compiled_year}·{published_year} 같은 서지 정보가 연결되며, "
        "질문에 대한 답은 이 범위 안에서만 설명할 수 있습니다. [출처: 동의보감]"
    )


def build_messages(system_prompt: str, question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def validate_entities(answer: str, allow: set[str], deny: set[str]) -> dict[str, Any]:
    names = {m.group(0) for m in NAME_RE.finditer(answer)}
    deny_hits = sorted(name for name in names if name in deny)
    unknown = sorted(name for name in names if name not in allow and name not in deny)
    return {"passed": not deny_hits, "blacklist_hits": deny_hits, "unknown_names": unknown}


def fact_diff_check(original: str, candidate: str) -> dict[str, Any]:
    orig_years = set(YEAR_RE.findall(original))
    cand_years = set(YEAR_RE.findall(candidate))
    orig_books = {match.group(1) for match in BOOK_RE.finditer(original)}
    cand_books = {match.group(1) for match in BOOK_RE.finditer(candidate)}
    new_years = sorted(cand_years - orig_years)
    new_books = sorted(cand_books - orig_books)
    return {"passed": not (new_years or new_books), "new_years": new_years, "new_books": new_books}


def trigram_jaccard(a: str, b: str) -> float:
    def grams(text: str) -> set[str]:
        if len(text) < 3:
            return {text}
        return {text[i : i + 3] for i in range(len(text) - 2)}

    ga, gb = grams(a), grams(b)
    return len(ga & gb) / max(len(ga | gb), 1)


def min_tokens_for(category: str) -> int:
    if category == "safety_refusal":
        return 100
    if category == "medical_literature":
        return 180
    if category == "out_of_scope":
        return 100
    return 80


def is_titleish_record(row: dict[str, Any]) -> bool:
    level = row.get("content_level") or ""
    if level not in TITLEISH_LEVELS:
        return False
    content = clean_text(row.get("trans_ko") or row.get("original"))
    if not content:
        return False
    if len(content) > 64:
        return False
    return not any(token in content for token in [".", "!", "?", '"'])


def is_symptom_record(row: dict[str, Any], parts: list[str]) -> bool:
    joined_path = " ".join(parts)
    ko = clean_text(row.get("trans_ko"))
    orig = clean_text(row.get("original"))
    blob = " ".join([joined_path, ko, orig])
    if parts and "雜病篇" in parts[0]:
        if not any(skip in joined_path for skip in ["天地運氣", "運氣", "脉訣", "六十年客氣旁通圖"]):
            return True
    if any(keyword in blob for keyword in DISEASE_PATH_KEYWORDS):
        return True
    if any(keyword in ko for keyword in DISEASE_KO_KEYWORDS):
        return True
    return False


def classify_record_type(row: dict[str, Any]) -> str:
    parts = split_path_parts(row.get("up_path_nm"))
    level = row.get("content_level") or ""
    ko = clean_text(row.get("trans_ko"))
    orig = clean_text(row.get("original"))
    path_blob = " ".join(parts)

    if level == "OO" or any("序" in part for part in parts) or "서문" in ko:
        return "서문"
    if any("總目" in part for part in parts) or "총목" in ko or level in LIST_LEVELS:
        return "총목"
    if level in {"AA", "BB"} or is_titleish_record(row):
        return "편명"
    if "集例" in path_blob or "養生" in path_blob or "鍼灸篇" in path_blob or "湯液篇" in path_blob:
        return "본문 설명"
    if is_symptom_record(row, parts):
        return "병증 설명"
    if len(parts) >= 2 and any(token in parts[-1] for token in ["圖", "法", "論"]):
        return "본문 설명"
    if level == "XX":
        return "서문" if parts and parts[0].startswith("內景篇卷之一") else "본문 설명"
    return "본문 설명"


def record_topic(row: dict[str, Any]) -> str:
    parts = split_path_parts(row.get("up_path_nm"))
    content = clean_text(row.get("trans_ko") or row.get("original"))
    if parts:
        leaf = parts[-1]
        if leaf not in {"東醫寶鑑序", "東醫寶鑑 總目"}:
            return leaf
    return content or f"seq_{row.get('content_seq')}"


def record_path_label(row: dict[str, Any]) -> str:
    parts = split_path_parts(row.get("up_path_nm"))
    if not parts:
        return f"book_008 / vol_{int(row.get('volume_id', 0)):02d}"
    if len(parts) <= 4:
        return " > ".join(parts)
    return " > ".join([parts[0], parts[-2], parts[-1]])


def explain_record_context(record_type: str, row: dict[str, Any]) -> str:
    parts = split_path_parts(row.get("up_path_nm"))
    root = parts[0] if parts else f"권{row.get('volume_id')}"
    topic = record_topic(row)
    path_label = record_path_label(row)
    if record_type == "서문":
        return (
            f"이 대목은 동의보감 서문 계열 기록으로, 상위 경로는 {path_label}입니다. "
            f"편찬 배경, 간행 목적, 책의 의미를 밝히는 앞부분 자료로 읽으면 됩니다. "
            "문헌 연구용 정리이며 개인 진단이나 처방 지침으로 쓰면 안 됩니다."
        )
    if record_type == "편명":
        return (
            f"이 대목은 동의보감의 편명·항목명 역할을 하는 표제입니다. 상위 경로는 {path_label}이고, "
            f"{root} 안에서 {topic} 항목이 어떤 위치에 놓이는지 보여 줍니다. "
            "뒤이어 나오는 본문을 읽기 위한 분류 표식으로 이해하면 됩니다."
        )
    if record_type == "총목":
        return (
            f"이 대목은 동의보감 총목·도표·목차 계열 기록입니다. 상위 경로는 {path_label}이며, "
            f"{root} 안에서 하위 항목이 어떻게 배열되는지 드러냅니다. "
            "책의 분류 체계를 파악하는 자료로 보아야 합니다."
        )
    if record_type == "병증 설명":
        return (
            f"이 대목은 {path_label}에 속한 병증 설명입니다. 중심 항목은 {topic}이며, "
            "동의보감은 이를 당시의 장부·병인·치법 체계 안에서 서술합니다. "
            "현대 의료 조언이 아니라 17세기 문헌 설명으로 읽어야 합니다."
        )
    return (
        f"이 대목은 {path_label}에 속한 본문 설명입니다. 중심 항목은 {topic}이며, "
        "동의보감이 해당 개념·약재·경혈·방법을 문헌적으로 어떻게 기록하는지 보여 줍니다. "
        "현대 의료 조언이 아니라 고전 텍스트 해설입니다."
    )


def compose_full_corpus_question(record_type: str, row: dict[str, Any], max_chars: int) -> str:
    path_label = record_path_label(row)
    topic = record_topic(row)
    source_excerpt = clip_text(row.get("original") or row.get("trans_ko"), max_chars)
    prompts = {
        "서문": "다음 동의보감 서문 대목을 현대 한국어로 풀고, 편찬 맥락에서 어떤 의미인지 설명해 주세요.",
        "편명": "다음 동의보감 표제를 보고, 이 항목이 책의 어디에 놓이는지 정리해 주세요.",
        "총목": "다음 동의보감 총목·목차 기록을 풀어서 쓰고, 분류 체계에서 어떤 역할인지 설명해 주세요.",
        "본문 설명": "다음 동의보감 본문 기록을 현대 한국어로 정리하고, 문헌상 핵심을 설명해 주세요.",
        "병증 설명": "다음 동의보감 병증 관련 기록을 현대 한국어로 정리하고, 어떤 증상·치법 문맥인지 문헌적으로 설명해 주세요.",
    }
    return (
        f"{prompts[record_type]}\n"
        f"유형: {record_type}\n"
        f"항목: {topic}\n"
        f"경로: {path_label}\n"
        f"원문 발췌: {source_excerpt}"
    )


def compose_full_corpus_answer(record_type: str, row: dict[str, Any], max_chars: int) -> str:
    translated = clip_text(row.get("trans_ko") or row.get("original"), max_chars)
    label = "정리" if record_type in {"편명", "총목"} else "현대 한국어"
    context = explain_record_context(record_type, row)
    return f"{label}:\n{translated}\n\n해설:\n{context}"


def min_tokens_for_record_type(record_type: str, floor: int) -> int:
    if record_type in {"편명", "총목"}:
        return max(18, floor // 2)
    return floor


def render_full_corpus_rows(
    record_index: dict[str, dict[str, Any]],
    system_prompt: str,
    token_count,
    record_types_filter: set[str] | None,
    limit: int | None,
    max_question_chars: int,
    max_answer_chars: int,
    min_answer_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "record_count": len(record_index),
        "accepted": 0,
        "rejected": 0,
        "by_record_type": {},
        "by_content_level": {},
    }

    for ref, row in sorted(record_index.items()):
        record_type = classify_record_type(row)
        if record_types_filter and record_type not in record_types_filter:
            continue

        question = compose_full_corpus_question(record_type, row, max_question_chars)
        answer = compose_full_corpus_answer(record_type, row, max_answer_chars)
        n_tokens = token_count(answer)
        source_text = clean_text(row.get("trans_ko") or row.get("original"))
        validation = {
            "id": ref.replace("/", "_"),
            "record_ref": ref,
            "record_type": record_type,
            "content_level": row.get("content_level"),
            "answer_tokens": n_tokens,
            "source_chars": len(source_text),
        }
        validations.append(validation)
        row_payload = {
            "id": ref.replace("/", "_"),
            "category": "medical_literature",
            "subcat": record_type,
            "record_type": record_type,
            "record_ref": ref,
            "content_level": row.get("content_level"),
            "question": question,
            "assistant": answer,
            "messages": build_messages(system_prompt, question, answer),
            "source_records": [ref],
            "path": split_path_parts(row.get("up_path_nm")),
            "answer_tokens": n_tokens,
        }
        passed = bool(source_text) and n_tokens >= min_tokens_for_record_type(record_type, min_answer_tokens)
        if passed:
            rows.append(row_payload)
            stats["accepted"] += 1
            stats["by_record_type"][record_type] = stats["by_record_type"].get(record_type, 0) + 1
            level = row.get("content_level") or "UNKNOWN"
            stats["by_content_level"][level] = stats["by_content_level"].get(level, 0) + 1
        else:
            rejects.append({**row_payload, "reject_reason": "full_corpus_validation_failed"})
            stats["rejected"] += 1
        if limit and len(rows) >= limit:
            break

    return rows, rejects, validations, stats


def render_template_rows(
    seed_doc: dict[str, Any],
    record_index: dict[str, dict[str, Any]],
    book: dict[str, Any],
    allow: set[str],
    deny: set[str],
    token_count,
    categories_filter: set[str] | None,
    limit: int | None,
    auto_source_top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    system_prompt = seed_doc["system_prompt"].strip()
    source_catalog = build_source_catalog(record_index)

    for category_name, payload in seed_doc["categories"].items():
        if categories_filter and category_name not in categories_filter:
            continue
        for idx, seed in enumerate(payload.get("seeds", [])):
            question = maybe_variation(choose_question(seed, idx), idx)
            source_records = auto_discover_source_records(
                seed=seed,
                question=question,
                record_index=record_index,
                catalog=source_catalog,
                book=book,
                top_k=auto_source_top_k,
            )
            quotes = extract_source_quotes(source_records, record_index)
            answer = compose_answer(category_name, seed.get("subcat", ""), question, seed, book, quotes)
            n_tokens = token_count(answer)
            entity_check = validate_entities(answer, allow, deny)
            validation = {
                "id": seed["id"],
                "category": category_name,
                "tokens": n_tokens,
                "source_records_used": source_records,
                **entity_check,
            }
            passed = entity_check["passed"] and n_tokens >= min_tokens_for(category_name)
            validations.append(validation)
            row = {
                "id": seed["id"],
                "category": category_name,
                "subcat": seed.get("subcat"),
                "question": question,
                "assistant": answer,
                "messages": build_messages(system_prompt, question, answer),
                "source_records": source_records,
                "explicit_source_records": seed.get("source_records", []),
                "factsheet_keys": seed.get("factsheet_keys", []),
                "answer_tokens": n_tokens,
                "entity_validation": entity_check,
            }
            if passed:
                rows.append(row)
            else:
                rejects.append({**row, "reject_reason": "validation_failed"})
            if limit and len(rows) >= limit:
                return rows, rejects, validations
    return rows, rejects, validations


def paraphrase_question(text: str, idx: int) -> str:
    out = maybe_variation(text, idx + 1)
    if "알려주세요" in out and idx % 2 == 0:
        out = out.replace("알려주세요", "설명해 주세요")
    return out


def paraphrase_answer(text: str, idx: int) -> str:
    out = text
    for src, dst in PARAPHRASE_REWRITES:
        if src in out and idx % 2 == 0:
            out = out.replace(src, dst)
            break
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", out) if p.strip()]
    if len(parts) >= 3 and idx % 2 == 1:
        parts = [parts[1], parts[0], *parts[2:]]
    return " ".join(parts)


def render_paraphrase_rows(
    template_rows: list[dict[str, Any]],
    allow: set[str],
    deny: set[str],
    token_count,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []

    for idx, base in enumerate(template_rows):
        question = paraphrase_question(base["question"], idx)
        answer = paraphrase_answer(base["assistant"], idx)
        n_tokens = token_count(answer)
        entity_check = validate_entities(answer, allow, deny)
        fact_check = fact_diff_check(base["assistant"], answer)
        diversity = trigram_jaccard(base["assistant"], answer)
        validation = {
            "id": f"{base['id']}-para",
            "category": base["category"],
            "tokens": n_tokens,
            "jaccard": diversity,
            **entity_check,
            **fact_check,
        }
        validations.append(validation)
        row = {
            **base,
            "id": f"{base['id']}-para",
            "question": question,
            "assistant": answer,
            "messages": build_messages(base["messages"][0]["content"], question, answer),
            "answer_tokens": n_tokens,
            "entity_validation": entity_check,
            "fact_diff": fact_check,
            "source_id": base["id"],
        }
        passed = (
            entity_check["passed"]
            and fact_check["passed"]
            and n_tokens >= min_tokens_for(base["category"])
            and diversity < 0.95
        )
        if passed:
            rows.append(row)
        else:
            rejects.append({**row, "reject_reason": "paraphrase_validation_failed"})
        if limit and len(rows) >= limit:
            return rows, rejects, validations
    return rows, rejects, validations


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    allow, deny = load_entity_policy(args.whitelist)
    token_count = build_token_counter(args.tokenizer)
    stats = None

    if args.mode == "template":
        if not args.seeds.exists():
            raise FileNotFoundError(f"seed yaml not found: {args.seeds}")
        seed_doc = load_yaml(args.seeds)
        record_index = load_record_index(RAW_DIR)
        book = load_book_facts(FACTSHEET_PATH)
        rows, rejects, validations = render_template_rows(
            seed_doc=seed_doc,
            record_index=record_index,
            book=book,
            allow=allow,
            deny=deny,
            token_count=token_count,
            categories_filter=set(args.categories) if args.categories else None,
            limit=args.limit,
            auto_source_top_k=args.auto_source_top_k,
        )
        if args.source_catalog_out:
            args.source_catalog_out.parent.mkdir(parents=True, exist_ok=True)
            with args.source_catalog_out.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "raw_dir": str(RAW_DIR.resolve()),
                        "record_count": len(record_index),
                        "catalog": build_source_catalog(record_index),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
    elif args.mode == "paraphrase":
        if not args.seeds.exists():
            raise FileNotFoundError(f"template jsonl not found: {args.seeds}")
        template_rows = load_jsonl(args.seeds)
        rows, rejects, validations = render_paraphrase_rows(
            template_rows=template_rows,
            allow=allow,
            deny=deny,
            token_count=token_count,
            limit=args.limit,
        )
        stats = None
    else:
        record_index = load_record_index(RAW_DIR)
        system_prompt = DEFAULT_FULL_SYSTEM_PROMPT
        if args.seeds and args.seeds.exists():
            seed_doc = load_yaml(args.seeds)
            system_prompt = seed_doc.get("system_prompt", system_prompt).strip()
        rows, rejects, validations, stats = render_full_corpus_rows(
            record_index=record_index,
            system_prompt=system_prompt,
            token_count=token_count,
            record_types_filter=set(args.record_types) if args.record_types else None,
            limit=args.limit,
            max_question_chars=args.max_question_chars,
            max_answer_chars=args.max_answer_chars,
            min_answer_tokens=args.min_answer_tokens,
        )
        if args.source_catalog_out:
            args.source_catalog_out.parent.mkdir(parents=True, exist_ok=True)
            catalog = build_source_catalog(record_index)
            for item in catalog:
                item["record_type"] = classify_record_type(record_index[item["ref"]])
            with args.source_catalog_out.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "raw_dir": str(RAW_DIR.resolve()),
                        "record_count": len(record_index),
                        "catalog": catalog,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    write_jsonl(args.out, rows)
    if args.rejection_log:
        write_jsonl(args.rejection_log, rejects)
    if args.validation_out:
        args.validation_out.parent.mkdir(parents=True, exist_ok=True)
        with args.validation_out.open("w", encoding="utf-8") as f:
            json.dump({"rows": validations, "accepted": len(rows), "rejected": len(rejects)}, f, ensure_ascii=False, indent=2)
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        payload = stats or {"accepted": len(rows), "rejected": len(rejects)}
        with args.stats_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[build_sft_qa] mode={args.mode} accepted={len(rows)} rejected={len(rejects)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
