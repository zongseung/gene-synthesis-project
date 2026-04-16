"""여러 tokenizer 한문/한글 효율 비교."""
import json
import random
from pathlib import Path
from transformers import AutoTokenizer

CANDIDATES = [
    ("Solar-10.7B-Instruct",       "upstage/SOLAR-10.7B-Instruct-v1.0"),
    ("Qwen2.5-7B-Instruct",         "Qwen/Qwen2.5-7B-Instruct"),
    ("Bllossom-8B (Llama-3 ko)",    "MLP-KTLim/llama-3-Korean-Bllossom-8B"),
    ("Qwen2.5-14B-Instruct",        "Qwen/Qwen2.5-14B-Instruct"),
    ("Mistral-Nemo-Instruct",       "mistralai/Mistral-Nemo-Instruct-2407"),
    ("DeepSeek-V2-Lite",            "deepseek-ai/DeepSeek-V2-Lite"),
    ("EXAONE-3.5-7.8B",             "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"),
]

# 샘플: Core 14 hanmed_zh_only + hanmed_ko_only 에서 10K char 씩
def sample_text(jsonl_path: Path, n_chars: int) -> str:
    chunks = []
    total = 0
    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()
    random.seed(42)
    random.shuffle(lines)
    for line in lines:
        rec = json.loads(line)
        t = rec.get("text", "")
        chunks.append(t)
        total += len(t)
        if total >= n_chars:
            break
    return "".join(chunks)[:n_chars]

BASE = Path("data/cpt")
SAMPLE_CHARS = 10000
zh = sample_text(BASE / "hanmed_zh_only.jsonl", SAMPLE_CHARS)
ko = sample_text(BASE / "hanmed_ko_only.jsonl", SAMPLE_CHARS)
block_sample = f"<ZH>{zh[:500]}</ZH>\n<KO>{ko[:500]}</KO>\n\n"

print(f"샘플 크기: zh={len(zh):,}자 / ko={len(ko):,}자 / block={len(block_sample):,}자\n")

results = []
for name, hf_id in CANDIDATES:
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    except Exception as e:
        print(f"✗ {name:30s}  {hf_id}  load failed: {type(e).__name__}: {str(e)[:60]}")
        continue
    vocab = len(tok)
    zh_ids = tok.encode(zh, add_special_tokens=False)
    ko_ids = tok.encode(ko, add_special_tokens=False)
    block_ids = tok.encode(block_sample, add_special_tokens=False)
    zh_tc = len(zh_ids) / len(zh)
    ko_tc = len(ko_ids) / len(ko)
    block_tc = len(block_ids) / len(block_sample)
    # byte fallback 검출 (토큰 문자열에 '<0x' 포함되는지)
    zh_toks = tok.tokenize(zh[:1000])
    byte_frac = sum(1 for t in zh_toks if "<0x" in str(t)) / max(len(zh_toks), 1)
    results.append((name, vocab, zh_tc, ko_tc, block_tc, byte_frac))
    print(f"✓ {name:30s}  vocab={vocab:>7,}  zh={zh_tc:.3f}  ko={ko_tc:.3f}  block={block_tc:.3f}  byte_fallback={byte_frac*100:.0f}%")

print("\n" + "=" * 90)
print(f"{'Model':32s}  {'vocab':>7s}  {'zh t/c':>7s}  {'ko t/c':>7s}  {'block t/c':>9s}  {'byte%':>6s}")
print("-" * 90)
for name, vocab, zh_tc, ko_tc, block_tc, bf in sorted(results, key=lambda r: r[4]):
    print(f"{name:32s}  {vocab:>7,}  {zh_tc:>7.3f}  {ko_tc:>7.3f}  {block_tc:>9.3f}  {bf*100:>5.0f}%")

# Core 14 HanMed unique token 재추정
print("\n[Core 14 HanMed unique 재추정]")
print(f"{'Model':32s}  {'zh tok':>10s}  {'ko tok':>10s}  {'총 unique':>12s}")
for name, vocab, zh_tc, ko_tc, _, _ in sorted(results, key=lambda r: r[2]*1.20 + r[3]*1.97):
    zh_unique = 1.20e6 * zh_tc
    ko_unique = 1.97e6 * ko_tc
    total = zh_unique + ko_unique
    print(f"{name:32s}  {zh_unique:>10,.0f}  {ko_unique:>10,.0f}  {total:>12,.0f}")
