"""Bllossom-8B tokenizer quick probe (Solar probe 와 동일 포맷)."""
from transformers import AutoTokenizer

MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
SAMPLE_ZH = "乾鑿度云, 天形出乎乾, 有太易太初太始太素"
SAMPLE_KO = "《건착도》에, 하늘[天]의 형(形)은 건(乾)에서 나오니"
BLOCK = f"<ZH>{SAMPLE_ZH}</ZH>\n<KO>{SAMPLE_KO}</KO>\n\n"

print("=" * 70)
print(f"Loading tokenizer: {MODEL}")
print("=" * 70)
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"vocab size: {len(tok):,}")
print(f"model_max_length: {tok.model_max_length}")
print(f"special tokens: bos={tok.bos_token!r} eos={tok.eos_token!r} pad={tok.pad_token!r}")
print()

# A — vanilla tokenize
print("[A] Vanilla tokenizer — bilingual block")
print(f"  input ({len(BLOCK)} chars): {BLOCK.strip()[:60]!r}...")
toks = tok.tokenize(BLOCK)
ids = tok.encode(BLOCK, add_special_tokens=False)
print(f"  tokens ({len(toks)}): {toks[:30]}")
print(f"  ids    ({len(ids)}):    {ids[:30]}")

# B — tag split (vanilla)
print()
print("[B] How are <ZH>/<KO> tags split (vanilla)?")
for tag in ["<ZH>", "</ZH>", "<KO>", "</KO>"]:
    t = tok.tokenize(tag)
    print(f"  {tag!r:10s} → {len(t)} tokens: {t}")

# C — add special tokens
print()
print("[C] Adding <ZH> </ZH> <KO> </KO> as special tokens")
added = tok.add_special_tokens({"additional_special_tokens": ["<ZH>", "</ZH>", "<KO>", "</KO>"]})
print(f"  added: {added}, new vocab size: {len(tok):,}")
for tag in ["<ZH>", "</ZH>", "<KO>", "</KO>"]:
    t = tok.tokenize(tag)
    i = tok.encode(tag, add_special_tokens=False)
    print(f"  {tag!r:10s} → {len(t)} tokens: {t}  (ids={i})")

# D — re-tokenize with special
print()
print("[D] Re-tokenize bilingual block WITH special tokens")
toks2 = tok.tokenize(BLOCK)
ids2 = tok.encode(BLOCK, add_special_tokens=False)
print(f"  tokens ({len(toks2)}): {toks2[:30]}")
print(f"  ids    ({len(ids2)}):    {ids2[:30]}")

# E — tokens/char
print()
print("[E] tokens/char efficiency (샘플 문장)")
zh_ids = tok.encode(SAMPLE_ZH, add_special_tokens=False)
ko_ids = tok.encode(SAMPLE_KO, add_special_tokens=False)
print(f"  한문 ({len(SAMPLE_ZH)} chars):  {len(zh_ids)} tokens  =  {len(zh_ids)/len(SAMPLE_ZH):.3f} tok/char")
print(f"  한글 ({len(SAMPLE_KO)} chars):  {len(ko_ids)} tokens  =  {len(ko_ids)/len(SAMPLE_KO):.3f} tok/char")

# F — block compression
print()
print("[F] Block length reduction (vanilla vs with special tokens)")
print(f"  vanilla:  {len(ids)} ids")
print(f"  with tag: {len(ids2)} ids  (Δ = {len(ids) - len(ids2)})")

# G — byte fallback check
print()
print("[G] Byte fallback detection (첫 50 한자 토큰)")
zh_toks_50 = tok.tokenize(SAMPLE_ZH)
byte_count = sum(1 for t in zh_toks_50 if "<0x" in str(t))
print(f"  tokens: {zh_toks_50}")
print(f"  byte tokens: {byte_count}/{len(zh_toks_50)} = {byte_count/max(len(zh_toks_50),1)*100:.0f}%")
