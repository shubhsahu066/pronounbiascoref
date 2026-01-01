import json
import re

INPUT_FILE = "unclean_dataset.jsonl"
OUTPUT_FILE = "dataset.jsonl"

# -------------------------
# CONFIG
# -------------------------

# UPDATED: Added reflexives so they count as gendered tokens
GENDERED_PRONOUNS = {
    "he", "she", "him", "her", "his", "hers", "himself", "herself"
}

REFLEXIVE_PRONOUNS = {
    "himself", "herself", "themselves"
}

UNGRAAMMATICAL_PATTERNS = [
    r"\bthey\s+opens\b",
    r"\bthey\s+is\b",
    r"\bthey\s+was\b",
    r"\bthey\s+deserves\b",
    r"\ba\s+[aeiou]\w*",      
    r"\ban\s+[bcdfghjklmnpqrstvwxyz]\w*",
]

# -------------------------
# HELPERS
# -------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def clean_reflexive_spans(text, spans):
    tokens = tokenize(text)
    has_he_she = any(tok in {"he", "she"} for tok in tokens)

    cleaned = []
    for s in spans:
        tok = text[s["start"]:s["end"]].lower()

        if tok == "themselves":
            continue

        # RULE: If he/she exists, drop reflexive mark (keep only he/she)
        if tok in {"himself", "herself"} and has_he_she:
            continue

        cleaned.append(s)

    return cleaned

def reflexive_type(text):
    toks = tokenize(text)
    has_himself = "himself" in toks
    has_herself = "herself" in toks
    has_themselves = "themselves" in toks
    has_he_she = any(t in {"he", "she"} for t in toks)

    if has_themselves: return "THEMSELVES"
    if (has_himself or has_herself) and has_he_she: return "COREFERENT"
    if (has_himself or has_herself) and not has_he_she: return "BARE_REFLEXIVE"
    return "NONE"

def text_contains_gendered(text):
    return any(tok in GENDERED_PRONOUNS for tok in tokenize(text))

def span_contains_gendered(text, spans):
    for s in spans:
        tok = text[s["start"]:s["end"]].lower()
        if tok in GENDERED_PRONOUNS:
            return True
    return False

def count_gendered_pronouns(text):
    toks = tokenize(text)
    return sum(1 for tok in toks if tok in GENDERED_PRONOUNS and tok not in REFLEXIVE_PRONOUNS)

def is_ungrammatical(text):
    t = text.lower()
    return any(re.search(p, t) for p in UNGRAAMMATICAL_PATTERNS)

# -------------------------
# CLEANING PIPELINE
# -------------------------

seen = set()
stats = {
    "total": 0, "duplicates": 0, "reflexive": 0,
    "multi_pronoun": 0, "ungrammatical": 0,
    "span_mismatch": 0, "neutral_leak": 0, "kept": 0
}

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        stats["total"] += 1
        ex = json.loads(line)
        text = ex["text"]
        spans = ex.get("spans", [])
        bias_type = ex.get("bias_type", "")

        # Apply span rules
        spans = clean_reflexive_spans(text, spans)
        ex["spans"] = spans

        # 1. Duplicates
        if text in seen:
            stats["duplicates"] += 1
            continue
        seen.add(text)

        # 2. Reflexive Logic
        rtype = reflexive_type(text)
        if rtype == "THEMSELVES":
            if bias_type != "NEUTRAL":
                stats["neutral_leak"] += 1
                continue

        # 3. Multi pronoun check
        if count_gendered_pronouns(text) > 1:
            stats["multi_pronoun"] += 1
            continue

        # 4. Grammar check
        if is_ungrammatical(text):
            stats["ungrammatical"] += 1
            continue

        has_gendered_token = text_contains_gendered(text)
        has_gendered_span = span_contains_gendered(text, spans)

        # 5. Span Mismatch (Token exists but not marked)
        if has_gendered_token and not has_gendered_span:
            # Exception: If it is a NAMED reflexive (Bare reflexive + Neutral), allow it.
            if rtype == "BARE_REFLEXIVE" and bias_type == "NEUTRAL":
                pass 
            else:
                stats["span_mismatch"] += 1
                continue

        # 6. Neutral Leak (Marked Neutral but has gender)
        if bias_type == "NEUTRAL" and has_gendered_token:
            # RULE: Allow "John hurt himself" (Bare Reflexive + Neutral) to pass
            if rtype == "BARE_REFLEXIVE":
                pass
            else:
                stats["neutral_leak"] += 1
                continue

        fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
        stats["kept"] += 1

print("\nCLEANING COMPLETE\n")
for k, v in stats.items():
    print(f"{k:18}: {v}")