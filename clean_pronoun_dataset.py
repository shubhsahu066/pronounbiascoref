import json
import re

INPUT_FILE = "pronoun_dataset.jsonl"
OUTPUT_FILE = "pronoun_dataset_clean.jsonl"

# -------------------------
# CONFIG
# -------------------------

GENDERED_PRONOUNS = {
    "he", "she", "him", "her", "his", "hers"
}

REFLEXIVE_PRONOUNS = {
    "himself", "herself", "themselves"
}

UNGRAAMMATICAL_PATTERNS = [
    r"\bthey\s+opens\b",
    r"\bthey\s+is\b",
    r"\bthey\s+was\b",
    r"\bthey\s+deserves\b",
    r"\ba\s+[aeiou]\w*",      # "a employee"
    r"\ban\s+[bcdfghjklmnpqrstvwxyz]\w*",
]

# -------------------------
# HELPERS
# -------------------------

def clean_reflexive_spans(text, spans):
    tokens = tokenize(text)

    has_he_she = any(tok in {"he", "she"} for tok in tokens)

    cleaned = []
    for s in spans:
        tok = text[s["start"]:s["end"]].lower()

        # never mark themselves
        if tok == "themselves":
            continue

        # if he/she exists, drop reflexive mark
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

    if has_themselves:
        return "THEMSELVES"
    if (has_himself or has_herself) and has_he_she:
        return "COREFERENT"
    if (has_himself or has_herself) and not has_he_she:
        return "BARE_REFLEXIVE"
    return "NONE"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def has_reflexive(text):
    return any(tok in REFLEXIVE_PRONOUNS for tok in tokenize(text))

def count_gendered_pronouns(text):
    toks = tokenize(text)
    return sum(
        1 for tok in toks
        if tok in GENDERED_PRONOUNS
        and tok not in {"himself", "herself"}
    )


def is_ungrammatical(text):
    t = text.lower()
    return any(re.search(p, t) for p in UNGRAAMMATICAL_PATTERNS)

def span_contains_gendered(text, spans):
    for s in spans:
        tok = text[s["start"]:s["end"]].lower()
        if tok in GENDERED_PRONOUNS:
            return True
    return False

def text_contains_gendered(text):
    return any(tok in GENDERED_PRONOUNS for tok in tokenize(text))

# -------------------------
# CLEANING PIPELINE
# -------------------------

seen = set()
stats = {
    "total": 0,
    "duplicates": 0,
    "reflexive": 0,
    "multi_pronoun": 0,
    "ungrammatical": 0,
    "span_mismatch": 0,
    "neutral_leak": 0,
    "kept": 0
}

with open("unclean_dataset.jsonl", "r", encoding="utf-8") as fin, \
     open("dataset.jsonl", "w", encoding="utf-8") as fout:

    for line in fin:
        stats["total"] += 1
        ex = json.loads(line)
        text = ex["text"]
        spans = ex.get("spans", [])
        bias_type = ex.get("bias_type", "")
        #FIX REFLEXIVE SPANS
        spans = clean_reflexive_spans(text, spans)
        ex["spans"] = spans

        # 1. Duplicate texts
        if text in seen:
            stats["duplicates"] += 1
            continue
        seen.add(text)

        # 2. Reflexives (always dropped)
        rtype = reflexive_type(text)

        # Drop ONLY themselves-based bias cases
        if rtype == "THEMSELVES":
            # must be neutral
            if bias_type != "NEUTRAL":
                stats["neutral_leak"] += 1
                continue

        # Require span ONLY for bare reflexives
        if rtype == "BARE_REFLEXIVE":
            if not span_contains_gendered(text, spans):
                stats["span_mismatch"] += 1
                continue


        # 3. Multiple gendered pronouns
        if count_gendered_pronouns(text) > 1:
            stats["multi_pronoun"] += 1
            continue

        # 4. Ungrammatical generation noise
        if is_ungrammatical(text):
            stats["ungrammatical"] += 1
            continue

        has_gendered_token = text_contains_gendered(text)
        has_gendered_span = span_contains_gendered(text, spans)

        # 5. Gendered pronoun exists but model didn't mark it
        if has_gendered_token and not has_gendered_span:
            stats["span_mismatch"] += 1
            continue

        # 6. Neutral sentence leaking gender
        if bias_type == "NEUTRAL" and has_gendered_token:
            stats["neutral_leak"] += 1
            continue

        fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
        stats["kept"] += 1

# -------------------------
# REPORT
# -------------------------

print("\nCLEANING COMPLETE\n")
for k, v in stats.items():
    print(f"{k:18}: {v}")
