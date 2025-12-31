import json
import re

# -------------------------
# CONFIG
# -------------------------

PRONOUN_MAP = {
    "he": {
        "{PRESENT_BE}": "is",
        "{PAST_BE}": "was",
    },
    "she": {
        "{PRESENT_BE}": "is",
        "{PAST_BE}": "was",
    },
    "they": {
        "{PRESENT_BE}": "are",
        "{PAST_BE}": "were",
    }
}

PRONOUN_FORMS = {
    "he": {
        "{PRONOUN_SUBJ}": "he",
        "{PRONOUN_OBJ}": "him",
        "{PRONOUN_POS}": "his",
        "{PRONOUN_POS_PRO}": "his",
        "{PRONOUN_REFL}": "himself"
    },
    "she": {
        "{PRONOUN_SUBJ}": "she",
        "{PRONOUN_OBJ}": "her",
        "{PRONOUN_POS}": "her",
        "{PRONOUN_POS_PRO}": "hers",
        "{PRONOUN_REFL}": "herself"
    },
    "they": {
        "{PRONOUN_SUBJ}": "they",
        "{PRONOUN_OBJ}": "them",
        "{PRONOUN_POS}": "their",
        "{PRONOUN_POS_PRO}": "theirs",
        "{PRONOUN_REFL}": "themselves"
    }
}

ROLES = ["student", "nurse", "teacher", "citizen", "employee", "user"]

# 🔑 NAME SPLIT (THIS IS THE ONLY NEW THING)
MALE_NAMES = ["John", "Shubh", "Ravi", "Alex"]
FEMALE_NAMES = ["Mary", "Anita", "Priya", "Ellie"]

GENDERED = {"he", "she", "him", "her", "his", "hers"}
REFLEXIVES = {"himself", "herself"}

BIAS_CATEGORIES = {
    "PRONOUN",
    "GENERIC_ROLE_POSSESSIVE",
    "GENERIC_ROLE_REFLEXIVE",
    "GENERIC_ROLE_SUBJECT_CONTINUATION"
}

# -------------------------
# HELPERS
# -------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def mark_spans(text, targets):
    spans = []
    for m in re.finditer(r"\b\w+\b", text):
        if m.group(0).lower() in targets:
            spans.append({
                "start": m.start(),
                "end": m.end(),
                "type": "PRONOUN"
            })
    return spans

def replace_pronouns(template, pronoun):
    sent = template
    spans = []

    for k, v in PRONOUN_MAP[pronoun].items():
        sent = sent.replace(k, v)

    for ph, word in PRONOUN_FORMS[pronoun].items():
        while ph in sent:
            i = sent.index(ph)
            w = word.capitalize() if i == 0 else word
            sent = sent[:i] + w + sent[i+len(ph):]
            spans.append({
                "start": i,
                "end": i + len(w),
                "type": "PRONOUN"
            })

    return sent, spans

# -------------------------
# GENERATOR
# -------------------------

def generate_dataset(template_path, output_path):
    templates = json.load(open(template_path))

    with open(output_path, "w", encoding="utf-8") as fout:
        for cat, sents in templates.items():
            for tmpl in sents:

                # ---------- STATIC ----------
                if cat == "STATIC":
                    spans = mark_spans(tmpl, GENDERED)
                    fout.write(json.dumps({
                        "text": tmpl,
                        "spans": spans,
                        "bias_type": "PRONOUN" if spans else "NEUTRAL"
                    }) + "\n")
                    continue

                # ---------- NAMED (GENDER-CORRECT ONLY) ----------
                name_list=MALE_NAMES+FEMALE_NAMES
                if "{NAME}" in tmpl:
                    if "he" in tmpl or "his" in tmpl or "him" in tmpl:
                        name_list = MALE_NAMES
                    elif "she" in tmpl or "her" in tmpl:
                        name_list = FEMALE_NAMES
                    else:
                        continue  # safety

                    for name in name_list:
                        sent = tmpl.replace("{NAME}", name)
                        spans = mark_spans(sent, GENDERED)
                        fout.write(json.dumps({
                            "text": sent,
                            "spans": spans,
                            "bias_type": "PRONOUN"
                        }) + "\n")
                    continue

                # ---------- NAMED (STRICT GENDER CONTROL) ----------
                if "{NAME}" in tmpl:
                    # gender-flexible templates → paired generation
                    pairs = [
                        (MALE_NAMES, "he"),
                        (FEMALE_NAMES, "she")
                    ]

                    for names, pron in pairs:
                        sent_with_pron = tmpl.replace(
                            " he ", f" {pron} "
                        ).replace(
                            " she ", f" {pron} "
                        )

                        for name in names:
                            sent = sent_with_pron.replace("{NAME}", name)
                            spans = mark_spans(sent, GENDERED)

                            fout.write(json.dumps({
                                "text": sent,
                                "spans": spans,
                                "bias_type": "PRONOUN"
                            }) + "\n")
                    continue


                if "{MALE_NAME}" in tmpl:
                    for name in MALE_NAMES:
                        sent = tmpl.replace("{MALE_NAME}", name)
                        spans = mark_spans(sent, GENDERED)

                        fout.write(json.dumps({
                            "text": sent,
                            "spans": spans,
                            "bias_type": "PRONOUN"
                        }) + "\n")
                    continue


                if "{FEMALE_NAME}" in tmpl:
                    for name in FEMALE_NAMES:
                        sent = tmpl.replace("{FEMALE_NAME}", name)
                        spans = mark_spans(sent, GENDERED)

                        fout.write(json.dumps({
                            "text": sent,
                            "spans": spans,
                            "bias_type": "PRONOUN"
                        }) + "\n")
                    continue


                # ---------- ROLE ----------
                role_variants = [tmpl]
                if "{ROLE}" in tmpl:
                    role_variants = [tmpl.replace("{ROLE}", r) for r in ROLES]

                for rv in role_variants:
                    for p in ["he", "she", "they"]:
                        sent, spans = replace_pronouns(rv, p)

                        toks = tokenize(sent)
                        has_he_she = any(t in {"he", "she"} for t in toks)

                        # neutralize they
                        if p == "they" or cat not in BIAS_CATEGORIES:
                            spans = []

                        # bare reflexive → mark
                        if not has_he_she:
                            spans += mark_spans(sent, REFLEXIVES)

                        fout.write(json.dumps({
                            "text": sent,
                            "spans": spans,
                            "bias_type": "PRONOUN" if spans else "NEUTRAL"
                        }) + "\n")

    print("Dataset generation complete")

if __name__ == "__main__":
    generate_dataset("pronoun_templates.json", "unclean_dataset.jsonl")
