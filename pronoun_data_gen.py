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

MALE_NAMES = ["John", "Shubh"]
FEMALE_NAMES = ["Mary", "Anita"]

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


                # ---------- NAMED ----------
                if "{NAME}" in tmpl or "{MALE_NAME}" in tmpl or "{FEMALE_NAME}" in tmpl:

                    has_refl = "{PRONOUN_REFL}" in tmpl

                    # ---- MALE NAME TEMPLATES ----
                    if "{MALE_NAME}" in tmpl or "{NAME}" in tmpl:
                        for name in MALE_NAMES:
                            pronouns = ["he"] if has_refl else ["he", "they"]

                            for p in pronouns:
                                sent = tmpl.replace("{NAME}", name)\
                                          .replace("{MALE_NAME}", name)\
                                          .replace("{FEMALE_NAME}", name)

                                sent, spans = replace_pronouns(sent, p)

                                if p == "they":
                                    spans = []

                                fout.write(json.dumps({
                                    "text": sent,
                                    "spans": spans,
                                    "bias_type": "PRONOUN" if spans else "NEUTRAL"
                                }) + "\n")

                    # ---- FEMALE NAME TEMPLATES ----
                    if "{FEMALE_NAME}" in tmpl or "{NAME}" in tmpl:
                        for name in FEMALE_NAMES:
                            pronouns = ["she"] if has_refl else ["she", "they"]

                            for p in pronouns:
                                sent = tmpl.replace("{NAME}", name)\
                                          .replace("{MALE_NAME}", name)\
                                          .replace("{FEMALE_NAME}", name)

                                sent, spans = replace_pronouns(sent, p)

                                if p == "they":
                                    spans = []

                                fout.write(json.dumps({
                                    "text": sent,
                                    "spans": spans,
                                    "bias_type": "PRONOUN" if spans else "NEUTRAL"
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

                        if p == "they" or cat not in BIAS_CATEGORIES:
                            spans = []

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
