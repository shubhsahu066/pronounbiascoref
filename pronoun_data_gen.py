import json
import re

# -------------------------
# CONFIG
# -------------------------

PRONOUN_MAP = {
    "he": {
        "{PRESENT_BE}": "is",
        "{PAST_BE}": "was",
        "{HAVE_PRESENT}": "has",
        "{HAVE_PAST}": "had",
        "{DO_PRESENT}": "does",
        "{DO_PAST}": "did"
    },
    "she": {
        "{PRESENT_BE}": "is",
        "{PAST_BE}": "was",
        "{HAVE_PRESENT}": "has",
        "{HAVE_PAST}": "had",
        "{DO_PRESENT}": "does",
        "{DO_PAST}": "did"
    },
    "they": {
        "{PRESENT_BE}": "are",
        "{PAST_BE}": "were",
        "{HAVE_PRESENT}": "have",
        "{HAVE_PAST}": "had",
        "{DO_PRESENT}": "do",
        "{DO_PAST}": "did"
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
NAMES = ["John", "Shubh", "Ravi"]

GENDERED_TOKENS = {
    "he", "she", "him", "her", "his", "hers",
    "himself", "herself"
}

REFLEXIVES_TO_MARK = {"himself", "herself"}

BIAS_CATEGORIES = {
    "PRONOUN",
    "GENERIC_ROLE_POSSESSIVE",
    "GENERIC_ROLE_REFLEXIVE",
    "GENERIC_ROLE_SUBJECT_CONTINUATION"
}

# -------------------------
# HELPERS
# -------------------------



def mark_gendered_spans(text):
    spans = []
    for m in re.finditer(r"\b\w+\b", text):
        if m.group(0).lower() in GENDERED_TOKENS:
            spans.append({
                "start": m.start(),
                "end": m.end(),
                "type": "PRONOUN"
            })
    return spans


def add_reflexive_spans(text, spans):
    lower = text.lower()
    for refl in REFLEXIVES_TO_MARK:
        start = 0
        while True:
            idx = lower.find(refl, start)
            if idx == -1:
                break
            spans.append({
                "start": idx,
                "end": idx + len(refl),
                "type": "PRONOUN"
            })
            start = idx + len(refl)
    return spans


def replace_pronouns(template, pronoun_sub):
    sentence = template
    spans = []

    for k, v in PRONOUN_MAP[pronoun_sub].items():
        sentence = sentence.replace(k, v)

    for ph, word in PRONOUN_FORMS[pronoun_sub].items():
        while ph in sentence:
            idx = sentence.index(ph)
            pron = word.capitalize() if idx == 0 else word
            sentence = sentence[:idx] + pron + sentence[idx + len(ph):]
            spans.append({
                "start": idx,
                "end": idx + len(pron),
                "type": "PRONOUN"
            })

    return sentence, spans


# -------------------------
# GENERATOR
# -------------------------

def generate_dataset(template_path, output_path):
    with open(template_path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    dataset = []

    for category, sentence_list in templates.items():
        for template in sentence_list:

            # ---------- STATIC ----------
            if category == "STATIC":
                spans = mark_gendered_spans(template)
                dataset.append({
                    "text": template,
                    "spans": spans,
                    "bias_type": "PRONOUN" if spans else "NEUTRAL"
                })
                continue

            # ---------- NAME ----------
            if "{NAME}" in template:
                for name in NAMES:
                    sent = template.replace("{NAME}", name)
                    spans = mark_gendered_spans(sent)
                    dataset.append({
                        "text": sent,
                        "spans": spans,
                        "bias_type": "PRONOUN" if spans else "NEUTRAL"
                    })
                continue

            # ---------- ROLE ----------
            role_variants = [template]
            if "{ROLE}" in template:
                role_variants = [
                    template.replace("{ROLE}", role) for role in ROLES
                ]

            # ---------- PRONOUN SUB ----------
            for tmpl in role_variants:
                for pronoun_sub in ["he", "she", "they"]:
                    sent, spans = replace_pronouns(tmpl, pronoun_sub)

                    # neutralize standard pronouns
                    if pronoun_sub == "they" or category not in BIAS_CATEGORIES:
                        spans = []

                    # always re-add himself / herself
                    spans = add_reflexive_spans(sent, spans)

                    dataset.append({
                        "text": sent,
                        "spans": spans,
                        "bias_type": "PRONOUN" if spans else "NEUTRAL"
                    })

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset generated: {len(dataset)} samples")


if __name__ == "__main__":
    generate_dataset("pronoun_templates.json", "unclean_dataset.jsonl")
