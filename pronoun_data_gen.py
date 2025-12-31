import json

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

PRONOUN_PLACEHOLDERS = {
    "{PRONOUN_SUBJ}",
    "{PRONOUN_OBJ}",
    "{PRONOUN_POS}",
    "{PRONOUN_POS_PRO}",
    "{PRONOUN_REFL}"
}

ROLES = [
    "student", "nurse", "teacher",
    "citizen", "employee", "user"

]

NAMES = ["John", "Mary", "Alex", "Shubh", "Ravi"]

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

def replace_placeholders(template, pronoun_sub):
    sentence = template
    spans = []

    # verb agreement
    for k, v in PRONOUN_MAP[pronoun_sub].items():
        sentence = sentence.replace(k, v)

    # pronouns
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


def generate_dataset(template_path, output_path):
    with open(template_path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    dataset = []

    BIAS_CATEGORIES = {
        "PRONOUN",
        "GENERIC_ROLE_POSSESSIVE",
        "GENERIC_ROLE_REFLEXIVE",
        "GENERIC_ROLE_SUBJECT_CONTINUATION"
    }

    for category, sentence_list in templates.items():
        for template in sentence_list:

            role_variants = [template]
            if "{ROLE}" in template:
                role_variants = [
                    template.replace("{ROLE}", role) for role in ROLES
                ]

            for tmpl in role_variants:
                for pronoun_sub in ["he", "she", "they"]:
                    sent, spans = replace_placeholders(tmpl, pronoun_sub)

                    # NEUTRALIZE
                    if pronoun_sub == "they" or category not in BIAS_CATEGORIES:
                        spans = []

                    dataset.append({
                        "text": sent,
                        "spans": spans,
                        "bias_type": (
                            "PRONOUN"
                            if category in BIAS_CATEGORIES and pronoun_sub != "they"
                            else "NEUTRAL"
                        )
                    })

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset generated: {len(dataset)} samples")


if __name__ == "__main__":
    generate_dataset("pronoun_templates.json", "unclean_dataset.jsonl")
