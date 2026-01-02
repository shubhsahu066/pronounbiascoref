import json
import re

# -------------------------
# CONFIG
# -------------------------

PRONOUN_MAP = {
    "he": {"{PRESENT_BE}": "is", "{PAST_BE}": "was"},
    "she": {"{PRESENT_BE}": "is", "{PAST_BE}": "was"},
    "they": {"{PRESENT_BE}": "are", "{PAST_BE}": "were"}
}

PRONOUN_FORMS = {
    "he": {
        "{PRONOUN_SUBJ}": "he", "{PRONOUN_OBJ}": "him",
        "{PRONOUN_POS}": "his", "{PRONOUN_POS_PRO}": "his",
        "{PRONOUN_REFL}": "himself"
    },
    "she": {
        "{PRONOUN_SUBJ}": "she", "{PRONOUN_OBJ}": "her",
        "{PRONOUN_POS}": "her", "{PRONOUN_POS_PRO}": "hers",
        "{PRONOUN_REFL}": "herself"
    },
    "they": {
        "{PRONOUN_SUBJ}": "they", "{PRONOUN_OBJ}": "them",
        "{PRONOUN_POS}": "their", "{PRONOUN_POS_PRO}": "theirs",
        "{PRONOUN_REFL}": "themselves"
    }
}

ROLES = ["student", "nurse", "teacher", "citizen", "employee", "user"]
MALE_NAMES = ["John", "Shubh"]
FEMALE_NAMES = ["Mary", "Anita"]

# UPDATED: Includes reflexives so mark_spans can initially find them
GENDERED = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}
REFLEXIVES = {"himself", "herself"}

BIAS_CATEGORIES = {
    "PRONOUN", "GENERIC_ROLE_POSSESSIVE",
    "GENERIC_ROLE_REFLEXIVE", "GENERIC_ROLE_SUBJECT_CONTINUATION"
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
    
    # 1. Perform Substitutions (Standard logic)
    for k, v in PRONOUN_MAP[pronoun].items():
        sent = sent.replace(k, v)

    for ph, word in PRONOUN_FORMS[pronoun].items():
        while ph in sent:
            i = sent.index(ph)
            w = word.capitalize() if i == 0 else word
            sent = sent[:i] + w + sent[i+len(ph):]

            
    target_words = set(PRONOUN_FORMS[pronoun].values())
    spans = mark_spans(sent, target_words)

    return sent, spans

def filter_spans(text, spans, is_named=False):
    """
    Applies the specific user rules:
    1. If HE/SHE present, remove reflexive spans (Rule 1: Priority).
    2. If NO he/she, keep reflexive spans even if named (Rule 4: Name + Reflexive is Bias).
    """
    toks = set(tokenize(text))
    has_he_she = "he" in toks or "she" in toks
    
    new_spans = []
    for s in spans:
        word = text[s['start']:s['end']].lower()
        is_reflexive = word in REFLEXIVES
        
        # REMOVED: The check that skipped reflexives for named entities.
        # Now, "John hurt himself" will keep the "himself" span.

        # Rule: If he/she is present, don't mark reflexives (only mark the subject)
        if has_he_she and is_reflexive:
            continue
            
        new_spans.append(s)
        
    return new_spans

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
                    # Apply logic to drop reflexives if needed
                    spans = filter_spans(tmpl, spans, is_named=False)
                    
                    fout.write(json.dumps({
                        "text": tmpl,
                        "spans": spans,
                        "bias_type": "PRONOUN" if spans else "NEUTRAL"
                    }) + "\n")
                    continue


                # ---------- NAMED ----------
                if "{NAME}" in tmpl or "{MALE_NAME}" in tmpl or "{FEMALE_NAME}" in tmpl:
                    has_refl = "{PRONOUN_REFL}" in tmpl
                    
                    name_sets = []
                    if "{MALE_NAME}" in tmpl or "{NAME}" in tmpl:
                        name_sets.append((MALE_NAMES, ["he"] if has_refl else ["he", "they"]))
                    if "{FEMALE_NAME}" in tmpl or "{NAME}" in tmpl:
                        name_sets.append((FEMALE_NAMES, ["she"] if has_refl else ["she", "they"]))

                    for names, pronouns in name_sets:
                        for name in names:
                            for p in pronouns:
                                sent = tmpl.replace("{NAME}", name)\
                                          .replace("{MALE_NAME}", name)\
                                          .replace("{FEMALE_NAME}", name)

                                sent, raw_spans = replace_pronouns(sent, p)
                                
                                if p == "they":
                                    final_spans = []
                                else:
                                    # CRITICAL: Pass is_named=True to drop reflexives
                                    final_spans = filter_spans(sent, raw_spans, is_named=True)

                                fout.write(json.dumps({
                                    "text": sent,
                                    "spans": final_spans,
                                    "bias_type": "PRONOUN" if final_spans else "NEUTRAL"
                                }) + "\n")
                    continue


                # ---------- ROLE ----------
                role_variants = [tmpl]
                if "{ROLE}" in tmpl:
                    role_variants = [tmpl.replace("{ROLE}", r) for r in ROLES]

                for rv in role_variants:
                    for p in ["he", "she", "they"]:
                        sent, raw_spans = replace_pronouns(rv, p)

                        if p == "they" or cat not in BIAS_CATEGORIES:
                            raw_spans = []
                        else:
                            # Add reflexives if they weren't added by replace_pronouns
                            # (e.g. "A nurse prepared herself")
                            raw_spans += mark_spans(sent, REFLEXIVES)
                            # Deduplicate spans based on start index
                            unique_spans = {s['start']: s for s in raw_spans}.values()
                            raw_spans = list(unique_spans)

                        # Apply filtering rules (Rule 1 & 3)
                        final_spans = filter_spans(sent, raw_spans, is_named=False)

                        fout.write(json.dumps({
                            "text": sent,
                            "spans": final_spans,
                            "bias_type": "PRONOUN" if final_spans else "NEUTRAL"
                        }) + "\n")

    print("Dataset generation complete")

if __name__ == "__main__":
    generate_dataset("pronoun_templates.json", "unclean_dataset.jsonl")