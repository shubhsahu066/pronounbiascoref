import spacy
from fastcoref import FCoref

# =========================
# LOAD MODELS
# =========================

nlp = spacy.load("en_core_web_sm")
coref_model = FCoref()  # default = spanbert-base

# =========================
# CONSTANTS (UNCHANGED)
# =========================

GENDERED_PRONOUNS = {"he", "she", "him", "her", "his", "hers"}
REFLEXIVE_PRONOUNS = {"himself", "herself"}

COPULAR_VERBS = {"is", "was", "are", "were"}
GENERIC_DETERMINERS = {"a", "an", "each", "every", "any"}

INHERENTLY_GENDERED_NOUNS = {
    "man", "woman", "boy", "girl",
    "father", "mother", "son", "daughter",
    "monk", "nun", "king", "queen",
    "priest", "actor", "actress",
    "chairman", "policeman", "fireman"
}

# =========================
# GATEKEEPING (UNCHANGED)
# =========================

def is_episodic(sentence: str) -> bool:
    doc = nlp(sentence)
    for tok in doc:
        if tok.tag_ in {"VBD", "VBG"}:
            return True
        if tok.tag_ == "VBN":
            for child in tok.children:
                if child.lemma_ == "have":
                    return True
        if tok.lemma_.lower() in {"will", "shall", "would", "can", "could"}:
            return True
    return False


def gatekeep_sentence(text: str) -> bool:
    doc = nlp(text)
    if not any(tok.text.lower() in GENDERED_PRONOUNS | REFLEXIVE_PRONOUNS for tok in doc):
        return True
    if is_episodic(text):
        return True
    return False

# =========================
# HELPERS (UNCHANGED)
# =========================

def get_determiner(noun):
    for child in noun.children:
        if child.dep_ == "det":
            return child.lemma_.lower()
    return None


def is_named_person(noun):
    return noun.ent_type_ == "PERSON"


def is_inherently_gendered(noun):
    return noun.lemma_.lower() in INHERENTLY_GENDERED_NOUNS


def is_left_dislocation(noun, pronoun, doc):
    return noun.i + 1 < len(doc) and doc[noun.i + 1].text == ","


def is_contrastive(sentence):
    return any(tok.lemma_ in {"though", "although", "however", "no"} for tok in nlp(sentence))


def is_bare_conditional(token):
    if token.dep_ == "nsubj":
        for child in token.head.children:
            if child.dep_ == "mark" and child.lemma_ == "if":
                return True
    return False


def is_predicate_possessive(token):
    if token.text.lower() not in {"his", "hers", "theirs"}:
        return False
    if token.i > 0 and token.nbor(-1).lemma_.lower() in COPULAR_VERBS:
        return True
    return False

# =========================
# FASTCOREF ANTECEDENT RESOLUTION
# =========================

def resolve_antecedent_fastcoref(pronoun_idx, clusters, spacy_doc):
    """
    Find nearest non-pronoun antecedent in same cluster
    """
    for cluster in clusters:
        for start, end in cluster:
            if start <= pronoun_idx <= end:
                # candidate antecedents before pronoun
                candidates = [
                    (s, e) for (s, e) in cluster
                    if e < pronoun_idx
                ]
                if not candidates:
                    return None

                # take closest antecedent span
                s, e = candidates[-1]
                return spacy_doc[s:e+1].root

    return None

# =========================
# BIAS DECISION (UNCHANGED)
# =========================

def is_bias_pronoun(pronoun, antecedent, sentence):
    if antecedent is None:
        return False

    if is_named_person(antecedent):
        return False

    if is_inherently_gendered(antecedent):
        return False

    det = get_determiner(antecedent)

    if det in GENERIC_DETERMINERS:
        return True

    if det == "the":
        if is_left_dislocation(antecedent, pronoun, nlp(sentence)):
            return False
        if is_contrastive(sentence):
            return True
        return False

    return False

# =========================
# MAIN INFERENCE
# =========================

def detect_pronoun_bias(text: str):
    doc = nlp(text)

    if gatekeep_sentence(text):
        return {"text": text, "spans": []}

    coref_out = coref_model.predict(texts=[text])[0]
    clusters = coref_out.clusters

    spans = []

    for tok in doc:
        if tok.text.lower() not in GENDERED_PRONOUNS | REFLEXIVE_PRONOUNS:
            continue

        if tok.idx == 0:
            continue

        if is_bare_conditional(tok):
            continue

        if is_predicate_possessive(tok):
            continue

        antecedent = resolve_antecedent_fastcoref(tok.i, clusters, doc)

        if is_bias_pronoun(tok, antecedent, text):
            spans.append({
                "start": tok.idx,
                "end": tok.idx + len(tok.text),
                "type": "PRONOUN"
            })

    return {"text": text, "spans": spans}

# =========================
# TEST
# =========================

if __name__ == "__main__":
    tests = [
        "A teacher should prepare his lessons carefully.",
        "The teacher knows how to handle her students.",
        "The teacher is always kind, no matter how he reacts.",
        "The monk should preserve his sanity.",
        "The secretary will file the reports when she returns."
    ]

    for t in tests:
        print(detect_pronoun_bias(t))
