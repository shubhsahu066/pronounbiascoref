GENDERED_PRONOUNS = {
    "he", "she", "him", "her", "his", "hers",
    "himself", "herself"
}

INHERENTLY_GENDERED_NOUNS = {
    "man", "woman", "boy", "girl",
    "father", "mother", "son", "daughter",
    "monk", "nun", "king", "queen",
    "priest", "actor", "actress",
    "chairman", "policeman", "fireman"
}

GENERIC_DETERMINERS = {"a", "an", "each", "every", "any"}

def is_named_person(span_text: str) -> bool:
    # Simple placeholder; can be upgraded
    return span_text.istitle()

def is_inherently_gendered(span_text: str) -> bool:
    return span_text.lower().split()[-1] in INHERENTLY_GENDERED_NOUNS

def is_generic_role(span_text: str) -> bool:
    words = span_text.lower().split()
    return len(words) > 1 or words[0] not in {"the"}

def detect_pronoun_bias(document, clusters):
    """
    document: list of tokens
    clusters: span clusters from SpanBERT
    """
    results = []

    for cluster in clusters:
        pronouns = []
        antecedents = []

        for start, end in cluster:
            span = " ".join(document[start:end+1])
            if span.lower() in GENDERED_PRONOUNS:
                pronouns.append(span)
            else:
                antecedents.append(span)

        if not antecedents:
            continue

        head = antecedents[0]

        if is_named_person(head):
            continue

        if is_inherently_gendered(head):
            continue

        if is_generic_role(head):
            for p in pronouns:
                results.append({
                    "pronoun": p,
                    "antecedent": head,
                    "bias": True
                })

    return results
