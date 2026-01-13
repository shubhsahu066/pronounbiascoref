import spacy
from knowledge_base import (
    GENDERED_ROLES, PRONOUN_MAP, 
    MALE_MODIFIERS, FEMALE_MODIFIERS,
    FREQUENCY_ADVERBS, OBLIGATION_MODALS, 
    PREDICTION_MODALS, ALL_MODALS, CONDITIONAL_MARKERS
)

nlp = spacy.load("en_core_web_sm")

# =========================
# HELPER FUNCTIONS
# =========================

def get_best_head_span(spans):
    best_span = spans[0]
    best_score = 0 
    for span in spans:
        root = span.root
        score = 0
        if root.ent_type_:
            score = 2
        elif root.pos_ in ["NOUN", "PROPN"]:
            score = 1
        if score > best_score:
            best_score = score
            best_span = span
    return best_span


def get_governing_verb(token):
    head = token.head
    while head.pos_ not in ["VERB", "AUX"] and head.head != head:
        head = head.head
    return head if head.pos_ in ["VERB", "AUX"] else None


def get_modifier_gender(span):
    for child in span.root.children:
        if child.lower_ in MALE_MODIFIERS:
            return "M"
        if child.lower_ in FEMALE_MODIFIERS:
            return "F"
    return None


def has_present_aux(verb):
    for child in verb.children:
        if child.dep_ in ["aux", "auxpass"] and child.lemma_ == "be":
            if child.tag_ in ["VBZ", "VBP"]:
                return True
    return False


def has_frequency_adverb(verb):
    for child in verb.children:
        if child.dep_ == "advmod" and child.lemma_.lower() in FREQUENCY_ADVERBS:
            return True
    return False


def is_strictly_episodic(verb):
    if not verb:
        return False

    if has_frequency_adverb(verb):
        return False

    for child in verb.children:
        if child.dep_ == "aux" and child.lemma_.lower() in ALL_MODALS:
            return False

    if verb.tag_ == "VBD":
        return True

    if verb.tag_ == "VBG" and has_present_aux(verb):
        return True

    return False


def is_generic_context(verb, is_anchored_entity, is_definite, recursion_depth=0):
    if not verb or recursion_depth > 5:
        return False

    # ---- recursion ----
    if verb.dep_ in ["xcomp", "ccomp", "advcl", "conj"]:
        parent = get_governing_verb(verb)
        if parent and parent != verb:
            if not is_generic_context(parent, is_anchored_entity, is_definite, recursion_depth + 1):
                return False

    # ---- modals ----
    found_modal = None
    for child in verb.children:
        if child.dep_ == "aux" and child.lemma_.lower() in ALL_MODALS:
            found_modal = child.lemma_.lower()
            break

    if found_modal:
        if found_modal in OBLIGATION_MODALS:
            return not is_anchored_entity

        if found_modal in PREDICTION_MODALS:
            return not (is_anchored_entity or is_definite)

    # ---- conditionals ----
    for child in verb.children:
        if child.dep_ == "mark" and child.lemma_.lower() in CONDITIONAL_MARKERS:
            return True
        if child.dep_ == "advcl":
            for g in child.children:
                if g.dep_ == "mark" and g.lemma_.lower() in CONDITIONAL_MARKERS:
                    return True

    # ---- SIMPLE PRESENT = GENERIC (FIX #2) ----
    if verb.tag_ in ["VBP", "VBZ"]:
        if not is_anchored_entity:
            return True

    # ---- present passive ----
    if verb.tag_ == "VBN" and has_present_aux(verb):
        if not is_anchored_entity:
            return True

    return False


# =========================
# MAIN LOGIC
# =========================

def detect_pronoun_bias(text: str, clusters):
    doc = nlp(text)
    bias_report = []

    for cluster_indices in clusters:
        spans = [
            doc.char_span(s[0], s[1]) 
            for s in cluster_indices 
            if doc.char_span(s[0], s[1]) is not None
        ]
        if not spans:
            continue

        cluster_words = {s.text.lower() for s in spans}
        if not any(p in cluster_words for p in PRONOUN_MAP):
            continue

        head_span = get_best_head_span(spans)
        head_root = head_span.root


        # --- PHASE 1: ANCHORING (FIXED) ---
        is_anchored_entity = False
        is_definite = False

        for span in spans:
            root = span.root

            # 1. Named entities are always specific
            if root.ent_type_ in ["PERSON", "ORG", "GPE"]:
                is_anchored_entity = True
                break

            # 2. Strong determiners (this/that/my/etc.)
            if root.pos_ in ["NOUN", "PROPN"]:
                for child in root.children:
                    if child.lemma_ in ["this", "that", "my", "your", "our"]:
                        is_anchored_entity = True
                        break
                    if child.lemma_ == "the":
                        is_definite = True

            if is_anchored_entity:
                break

        # 3. Dynamic anchoring: ONLY episodic verbs can anchor
        if not is_anchored_entity:
            for span in spans:
                verb = get_governing_verb(span.root)
                if is_strictly_episodic(verb):
                    is_anchored_entity = True
                    break


        # ---- dynamic anchoring ----
        if not is_anchored_entity:
            for span in spans:
                verb = get_governing_verb(span.root)
                if is_strictly_episodic(verb):
                    is_anchored_entity = True
                    break

        # ---- PHASE 2: ROLE GENDER ----
        role_gender = GENDERED_ROLES.get(head_root.lemma_.lower())
        mod_gender = get_modifier_gender(head_span)
        if mod_gender:
            role_gender = mod_gender

        # ---- FIX #3: CONTRASTIVE GENDER SYMMETRY ----
        has_male = any(w in PRONOUN_MAP and PRONOUN_MAP[w] == "M" for w in cluster_words)
        has_female = any(w in PRONOUN_MAP and PRONOUN_MAP[w] == "F" for w in cluster_words)
        if has_male and has_female:
            continue

        for span in spans:
            token = span.root
            word = token.text.lower()

            if word not in PRONOUN_MAP:
                continue

            pronoun_gender = PRONOUN_MAP[word]

            if role_gender and role_gender == pronoun_gender:
                continue

            verb = get_governing_verb(token)

            # ---- FIX #4: BARE CONDITIONAL IMPERATIVES ----
            sent = span.sent
            if sent.text.lower().startswith("if"):
                main_verb = None
                for t in sent:
                    if t.pos_ == "VERB":
                        main_verb = t
                        break
                if main_verb and main_verb.tag_ == "VB":
                    continue

            # ---- FIX #5: LEFT DISLOCATION ----
            if token.dep_ == "nsubj" and token.head == token:
                continue

            if is_generic_context(verb, is_anchored_entity, is_definite):
                bias_report.append({
                    "start": span.start_char,
                    "end": span.end_char,
                    "text": word,
                    "context": span.sent.text,
                    "reason": "Generic context linked to role"
                })

    return bias_report
