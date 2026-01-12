import spacy

# Load spaCy once
nlp = spacy.load("en_core_web_sm")

# =========================
# KNOWLEDGE BASES
# =========================

GENDERED_ROLES = {
    # Male
    "rifleman": "M", "policeman": "M", "fireman": "M", "king": "M", "actor": "M",
    "waiter": "M", "steward": "M", "hero": "M", "uncle": "M", "father": "M", 
    "brother": "M", "son": "M", "man": "M", "boy": "M", "gentleman": "M",
    # Female
    "riflewoman": "F", "policewoman": "F", "firewoman": "F", "queen": "F", 
    "actress": "F", "waitress": "F", "stewardess": "F", "heroine": "F",
    "aunt": "F", "mother": "F", "sister": "F", "daughter": "F", "woman": "F",
    "girl": "F", "lady": "F"
}

PRONOUN_MAP = {
    "he": "M", "him": "M", "his": "M", "himself": "M",
    "she": "F", "her": "F", "hers": "F", "herself": "F"
}

FREQUENCY_ADVERBS = {"always", "constantly", "continually", "forever", "usually", "often", "never", "typically", "generally"}
MODAL_VERBS = {"should", "must", "can", "could", "would", "ought", "might", "may", "will", "shall"}
CONDITIONAL_MARKERS = {"if", "unless", "whenever", "whether"}

# =========================
# HELPER FUNCTIONS
# =========================

def get_governing_verb(token):
    """
    Finds the main verb that the token is the subject/object of.
    Now accepts AUX (e.g., "is", "was") as valid governors.
    """
    head = token.head
    while head.pos_ not in ["VERB", "AUX"] and head.head != head:
        head = head.head
    
    # Return if it's a Verb or Aux, otherwise None
    return head if head.pos_ in ["VERB", "AUX"] else None

def is_episodic_context(verb_token):
    """
    Returns True if the action is Specific (Safe).
    Returns False if the action is Generic/Habitual (Potential Bias).
    """
    if not verb_token:
        return False 

    # 1. Past Tense -> Episodic (SAFE)
    # VBD = Past Tense, VBN = Past Participle (often perfect)
    if verb_token.tag_ == "VBD": 
        return True

    # 2. Perfect Tense -> Episodic (SAFE)
    # Check for "have" auxiliary: "has eaten", "had gone"
    if verb_token.tag_ == "VBN":
        for child in verb_token.children:
            if child.lemma_ == "have":
                return True

    # 3. Present Continuous -> Episodic (SAFE) UNLESS Habitual
    # "is eating" (Safe) vs "is always eating" (Habitual/Generic)
    if verb_token.tag_ == "VBG":
        has_aux_be = False
        is_habitual = False
        
        for child in verb_token.children:
            if child.dep_ == "aux" and child.lemma_ == "be":
                has_aux_be = True
            if child.dep_ == "advmod" and child.lemma_.lower() in FREQUENCY_ADVERBS:
                is_habitual = True

        if has_aux_be and not is_habitual:
            return True

    return False

def is_generic_context(verb_token):
    """
    Checks for Generic markers (Modals, Simple Present, Conditionals).
    """
    if not verb_token:
        return False
        
    # Check A: Modals (Deontic) -> "He SHOULD work"
    for child in verb_token.children:
        if child.dep_ == "aux" and child.lemma_.lower() in MODAL_VERBS:
            return True

    # Check B: Conditionals (Bare or Explicit)
    # 1. Direct 'mark' dependency (rare on main verb, but possible)
    for child in verb_token.children:
        if child.dep_ == "mark" and child.lemma_.lower() in CONDITIONAL_MARKERS:
            return True
        
    # 2. Adverbial Clause ('advcl') containing a marker -> "If he runs, he gets tired"
    # The main verb "gets" has an 'advcl' child "runs", which has a 'mark' child "if".
    for child in verb_token.children:
        if child.dep_ == "advcl":
            for grandchild in child.children:
                if grandchild.dep_ == "mark" and grandchild.lemma_.lower() in CONDITIONAL_MARKERS:
                    return True

    # Check C: Simple Present (Gnomic) -> "He WORKS hard" / "The choice IS his"
    # VBP = non-3rd person singular present, VBZ = 3rd person singular present
    if verb_token.tag_ in ["VBP", "VBZ"]:
        return True
        
    return False

# =========================
# MAIN LOGIC
# =========================

def detect_pronoun_bias(text: str, clusters):
    """
    Analyzes document coreference clusters for pronoun bias.
    """
    doc = nlp(text)
    bias_report = []

    for cluster_indices in clusters:
        # Convert indices to spaCy Spans
        spans = [doc.char_span(s[0], s[1]) for s in cluster_indices if doc.char_span(s[0], s[1]) is not None]
        if not spans: continue

        # --- GATEKEEPING: CLUSTER LEVEL ---
        # If this cluster has NO gendered pronouns, we don't care about it.
        cluster_text_set = {span.text.lower() for span in spans}
        has_gendered_pronoun = any(p in cluster_text_set for p in PRONOUN_MAP)
        
        if not has_gendered_pronoun:
            continue 

        head_span = spans[0]
        head_root = head_span.root
        
        # --- PHASE 1: ANCHOR CHECK (Is the Entity Specific?) ---
        is_anchored_safe = False
        
        for span in spans:
            # Anchor A: Named Entity (Alice, Mr. Smith)
            if span.root.ent_type_ in ["PERSON", "ORG"]:
                is_anchored_safe = True
                break
            
            # Anchor B: Specific Modifiers ("The teacher who...", "This teacher")
            if span.root == head_root:
                for child in span.root.children:
                    if child.dep_ == "relcl" or child.lemma_ in ["this", "that", "my", "your"]:
                        is_anchored_safe = True
                        break
        
        if is_anchored_safe:
            continue # Safe cluster. Skip to next.

        # --- PHASE 2: PRONOUN BIAS CHECK ---
        role_gender = GENDERED_ROLES.get(head_root.lemma_.lower())

        for span in spans:
            token = span.root
            word = token.text.lower()
            
            # Filter: Check only pronouns
            if word not in PRONOUN_MAP: continue
                
            pronoun_gender = PRONOUN_MAP[word]

            # 1. GENDER MATCH CHECK (The "Rifleman" Rule)
            if role_gender and role_gender == pronoun_gender:
                continue 

            # 2. CONTEXT CHECK (The "Mixed Mode" Rule)
            verb = get_governing_verb(token)
            
            # "so he DRANK coffee" (Past) -> Safe
            if is_episodic_context(verb):
                continue 
            
            # "He SHOULD be ready", "He IS usually", "IF he works" -> BIAS
            if is_generic_context(verb):
                bias_report.append({
                    "start": span.start_char,
                    "end": span.end_char,
                    "text": word,
                    "context": span.sent.text,
                    "reason": f"Generic context ({verb.text if verb else '?'}) linked to unanchored role ({head_span.text})"
                })

    return bias_report