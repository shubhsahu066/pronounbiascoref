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
    """Finds the true noun (Role) in the cluster."""
    best_span = spans[0]
    best_score = 0 
    for span in spans:
        root = span.root
        current_score = 0
        if root.ent_type_: current_score = 2
        elif root.pos_ in ["NOUN", "PROPN"]: current_score = 1
        if current_score > best_score:
            best_score = current_score
            best_span = span
    return best_span

def get_governing_verb(token):
    head = token.head
    while head.pos_ not in ["VERB", "AUX"] and head.head != head:
        head = head.head
    return head if head.pos_ in ["VERB", "AUX"] else None

def get_modifier_gender(span):
    """Checks for 'male manager' / 'female reporter'."""
    for child in span.root.children:
        if child.lower_ in MALE_MODIFIERS: return "M"
        if child.lower_ in FEMALE_MODIFIERS: return "F"
    return None

def has_present_aux(verb_token):
    for child in verb_token.children:
        if child.dep_ in ["aux", "auxpass"] and child.lemma_ == "be":
            if child.tag_ in ["VBZ", "VBP"]: return True
    return False

def has_frequency_adverb(verb_token):
    for child in verb_token.children:
        if child.dep_ == "advmod" and child.lemma_.lower() in FREQUENCY_ADVERBS:
            return True
    return False

def is_strictly_episodic(verb_token):
    """
    Checks for unambiguous specific events.
    Used for Dynamic Anchoring ("The doctor IS examining...").
    """
    if not verb_token: return False
    
    # 0. Check Frequency (Habitual != Episodic)
    if has_frequency_adverb(verb_token): return False

    # 1. Check Modals (Disqualifies 'Strictly Episodic' status)
    for child in verb_token.children:
        if child.dep_ == "aux" and child.lemma_.lower() in ALL_MODALS:
            return False
            
    # 2. Past Tense ("ran") -> Yes
    if verb_token.tag_ == "VBD": return True
    
    # 3. Present Continuous ("is running") -> Yes
    if verb_token.tag_ == "VBG" and has_present_aux(verb_token): return True
    
    return False

def is_generic_context(verb_token, is_anchored_entity, is_definite, recursion_depth=0):
    """
    is_anchored_entity: Strong Anchor (Name/Specific). Saves EVERYTHING.
    is_definite: Weak Anchor ("The"). Saves 'Will', but NOT Simple Present.
    """
    if not verb_token or recursion_depth > 5: return False

    # 1. RECURSION (Nested Contexts)
    # If nested inside a verb, check the PARENT.
    if verb_token.dep_ in ["xcomp", "ccomp", "advcl", "conj"]:
        parent_verb = get_governing_verb(verb_token)
        if parent_verb and parent_verb != verb_token:
            parent_is_generic = is_generic_context(parent_verb, is_anchored_entity, is_definite, recursion_depth + 1)
            # If parent is Safe (Episodic), child is Safe.
            if not parent_is_generic:
                return False 

    # 2. MODAL CHECK
    found_modal = None
    for child in verb_token.children:
        if child.dep_ == "aux" and child.lemma_.lower() in ALL_MODALS:
            found_modal = child.lemma_.lower()
            break
            
    if found_modal:
        # A. Obligation ("should") -> Bias unless Strong Anchor
        # "The teacher should" -> Bias. "Julian should" -> Safe.
        if found_modal in OBLIGATION_MODALS:
            if is_anchored_entity: return False
            return True 
            
        # B. Prediction ("will") -> Safe if Definite ("The") or Strong Anchor
        # "The officer will" -> Safe. "A teacher will" -> Bias.
        if found_modal in PREDICTION_MODALS:
            if is_anchored_entity or is_definite: return False 
            return True 

    # 3. CONDITIONAL CHECK
    for child in verb_token.children:
        if child.dep_ == "mark" and child.lemma_.lower() in CONDITIONAL_MARKERS:
            return True
        if child.dep_ == "advcl": 
             for grandchild in child.children:
                if grandchild.dep_ == "mark" and grandchild.lemma_.lower() in CONDITIONAL_MARKERS:
                    return True

    # 4. TENSE CHECK
    # Simple Present ("He works") -> Safe ONLY if Strong Anchor (Name).
    # "The teacher works" -> Bias. "Julian works" -> Safe.
    # Note: 'is_definite' does NOT save simple present. 
    # (e.g. "The adolescent completes" -> Bias)
    if verb_token.tag_ in ["VBP", "VBZ"]:
        if is_anchored_entity: return False
        return True 
        
    # Present Passive ("He is penalized")
    if verb_token.tag_ == "VBN" and has_present_aux(verb_token):
        if is_anchored_entity: return False
        return True

    return False

# =========================
# MAIN LOGIC
# =========================

def detect_pronoun_bias(text: str, clusters):
    doc = nlp(text)
    bias_report = []

    for cluster_indices in clusters:
        spans = [doc.char_span(s[0], s[1]) for s in cluster_indices if doc.char_span(s[0], s[1]) is not None]
        if not spans: continue
        
        cluster_text_set = {span.text.lower() for span in spans}
        if not any(p in cluster_text_set for p in PRONOUN_MAP): continue 

        head_span = get_best_head_span(spans)
        head_root = head_span.root
        
        # --- PHASE 1: ANCHORING ---
        is_anchored_entity = False # Strong Anchor (Names, Specific Context)
        is_definite = False # Weak Anchor ("The")
        
        # A. Static Checks
        for span in spans:
            if span.root.ent_type_ in ["PERSON", "ORG", "GPE"]:
                is_anchored_entity = True; break
            if span.root.pos_ in ["NOUN", "PROPN"]:
                for child in span.root.children:
                    # Strong Determiners
                    if child.lemma_ in ["this", "that", "my", "your", "our"]:
                        is_anchored_entity = True
                    # Definite Article
                    if child.lemma_ == "the":
                        is_definite = True
                    # Relative Clause ("The teacher who...")
                    if child.dep_ == "relcl":
                        is_anchored_entity = True
            if is_anchored_entity: break
        
        # B. Dynamic Anchoring (Contextual Specificity)
        # If "The doctor" is doing something Specific ("is examining"), he is Anchored.
        if not is_anchored_entity:
            for span in spans:
                verb = get_governing_verb(span.root)
                if is_strictly_episodic(verb):
                    is_anchored_entity = True 
                    break

        # --- PHASE 2: GENDER CHECK ---
        role_gender = GENDERED_ROLES.get(head_root.lemma_.lower())
        mod_gender = get_modifier_gender(head_span)
        if mod_gender: role_gender = mod_gender

        # --- PHASE 3: BIAS DETECTION ---
        for span in spans:
            token = span.root
            word = token.text.lower()
            if word not in PRONOUN_MAP: continue
            
            pronoun_gender = PRONOUN_MAP[word]

            if role_gender and role_gender == pronoun_gender:
                continue 

            verb = get_governing_verb(token)
            
            # Pass both flags:
            if is_generic_context(verb, is_anchored_entity, is_definite):
                bias_report.append({
                    "start": span.start_char,
                    "end": span.end_char,
                    "text": word,
                    "context": span.sent.text,
                    "reason": f"Generic context ({verb.text if verb else '?'}) linked to role"
                })

    return bias_report