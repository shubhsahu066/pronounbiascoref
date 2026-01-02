import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import spacy
import os

# =========================
# CONFIG
# =========================

MODEL_PATH = "./pronoun_model_roberta"
LABELS = ["O", "B-PRONOUN"]

GENDERED_PRONOUNS = {"he", "she", "him", "her", "his", "hers"}
NEUTRAL_PRONOUNS = {"they", "them", "their", "theirs"}
REFLEXIVE_PRONOUNS = {"himself", "herself"}

COPULAR_VERBS = {"is", "was", "are", "were"}

GENERIC_DETERMINERS = {"a", "an", "each", "every" ,"any"}
GENERIC_QUANTIFIERS = {
    "everyone", "someone", "anyone",
    "everybody","anybody", "each"
}

DEBUG_MODE = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =========================
# LOAD MODEL
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

# =========================
# NLP
# =========================

nlp = spacy.load("en_core_web_sm")

# =========================
# HELPERS
# =========================

def is_specific_reflexive(text: str, start: int, end: int) -> bool:
    """
    FIX FOR REFLEXIVES:
    Distinguishes between "The author hurt himself" (Specific -> Ignore) 
    and "A nurse helps herself" (Generic -> Mark).
    Returns True if the reflexive refers to a 'The + Noun' subject.
    """
    doc = nlp(text)
    span = doc.char_span(start, end)
    
    if span is None: 
        return False
    
    token = span[0]
    
    # 1. Traverse up to find the main verb governing this reflexive
    head = token.head
    # Climb up if we are in a prepositional phrase (e.g. "take care OF herself")
    while head.pos_ != "VERB" and head.dep_ != "ROOT" and head != head.head:
        head = head.head
        
    # 2. Find the subject (nsubj) of that verb
    subject = None
    for child in head.children:
        if child.dep_ == "nsubj":
            subject = child
            break
            
    # 3. Check if subject has "The" as determiner
    if subject:
        for child in subject.children:
            if child.dep_ == "det" and child.lemma_.lower() == "the":
                return True # "The author..." -> Specific
                
    return False

def is_bare_conditional(text: str, start: int, end: int) -> bool:
    """
    FIX FOR ISSUE 2: "If he calls..."
    """
    doc = nlp(text)
    span = doc.char_span(start, end)
    
    if span is None: 
        return False
    
    token = span[0]

    if token.dep_ == "nsubj":
        head_verb = token.head
        for child in head_verb.children:
            if child.dep_ == "mark" and child.lemma_.lower() == "if":
                return True
                
    return False

def is_episodic(sentence: str) -> bool:
    # REVERTED TO ORIGINAL
    doc = nlp(sentence)
    
    SPECIFIC_MODALS = {"will", "wo", "shall", "'ll", "can"} 

    for tok in doc:
        # 1. PAST TENSE (VBD)
        if tok.tag_ == "VBD": return True

        # 2. PROGRESSIVE ASPECT (VBG)
        if tok.tag_ == "VBG": return True
        
        # 3. PERFECT ASPECT (VBN + Have)
        if tok.tag_ == "VBN":
            for child in tok.children:
                if child.lemma_ == "have": return True

        # 4. "THE + SPECIFIC MODAL"
        if tok.lemma_.lower() in SPECIFIC_MODALS:
            verb = tok.head
            for child in verb.children:
                if child.dep_ == "nsubj":
                    # CASE A: Subject is "The [Noun]"
                    for grandchild in child.children:
                        if grandchild.dep_ == "det" and grandchild.lemma_.lower() == "the":
                            return True
                    # CASE B: Subject is Pronoun referring to "The"
                    if child.pos_ == "PRON":
                        for potential_anchor in doc:
                            if potential_anchor.pos_ == "NOUN":
                                for det in potential_anchor.children:
                                    if det.lemma_.lower() == "the":
                                        return True
    return False

def contains_named_person(sentence: str) -> bool:
    doc = nlp(sentence)
    return any(ent.label_ == "PERSON" for ent in doc.ents)

def has_generic_subject(sentence: str) -> bool:
    doc = nlp(sentence)
    for token in doc:
        if token.lemma_.lower() in GENERIC_QUANTIFIERS:
            return True
        if token.pos_ == "NOUN":
            for child in token.children:
                if child.dep_ == "det":
                    det = child.lemma_.lower()
                    if det in GENERIC_DETERMINERS or det == "the":
                        return True
    return False

def is_predicate_possessive(sentence: str, pronoun: str) -> bool:
    # 1. Filter irrelevant pronouns
    if pronoun.lower() not in {"his", "hers", "theirs"}:
        return False

    # 2. Use SpaCy tokens to safely separate "hers" from "."
    doc = nlp(sentence)
    
    for i, token in enumerate(doc):
        # Match the detected pronoun (SpaCy knows "hers" != ".")
        if token.text.lower() == pronoun.lower():
            # Check the previous word
            if i > 0:
                prev_token = doc[i - 1]
                if prev_token.text.lower() in COPULAR_VERBS:
                    return True
                    
    return False

def is_candidate_pronoun(token_text: str) -> bool:
    # MODIFIED: ALLOW REFLEXIVES NOW
    t = token_text.lower()
    if t in NEUTRAL_PRONOUNS: return False
    # if t in REFLEXIVE_PRONOUNS: return False  <-- REMOVED BLOCK
    return t in GENDERED_PRONOUNS or t in REFLEXIVE_PRONOUNS

# =========================
# MAIN INFERENCE
# =========================

def detect_pronoun_bias(text: str):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True
    )
    offset_mapping = encoding.pop("offset_mapping")[0]

    with torch.no_grad():
        outputs = model(**encoding)
        preds = torch.argmax(outputs.logits[0], dim=-1)

    spans = []
    
    generic_subject = has_generic_subject(text)
    named_person = contains_named_person(text)

    for idx, ((start, end), pred) in enumerate(zip(offset_mapping, preds)):
        start, end = start.item(), end.item()
        if start == end: continue

        token_text = text[start:end]
        label = LABELS[pred]

        if not is_candidate_pronoun(token_text): continue

        # ---- LOGIC FLOW ----
        
        # 1. Start of sentence check (Strict Old Rule)
        if start == 0: continue

        # 2. Predicate Possessive check
        if is_predicate_possessive(text, token_text): continue

        # 3. Named Person check ("John... himself")
        if named_person: continue

        # 4. Generic Subject check
        if not generic_subject: continue

        # 5. Episodic check
        if is_episodic(text): continue
            
        # 6. Specific Syntax Fixes
        if is_bare_conditional(text, start, end): continue # Fix for "If he calls"
        
        # 7. Reflexive Specificity Fix
        # If it's a reflexive, we apply the "The vs A" logic
        if token_text.lower() in REFLEXIVE_PRONOUNS:
            if is_specific_reflexive(text, start, end): # If subject is "The..."
                continue

        spans.append({
            "start": start,
            "end": end,
            "type": "PRONOUN"
        })

    return {
        "text": text,
        "spans": spans
    }

if __name__ == "__main__":
    examples = [
        # ====================================================
        # 1. GENERIC BIAS (Target: DETECTED / MARKED)
        # ====================================================
        "A user should always update his password regularly.",
        "Every doctor must wash her hands before surgery.",
        "When a student fails, he must retake the exam.",
        "A developer usually pushes his code to the main branch.",
        "Each employee is responsible for his own keycard.",
        "If a customer complains, he should be treated with respect.",
        "A politician must keep his promises to the public.",
        "Every citizen must pay his taxes on time.",
        "A teacher often spends her own money on classroom supplies.",
        "When a child learns to read, he opens up a new world.",
        "A pilot checks his instruments before takeoff.",
        "Any user can reset his settings from the dashboard.", # Includes "Any"
        "A good CEO puts his employees first.",
        "A pedestrian must look both ways before he crosses.",
        "If anyone has a question, he should raise a hand.",
        "When a nurse arrives, she will check the vitals.",
        "Each student must submit his assignment by Friday.",
        "A firefighter risks his life for others.",
        "A lawyer should always defend his client.",
        "Every parent loves his child.",
        "A nurse should take care of herself.",   # Reflexive (Generic)
        "A cricketer takes care of himself.",    # Reflexive (Generic)
        "If a user fails to log in, he should reset the password.",
        "When a student applies, his guardian should take care.",

        # ====================================================
        # 2. EPISODIC / SPECIFIC (Target: IGNORED / EMPTY)
        # ====================================================
        "The nurse is checking the patient's vitals right now.",
        "The developer was coding all night long.",
        "A teacher has graded the exams already.",
        "The doctor checked the chart and left the room.",
        "The pilot is going to land the plane soon.",
        "The secretary will file the report tomorrow.",
        "The manager will review the documents later.",
        "A student was studying in the library.",
        "The user has logged in successfully.",
        "The customer is complaining about the service.",
        "The officer wrote a ticket and drove away.",
        "The nurse will return to the station shortly.",
        "A user was trying to access the file.",
        "The author has written a new book.",
        "The athlete is training for the olympics.",
        "The final decision was made yesterday.",
        "The project was completed on time.",
        "Responsibility was clearly defined.",
        "The meeting has ended.",
        "A decision was made after the discussion.",
        "The nurse said she would return.",
        "The assistant confirmed that she had sent the email.",
        "The customer is checking his receipt.",
        "The author hurt himself.",             # Reflexive (Specific 'The')
        "The boy washes himself.",              # Reflexive (Specific 'The')

        # ====================================================
        # 3. START OF SENTENCE (Target: IGNORED / EMPTY)
        # ====================================================
        "He is responsible for managing the project.",
        "She naturally took charge of the discussion.",
        "He led the initiative successfully.",
        "She oversaw the operation last year.",
        "He should review the document.",
        "She can resolve complex issues.",
        "He told her that she should decide.",
        "She reminded him that he was responsible.",
        "He challenged the decision.",
        "She prepared for the presentation.",
        "He walked into the room quietly.",
        "She opened the window and left.",
        "He said that she would decide.",
        "She insisted that he take control.",
        "He works hard every day.",
        "He hurt himself.",                     # Start=0 rule

        # ====================================================
        # 4. PREDICATE POSSESSIVES (Target: IGNORED / EMPTY)
        # ====================================================
        "The final authority was his.",
        "Final accountability lay with her.",
        "The book was his.",
        "The book is his.",
        "The bag is hers.",
        "The keys are theirs.",
        "Control over the outcome was entirely hers.",
        "Ownership of the decision was clearly his.",
        "The responsibility for approval was solely hers.",
        "The mistake was his.",
        "The victory is hers.",

        # ====================================================
        # 5. NAMED ENTITIES (Target: IGNORED / EMPTY)
        # ====================================================
        "John said he would arrive later.",
        "Mary thinks she is right.",
        "Mr. Smith checked his watch.",
        "Alice lost her keys.",
        "Dr. Jones cares about his patients.",
        "Elon Musk tweeted his opinion.",
        "Sarah finished her homework.",
        "Bob asked if he could leave.",
        "The customer John said that he will come soon.",
        "Shubh thought he would win.",
        "John is really full of himself.",

        # ====================================================
        # 6. FIX: BARE CONDITIONAL (Target: IGNORED / EMPTY)
        # ====================================================
        "If he calls, take a message.",
        "If she writes, tell her I am busy.",
        "If he arrives, show him in.",
        "If she asks, say nothing.",
        "If he fails, we will try again.",
        "If she leaves, lock the door.",
        "If he helps, thank him.",

        # ====================================================
        # 7. LEFT DISLOCATION (Target: MARKED)
        # ====================================================
        "The nurse, she is kind.",
        "The teacher, he is strict.",
        "The pilot, he is skilled.",
        "The mother, she knows best."
    ]

    for e in examples:
        print(detect_pronoun_bias(e))

    # print(f"{'ID':<4} | {'STATUS':<10} | {'SENTENCE'}")
    # print("-" * 80)
    
    # for i, text in enumerate(examples):
    #     result = detect_pronoun_bias(text)
    #     spans = result["spans"]
    #     status = "🔴 MARKED" if spans else "🟢 EMPTY"
    #     print(f"{i+1:03}  | {status:<10} | {text}")
    #     print(spans)