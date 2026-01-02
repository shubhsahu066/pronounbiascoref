# print("STEP 1: starting script")

# import torch
# print("STEP 2: torch imported")

# from transformers import AutoTokenizer, AutoModelForTokenClassification
# print("STEP 3: transformers imported")

# print("STEP 4: loading tokenizer")
# tokenizer = AutoTokenizer.from_pretrained("./pronoun_model_roberta", add_prefix_space=True)
# print("STEP 5: tokenizer loaded")

# print("STEP 6: loading model")
# model = AutoModelForTokenClassification.from_pretrained("./pronoun_model_roberta")
# model.eval()
# print("STEP 7: model loaded")

# import spacy
# print("STEP 8: spaCy imported")

# print("STEP 9: loading spaCy model")
# nlp = spacy.load("en_core_web_sm")
# print("STEP 10: spaCy model loaded")
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

GENERIC_DETERMINERS = {"a", "an", "each", "every"}
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

def is_episodic(sentence: str) -> bool:
    doc = nlp(sentence)
    
    SPECIFIC_MODALS = {"will", "wo", "shall", "'ll", "can"} 

    # We iterate over EVERY token, not just the ROOT.
    # If we find ANY evidence that this is a specific story, we return True (Ignore).
    for tok in doc:
        
        # 1. PAST TENSE (VBD) -> Specific
        # e.g., "The nurse checked..." or "The nurse was checking"
        if tok.tag_ == "VBD":
            return True

        # 2. PROGRESSIVE ASPECT (VBG) -> Specific
        # e.g., "The nurse is checking..."
        # This fixes your "checking vitals" sentence immediately.
        if tok.tag_ == "VBG": 
            return True
        
        # 3. PERFECT ASPECT (VBN + Have) -> Specific
        # e.g., "The nurse has checked..."
        if tok.tag_ == "VBN":
            for child in tok.children:
                if child.lemma_ == "have":
                    return True

        # 4. "THE + SPECIFIC MODAL" (Modified for multi-clause)
        # Check if this token is a specific modal (will/can)
        if tok.lemma_.lower() in SPECIFIC_MODALS:
            # Look at the verb this modal is attached to (the head)
            verb = tok.head
            # Now find the subject of that verb
            for child in verb.children:
                if child.dep_ == "nsubj":
                    # CASE A: Subject is "The [Noun]" (e.g., "The secretary will...")
                    for grandchild in child.children:
                        if grandchild.dep_ == "det" and grandchild.lemma_.lower() == "the":
                            return True
                    
                    # CASE B: Subject is a Pronoun ("she"), but refers to "The [Noun]"
                    # e.g. "The nurse is busy, she will return."
                    # If we see "will" + "she", we scan the rest of the doc for "The [Noun]"
                    if child.pos_ == "PRON":
                        # Scan previous tokens for a "The [Noun]" anchor
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
        # Quantifiers anywhere
        if token.lemma_.lower() in GENERIC_QUANTIFIERS:
            return True

        # Generic common nouns with determiners anywhere
        if token.pos_ == "NOUN":
            for child in token.children:
                if child.dep_ == "det":
                    det = child.lemma_.lower()
                    if det in GENERIC_DETERMINERS or det == "the":
                        return True
    return False



def is_predicate_possessive(sentence: str, pronoun: str) -> bool:
    if pronoun.lower() not in {"his", "hers", "theirs"}:
        return False

    words = sentence.lower().split()
    for i in range(len(words) - 1):
        if words[i] in COPULAR_VERBS and words[i + 1] == pronoun.lower():
            return True
    return False


def is_candidate_pronoun(token_text: str) -> bool:
    t = token_text.lower()
    if t in NEUTRAL_PRONOUNS:
        return False
    if t in REFLEXIVE_PRONOUNS:
        return False
    return t in GENDERED_PRONOUNS
    # return t in GENDERED_PRONOUNS or t in REFLEXIVE_PRONOUNS


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
    debug = []

    generic_subject = has_generic_subject(text)
    named_person = contains_named_person(text)

    for idx, ((start, end), pred) in enumerate(zip(offset_mapping, preds)):
        start, end = start.item(), end.item()
        if start == end:
            continue

        token_text = text[start:end]
        label = LABELS[pred]
        # print(token_text, LABELS[pred])


        if DEBUG_MODE:
            debug.append({
                "token": token_text,
                "label": label
            })

        if not is_candidate_pronoun(token_text):
          continue


        # ---- restored rules ----
        if start == 0:
            continue

        if not is_candidate_pronoun(token_text):
            continue

        if is_predicate_possessive(text, token_text):
            continue

        if named_person:
            continue

        if not generic_subject:
            continue

        # 🔑 EPISODIC FILTER 
        if is_episodic(text):
            continue

        spans.append({
            "start": start,
            "end": end,
            "type": "PRONOUN"
        })

    result = {
        "text": text,
        "spans": spans
    }

    if DEBUG_MODE:
        result["debug"] = debug

    return result

if __name__ == "__main__":
    examples = [
        "He is responsible for managing the project.",
        "She naturally took charge of the discussion.",
        "He led the initiative successfully.",
        "She oversaw the operation last year.",
        "He should review the document.",
        "She can resolve complex issues.",

        "The final authority was his.",
        "Final accountability lay with her.",
        "Ultimate responsibility rested with him.",
        "Control over the outcome was entirely hers.",
        "Ownership of the decision was clearly his.",
        "The responsibility for approval was solely hers.",

        "HE is clearly in charge.",
        "SHE made the final call.",
        "He said that SHE would decide.",
        "She insisted that HE take control.",

        "He told her that she should decide.",
        "She reminded him that he was responsible.",
        "He challenged himself to improve.",
        "She prepared herself for the presentation.",

        "He walked into the room quietly.",
        "She opened the window and left.",
        "They arrived at the station early.",
        "If he calls, take a message.",
        "After she left, the lights were turned off.",

        "They are responsible for overseeing the process.",
        "The manager said they would review the report.",
        "Control over the outcome was entirely theirs.",
        "The final decision was theirs.",
        "Responsibility was clearly defined as theirs.",

        "The therapist said he would call later.",
        "The nurse explained that she would return shortly.",
        "A teacher mentioned that he was unavailable.",
        "The assistant confirmed that she had sent the email.",

        "When he arrived, the meeting had already started.",
        "Before she entered, the discussion had ended.",
        "After they finished, the results were published.",

        "The final decision was made yesterday.",
        "The project was completed on time.",
        "Responsibility was clearly defined.",
        "A decision was made after discussion.",
        "A student must submit his homework.",
        "The nurse said she would return.",
        "The book was his.",
        "The bag is hers.",
        "The keys are theirs.",

        "Someone said he might arrive later.",
        "Someone said she might arrive later.",
        "Someone said they might arrive later.",
        "A monk should preserve his sanity.",
        "They completed their assignment.",
        "John said he would arrive.",
        "when a student applies, his father should take care of his admission",
        "when a student applies, his guardian should take care of his admission",
        "When a student applies, the guardian should show full dedication to ensure his student's admission",


        "The customer is always right, and he deserves respect.",
        "When a child learns to read, he opens up a new world.",
        "Every citizen must pay his taxes on time.",
        "The developer pushed his code to the repository.",
        "A developer pushes his code to the repo.",
        "Every developer works on his github account.",
        "Consult your doctor and ask him about side effects.",
        "A good CEO puts his employees first.",
        "The scientist published his findings in the journal.",
        "When the police officer arrives, he will take a statement.",
        "The pilot announced that he was beginning the descent.",
        "A politician should always keep his promises.",
        "Ask the nurse if she has checked the patient's vitals.",
        "The secretary will file the report when she returns.",
        "A teacher often spends her own money on supplies.",
        "The flight attendant asked if I needed help, and she smiled.",
        "Call the receptionist and tell her to hold my calls.",
        "Everyone must submit his application by Friday.",
        "A pedestrian must look both ways before he crosses the street.",
        "If anyone has a question, he should raise a hand.",
        "If a user fails to log in, he should reset the password.",
        "Each student is responsible for his own locker.",
        "The customer john said that he will come soon",
        "The customer said that he will come soon",
        "The teacher is always right, no matter how he reacts",
        "The customer who complained, he is such a douchebag.",
        "The customer said he wanted a refund",
        "The customer should always check the receipt, and he should keep it safe.",
        "The president said he would address the nation.",
        "A nurse should take care of herself",#coreferencing needed maybe,
        #  probable heuristis -> sen. containing re.
        #  pronouns are not marked if they do not refer to generic thing
        "A nurse should take care of her health",
        "He hurt himself",
        "The author hurt himself",
        "A cricketer takes care of himself.",#same thing
        "The final authority is his.",
        "The kitchen duties are hers.",
        "This is his decision.",#attributive possessive,
        "Yesterday I met a mother, the mother really thought she was a psycho",
        "Shubh hurt himself",
        "Shubh thought he would win.",
        "John hurt himself",
        "The customer, he hurt himself.",
        "HE should really avoid being full of himself all the time.",

        "The nurse is checking the patient's vitals, she will return to us ASAP.",
        "The nurse is checking the charts, she is working well.",
        "The nurse has checked charts, she was really fast.",
        "The nurse, she checks charts.",
        "The nurse checks charts, she...",
        "The nurse, she is kind",
        "The pilot, he is going to fly."



    ]

    for e in examples:
        print(detect_pronoun_bias(e))


# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification

# MODEL_PATH = "./pronoun_model_roberta"
# LABELS = ["O", "B-PRONOUN"]

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, add_prefix_space=True)
# model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
# model.eval()

# # Canonical lowercase sets
# GENDERED_PRONOUNS = {
#     "he", "she",
#     "him", "her",
#     "his", "hers",
#     "himself", "herself"
# }

# NEUTRAL_PRONOUNS = {
#     "they", "them",
#     "their", "theirs",
#     "themselves", "themself"
# }

# def detect_pronoun_bias(text):
#     encoding = tokenizer(
#         text,
#         return_offsets_mapping=True,
#         return_tensors="pt"
#     )

#     offset_mapping = encoding.pop("offset_mapping")[0]

#     with torch.no_grad():
#         outputs = model(**encoding)
#         preds = torch.argmax(outputs.logits, dim=-1)[0]

#     spans = []

#     for pred, (start, end) in zip(preds, offset_mapping):
#         start = start.item()
#         end = end.item()

#         if start == end:
#             continue

#         if LABELS[pred] != "B-PRONOUN":
#             continue

#         span_text = text[start:end]
#         span_lower = span_text.lower()

#         # Skip neutral pronouns
#         if span_lower in NEUTRAL_PRONOUNS:
#             continue

#         # Accept ALL gender-marked pronouns (including reflexives)
#         if span_lower in GENDERED_PRONOUNS:
#             spans.append({
#                 "start": start,
#                 "end": end,
#                 "type": "PRONOUN"
#             })

#     return {
#         "text": text,
#         "spans": spans
#     }


# if __name__ == "__main__":
#     examples = [
#         "He is responsible for managing the project.",
#         "She naturally took charge of the discussion.",
#         "He led the initiative successfully.",
#         "She oversaw the operation last year.",
#         "He should review the document.",
#         "She can resolve complex issues.",

#         "The final authority was his.",
#         "Final accountability lay with her.",
#         "Ultimate responsibility rested with him.",
#         "Control over the outcome was entirely hers.",
#         "Ownership of the decision was clearly his.",
#         "The responsibility for approval was solely hers.",

#         "HE is clearly in charge.",
#         "SHE made the final call.",
#         "He said that SHE would decide.",
#         "She insisted that HE take control.",

#         "He told her that she should decide.",
#         "She reminded him that he was responsible.",
#         "He challenged himself to improve.",
#         "She prepared herself for the presentation.",

#         "He walked into the room quietly.",
#         "She opened the window and left.",
#         "They arrived at the station early.",
#         "If he calls, take a message.",
#         "After she left, the lights were turned off.",

#         "They are responsible for overseeing the process.",
#         "The manager said they would review the report.",
#         "Control over the outcome was entirely theirs.",
#         "The final decision was theirs.",
#         "Responsibility was clearly defined as theirs.",

#         "The therapist said he would call later.",
#         "The nurse explained that she would return shortly.",
#         "A teacher mentioned that he was unavailable.",
#         "The assistant confirmed that she had sent the email.",

#         "When he arrived, the meeting had already started.",
#         "Before she entered, the discussion had ended.",
#         "After they finished, the results were published.",

#         "The final decision was made yesterday.",
#         "The project was completed on time.",
#         "Responsibility was clearly defined.",
#         "A decision was made after discussion.",
#         "A student must submit his homework.",
#         "The nurse said she would return.",
#         "The book was his.",
#         "The bag is hers.",
#         "The keys are theirs.",

#         "Someone said he might arrive later.",
#         "Someone said she might arrive later.",
#         "Someone said they might arrive later.",
#         "A monk should preserve his sanity.",
#         "They completed their assignment.",
#         "John said he would arrive.",
#         "when a student applies, his father should take care of his admission",
#         "when a student applies, his guardian should take care of his admission",
#         "When a student applies, the guardian should show full dedication to ensure his student's admission",


#         "The customer is always right, and he deserves respect.",
#         "When a child learns to read, he opens up a new world.",
#         "Every citizen must pay his taxes on time.",
#         "The developer pushed his code to the repository.",
#         "A developer pushes his code to the repo.",
#         "Every developer works on his github account.",
#         "Consult your doctor and ask him about side effects.",
#         "A good CEO puts his employees first.",
#         "The scientist published his findings in the journal.",
#         "When the police officer arrives, he will take a statement.",
#         "The pilot announced that he was beginning the descent.",
#         "A politician should always keep his promises.",
#         "Ask the nurse if she has checked the patient's vitals.",
#         "The secretary will file the report when she returns.",
#         "A teacher often spends her own money on supplies.",
#         "The flight attendant asked if I needed help, and she smiled.",
#         "Call the receptionist and tell her to hold my calls.",
#         "Everyone must submit his application by Friday.",
#         "A pedestrian must look both ways before he crosses the street.",
#         "If anyone has a question, he should raise a hand.",
#         "If a user fails to log in, he should reset the password.",
#         "Each student is responsible for his own locker.",
#         "The customer john said that he will come soon",
#         "The customer said that he will come soon",
#         "The teacher is always right, no matter how he reacts",
#         "The customer who complained, he is such a douchebag.",
#         "The customer said he wanted a refund",
#         "The customer should always check the receipt, and he should keep it safe.",
#         "The president said he would address the nation.",
#         "A nurse should take care of herself",#coreferencing needed maybe,
#         #  probable heuristis -> sen. containing re.
#         #  pronouns are not marked if they do not refer to generic thing
#         "A nurse should take care of her health",
#         "He hurt himself",
#         "The author hurt himself",
#         "A cricketer takes care of himself.",#same thing
#         "The final authority is his.",
#         "The kitchen duties are hers.",
#         "This is his decision.",#attributive possessive,
#         "Yesterday I met a mother, the mother really thought she was a psycho",
#         "Shubh hurt himself",
#         "Shubh thought he would win.",
#         "John hurt himself",
#         "The customer, he hurt himself.",
#         "HE should really avoid being full of himself all the time."
#     ]

#     for e in examples:
#         print(detect_pronoun_bias(e))


