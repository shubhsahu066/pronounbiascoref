# knowledge_base.py

# =========================
# GENDERED ROLES & PRONOUNS
# =========================

GENDERED_ROLES = {
    # Male roles
    "rifleman": "M", "policeman": "M", "fireman": "M", "king": "M", "actor": "M",
    "waiter": "M", "steward": "M", "hero": "M", "uncle": "M", "father": "M", 
    "brother": "M", "son": "M", "man": "M", "boy": "M", "gentleman": "M",
    "lad": "M", "guy": "M", "fellow": "M", "chap": "M", "bloke": "M", "mr": "M", 
    "sir": "M", "lord": "M", "prophet": "M", "monk": "M", "prince": "M",
    "husband": "M", "groom": "M", "grandson": "M",
    
    # Female roles
    "riflewoman": "F", "policewoman": "F", "firewoman": "F", "queen": "F", 
    "actress": "F", "waitress": "F", "stewardess": "F", "heroine": "F",
    "aunt": "F", "mother": "F", "sister": "F", "daughter": "F", "woman": "F",
    "girl": "F", "lady": "F", "mrs": "F", "ms": "F", "madam": "F", "nun": "F",
    "princess": "F", "wife": "F", "bride": "F", "granddaughter": "F"
}

PRONOUN_MAP = {
    "he": "M", "him": "M", "his": "M", "himself": "M",
    "she": "F", "her": "F", "hers": "F", "herself": "F"
}

# Modifiers that explicitly gender a noun (e.g., "Male nurse")
MALE_MODIFIERS = {"male", "man", "boy", "gentleman", "mr", "mr.", "masculine"}
FEMALE_MODIFIERS = {"female", "woman", "lady", "girl", "mrs", "mrs.", "ms", "ms.", "feminine"}

# =========================
# LINGUISTIC MARKERS
# =========================

FREQUENCY_ADVERBS = {
    "always", "constantly", "continually", "forever", "usually", "often", 
    "never", "typically", "generally", "frequently", "rarely", "seldom"
}

# OBLIGATION: Bias likely (unless anchored by Name).
# Removed "have" to avoid flagging perfect tense ("has eaten").
OBLIGATION_MODALS = {"should", "must", "ought", "need"} 

# PREDICTION: Bias only if Indefinite ('A'), Safe if Definite ('The').
PREDICTION_MODALS = {"will", "would", "can", "could", "may", "might", "shall"}

ALL_MODALS = OBLIGATION_MODALS | PREDICTION_MODALS

CONDITIONAL_MARKERS = {"if", "unless", "whenever", "whether"}