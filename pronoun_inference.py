import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "./pronoun_model_roberta"
LABELS = ["O", "B-PRONOUN"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

VALID_BIASED_PRONOUNS = {
    "he", "she", "him", "her", "his", "hers", "himself", "herself",
    "He", "She", "Him", "Her", "His", "Hers", "Himself", "Herself",
    "HE", "SHE", "HIM", "HER", "HIS", "HERS"
}

NEUTRAL_PRONOUNS = {
    "they", "them", "their", "theirs", "themself",
    "They", "Them", "Their", "Theirs", "Themself",
    "THEY", "THEM", "THEIR", "THEIRS"
}

REFLEXIVE_PRONOUNS = {
    "himself", "herself",
    "Himself", "Herself",
    "HIMSELF", "HERSELF"
}

def detect_pronoun_bias(text):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offset_mapping = encoding.pop("offset_mapping")[0]

    with torch.no_grad():
        outputs = model(**encoding)
        preds = torch.argmax(outputs.logits, dim=-1)[0]

    spans = []
    for pred, (start, end) in zip(preds, offset_mapping):
        start = start.item()
        end = end.item()

        if start == end:
            continue

        if LABELS[pred] != "B-PRONOUN":
            continue

        span_text = text[start:end]

        if span_text in NEUTRAL_PRONOUNS or span_text in REFLEXIVE_PRONOUNS:
            continue

        if span_text not in VALID_BIASED_PRONOUNS:
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
    ]

    for e in examples:
        print(detect_pronoun_bias(e))