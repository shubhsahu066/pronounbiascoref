# main.py
from coref_solver import CorefResolver
from bias_detector import detect_pronoun_bias

# 1. Initialize Model
# (We print this once so you know it's starting, but the loop will be silent)
print("Loading FastCoref model...")
resolver = CorefResolver(device='cpu') 

# 2. Test Cases (Using the advanced set we verified)
test_docs = [
    # Safe
    "The doctor arrived late. He was tired from the shift.",
    # Bias (Modal)
    "A teacher must always prepare his lessons carefully.",
    # Bias (Conditional)
    "If a teacher is late to class, he is usually penalized.",
    # Bias (Predicate Possessive)
    "A true leader accepts responsibility. The final choice is his alone.",
    # Safe (Specific)
    "The leader looked at the map. The final choice was his.",
    # TEST CASE 1: Long Narrative with Context Shifts
    """
    Dr. Aravind entered the emergency room at 2 AM. He had been working for 
    twelve hours straight. He checked the patient's vitals and ordered a scan immediately. 
    It is widely accepted that a doctor must always be vigilant, even when he is exhausted. 
    If he misses a single detail, he faces severe consequences. 
    However, Dr. Aravind remained focused. He knew that this specific patient needed him.
    """,

    # TEST CASE 2: Resume / Professional Summary
    """
    Candidate: Jane Doe
    Role: Senior Software Engineer
    
    Professional Summary:
    Jane is a dedicated professional who has led multiple backend teams. 
    She improved system latency by 40% last quarter. She successfully migrated 
    our legacy database to the cloud.
    
    Work Philosophy:
    I believe that a great software engineer should always take ownership of her code. 
    She must ensure that her documentation is clear for junior developers. 
    If she deploys to production, she is ultimately responsible for the system's stability.
    """


]

# 3. Run Pipeline
print("\n--- BIAS REPORT ---\n")

for i, text in enumerate(test_docs):
    # Step A: Resolve Coreference (Now Silent)
    clusters = resolver.resolve(text)
    
    # Step B: Detect Bias
    biases = detect_pronoun_bias(text, clusters)
    
    # Step C: Clean Output
    if biases:
        print(f"Document {i+1}: [BIAS DETECTED]")
        print(f"  Context: \"{text.strip()}\"")
        
        # Extract and print just the spans for downstream use
        bias_spans = [(b['start'], b['end'], b['text']) for b in biases]
        print(f"  Biased Spans (Start, End, Text): {bias_spans}")
        
        # Optional: Detailed reasoning
        for b in biases:
            print(f"    - Reason: {b['reason']}")
    else:
        print(f"Document {i+1}: [SAFE]")
    
    print("-" * 30)