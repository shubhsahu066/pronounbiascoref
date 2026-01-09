from coref_solver import CorefResolver
from inference import detect_pronoun_bias

text = """
Senior Software Engineer with 6 years experience.
He led backend migration and improved system reliability.
"""

coref = CorefResolver()
document, clusters = coref.resolve(text)

biases = detect_pronoun_bias(document, clusters)
print(biases)
