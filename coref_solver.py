from fastcoref import FCoref

class CorefResolver:
    def __init__(self):
        self.model = FCoref()

    def resolve(self, text: str):
        pred = self.model.predict(
            texts=[text],
            is_split_into_words=False
        )[0]

        # FastCoref-safe outputs
        clusters = pred.get_clusters(as_strings=False)
        document = text  # we keep raw text, not tokens

        return document, clusters
