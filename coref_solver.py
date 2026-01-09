from allennlp.predictors import Predictor
import allennlp_models.coref

class CorefResolver:
    def __init__(self):
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/"
            "coref-spanbert-large-2021.03.10.tar.gz"
        )

    def resolve(self, text: str):
        """
        Returns:
          document: list of tokens
          clusters: list of [start, end] token spans
        """
        output = self.predictor.predict(document=text)
        return output["document"], output["clusters"]
