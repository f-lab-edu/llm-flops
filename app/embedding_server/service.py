import os
import bentoml
import numpy as np
import time

from dotenv import load_dotenv
from typing import Union

load_dotenv()


@bentoml.service
class SentenceEmbeddingService:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.embedding_model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL"), trust_remote_code=True
        )

    @bentoml.api
    def embed(self, sentences: Union[str, list]) -> np.ndarray:
        embeddings = self.embedding_model.encode(sentences)

        return embeddings
