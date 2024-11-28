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
        """초기화 메서드
        Embedding BentoML 서비스 클래스의 인스턴스를 초기화합니다.
        """
        from sentence_transformers import SentenceTransformer

        # bentofile.yaml에 env 변수를 참조하여 특정 모델을 초기화합니다
        self.embedding_model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL"), trust_remote_code=True
        )

    @bentoml.api
    def multiple_embed(self, sentences: list[str]) -> np.ndarray:
        """list에 담긴 각 sentence를 embedding 모델로 embed하는 API 함수

        Args:
            sentences (list[str]): Embedding하고자 하는 input 문장/문단이 담긴 list

        Returns:
            np.ndarray: embedding 결과 [(len(sentences), embed_dim) shape의 ndarray 반환]
        """
        embeddings = self.embedding_model.encode(sentences)

        return embeddings

    def embed(self, sentences: str) -> np.ndarray:
        """string input 문장을 embedding 모델로 embed하는 API 함수

        Args:
            sentences (str): Embedding하고자 하는 input 문장/문단

        Returns:
            np.ndarray: embedding 결과 [(embed_dim, ) shape의 ndarray 반환]
        """
        embeddings = self.embedding_model.encode(sentences)
