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
    def embed(self, sentences: Union[str, list]) -> np.ndarray:
        """Input 문장을 embedding 모델로 embed하는 API 함수

        Args:
            sentences (Union[str, list]): Embedding하고자 하는 input 문장/문단

        Returns:
            np.ndarray: embedding 결과
                - sentences가 str이면 (embed_dim, ) shape의 ndarray 반환
                - sentences가 list면 (len(sentences), embed_dim) shape의 ndarray 반환
        """
        embeddings = self.embedding_model.encode(sentences)

        return embeddings
