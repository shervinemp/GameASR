from typing import List, Optional

from ..common.config import config
from ..common.utils import get_logger


class Embedder:

    def __init__(self, model_name: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.model_name = model_name or config.get(
            "llm.models.embedding", "google/embeddinggemma-300m"
        )
        self._model = None

    def _load(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        self.logger.info("Embedding model loaded: %s", self.model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            self._load()
        return self._model.encode(texts, normalize_embeddings=True).tolist()
