from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple


class StorageBackend(ABC):
    """Abstract storage backend for graph/vector operations.

    Each backend implements labeled entity storage, vector + keyword search,
    graph traversal (expansion, shortest-path), and triplet persistence.
    """

    @abstractmethod
    def exact_label_search(self, labels: List[str]) -> Dict[str, Dict]:
        ...

    @abstractmethod
    def vector_search(
        self, embeddings: List[List[float]], top_k: int = 5,
        source_filter: str | None = None,
    ) -> List[List[Dict]]:
        ...

    @abstractmethod
    def keyword_search(
        self, keywords: List[str], top_k: int = 5
    ) -> List[List[Dict]]:
        ...

    @abstractmethod
    def subgraph(self, node_ids: List[str]) -> Dict[str, List[Dict]]:
        ...

    @abstractmethod
    def expansion(
        self,
        frontier_ids: List[str],
        excluded_ids: List[str],
        n_hops: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        ...

    @abstractmethod
    def k_shortest_paths_batch(
        self, pairs: List[Tuple[str, str]], k: int = 3
    ) -> List[Dict]:
        ...

    @abstractmethod
    def triplet_search(self, triplet: Dict) -> List[Dict]:
        ...

    @abstractmethod
    def add_triplets(self, triplets: List[Dict[str, str]]):
        ...

    def store_conversation(self, role: str, content: str):
        """Optional: store a conversation turn. Default no-op."""
        pass

    @abstractmethod
    def close(self):
        ...

    def verify_connectivity(self) -> bool:
        """Optional connectivity check. Default no-op."""
        return True

    def deadline(self, value: float | None):
        """Optional deadline context manager. Default no-op."""
        return nullcontext()
