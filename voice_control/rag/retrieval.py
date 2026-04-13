from abc import ABC, abstractmethod
import json
import re
from typing import Any, List, Dict, Tuple
from itertools import chain, cycle, combinations

from ..llm.session import Session
from .knowledge import KnowledgeGraph
from ..common.utils import get_logger


class Reranker:
    def __init__(
        self,
        cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.cross_encoder_name = cross_encoder
        self.reranker = None

    def __call__(
        self, query: str, results: List[str]
    ) -> Tuple[List[str], List[float]]:
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.cross_encoder_name)

        truncated_results = [r[:500] for r in results]
        pairs = list(zip(cycle([query]), truncated_results))
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        sorted_pairs = sorted(
            zip(results, scores), key=lambda x: x[1], reverse=True
        )

        sorted_results, scores = zip(*sorted_pairs)

        return sorted_results, scores


class Retriever(ABC):
    """The unified interface expected by the RAG orchestrator."""
    @abstractmethod
    def __call__(self, query: str, **kwargs) -> List[str]:
        pass


class GraphSearchStrategy(ABC):
    """Defines the specific mathematical approach to traversing the Knowledge Graph."""
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    @abstractmethod
    def search(self, keywords: List[str], **kwargs) -> List[Dict]:
        pass

    @abstractmethod
    def format_results(self, results: List[Dict]) -> List[str]:
        pass


class NeighborhoodStrategy(GraphSearchStrategy):
    """Standard 1-hop to N-hop semantic neighborhood expansion."""

    def search(self, keywords: List[str], top_k_vector: int = 7, top_k_keyword: int = 5, n_hops: int = 1, **kwargs) -> List[Dict]:
        if not self.graph:
            return []

        embeddings = self.graph.embedding_model.encode(keywords).tolist()
        vector_results = self.graph.vector_search(embeddings, top_k=top_k_vector)
        keyword_results = self.graph.keyword_search(keywords, top_k=top_k_keyword)

        combined = {n["id"]: n for n in chain.from_iterable(vector_results)}
        for item in chain.from_iterable(keyword_results):
            if (id_ := item["id"]) not in combined:
                combined[id_] = item

        results = list(combined.values())
        if n_hops:
            seed_nodes = self.graph.subgraph([n["id"] for n in results]).get("nodes", [])
            expansion = self.graph.expansion(
                frontier_ids=[n["id"] for n in results], excluded_ids=[n["id"] for n in results], n_hops=n_hops
            )
            results = seed_nodes + [item for sublist in expansion for item in sublist]

        return results

    def format_results(self, results: List[Dict]) -> List[str]:
        formatted = []
        for item in results:
            if "parent" in item and "node" in item:
                formatted.append(f"{item.get('parent', {}).get('label', '')} -[{item.get('relationship', {}).get('type', 'RELATED_TO')}]-> {item.get('node', {}).get('label', '')}")
            else:
                formatted.append(f"{item.get('label', '')}: {item.get('description', '')}")
        return list(dict.fromkeys(formatted))


class ShortestPathStrategy(GraphSearchStrategy):
    """Semantic-aware multi-hop shortest path logic between multiple anchors."""

    def search(self, keywords: List[str], top_k_vector: int = 3, max_paths: int = 3, **kwargs) -> List[Dict]:
        if not self.graph or len(keywords) < 2:
            return [] # S-Path mathematically requires 2+ anchors

        embeddings = self.graph.embedding_model.encode(keywords).tolist()
        batch_results = self.graph.vector_search(embeddings, top_k=top_k_vector)

        anchor_nodes = list(set(
            node["id"] for res_list in batch_results for node in res_list if node.get("id")
        ))

        all_paths = []
        for src, tgt in combinations(anchor_nodes, 2):
            all_paths.extend(self.graph.k_shortest_paths(source_id=src, target_id=tgt, k=max_paths))
        return all_paths

    def format_results(self, results: List[Dict]) -> List[str]:
        formatted = []
        for path_data in results:
            if "nodes" not in path_data or "relations" not in path_data:
                continue
            trace = []
            for i, node in enumerate(path_data["nodes"]):
                trace.append(node.get("label", "Unknown"))
                if i < len(path_data["relations"]):
                    trace.append(f"-[{path_data['relations'][i].get('type', 'RELATED_TO')}]->")
            formatted.append(" ".join(trace))
        return list(dict.fromkeys(formatted))


class SmartGraphRetriever(Retriever):
    """
    Coordinates LLM intent extraction and delegates to the injected Graph Strategies.
    """
    def __init__(self, session: Session, primary_strategy: GraphSearchStrategy, fallback_strategy: GraphSearchStrategy | None = None):
        self.logger = get_logger(__file__)
        self.session = session
        self.primary = primary_strategy
        self.fallback = fallback_strategy

    def _extract_keywords(self, query: str) -> List[str]:
        from ..common.utils import safe_json_loads
        prompt = (
            "Extract key entities and keywords from the following query. "
            "Focus on the most important terms representing the core intent. Return a JSON array of strings."
            f'\nQuery: "{query}"'
        )
        response = "".join(self.session(prompt)).strip()
        return safe_json_loads(response, fallback=query.split())

    def __call__(self, query: str, **kwargs) -> List[str]:
        try:
            keywords = self._extract_keywords(query)
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed. Error: {e}")
            keywords = [query]

        # 1. Execute Primary Strategy
        raw_results = self.primary.search(keywords, **kwargs)

        # 2. Seamless Fallback (e.g., if SPath lacks 2+ keywords)
        if not raw_results and self.fallback:
            self.logger.info(f"Primary strategy yielded no results. Engaging fallback.")
            raw_results = self.fallback.search(keywords, **kwargs)
            return self.fallback.format_results(raw_results)

        return self.primary.format_results(raw_results)


class WebRetriever(Retriever):
    def __init__(self, session: Session):
        from ddgs import DDGS

        self.logger = get_logger(__file__)

        self.session = session
        self.ddgs = DDGS()

    def __call__(self, query: str, **kwargs) -> List[str]:
        search_results = self.search(query=query, **kwargs)
        return self.format_results(search_results)

    def search(self, query: str, *, top_k: int = 3) -> List[dict]:
        """Performs a web search for the given query using the DuckDuckGo API."""
        search_query = query
        try:
            search_query = self.transform_query(query=query)
        except Exception as e:
            self.logger.warning(
                f"Failed to extract keywords due to an error. Using original query as a keyword. Error: {e}"
            )

        self.logger.info(f"Searching for '{search_query}'...")
        results = []
        try:
            results = self.ddgs.text(search_query, max_results=top_k)
            if not results:
                self.logger.warning(
                    f"Web search for '{search_query}' yielded no results."
                )
        except Exception as e:
            self.logger.error(
                f"Web search failed for query '{search_query}': {e}"
            )

        return results

    def transform_query(self, query: str) -> List[str]:
        prompt = (
            "Transform the following conversational query into a concise, keyword-based search engine query. "
            "For example, 'Can you tell me who the members of the band Coldplay are?' should become {'search_query': 'Coldplay band members'}."
            f'\nConversational Query: "{query}"'
            f"\nSearch Engine Query: "
        )

        response = "".join(self.session(prompt))
        search_query = json.loads(response)["search_query"].strip()

        return search_query

    def format_results(self, results: List[dict]) -> List[str]:
        return [
            f"{r.get('title', '')}: {r.get('body', '')} -- <href>{r.get('href', '#')}</href>"
            for r in results
        ]
