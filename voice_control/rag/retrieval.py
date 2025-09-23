from abc import ABC, abstractmethod
import json
from typing import Any, List, Dict, Tuple
from itertools import chain, cycle

from ..llm.session import Session
from .knowledge import KnowledgeGraph
from ..common.utils import get_logger


class Reranker:
    def __init__(
        self,
        cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        from sentence_transformers import CrossEncoder

        self.reranker = CrossEncoder(cross_encoder)

    def __call__(
        self, query: str, results: List[str]
    ) -> Tuple[List[str], List[float]]:
        pairs = list(zip(cycle(query), results))
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        sorted_pairs = sorted(
            zip(results, scores), key=lambda x: x[1], reverse=True
        )

        sorted_results, scores = zip(*sorted_pairs)

        return sorted_results, scores


class Retriever(ABC):
    def __call__(
        self, query: str, **kwargs
    ) -> List[str] | Tuple[List[str], List[float]]:
        search_results = self.search(query=query, **kwargs)
        results = self.format_results(search_results)
        return results

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Dict]: ...

    @abstractmethod
    def format_results(self, results: List[dict]) -> List[str]: ...


class GraphRetriever(Retriever):
    def __init__(
        self,
        session: Session,
        graph: KnowledgeGraph,
    ):
        self.logger = get_logger(__file__)
        self.graph = graph
        self.session = session

    def search(
        self,
        query: str,
        *,
        top_k_vector: int = 7,
        top_k_keyword: int = 5,
        n_hops: int = 1,
    ) -> List[Dict]:
        keywords = [query]
        try:
            keywords = self.extract_keywords(query=query)
            self.logger.debug(f"Extracted keywords: {keywords}")
        except Exception as e:
            self.logger.warning(
                f"Failed to extract keywords due to an error. Using original query as a keyword. Error: {e}"
            )

        embeddings = self.graph.embedding_model.encode(keywords)
        vector_results = self.graph.vector_search(
            embeddings, top_k=top_k_vector
        )

        keyword_results = self.graph.keyword_search(
            keywords, top_k=top_k_keyword
        )

        combined = {n["id"]: n for n in chain.from_iterable(vector_results)}
        for item in chain.from_iterable(keyword_results):
            if (id_ := item["id"]) not in combined:
                combined[id_] = item

        results = list(combined.values())

        if n_hops:
            node_ids = [n["id"] for n in results]
            results = self.expand(node_ids=node_ids, n_hops=n_hops)

        return results

    def extract_keywords(self, query: str) -> List[str]:
        prompt = (
            "Extract key entities and keywords from the following query. "
            "Focus on the most important terms that represent the core of the user's intent. "
            "Return a JSON array of strings."
            f'\nQuery: "{query}"'
            ""
        )

        response = "".join(self.session(prompt)).strip()
        keywords = json.loads(response)

        return keywords

    def expand(self, node_ids: List[str], n_hops: int) -> List[Dict[str, Any]]:
        self.logger.info(
            f"Starting {n_hops}-hop exploration from {len(node_ids)} initial nodes."
        )

        subgraph = self.graph.subgraph(node_ids)
        node_data = subgraph.get("nodes", [])

        neighbor_data_lists = self.graph.expansion(
            frontier_ids=node_ids,
            excluded_ids=node_ids,
            n_hops=n_hops,
        )

        neighbor_nodes = [
            item["node"] for sublist in neighbor_data_lists for item in sublist
        ]

        self.logger.info(
            f"Found {len(neighbor_nodes)} unique neighbors within {n_hops} hops."
        )

        combined_nodes = {node["id"]: node for node in node_data}
        for node in neighbor_nodes:
            if node["id"] not in combined_nodes:
                combined_nodes[node["id"]] = node

        return list(combined_nodes.values())

    def format_results(self, results: List[dict]) -> List[str]:
        return [
            f"{r.get('label', '')}: {r.get('description', '')}"
            for r in results
        ]


class WebRetriever(Retriever):
    def __init__(self, session: Session):
        from ddgs import DDGS

        self.logger = get_logger(__file__)

        self.session = session
        self.ddgs = DDGS()

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
        prefix = '{"search_query": "'
        prompt = (
            "Transform the following conversational query into a concise, keyword-based search engine query. "
            "For example, 'Can you tell me who the members of the band Coldplay are?' should become {'search_query': 'Coldplay band members'}."
            f'\nConversational Query: "{query}"'
            f"\nSearch Engine Query: {prefix}"
        )

        response = prefix + "".join(self.session(prompt))
        search_query = json.loads(response)["search_query"].strip()

        return search_query

    def format_results(self, results: List[dict]) -> List[str]:
        return [
            f"{r.get('title', '')}: {r.get('body', '')} -- <href>{r.get('href', '#')}</href>"
            for r in results
        ]
