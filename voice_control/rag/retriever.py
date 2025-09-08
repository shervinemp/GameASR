from itertools import chain
import json
from typing import List, Dict

import requests

from sentence_transformers import CrossEncoder

from ..llm.model import LLM
from ..llm.session import Session
from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger


class RetrievalManager:
    """Manages the retrieval of information from the knowledge graph and the web."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        llm: LLM,
        max_keywords: int = 4,
        top_k_rerank: int = 4,
        top_k_web: int = 4,
    ):
        self.logger = get_logger(__file__)
        self.graph = graph
        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1
        self.max_keywords = max_keywords
        self.top_k_rerank = top_k_rerank
        self.top_k_web = top_k_web

        try:
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load CrossEncoder model. Reranking will be disabled. Error: {e}"
            )
            self.reranker = None

    def _query_for_web(self, query: str) -> str:
        """Transforms a conversational query into a keyword-based search engine query."""
        prompt = (
            "Transform the following conversational query into a concise, keyword-based search engine query. "
            "For example, 'Can you tell me who the members of the band Coldplay are?' should become 'Coldplay band members'."
            f'\nConversational Query: "{query}"'
            "\nSearch Engine Query:"
        )
        try:
            transformed_query = "".join(self.session(prompt)).strip()
            self.logger.info(f"Web search: '{transformed_query}'")
            return transformed_query
        except Exception as e:
            self.logger.warning(
                f"Failed to transform query, using original. Error: {e}"
            )
            return query

    def search_web(self, query: str) -> str:
        """Performs a web search for the given query using the DuckDuckGo API."""
        self.logger.info(
            f"Performing web search for original query: '{query}'"
        )
        search_query = self._query_for_web(query)

        DDG_API_URL = "https://api.duckduckgo.com/"
        params = {
            "q": search_query,
            "format": "json",
            "no_html": 1,
        }

        try:
            response = requests.get(DDG_API_URL, params=params)
            response.raise_for_status()
            results = response.json()

            context_parts = []

            if results.get("AbstractText"):
                context_parts.append(
                    f"--- Source 1 (Abstract): {results['AbstractText']} ---"
                )

            if results.get("RelatedTopics"):
                for topic in results["RelatedTopics"][: self.top_k_web]:
                    if topic.get("Text"):
                        context_parts.append(
                            f"--- Source {len(context_parts) + 1}: {topic['Text']} ---"
                        )

            if not context_parts:
                self.logger.warning(
                    f"Web search for '{search_query}' yielded no usable results."
                )
                return None

            return "\n".join(context_parts)

        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Web search failed for query '{search_query}': {e}"
            )
            return None
        except json.JSONDecodeError:
            self.logger.error(
                f"Failed to parse JSON response from web search for query '{search_query}'."
            )
            return None

    def _rerank_nodes(self, query: str, nodes: List[Dict]) -> List[Dict]:
        """Reranks a list of nodes based on their relevance to the query using a cross-encoder."""
        if not nodes:
            return []

        if not self.reranker:
            self.logger.warning(
                "Reranker not available. Returning top nodes without reranking."
            )
            return nodes[: self.top_k_rerank]

        pairs = [
            [query, f"{node.get('label', '')}: {node.get('description', '')}"]
            for node in nodes
        ]

        scores = self.reranker.predict(pairs, show_progress_bar=False)

        for i, node in enumerate(nodes):
            node["relevance_score"] = scores[i]

        sorted_nodes = sorted(
            nodes, key=lambda x: x["relevance_score"], reverse=True
        )

        return sorted_nodes[: self.top_k_rerank]

    def _retrieve(self, keywords: List[str]) -> List[Dict]:
        embeddings = self.graph.embedding_model.encode(keywords)

        vector_results = self.graph.vector_search(
            embeddings, top_k=self.top_k_rerank * 2
        )
        keyword_results = self.graph.keyword_search(
            keywords, top_k=self.top_k_rerank
        )

        combined_results = {
            n["id"]: n for n in chain.from_iterable(vector_results)
        }
        for item in chain.from_iterable(keyword_results):
            if item["id"] not in combined_results:
                combined_results[item["id"]] = item
        nodes = list(combined_results.values())

        return nodes

    def retrieve_and_rerank_nodes(self, query: str) -> List[Dict]:
        """Retrieves and reranks initial nodes from the knowledge graph."""
        self.logger.debug(f"Original query for node retrieval: {query}")

        keywords = self._extract_keywords(query)[: self.max_keywords]
        self.logger.debug(f"Extracted keywords: {keywords}")

        if not keywords:
            return []

        nodes = self._retrieve(keywords)
        for node in nodes:
            self.logger.debug(
                f"  - ID: {node['id']}, Label: {node.get('label', '')}"
            )

        reranked = self._rerank_nodes(query, nodes)

        self.logger.debug(
            f"Reranking complete. Top {self.top_k_rerank} results:"
        )
        for node in reranked:
            self.logger.debug(
                f"  - ID: {node['id']}, Label: {node.get('label', '')}, Score: {node['relevance_score']:.4f}"
            )

        return reranked

    def _extract_keywords(self, query: str) -> List[str]:
        """Extracts key entities and keywords from a query using the LLM."""
        prompt = (
            "Extract key entities and keywords from the following query. "
            "Focus on the most important terms that represent the core of the user's intent. "
            "Return a JSON array of strings."
            f'\nQuery: "{query}"'
        )
        try:
            response = "".join(self.session(prompt))
            keywords = json.loads(response)
            return keywords
        except Exception as e:
            self.logger.warning(
                f"Failed to extract keywords due to an error. Using original query as a keyword. Error: {e}"
            )
            return [query]
