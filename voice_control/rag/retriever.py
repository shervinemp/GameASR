import json
from typing import List, Dict

from sentence_transformers import CrossEncoder

from ..llm.model import LLM
from ..llm.session import Session
from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger


class RetrievalManager:
    """Manages the retrieval of information from the knowledge graph and the web."""
    def __init__(self, graph: KnowledgeGraph, llm: LLM, max_keywords: int = 5, top_k_rerank: int = 5):
        self.logger = get_logger(__file__)
        self.graph = graph
        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1
        self.max_keywords = max_keywords
        self.top_k_rerank = top_k_rerank

        # Initialize the cross-encoder model for reranking, with error handling
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            self.logger.error(f"Failed to load CrossEncoder model. Reranking will be disabled. Error: {e}")
            self.reranker = None

    def _transform_query_for_web(self, query: str) -> str:
        """Transforms a conversational query into a keyword-based search engine query."""
        prompt = (
            "Transform the following conversational query into a concise, keyword-based search engine query. "
            "For example, 'Can you tell me who the members of the band Coldplay are?' should become 'Coldplay band members'."
            f"\n\nConversational Query: \"{query}\""
            "\n\nSearch Engine Query:"
        )
        try:
            transformed_query = "".join(self.session(prompt)).strip()
            self.logger.info(f"Transformed query for web search: '{transformed_query}'")
            return transformed_query
        except Exception as e:
            self.logger.warning(f"Failed to transform query, using original. Error: {e}")
            return query

    def search_web(self, query: str) -> str:
        """[Placeholder] Performs an enhanced web search for the given query."""
        self.logger.info(f"Web search called for original query: '{query}'")
        search_query = self._transform_query_for_web(query)
        self.logger.warning("Web search is not implemented. Returning placeholder multi-source context.")
        placeholder_content = (
            f"Placeholder content for web search on '{search_query}'.\n"
            "--- Source 1: The first simulated search result would contain relevant text here. ---\n"
            "--- Source 2: The second simulated search result would provide additional details. ---\n"
            "--- Source 3: A third source could offer a different perspective or more data. ---"
        )
        return placeholder_content

    def _rerank_nodes(self, query: str, nodes: List[Dict]) -> List[Dict]:
        """Reranks a list of nodes based on their relevance to the query using a cross-encoder."""
        if not nodes:
            return []

        self.logger.info(f"Reranking {len(nodes)} nodes...")

        if not self.reranker:
            self.logger.warning("Reranker not available. Returning top nodes without reranking.")
            return nodes[:self.top_k_rerank]

        # Create pairs of [query, node_description] for the cross-encoder
        pairs = [[query, f"{node.get('label', '')}: {node.get('description', '')}"] for node in nodes]

        # Get scores from the model
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        # Add scores to the nodes and sort
        for i, node in enumerate(nodes):
            node['relevance_score'] = scores[i]

        sorted_nodes = sorted(nodes, key=lambda x: x['relevance_score'], reverse=True)

        self.logger.debug("Reranking complete. Top results:")
        for node in sorted_nodes[:self.top_k_rerank]:
            self.logger.debug(f"  - ID: {node['id']}, Score: {node['relevance_score']:.4f}")

        return sorted_nodes[:self.top_k_rerank]

    def retrieve_and_rerank_nodes(self, query: str) -> List[Dict]:
        """Retrieves and reranks initial nodes from the knowledge graph."""
        self.logger.debug(f"Original query for node retrieval: {query}")

        keywords = self._extract_keywords(query)[: self.max_keywords]
        self.logger.debug(f"Extracted keywords: {keywords}")

        if not keywords:
            return []

        embeddings = self.graph.embedding_model.encode(keywords)

        vector_results = self.graph.vector_search(embeddings, top_k=10) # Retrieve more to give reranker more to work with
        keyword_results = self.graph.keyword_search(keywords, top_k=10)

        flat_vector_results = [node for sublist in vector_results for node in sublist]
        flat_keyword_results = [node for sublist in keyword_results for node in sublist]

        combined_results = {item['id']: item for item in flat_vector_results}
        for item in flat_keyword_results:
            if item['id'] not in combined_results:
                combined_results[item['id']] = item

        if not combined_results:
            return []

        # Rerank the combined list of unique nodes
        reranked_nodes = self._rerank_nodes(query, list(combined_results.values()))

        return reranked_nodes

    def _extract_keywords(self, query: str) -> List[str]:
        """Extracts key entities and keywords from a query using the LLM."""
        prompt = (
            "Extract key entities and keywords from the following query. "
            "Focus on the most important terms that represent the core of the user's intent. "
            "Return a JSON array of strings."
            f"\n\nQuery: \"{query}\""
        )
        try:
            response = "".join(self.session(prompt))
            keywords = json.loads(response)
            return keywords
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.warning(
                f"Failed to extract keywords, using original query as a keyword. Error: {e}"
            )
            return [query]
