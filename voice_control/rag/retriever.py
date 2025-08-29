import json
from typing import List, Dict

from ..llm.model import LLM
from ..llm.session import Session
from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger


class RetrievalManager:
    def __init__(self, graph: KnowledgeGraph, llm: LLM, max_keywords: int = 5):
        self.logger = get_logger(__file__)
        self.graph = graph
        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1
        self.max_keywords = max_keywords

    def search_web(self, query: str) -> str:
        """
        [Placeholder] Performs a web search for the given query.

        This method is a placeholder and needs to be implemented with a
        proper web search API (e.g., Google, Bing, etc.).
        """
        self.logger.info(f"Web search called for query: '{query}'")
        self.logger.warning("Web search is not implemented. Returning empty context.")
        # To implement this, you would use a library like 'requests' or a
        # dedicated search API client to perform a search and extract content.
        # Example:
        #   search_results = some_search_api.search(query)
        #   if search_results:
        #       top_url = search_results[0]['url']
        #       content = extract_text_from_url(top_url)
        #       return content
        return ""

    def retrieve_initial_nodes(self, query: str) -> List[str]:
        self.logger.debug(f"Original query: {query}")

        keywords = self._extract_keywords(query)[: self.max_keywords]
        self.logger.debug(f"Extracted keywords: {keywords}")

        if not keywords:
            return []

        embeddings = self.graph.embedding_model.encode(keywords)

        vector_results = self.graph.vector_search(embeddings, top_k=5)
        keyword_results = self.graph.keyword_search(keywords, top_k=5)

        # Combine and deduplicate results
        combined_results = {item['id']: item for item in vector_results}
        for item in keyword_results:
            if item['id'] not in combined_results:
                combined_results[item['id']] = item

        initial_nodes = list(combined_results.keys())

        self.logger.debug("Found initial candidates:")
        for node_id in initial_nodes:
            self.logger.debug(f"  - ID: {node_id}")

        return initial_nodes

    def _extract_keywords(self, query: str) -> List[str]:
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
