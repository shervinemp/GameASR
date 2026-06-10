from typing import List

from ..llm.session import Session
from ..llm.model import LLM
from abc import ABC, abstractmethod

RERANKER_INPUT_LIMIT = 20

from .retrieval import (
    Retriever,
    WebRetriever,
    Reranker,
    SmartGraphRetriever,
    NeighborhoodStrategy,
    ShortestPathStrategy,
)
from .generation import Composer
from .knowledge import KnowledgeGraph
from ..common.utils import get_logger


class BaseRAG(ABC):
    def __init__(
        self,
        session: Session,
        web_search: bool = False,
    ):
        self.logger = get_logger(__file__)
        self.session = session
        self.session.conversation.cutoff_idx = 0

        self.retrievers: List[Retriever] = []
        if web_search:
            self.retrievers.append(WebRetriever(session=self.session))

        self.reranker = Reranker()
        self.composer = Composer(session=self.session)

    def get_state(self) -> str:
        """Override to inject dynamic game state into the system prompt on reset."""
        return ""

    @abstractmethod
    def _attach_graph_retriever(self, graph: KnowledgeGraph | None):
        pass

    def _get_top_k_context(self, query: str, top_k: int) -> List[str]:
        results = []
        for fn in self.retrievers:
            results.extend(fn(query))

        if not results:
            return []

        # Truncate before reranker: top 20 is enough to find the best candidates
        reranked, _ = self.reranker(query, results=results[:20])

        top_results = []
        for r in reranked[:top_k]:
            top_results.append(str(r))
        return top_results

    @abstractmethod
    def __call__(self, query: str, **kwargs) -> str:
        pass


class SimpleRAG(BaseRAG):
    """Standard single-pass retrieval augmented generation."""

    def __init__(self, llm: LLM | None = None, graph: KnowledgeGraph | None = None, web_search: bool = True):
        super().__init__(session=Session(llm=llm), web_search=web_search)
        self._attach_graph_retriever(graph)

    def _attach_graph_retriever(self, graph: KnowledgeGraph | None):
        if graph:
            # Simple RAG only uses Neighborhood expansion
            strategy = NeighborhoodStrategy(graph)

            retriever = SmartGraphRetriever(
                session=self.session,
                primary_strategy=strategy
            )
            self.retrievers.append(retriever)

    def __call__(self, query: str, top_k: int = 5) -> str:
        context_str = "\n".join(self._get_top_k_context(query, top_k))
        return self.composer(query, context=context_str)


class SPathRAG(BaseRAG):
    """Implements the Neural-Socratic Graph Dialogue loop utilizing S-Path-RAG principles."""

    def __init__(self, llm: LLM | None = None, graph: KnowledgeGraph | None = None, web_search: bool = False):
        super().__init__(session=Session(llm=llm), web_search=web_search)
        self._graph = graph
        self._attach_graph_retriever(graph)

    def _attach_graph_retriever(self, graph: KnowledgeGraph | None):
        if graph:
            # S-Path RAG uses Shortest Paths primarily, but falls back to Neighborhoods
            primary = ShortestPathStrategy(graph)
            fallback = NeighborhoodStrategy(graph)

            retriever = SmartGraphRetriever(
                session=self.session,
                primary_strategy=primary,
                fallback_strategy=fallback
            )
            self.retrievers.append(retriever)

    def __call__(
        self, query: str, top_k: int = 5, max_iterations: int = 3
    ) -> str:
        self.logger.info("Starting S-Path-RAG Neural-Socratic Dialogue")
        current_query = query
        accumulated_context = []
        seen_hashes = set()
        has_exact_label_matches = False

        for iteration in range(max_iterations):
            self.logger.info(f"Retrieval Iteration {iteration + 1}")

            # Retrieve structural paths
            results = []
            for fn in self.retrievers:
                results.extend(fn(current_query))

            if not results:
                break

            reranked, _ = self.reranker(query, results=results[:RERANKER_INPUT_LIMIT])

            new_count = 0
            for r in reranked[:top_k]:
                if r not in seen_hashes:
                    seen_hashes.add(r)
                    accumulated_context.append(r)
                    new_count += 1

            context_str = "\n".join(accumulated_context)

            if iteration == 0 and new_count >= top_k:
                has_exact_label_matches = True
                self.logger.info(
                    "Graph-based results found. Skipping Socratic correction."
                )
                break

            # Ask the LLM to verify if the context is sufficient (The Socratic Check)
            draft_answer = self.composer.generate_answer(query, context_str)
            critique, is_correct = self.composer.critique_answer(
                query=query,
                context=context_str,
                answer=draft_answer,
            )

            if is_correct:
                self.logger.info("LLM is confident. Halting path expansion.")
                break
            else:
                self.logger.info(
                    f"Uncertainty detected. Expanding search query based on critique: {critique}"
                )
                current_query = f"{query}. We are missing: {critique}"

        # Final generation
        final_context = "\n".join(accumulated_context)
        final_answer = self.composer(query, context=final_context)

        # Active learning: only extract triplets when we have meaningful context
        has_info = final_context.strip() and "Insufficient information" not in final_answer
        if self._graph and has_info:
            try:
                triplets = self.composer.extract_new_triplets(
                    final_answer, final_context
                )
                if triplets:
                    self._graph.add_triplets(triplets)
                    self.logger.info(
                        f"Added {len(triplets)} new triplets to knowledge graph."
                    )
            except Exception as e:
                self.logger.warning(
                    f"Active learning triplet extraction failed: {e}"
                )

        return final_answer
