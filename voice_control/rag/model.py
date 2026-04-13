from typing import List

from ..llm.session import Session
from ..llm.model import LLM
from abc import ABC, abstractmethod

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
        self.session.conversation.cutoff_idx = -1

        self.retrievers: List[Retriever] = []
        if web_search:
            self.retrievers.append(WebRetriever(session=self.session))

        self.reranker = Reranker()
        self.composer = Composer(session=self.session)

    @abstractmethod
    def _attach_graph_retriever(self, graph: KnowledgeGraph | None):
        pass

    def _get_top_k_context(self, query: str, top_k: int) -> List[str]:
        results = []
        for fn in self.retrievers:
            results.extend(fn(query))

        if not results:
            return []

        reranked, scores = self.reranker(query, results=results)

        top_results = []
        for _, r, s in zip(range(top_k), reranked, scores):
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

        for iteration in range(max_iterations):
            self.logger.info(f"Retrieval Iteration {iteration + 1}")

            # Retrieve structural paths
            results = []
            for fn in self.retrievers:
                results.extend(fn(current_query))

            if not results:
                break

            reranked, scores = self.reranker(query, results=results)
            top_results = [
                r for _, r, s in zip(range(top_k), reranked, scores)
            ]
            accumulated_context.extend(top_results)

            context_str = "\n".join(set(accumulated_context))

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
                # Append the critique to guide the next S-Path anchor selection
                current_query = f"{query}. We are missing: {critique}"

        # Final generation
        final_context = "\n".join(set(accumulated_context))
        return self.composer(query, context=final_context)
