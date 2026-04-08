from typing import List

from ..llm.session import Session
from ..llm.model import LLM
from .retrieval import (
    GraphRetriever,
    SPathRetriever,
    Retriever,
    WebRetriever,
    Reranker,
)
from .generation import Composer
from .knowledge import KnowledgeGraph
from ..common.utils import get_logger


class RAG:
    def __init__(
        self,
        retrievers: List[Retriever],
        reranker: Reranker,
        composer: Composer,
    ):
        self.logger = get_logger(__file__)

        self.retrievers = retrievers
        self.reranker = reranker
        self.composer = composer

    async def __call__(self, query: str, top_k: int = 5) -> str:
        results = []
        for fn in self.retrievers:
            results.extend(await fn(query))
        reranked, scores = await self.reranker(query, results=results)
        context = "\n".join(
            str(r)
            # " (score: {s})"
            for _, r, s in zip(range(top_k), reranked, scores)
        )
        answer = self.composer(query, context=context)

        return answer


class SimpleRAG(RAG):
    def __init__(
        self,
        llm: LLM | None = None,
        graph: KnowledgeGraph | None = None,
        web_search: bool = True,
    ):
        session = Session(llm=llm)
        session.conversation.cutoff_idx = -1

        retrievers = []

        if graph:
            retrievers.append(GraphRetriever(session=session, graph=graph))

        if web_search:
            retrievers.append(WebRetriever(session=session))

        reranker = Reranker()
        composer = Composer(session=session)

        super().__init__(
            retrievers=retrievers,
            reranker=reranker,
            composer=composer,
        )


class SPathRAG(RAG):
    """
    Implements the Neural-Socratic Graph Dialogue loop utilizing S-Path-RAG principles.
    """

    def __init__(
        self, llm: LLM, graph: KnowledgeGraph, web_search: bool = False
    ):
        session = Session(llm=llm)
        session.conversation.cutoff_idx = -1

        # Use the S-Path Retriever instead of the standard GraphRetriever
        retrievers = [SPathRetriever(session=session, graph=graph)]
        if web_search:
            retrievers.append(WebRetriever(session=session))

        super().__init__(
            retrievers=retrievers,
            reranker=Reranker(),
            composer=Composer(session=session),
        )

    async def __call__(
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
                results.extend(await fn(current_query))

            if not results:
                break

            reranked, scores = await self.reranker(query, results=results)
            top_results = [
                r for _, r, s in zip(range(top_k), reranked, scores)
            ]
            accumulated_context.extend(top_results)

            context_str = "\n".join(set(accumulated_context))

            # Ask the LLM to verify if the context is sufficient (The Socratic Check)
            critique, is_correct = self.composer.critique_answer(
                query=query,
                context=context_str,
                answer="Draft answer based on current knowledge.",
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
