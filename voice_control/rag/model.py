from typing import List

from ..llm.session import Session
from ..llm.model import LLM
from .retrieval import GraphRetriever, Retriever, WebRetriever, Reranker
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

    def __call__(self, query: str, top_k: int = 5) -> str:
        results = [r for fn in self.retrievers for r in fn(query)]
        reranked, scores = self.reranker(query, results=results)
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
