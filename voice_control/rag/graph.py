from dotenv import dotenv_values
from typing import Union

from ..llm.model import LLM
from .retriever import RetrievalManager
from .explorer import ExplorationEngine
from .generator import GenerationService
from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger, setup_logging


class RAG:
    def __init__(
        self,
        graph: KnowledgeGraph,
        llm: Union[LLM, None] = None,
        max_iterations: int = 5,
        max_keywords: int = 3,
        max_retries: int = 3,
    ):
        self.logger = get_logger(__file__)
        self.retrieval_manager = RetrievalManager(graph, llm, max_keywords)
        self.exploration_engine = ExplorationEngine(
            graph, llm, max_iterations, max_retries
        )
        self.generation_service = GenerationService(llm)

    def __call__(self, query: str) -> str:
        return self._execute_query(query)

    def _execute_query(self, query: str) -> str:
        report = {
            "state": "Starting search for clues with initial nodes...",
            "context": "",
            "explicit_mention": [],
        }

        initial_nodes = self.retrieval_manager.retrieve_initial_nodes(query)

        if not initial_nodes:
            return "Could not find any relevant information."

        final_answer, report = self.exploration_engine.explore(
            query, initial_nodes, report, self.generation_service
        )

        final_answer = self.generation_service.verify(final_answer, report)

        return {"answer": final_answer, "report": report}


def main():
    setup_logging("DEBUG")
    logger = get_logger(__file__)

    env = dotenv_values(".env")
    NEO4J_URI = env.get("NEO4J_URI")
    NEO4J_USER = env.get("NEO4J_USER")
    NEO4J_PASSWORD = env.get("NEO4J_PASSWORD")

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise ValueError("Neo4j credentials not found in .env file.")

    user_queries = [
        "Which American presidents had a background in law before taking office, like Obama?",
        "Who are the members of the band Coldplay?",
        "Give me all the information you have on Justin Bieber, including his personal life.",
    ]

    graph = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    rag = RAG(graph)

    try:
        for user_query in user_queries:
            final_answer = rag(user_query)
            logger.info(final_answer)
    finally:
        graph.close()


if __name__ == "__main__":
    main()
