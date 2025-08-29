from typing import Union

from ..llm.model import LLM
from .retriever import RetrievalManager
from .explorer import ExplorationEngine
from .generator import GenerationService
from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger


class RAG:
    """
    Implements the simplified Retrieve-Expand-Generate RAG pipeline with an
    optional web search capability.
    """
    def __init__(
        self,
        graph: KnowledgeGraph,
        llm: Union[LLM, None] = None,
        use_web_search: bool = True,
    ):
        self.logger = get_logger(__file__)
        self.retriever = RetrievalManager(graph, llm)
        self.explorer = ExplorationEngine(graph)
        self.generator = GenerationService(llm)
        self.use_web_search = use_web_search

    def __call__(self, query: str) -> str:
        """
        Executes the RAG pipeline for a given query.

        Args:
            query: The user's query.

        Returns:
            The generated answer.
        """
        # 1. Retrieve: Get initial nodes from the knowledge graph
        initial_nodes = self.retriever.retrieve_initial_nodes(query)
        if not initial_nodes:
            self.logger.info("No initial nodes found in the knowledge graph.")

        # 2. Expand: Perform a single-hop expansion on the initial nodes
        graph_context_nodes = self.explorer.explore(initial_nodes)
        if not graph_context_nodes:
            self.logger.info("Exploration did not yield any graph context.")

        # 3. Web Search: Optionally perform a web search for additional context
        web_context = None
        if self.use_web_search:
            web_context = self.retriever.search_web(query)
            if not web_context:
                self.logger.info("Web search did not yield any context.")

        # Check if any context was found at all
        if not graph_context_nodes and not web_context:
            self.logger.warning("No context found from any source.")
            return "I could not find any relevant information to answer your query."

        # 4. Generate: Create a final answer based on the combined context
        final_answer = self.generator.generate_answer(
            query,
            context_nodes=graph_context_nodes,
            web_context=web_context
        )

        return final_answer
