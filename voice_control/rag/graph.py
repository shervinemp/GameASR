from typing import Union

from ..llm.model import LLM
from .retriever import RetrievalManager
from .explorer import ExplorationEngine
from .generator import GenerationService
from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger


class RAG:
    """Implements the advanced RAG pipeline."""
    def __init__(
        self,
        graph: KnowledgeGraph,
        llm: Union[LLM, None] = None,
        use_web_search: bool = True,
        use_graph_writer: bool = False, # Default to off
        max_iterations: int = 2,
        max_hops: int = 2,
    ):
        """Initializes the RAG pipeline and its components."""
        self.logger = get_logger(__file__)
        self.retriever = RetrievalManager(graph, llm)
        self.explorer = ExplorationEngine(graph)
        self.generator = GenerationService(llm, graph, max_iterations=max_iterations)
        self.use_web_search = use_web_search
        self.use_graph_writer = use_graph_writer
        self.max_hops = max_hops

    def __call__(self, query: str) -> str:
        """Executes the full RAG pipeline for a given query."""
        # 1. Retrieve and Rerank
        reranked_nodes = self.retriever.retrieve_and_rerank_nodes(query)
        if not reranked_nodes:
            self.logger.info("No initial nodes found in the knowledge graph.")

        # 2. Expand
        reranked_node_ids = [node['id'] for node in reranked_nodes]
        graph_context_nodes = self.explorer.explore(reranked_node_ids, max_hops=self.max_hops)

        if not graph_context_nodes:
            self.logger.info("Exploration did not yield any additional graph context.")
            graph_context_nodes = reranked_nodes

        # 3. Web Search
        web_context = None
        if self.use_web_search:
            web_context = self.retriever.search_web(query)
            if not web_context:
                self.logger.info("Web search did not yield any context.")

        if not graph_context_nodes and not web_context:
            self.logger.warning("No context found from any source.")
            return "I could not find any relevant information to answer your query."

        # 4. Generate
        final_answer = self.generator.generate_answer(
            query,
            context_nodes=graph_context_nodes,
            web_context=web_context,
            write_to_graph=self.use_graph_writer
        )

        return final_answer
