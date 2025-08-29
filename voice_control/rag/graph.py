from typing import Union

from ..llm.model import LLM
from .retriever import RetrievalManager
from .explorer import ExplorationEngine
from .generator import GenerationService
from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger


class RAG:
    """
    Implements the enhanced Retrieve-Expand-Generate RAG pipeline with
    reranking, web search, context summarization, and self-correction.
    """
    def __init__(
        self,
        graph: KnowledgeGraph,
        llm: Union[LLM, None] = None,
        use_web_search: bool = True,
        max_iterations: int = 2,
    ):
        self.logger = get_logger(__file__)
        self.retriever = RetrievalManager(graph, llm)
        self.explorer = ExplorationEngine(graph)
        self.generator = GenerationService(llm, max_iterations=max_iterations)
        self.use_web_search = use_web_search

    def __call__(self, query: str) -> str:
        """
        Executes the RAG pipeline for a given query.
        """
        # 1. Retrieve and Rerank: Get the most relevant initial nodes.
        # This method now returns a list of node dictionaries, already reranked.
        reranked_nodes = self.retriever.retrieve_and_rerank_nodes(query)
        if not reranked_nodes:
            self.logger.info("No initial nodes found in the knowledge graph.")

        # 2. Expand: Perform a single-hop expansion on the reranked nodes.
        # We need to extract the IDs to pass to the explorer.
        reranked_node_ids = [node['id'] for node in reranked_nodes]
        graph_context_nodes = self.explorer.explore(reranked_node_ids)

        # The explorer returns all nodes (initial + neighbors).
        # We'll trust the reranked nodes are the most important, but the context is richer with neighbors.
        if not graph_context_nodes:
            self.logger.info("Exploration did not yield any additional graph context.")
            # If expansion fails, we can fall back to just the reranked nodes.
            graph_context_nodes = reranked_nodes

        # 3. Web Search: Optionally perform a web search for additional context.
        web_context = None
        if self.use_web_search:
            web_context = self.retriever.search_web(query)
            if not web_context:
                self.logger.info("Web search did not yield any context.")

        # Check if any context was found at all
        if not graph_context_nodes and not web_context:
            self.logger.warning("No context found from any source.")
            return "I could not find any relevant information to answer your query."

        # 4. Generate: Create a final answer using the new, powerful generator.
        # The generator now handles summarization and self-correction internally.
        final_answer = self.generator.generate_answer(
            query,
            context_nodes=graph_context_nodes, # Pass the full context from the explorer
            web_context=web_context
        )

        return final_answer
