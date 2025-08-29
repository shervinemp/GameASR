from typing import Union

from ..llm.model import LLM
from .retriever import RetrievalManager
from .explorer import ExplorationEngine
from .generator import GenerationService
from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger


class RAG:
    """
    Implements an advanced, multi-step RAG pipeline designed for high-quality,
    reasoned answers, especially with smaller language models.

    The pipeline follows a sequence of:
    1.  **Retrieve & Rerank:** Fetches initial candidate nodes from the knowledge
        graph and reranks them for relevance using a cross-encoder.
    2.  **Explore:** Performs a multi-hop traversal from the top-ranked nodes to
        gather a rich, contextual neighborhood of information.
    3.  **Web Search:** Optionally performs a web search to augment the graph
        context with real-time information.
    4.  **Generate:** A powerful generation step that first summarizes all the
        collected context, then uses a self-correction loop to iteratively
        generate and critique an answer for accuracy.
    5.  **Graph-Writer (Optional):** After generating a verified answer, this
        feature allows the pipeline to extract new facts and write them back
        to the knowledge graph, making the system self-improving.

    Attributes:
        use_web_search (bool): Flag to enable/disable web search.
        use_graph_writer (bool): Flag to enable/disable writing back to the graph.
        max_iterations (int): The number of self-correction loops for the generator.
        max_hops (int): The depth of the graph traversal for the explorer.

    Performance Considerations:
        This pipeline is optimized for quality, which comes at the cost of latency
        due to multiple sequential LLM calls and expensive operations like
        reranking. A single query can result in 5-7 LLM calls.

        To manage performance, you can configure the pipeline at initialization:
        - `use_web_search=False`: Disables the web search and its associated LLM call.
        - `use_graph_writer=False`: Disables the final triplet extraction LLM call.
        - `max_iterations=1`: Effectively disables the self-correction loop,
          reducing the generation step from 3+ LLM calls to 1-2.
        - `max_hops=1`: Reduces graph exploration to a single-hop traversal, which
          is significantly faster on large or dense graphs.
    """
    def __init__(
        self,
        graph: KnowledgeGraph,
        llm: Union[LLM, None] = None,
        use_web_search: bool = True,
        use_graph_writer: bool = False, # Default to off
        max_iterations: int = 2,
        max_hops: int = 2,
    ):
        """
        Initializes the RAG pipeline and its components.

        Args:
            graph (KnowledgeGraph): The knowledge graph instance to use.
            llm (Union[LLM, None], optional): The language model instance.
                If None, one will be created. Defaults to None.
            use_web_search (bool, optional): Whether to enable the web search feature.
                Defaults to True.
            use_graph_writer (bool, optional): Whether to enable writing new facts
                back to the graph. Defaults to False.
            max_iterations (int, optional): The number of self-correction iterations
                for the generator. Defaults to 2.
            max_hops (int, optional): The maximum number of hops for graph exploration.
                Defaults to 2.
        """
        self.logger = get_logger(__file__)
        self.retriever = RetrievalManager(graph, llm)
        self.explorer = ExplorationEngine(graph)
        self.generator = GenerationService(llm, graph, max_iterations=max_iterations)
        self.use_web_search = use_web_search
        self.use_graph_writer = use_graph_writer
        self.max_hops = max_hops

    def __call__(self, query: str) -> str:
        """
        Executes the full RAG pipeline for a given query.

        Args:
            query (str): The user's query.

        Returns:
            str: The final, generated answer.
        """
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
