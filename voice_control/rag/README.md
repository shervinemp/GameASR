# Advanced RAG Pipeline Module

This document provides an overview of the advanced Retrieval-Augmented Generation (RAG) pipeline designed for high-quality, reasoned answers, especially when working with smaller language models.

## Pipeline Architecture

The pipeline follows a sophisticated, multi-step process to generate answers:

1.  **Retrieve & Rerank:** Fetches initial candidate nodes from the knowledge graph using both keyword and vector searches. It then reranks these candidates for relevance using a cross-encoder model to ensure only the most promising information proceeds.
2.  **Explore:** Performs a multi-hop traversal (defaulting to 2 hops) from the top-ranked nodes. This gathers a rich, contextual neighborhood of information from the graph, uncovering deeper relationships.
3.  **Web Search (Optional):** Augments the graph context with real-time information from the web. This step includes an LLM-powered transformation to convert the user's conversational query into a more effective search engine query.
4.  **Generate:** A powerful, two-stage generation process:
    *   **Summarize:** First, an LLM call condenses all the collected context (from the graph and web) into a concise summary. This prompt also explicitly instructs the model to identify and report any contradictions between the data sources.
    *   **Self-Correct:** The summarized context is then passed to an iterative loop. In each cycle, the LLM generates an answer, critiques its own work for factual accuracy against the context, and then refines the answer based on that critique.
5.  **Graph-Writer (Optional):** After generating a final, verified answer, this feature allows the pipeline to use an LLM to extract new, high-confidence facts from the web context and write them back to the knowledge graph as new triplets, making the system self-improving.

---

## Components

-   **`RAG` (in `graph.py`):** The main orchestrator class that executes the full pipeline. It is highly configurable.
-   **`RetrievalManager` (in `retriever.py`):** Handles keyword extraction, initial document retrieval, reranking, and the web search functionality.
-   **`ExplorationEngine` (in `explorer.py`):** Manages the multi-hop graph traversal.
-   **`GenerationService` (in `generator.py`):** Manages the advanced generation process, including summarization, self-correction, and the optional graph-writer logic.
-   **`KnowledgeGraph` (in `knowledge_base.py`):** Provides the interface for all interactions with the Neo4j database, including search, traversal, and writing new data.

---

## Performance and Configuration

This pipeline is optimized for **quality**, which comes at the cost of **latency**. A single query can result in 5-7 LLM calls and several expensive database and model inference operations.

You can manage this trade-off by configuring the `RAG` class at initialization. The key parameters are:

-   `use_web_search` (bool): Set to `False` to disable the web search and its associated LLM call.
-   `use_graph_writer` (bool): Set to `False` (the default) to disable writing new facts to the graph and its associated LLM call.
-   `max_iterations` (int): Controls the self-correction loop. Set to `1` to effectively disable self-correction, providing a major speed-up by reducing the generation step from 3+ LLM calls to 1-2.
-   `max_hops` (int): Controls the graph exploration depth. Set to `1` for a faster single-hop traversal, which is recommended if deep, complex relationships are not required for your use case.
