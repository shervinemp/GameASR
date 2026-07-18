from abc import ABC, abstractmethod
from collections import OrderedDict
import threading
import time
from typing import List

from ..common.config import config
from ..common.utils import get_logger
from ..llm.model import LLM
from ..llm.session import Session
from .generation import Composer
from .knowledge import KnowledgeGraph
from .retrieval import (
    Retriever,
    WebRetriever,
    Reranker,
    SmartGraphRetriever,
    NeighborhoodStrategy,
    ShortestPathStrategy,
)
from .validation import normalize_triplets, queue_triplets

class BaseRAG(ABC):
    def __init__(
        self,
        session: Session,
        web_search: bool = False,
    ):
        self.logger = get_logger(__file__)
        self.session = session
        self.session.conversation.cutoff_idx = 0
        self.runtime = config.get("rag.runtime")

        self.retrievers: List[Retriever] = []
        self.web_retriever = (
            WebRetriever(session=self.session) if web_search else None
        )

        self.reranker = Reranker()
        self.composer = Composer(session=self.session)
        self._context_cache = OrderedDict()
        self._cache_lock = threading.Lock()

    def close(self):
        self.session.close()

    def get_state(self) -> str:
        """Override to inject dynamic game state into the system prompt on reset."""
        return ""

    @abstractmethod
    def _attach_graph_retriever(self, graph: KnowledgeGraph | None):
        pass

    def _validate_request(self, query: str, top_k: int) -> str:
        max_query_chars = getattr(self.runtime, "max_query_chars", 2_000)
        if not isinstance(query, str):
            raise TypeError("RAG query must be a string.")
        query = query.strip()
        if not query or len(query) > max_query_chars:
            raise ValueError(
                f"RAG query must contain 1 to {max_query_chars} characters."
            )
        if not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
        return query

    @staticmethod
    def _deduplicate_candidates(results: List[str]) -> List[str]:
        deduplicated = []
        seen = set()
        for result in results:
            if not isinstance(result, str):
                continue
            # ASVS 2.2.1 / 15.2.2: bound external and model-generated
            # evidence before cross-encoding and prompt construction.
            result = " ".join(result.split())[:4_000]
            key = result.casefold()
            if result and key not in seen:
                seen.add(key)
                deduplicated.append(result)
        return deduplicated

    def _cache_get(self, key):
        now = time.monotonic()
        with self._cache_lock:
            cached = self._context_cache.get(key)
            if cached and cached[0] >= now:
                self._context_cache.move_to_end(key)
                return list(cached[1])
            if cached:
                del self._context_cache[key]
        return None

    def _cache_put(self, key, value: List[str]):
        cache_size = getattr(self.runtime, "cache_size", 128)
        if not cache_size:
            return
        ttl = getattr(self.runtime, "cache_ttl_seconds", 300.0)
        with self._cache_lock:
            self._context_cache[key] = (
                time.monotonic() + ttl,
                tuple(value),
            )
            self._context_cache.move_to_end(key)
            while len(self._context_cache) > cache_size:
                self._context_cache.popitem(last=False)

    def _clear_cache(self):
        with self._cache_lock:
            self._context_cache.clear()

    def _get_top_k_context(
        self,
        query: str,
        top_k: int,
        *,
        deadline: float | None = None,
    ) -> List[str]:
        query = self._validate_request(query, top_k)
        cache_key = (" ".join(query.casefold().split()), top_k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if deadline is None:
            deadline = time.monotonic() + getattr(
                self.runtime, "retrieval_deadline_seconds", 8.0
            )

        results = []
        source = "graph"
        for fn in self.retrievers:
            if time.monotonic() >= deadline:
                break
            try:
                results.extend(fn(query, deadline=deadline))
            except Exception as exc:
                self.logger.warning("Graph retrieval failed: %s", exc)

        # Graph evidence is authoritative for game knowledge. Web retrieval is
        # a graceful fallback, not an always-on source of latency and noise.
        if not results and self.web_retriever and time.monotonic() < deadline:
            source = "web"
            try:
                results.extend(
                    self.web_retriever(query, deadline=deadline)
                )
            except Exception as exc:
                # ASVS 16.5.2: external search failure degrades to no evidence.
                self.logger.warning("Web retrieval failed: %s", exc)

        if not results:
            self._cache_put(cache_key, [])
            return []

        results = self._deduplicate_candidates(results)
        reranker_limit = getattr(
            self.runtime, "reranker_input_limit", 20
        )
        candidates = results[:reranker_limit]
        if time.monotonic() < deadline:
            reranked, _ = self.reranker(query, results=candidates)
        else:
            # Preserve already-retrieved evidence when the local reranker
            # cannot start inside the request budget.
            reranked = candidates

        top_results = [
            f"[{source}] {result}" for result in reranked[:top_k]
        ]
        self._cache_put(cache_key, top_results)
        return top_results

    def retrieve_context(self, query: str, top_k: int | None = None) -> str:
        if top_k is None:
            top_k = getattr(self.runtime, "top_k", 5)
        evidence = self._get_top_k_context(query, top_k)
        return "\n".join(evidence)

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

    def __call__(self, query: str, top_k: int | None = None) -> str:
        context_str = self.retrieve_context(query, top_k)
        return self.composer(query, context=context_str)


class SPathRAG(BaseRAG):
    """Implements the Neural-Socratic Graph Dialogue loop utilizing S-Path-RAG principles."""

    def __init__(self, llm: LLM | None = None, graph: KnowledgeGraph | None = None, web_search: bool = False):
        super().__init__(session=Session(llm=llm), web_search=web_search)
        self._graph = graph
        self._web_search_enabled = web_search
        self._attach_graph_retriever(graph)

    def close(self):
        super().close()
        if self._graph:
            self._graph.close()

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

    def _retrieve_with_optional_draft(
        self,
        query: str,
        top_k: int,
        max_iterations: int,
    ) -> tuple[str, str | None]:
        query = self._validate_request(query, top_k)
        configured_max = getattr(self.runtime, "max_iterations", 3)
        if (
            not isinstance(max_iterations, int)
            or not 1 <= max_iterations <= configured_max
        ):
            raise ValueError(
                f"max_iterations must be between 1 and {configured_max}."
            )

        self.logger.info("Starting S-Path-RAG Neural-Socratic Dialogue")
        current_query = query
        accumulated_context = []
        seen_evidence = set()
        verified_answer = None
        deadline = time.monotonic() + getattr(
            self.runtime, "retrieval_deadline_seconds", 8.0
        )

        for iteration in range(max_iterations):
            if time.monotonic() >= deadline:
                break
            self.logger.info(f"Retrieval Iteration {iteration + 1}")
            results = self._get_top_k_context(
                current_query,
                top_k,
                deadline=deadline,
            )
            if not results:
                break

            new_count = 0
            for result in results:
                key = result.casefold()
                if key not in seen_evidence:
                    seen_evidence.add(key)
                    accumulated_context.append(result)
                    new_count += 1

            context_str = "\n".join(accumulated_context)

            if iteration == 0 and new_count >= top_k:
                self.logger.info(
                    "Graph-based results found. Skipping Socratic correction."
                )
                break

            # Do not pay for a critique if no later retrieval iteration can use
            # it. Evidence-only tool calls default to one iteration.
            if iteration >= max_iterations - 1:
                break

            draft_answer = self.composer.generate_answer(query, context_str)
            critique, is_correct = self.composer.critique_answer(
                query=query,
                context=context_str,
                answer=draft_answer,
            )

            if is_correct:
                self.logger.info("LLM is confident. Halting path expansion.")
                verified_answer = draft_answer
                break
            self.logger.info("Uncertainty detected; expanding graph retrieval.")
            # ASVS 2.2.1 / 15.2.2: bound model-generated refinement before it
            # controls another embedding, database, or web request.
            refinement = str(critique).strip()[:512]
            current_query = (
                f"{query}. Missing evidence: {refinement}"
            )[: getattr(self.runtime, "max_query_chars", 2_000)]

        return "\n".join(accumulated_context), verified_answer

    def retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
    ) -> str:
        """Return evidence only; never invoke nested answer-generation calls."""
        if top_k is None:
            top_k = getattr(self.runtime, "top_k", 5)
        context, _ = self._retrieve_with_optional_draft(
            query, top_k, 1
        )
        return context

    def __call__(
        self,
        query: str,
        top_k: int | None = None,
        max_iterations: int | None = None,
    ) -> str:
        if top_k is None:
            top_k = getattr(self.runtime, "top_k", 5)
        if max_iterations is None:
            max_iterations = getattr(self.runtime, "max_iterations", 3)
        final_context, verified_answer = self._retrieve_with_optional_draft(
            query, top_k, max_iterations
        )

        # A draft that already passed the evidence critique is the final
        # answer; regenerating it adds latency and can only lose fidelity.
        final_answer = verified_answer or self.composer(
            query, context=final_context
        )

        # Active learning is opt-in and review-first. Model output and web
        # content are untrusted and must not become durable facts implicitly.
        has_info = final_context.strip() and "Insufficient information" not in final_answer
        learning = config.get("rag.active_learning")
        learning_enabled = bool(learning and learning.enabled)
        web_allowed = bool(learning and learning.allow_web_context)
        used_web = any(
            line.startswith("[web]") for line in final_context.splitlines()
        )
        can_learn = (
            self._graph
            and has_info
            and learning_enabled
            and (not used_web or web_allowed)
        )
        if can_learn:
            try:
                raw_triplets = self.composer.extract_new_triplets(
                    final_answer, final_context
                )
                triplets = normalize_triplets(
                    raw_triplets,
                    max_items=learning.max_triplets_per_answer,
                )
                if triplets:
                    if learning.review_required:
                        path = queue_triplets(
                            triplets,
                            learning.review_queue_path,
                            query=query,
                            provenance=(
                                "web" if used_web else "graph"
                            ),
                        )
                        self.logger.info(
                            "Queued %s triplets for review in %s.",
                            len(triplets),
                            path,
                        )
                    else:
                        self._graph.add_triplets(triplets)
                        self._clear_cache()
                        self.logger.info(
                            "Added %s reviewed-policy triplets to the graph.",
                            len(triplets),
                        )
            except Exception as e:
                self.logger.warning(
                    f"Active learning triplet extraction failed: {e}"
                )
        elif learning_enabled and used_web and not web_allowed:
            self.logger.info(
                "Skipped active learning because web-derived context is not approved."
            )

        return final_answer
