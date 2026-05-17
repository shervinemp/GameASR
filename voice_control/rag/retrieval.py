from abc import ABC, abstractmethod
import json
from typing import List, Dict, Tuple
from itertools import chain, repeat, combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm.session import Session
from .knowledge import KnowledgeGraph
from ..common.utils import get_logger


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.reranker = CrossEncoder(model_name)

    def __call__(
        self, query: str, results: List[str]
    ) -> Tuple[List[str], List[float]]:
        if not results:
            return [], []

        pairs = list(zip(repeat(query), results))
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        sorted_pairs = sorted(
            zip(results, scores), key=lambda x: x[1], reverse=True
        )

        sorted_results, scores = zip(*sorted_pairs)

        return list(sorted_results), list(scores)


class Retriever(ABC):
    """The unified interface expected by the RAG orchestrator."""

    @abstractmethod
    def __call__(self, query: str, **kwargs) -> List[str]:
        pass


class GraphSearchStrategy(ABC):
    """Defines the specific mathematical approach to traversing the Knowledge Graph."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    @abstractmethod
    def search(self, keywords: List[str], **kwargs) -> List[Dict]:
        pass

    @abstractmethod
    def format_results(self, results: List[Dict]) -> List[str]:
        pass


class NeighborhoodStrategy(GraphSearchStrategy):
    """Standard 1-hop to N-hop semantic neighborhood expansion."""

    def search(
        self,
        keywords: List[str],
        top_k_vector: int = 7,
        top_k_keyword: int = 5,
        n_hops: int = 1,
        **kwargs,
    ) -> List[Dict]:
        if not self.graph:
            return []

        # 1. Exact-label lookup: fast path for known entities
        exact_matches = self.graph.exact_label_search(keywords)
        seen_ids = {n["id"] for n in exact_matches.values() if n.get("id")}
        all_matched = len(exact_matches) >= sum(1 for k in keywords if k.strip())

        combined = {n["id"]: n for n in exact_matches.values()}

        # 2. Semantic search only for keywords not resolved by exact match
        if not all_matched:
            # Run vector and keyword search in parallel
            with ThreadPoolExecutor(max_workers=2) as pool:
                vector_future = pool.submit(
                    self._do_vector_search, keywords, top_k_vector
                )
                keyword_future = pool.submit(
                    self.graph.keyword_search, keywords, top_k_keyword
                )
                for future in as_completed([vector_future, keyword_future]):
                    try:
                        batch = future.result()
                        for item in chain.from_iterable(batch):
                            if item["id"] not in seen_ids:
                                combined[item["id"]] = item
                    except Exception as e:
                        self.graph.logger.warning(
                            f"Parallel search failed: {e}"
                        )

        results = list(combined.values())
        if n_hops:
            results = self._expand(results, n_hops)

        # Adaptive: if n_hops=1 returned too little, retry with n_hops=2
        if n_hops == 1 and len(results) < 3:
            results = self._expand(list(combined.values()), n_hops=2)

        return results

    def _do_vector_search(self, keywords, top_k):
        embeddings = self.graph.embedding_model.encode(keywords).tolist()
        return self.graph.vector_search(embeddings, top_k=top_k)

    def _expand(self, seed_results, n_hops):
        seed_nodes = self.graph.subgraph(
            [n["id"] for n in seed_results]
        ).get("nodes", [])
        expansion = self.graph.expansion(
            frontier_ids=[n["id"] for n in seed_results],
            excluded_ids=[n["id"] for n in seed_results],
            n_hops=n_hops,
        )
        return seed_nodes + [
            item for sublist in expansion for item in sublist
        ]

    def format_results(self, results: List[Dict]) -> List[str]:
        formatted = []
        for item in results:
            if "parent" in item and "node" in item:
                parent = item.get("parent", {}).get("label", "")
                node_label = item.get("node", {}).get("label", "")
                rel_type = item.get("relationship", {}).get("type", "RELATED_TO").replace("_", " ").title()
                formatted.append(f"{parent} is {rel_type} {node_label}.")
            else:
                label = item.get("label", "")
                desc = item.get("description", "")
                formatted.append(f"{label}: {desc}" if desc else label)
        return list(dict.fromkeys(formatted))


class ShortestPathStrategy(GraphSearchStrategy):
    """Semantic-aware multi-hop shortest path logic between multiple anchors."""

    def search(
        self,
        keywords: List[str],
        top_k_vector: int = 3,
        max_paths: int = 3,
        **kwargs,
    ) -> List[Dict]:
        if not self.graph or len(keywords) < 2:
            return []  # S-Path mathematically requires 2+ anchors

        # 1. Exact-label lookup: fast path
        exact_matches = self.graph.exact_label_search(keywords)
        exact_ids = {n["id"] for n in exact_matches.values() if n.get("id")}

        # 2. Vector search for additional anchors
        embeddings = self.graph.embedding_model.encode(keywords).tolist()
        batch_results = self.graph.vector_search(
            embeddings, top_k=top_k_vector
        )

        # 3. Deduplicate anchors by label (same entity from different keywords)
        anchor_map = {}  # label -> id (first occurrence wins)
        # Exact matches take priority
        for node in exact_matches.values():
            if node.get("label") and node.get("id"):
                anchor_map[node["label"].lower()] = node["id"]
        # Vector results fill gaps
        for res_list in batch_results:
            for node in res_list:
                label = node.get("label", "").lower()
                nid = node.get("id")
                if label and nid and label not in anchor_map:
                    anchor_map[label] = nid

        anchor_ids = list(anchor_map.values())
        if len(anchor_ids) < 2:
            return []

        # Batch all pairs into a single Cypher query
        pairs = list(combinations(anchor_ids, 2))
        return self.graph.k_shortest_paths_batch(pairs, k=max_paths)

    def format_results(self, results: List[Dict]) -> List[str]:
        formatted = []
        for path_data in results:
            if "nodes" not in path_data or "relations" not in path_data:
                continue
            nodes = path_data["nodes"]
            relations = path_data["relations"]
            parts = []
            for i, rel in enumerate(relations):
                src = nodes[i].get("label", "Unknown")
                tgt = nodes[i + 1].get("label", "Unknown")
                rel_type = rel.get("type", "RELATED_TO").replace("_", " ").title()
                parts.append(f"{src} is {rel_type} {tgt}")
            if parts:
                formatted.append(". ".join(parts) + ".")
        return list(dict.fromkeys(formatted))


class SmartGraphRetriever(Retriever):
    """
    Coordinates LLM intent extraction and delegates to the injected Graph Strategies.
    """

    def __init__(
        self,
        session: Session,
        primary_strategy: GraphSearchStrategy,
        fallback_strategy: GraphSearchStrategy | None = None,
    ):
        self.logger = get_logger(__file__)
        self.session = session
        self.primary = primary_strategy
        self.fallback = fallback_strategy

    def _extract_keywords(self, query: str) -> List[str]:
        from ..common.utils import safe_json_loads

        prompt = (
            "Extract named entities from the query below.\n"
            "Rules:\n"
            "- Pick specific names: people, places, things\n"
            "- Skip generic words: relationship, connection, difference, meaning, about\n"
            "- Skip the word RAG\n"
            "- Return a JSON list of strings\n"
            f'Query: "{query}"\n'
            "Output:"
        )
        response = "".join(self.session(prompt)).strip()
        return safe_json_loads(response, fallback=[query])

    def __call__(self, query: str, **kwargs) -> List[str]:
        try:
            keywords = self._extract_keywords(query)
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed. Error: {e}")
            keywords = [query]

        raw_results = self.primary.search(keywords, **kwargs)

        if not raw_results and self.fallback:
            self.logger.info(
                "Primary strategy yielded no results. Engaging fallback."
            )
            raw_results = self.fallback.search(keywords, **kwargs)
            return self.fallback.format_results(raw_results)

        return self.primary.format_results(raw_results)


class WebRetriever(Retriever):
    def __init__(self, session: Session):
        self.logger = get_logger(__file__)
        self.session = session
        self._ddgs = None

    def _get_ddgs(self):
        if self._ddgs is None:
            try:
                from ddgs import DDGS
                self._ddgs = DDGS()
            except Exception:
                self._ddgs = False
        return self._ddgs if self._ddgs is not False else None

    def __call__(self, query: str, **kwargs) -> List[str]:
        search_results = self.search(query=query, **kwargs)
        return self.format_results(search_results)

    def search(self, query: str, *, top_k: int = 3) -> List[dict]:
        """Performs a web search with DuckDuckGo fallback chain."""
        search_query = query
        try:
            search_query = self.transform_query(query=query)
        except Exception as e:
            self.logger.warning(
                f"Failed to extract keywords. Using original query. Error: {e}"
            )

        self.logger.info(f"Searching for '{search_query}'...")

        # Try DDGS library first
        ddgs = self._get_ddgs()
        results = self._search_ddgs(ddgs, search_query, top_k) if ddgs else []

        # Fallback: try DDG Instant Answer API, then lite HTML
        if not results:
            results = self._search_instant_answer(search_query, top_k)
        if not results:
            results = self._search_lite_fallback(search_query, top_k)

        if not results:
            self.logger.warning(f"Web search for '{search_query}' yielded no results.")

        return results

    def _search_ddgs(self, ddgs, query: str, top_k: int) -> List[dict]:
        import time
        for attempt in range(3):
            try:
                results = ddgs.text(query, max_results=top_k)
                if results:
                    return results
            except Exception as e:
                self.logger.warning(f"DDGS attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return []

    def _search_instant_answer(self, query: str, top_k: int) -> List[dict]:
        """Fallback using DuckDuckGo Instant Answer JSON API."""
        try:
            import urllib.request
            import urllib.parse
            import json

            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            # Abstract (infobox summary)
            abstract = data.get("AbstractText", "")
            source = data.get("AbstractSource", "")
            if abstract:
                results.append({"title": source or "Summary", "body": abstract, "href": data.get("AbstractURL", "")})

            # Related topics
            for topic in data.get("RelatedTopics", []):
                if "Text" in topic and len(results) < top_k:
                    results.append({"title": topic.get("Text", "").split(" - ")[0], "body": topic.get("Text", ""), "href": topic.get("FirstURL", "")})
                if len(results) >= top_k:
                    break

            return results
        except Exception as e:
            self.logger.warning(f"Instant Answer API failed: {e}")
            return []

    def _search_lite_fallback(self, query: str, top_k: int) -> List[dict]:
        """Fallback using DuckDuckGo's HTML-only lite endpoint."""
        try:
            import urllib.request
            import urllib.parse

            url = "https://lite.duckduckgo.com/lite/"
            data = urllib.parse.urlencode({"q": query}).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"User-Agent": "Mozilla/5.0 (compatible; VoiceBot/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Parse DDG lite HTML results
            import re
            results = []
            seen_urls = set()
            for match in re.finditer(
                r'<a[^>]*href="(https?://[^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>',
                html,
            ):
                href = match.group(1)
                title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
                # Skip DDG homepage/brand links
                if title and href not in seen_urls and "duckduckgo.com" not in href:
                    seen_urls.add(href)
                    results.append({"title": title, "href": href, "body": ""})
                if len(results) >= top_k:
                    break
            # Fallback: try broader pattern if class-based failed
            if not results:
                for match in re.finditer(
                    r'<a[^>]*href="(https?://(?!duckduckgo\.com)[^"]+)"[^>]*>(.*?)</a>',
                    html,
                ):
                    href = match.group(1)
                    title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
                    if title and href not in seen_urls:
                        seen_urls.add(href)
                        results.append({"title": title, "href": href, "body": ""})
                    if len(results) >= top_k:
                        break
            return results
        except Exception as e:
            self.logger.warning(f"Web search fallback also failed: {e}")
            return []

    def transform_query(self, query: str) -> str:
        if len(query.split()) < 6:
            return query

        prompt = (
            "Turn the query into a short keyword search.\n"
            "Example:\n"
            'Input: "Can you tell me who the members of the band Coldplay are?"\n'
            'Output: {"search_query": "Coldplay band members"}\n\n'
            f'Input: "{query}"\n'
            "Output:"
        )

        response = "".join(self.session(prompt))
        search_query = json.loads(response).get("search_query", query).strip()

        return search_query

    def format_results(self, results: List[dict]) -> List[str]:
        formatted = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            if title and body:
                formatted.append(f"{title}: {body}")
            elif title:
                formatted.append(title)
        return formatted
