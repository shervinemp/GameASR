from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import nullcontext
import json
import re
import threading
import time
from typing import List, Dict, Tuple
from itertools import chain, repeat, combinations, product
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..common.config import config
from ..common.utils import get_logger
from ..exceptions import StorageError
from ..llm.session import Session
from .backends.base import StorageBackend
from .embeddings import Embedder


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.reranker = CrossEncoder(model_name)
        self._cache = OrderedDict()
        self._cache_size = 128
        self._cache_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def __call__(
        self, query: str, results: List[str]
    ) -> Tuple[List[str], List[float]]:
        if not results:
            return [], []

        unique_results = list(dict.fromkeys(results))
        cache_key = (query, tuple(unique_results))
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return list(cached[0]), list(cached[1])

        pairs = list(zip(repeat(query), unique_results))
        # ASVS 15.4.1: model inference and cache mutation are synchronized;
        # CrossEncoder implementations are not guaranteed to be thread-safe.
        with self._predict_lock:
            scores = self.reranker.predict(pairs, show_progress_bar=False)

        sorted_pairs = sorted(
            zip(unique_results, scores), key=lambda x: x[1], reverse=True
        )

        sorted_results, scores = zip(*sorted_pairs)
        result = list(sorted_results), [float(score) for score in scores]
        with self._cache_lock:
            self._cache[cache_key] = (tuple(result[0]), tuple(result[1]))
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result


class Retriever(ABC):
    """The unified interface expected by the RAG orchestrator."""

    @abstractmethod
    def __call__(self, query: str, **kwargs) -> List[str]:
        pass


class GraphSearchStrategy(ABC):
    """Defines the specific mathematical approach to traversing the backend graph."""

    def __init__(self, backend: StorageBackend, embedder: Embedder | None = None):
        self.backend = backend
        self.embedder = embedder or Embedder()
        self.logger = get_logger(f"{__name__}.{type(self).__name__}")

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
        if not self.backend:
            return []

        # 1. Exact-label lookup: fast path for known entities
        exact_matches = self.backend.exact_label_search(keywords)
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
                    self.backend.keyword_search, keywords, top_k_keyword
                )
                for future in as_completed([vector_future, keyword_future]):
                    try:
                        batch = future.result()
                        for item in chain.from_iterable(batch):
                            if item["id"] not in seen_ids:
                                combined[item["id"]] = item
                    except Exception as e:
                        self.logger.warning(
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
        embeddings = self.embedder.encode(keywords)
        return self.backend.vector_search(embeddings, top_k=top_k)

    def _expand(self, seed_results, n_hops):
        seed_nodes = self.backend.subgraph(
            [n["id"] for n in seed_results]
        ).get("nodes", [])
        expansion = self.backend.expansion(
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
        if not self.backend or len(keywords) < 2:
            return []  # S-Path mathematically requires 2+ anchors

        # 1. Exact-label lookup: fast path
        exact_matches = self.backend.exact_label_search(keywords)

        unresolved = [
            keyword
            for keyword in keywords
            if keyword.strip().lower() not in exact_matches
        ]
        vector_by_keyword = {}
        if unresolved:
            embeddings = self.embedder.encode(unresolved)
            batch_results = self.backend.vector_search(
                embeddings, top_k=top_k_vector
            )
            vector_by_keyword = dict(zip(unresolved, batch_results))

        # Keep candidates grouped by the query entity that produced them.
        # Pairing alternatives from the same entity creates irrelevant paths
        # and quadratic work without adding relationship evidence.
        anchor_groups = []
        for keyword in keywords:
            exact = exact_matches.get(keyword.strip().lower())
            candidates = [exact] if exact else vector_by_keyword.get(keyword, [])
            ids = []
            for node in candidates:
                node_id = node.get("id") if node else None
                if node_id and node_id not in ids:
                    ids.append(node_id)
            if ids:
                anchor_groups.append(ids)

        if len(anchor_groups) < 2:
            return []

        pairs = []
        seen_pairs = set()
        for left, right in combinations(anchor_groups, 2):
            for source, target in product(left, right):
                if source == target:
                    continue
                key = tuple(sorted((source, target)))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    pairs.append((source, target))
                if len(pairs) >= 64:
                    break
            if len(pairs) >= 64:
                break
        return self.backend.k_shortest_paths_batch(pairs, k=max_paths)

    def format_results(self, results: List[Dict]) -> List[str]:
        formatted = []
        for path_data in results:
            if "nodes" not in path_data or "relations" not in path_data:
                continue
            nodes = path_data["nodes"]
            relations = path_data["relations"]
            parts = []
            for i, rel in enumerate(relations):
                left = nodes[i]
                right = nodes[i + 1]
                if rel.get("source") == right.get("id"):
                    left, right = right, left
                src = left.get("label", "Unknown")
                tgt = right.get("label", "Unknown")
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
        self._nlp = None
        self._keyword_cache = OrderedDict()
        self._cache_lock = threading.Lock()

    def _ask(self, prompt: str) -> str:
        if isinstance(self.session, Session):
            return self.session.complete_once(prompt)
        return "".join(self.session(prompt)).strip()

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except Exception:
                self._nlp = False
        return self._nlp if self._nlp is not False else None

    def _extract_keywords_ner(self, query: str) -> List[str] | None:
        nlp = self._get_nlp()
        if nlp is None:
            return None
        doc = nlp(query)
        entities = list(set(ent.text for ent in doc.ents))
        return entities if entities else None

    @staticmethod
    def _candidate_phrases(query: str) -> List[str]:
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", query)
        if not words:
            return []
        stop_words = {
            "a", "an", "and", "are", "about", "for", "from", "how",
            "in", "is", "of", "on", "the", "to", "what", "when",
            "where", "which", "who", "why", "with",
        }
        phrases = []
        max_width = min(5, len(words))
        for width in range(max_width, 0, -1):
            for start in range(len(words) - width + 1):
                phrase_words = words[start : start + width]
                if width == 1 and phrase_words[0].lower() in stop_words:
                    continue
                phrases.append(" ".join(phrase_words))
                if len(phrases) >= 64:
                    return phrases
        return phrases

    @staticmethod
    def _validated_keywords(value, fallback: str) -> List[str]:
        if not isinstance(value, list):
            return [fallback]
        keywords = []
        for item in value[:8]:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if item and len(item) <= 256 and item not in keywords:
                keywords.append(item)
        return keywords or [fallback]

    def _extract_keywords(
        self,
        query: str,
        *,
        allow_model_fallback: bool = True,
    ) -> List[str]:
        from ..common.utils import safe_json_loads

        cache_key = " ".join(query.lower().split())
        with self._cache_lock:
            cached = self._keyword_cache.get(cache_key)
            if cached is not None:
                self._keyword_cache.move_to_end(cache_key)
                return list(cached)

        # Resolve entity spans against the indexed graph before spending an
        # LLM call. This also works with lowercase ASR transcripts.
        phrases = self._candidate_phrases(query)
        exact_matches = self.primary.backend.exact_label_search(phrases)
        keywords = []
        for phrase in phrases:
            node = exact_matches.get(phrase.strip().lower())
            label = node.get("label") if node else None
            if label and label not in keywords:
                keywords.append(label)
            if len(keywords) >= 8:
                break

        if not keywords:
            ner_keywords = self._extract_keywords_ner(query)
            if ner_keywords:
                self.logger.debug(f"NER extracted keywords: {ner_keywords}")
                keywords = self._validated_keywords(ner_keywords, query)

        if not keywords and allow_model_fallback:
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
            response = self._ask(prompt)
            parsed = safe_json_loads(response, fallback=[query])
            # ASVS 2.2.1 / 15.3.5: model output is schema- and size-bounded
            # before it controls vector queries and graph expansion.
            keywords = self._validated_keywords(parsed, query)

        if not keywords:
            # Bounded retrieval calls must not inherit the provider's much
            # longer generation timeout. The whole query is still useful to
            # vector/full-text neighborhood search when exact linking and NER
            # find no graph entity.
            keywords = [query]

        with self._cache_lock:
            self._keyword_cache[cache_key] = tuple(keywords)
            self._keyword_cache.move_to_end(cache_key)
            while len(self._keyword_cache) > 128:
                self._keyword_cache.popitem(last=False)
        return keywords

    def __call__(self, query: str, **kwargs) -> List[str]:
        deadline = kwargs.get("deadline")
        deadline_scope = (
            self.primary.backend.deadline(deadline)
            if hasattr(self.primary.backend, "deadline")
            else nullcontext()
        )
        with deadline_scope:
            try:
                keywords = self._extract_keywords(
                    query,
                    allow_model_fallback=deadline is None,
                )
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
        self._last_search_time = 0.0
        self._search_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        runtime = config.get("rag.runtime")
        self._cache_ttl = getattr(runtime, "cache_ttl_seconds", 300.0)
        self._cache_size = getattr(runtime, "cache_size", 128)
        self._web_timeout = getattr(runtime, "web_timeout_seconds", 4.0)

    @staticmethod
    def _remaining(deadline: float | None, maximum: float) -> float:
        if deadline is None:
            return maximum
        return max(0.0, min(maximum, deadline - time.monotonic()))

    def _rate_limit(self, deadline: float | None) -> bool:
        with self._cache_lock:
            now = time.monotonic()
            scheduled = max(now, self._last_search_time + 1.0)
            self._last_search_time = scheduled
        delay = scheduled - now
        if deadline is not None and delay >= deadline - now:
            return False
        if delay:
            time.sleep(delay)
        return True

    @staticmethod
    def _get_ddgs(timeout: int):
        try:
            from ddgs import DDGS
            return DDGS(timeout=timeout, verify=True)
        except Exception:
            return None

    def _ask(self, prompt: str) -> str:
        if isinstance(self.session, Session):
            return self.session.complete_once(prompt)
        return "".join(self.session(prompt)).strip()

    def __call__(self, query: str, **kwargs) -> List[str]:
        search_results = self.search(query=query, **kwargs)
        return self.format_results(search_results)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        deadline: float | None = None,
    ) -> List[dict]:
        """Performs a web search with DuckDuckGo fallback chain."""
        if not isinstance(query, str) or not query.strip():
            return []
        if not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise StorageError("Web top_k must be between 1 and 10.")
        # Search engines accept natural-language queries directly. Avoid an
        # unbounded LLM rewrite before the bounded network deadline.
        search_query = " ".join(query.split())

        # Check cache
        cache_key = (search_query, top_k)
        now = time.monotonic()
        with self._cache_lock:
            cached = self._search_cache.get(cache_key)
            if cached and cached[0] >= now:
                self._search_cache.move_to_end(cache_key)
                self.logger.info(f"Cache hit for '{search_query}'")
                return list(cached[1])
            if cached:
                del self._search_cache[cache_key]

        self.logger.info(f"Searching for '{search_query}'...")

        # Try DDGS library first
        results = self._search_ddgs(search_query, top_k, deadline)

        # Fallback: try DDG Instant Answer API, then lite HTML
        if not results:
            results = self._search_instant_answer(
                search_query, top_k, deadline
            )
        if not results:
            results = self._search_lite_fallback(
                search_query, top_k, deadline
            )

        if not results:
            self.logger.warning(f"Web search for '{search_query}' yielded no results.")

        # Cache result (shallow copy to avoid mutation)
        with self._cache_lock:
            if self._cache_size:
                self._search_cache[cache_key] = (
                    time.monotonic() + self._cache_ttl,
                    tuple(results),
                )
                self._search_cache.move_to_end(cache_key)
                while len(self._search_cache) > self._cache_size:
                    self._search_cache.popitem(last=False)

        return results

    def _search_ddgs(
        self,
        query: str,
        top_k: int,
        deadline: float | None,
    ) -> List[dict]:
        for attempt in range(3):
            remaining = self._remaining(deadline, self._web_timeout)
            if remaining < 1.0 or not self._rate_limit(deadline):
                break
            remaining = self._remaining(deadline, self._web_timeout)
            if remaining < 1.0:
                break
            try:
                ddgs = self._get_ddgs(max(1, int(remaining)))
                if ddgs is None:
                    break
                # ASVS 13.2.4: pin the library to the intended outbound
                # provider instead of its multi-provider "auto" backend.
                results = ddgs.text(
                    query,
                    max_results=top_k,
                    backend="duckduckgo",
                )
                if results:
                    return list(results)[:top_k]
            except Exception as e:
                self.logger.warning(f"DDGS attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    delay = min(2 ** attempt, self._remaining(deadline, 2.0))
                    if delay <= 0:
                        break
                    time.sleep(delay)
        return []

    def _search_instant_answer(
        self,
        query: str,
        top_k: int,
        deadline: float | None,
    ) -> List[dict]:
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
            timeout = self._remaining(deadline, self._web_timeout)
            if timeout <= 0:
                return []
            with self._open_allowed(req, timeout) as resp:
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

    def _search_lite_fallback(
        self,
        query: str,
        top_k: int,
        deadline: float | None,
    ) -> List[dict]:
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
            timeout = self._remaining(deadline, self._web_timeout)
            if timeout <= 0:
                return []
            with self._open_allowed(req, timeout) as resp:
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

        response = self._ask(prompt)
        parsed = json.loads(response)
        value = parsed.get("search_query") if isinstance(parsed, dict) else None
        if not isinstance(value, str):
            return query
        value = " ".join(value.split())
        return value[:256] or query

    @staticmethod
    def _open_allowed(request, timeout: float):
        import urllib.error
        import urllib.parse
        import urllib.request

        allowed_hosts = frozenset(
            {"api.duckduckgo.com", "lite.duckduckgo.com"}
        )

        class HTTPSAllowlistRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(
                self, req, fp, code, msg, headers, newurl
            ):
                parsed = urllib.parse.urlparse(newurl)
                # ASVS 12.3.1 / 13.2.4 / 15.3.2: outbound redirects stay
                # on the fixed HTTPS search-service allowlist.
                if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
                    raise urllib.error.HTTPError(
                        newurl, code, "Redirect target is not allowed", headers, fp
                    )
                return super().redirect_request(
                    req, fp, code, msg, headers, newurl
                )

        parsed = urllib.parse.urlparse(request.full_url)
        if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
            raise StorageError("Web search endpoint is not allowed.")
        opener = urllib.request.build_opener(HTTPSAllowlistRedirect())
        return opener.open(request, timeout=timeout)

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
