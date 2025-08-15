import json
from typing import List, Dict

from sentence_transformers import util

from ..llm.model import LLM
from ..llm.session import Session
from .triplet import KnowledgeExtractor
from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger

class RetrievalManager:
    def __init__(self, graph: KnowledgeGraph, llm: LLM, max_keywords: int = 3):
        self.logger = get_logger(__file__)
        self.graph = graph
        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1
        self.triplet_extractor = KnowledgeExtractor(llm)
        self.max_keywords = max_keywords

    def retrieve_initial_nodes(self, query: str) -> List[str]:
        # --- Triplet-based Search ---
        triplet_results = []
        try:
            extracted_triplets_str = self.triplet_extractor.extract_triplets(query, retrieval=True)
            extracted_triplets = json.loads(extracted_triplets_str)
            self.logger.debug(f"Extracted triplets: {extracted_triplets}")
            for triplet in extracted_triplets:
                if "?" in str(triplet):  # Check if it's a question
                    triplet_results.extend(self.graph.triplet_search(triplet))
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.warning(f"Could not extract or process triplets: {e}")

        # --- Keyword and Vector Search ---
        expanded_query = self._expand_query(query)
        self.logger.debug(f"Expanded query: {expanded_query}")

        keywords = self._extract_keywords(expanded_query)[:self.max_keywords]
        embeddings = self.graph.embedding_model.encode(keywords)

        initial_results = [
            a + [n for n in b if n["id"] not in (e["id"] for e in a)]
            for a, b in zip(
                self.graph.vector_search(embeddings, top_k=4),
                self.graph.keyword_search(keywords, top_k=2),
            )
        ]

        # --- Combine and Rerank ---
        flat_results = [item for sublist in initial_results for item in sublist]
        if triplet_results:
            flat_results.extend(triplet_results)

        unique_results = list({item['id']: item for item in flat_results}.values())
        reranked_results = self._rerank_results(unique_results, query)

        self.logger.debug("Found initial candidates (post-reranking):")
        initial_nodes = []
        for r in reranked_results:
            self.logger.debug(f"  - ID: {r['id']}, Label: {r['label']}, Score: {r['score']}")
            initial_nodes.append(r["id"])

        return initial_nodes

    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        if not results:
            return []
        query_embedding = self.graph.embedding_model.encode(query, convert_to_tensor=True)
        result_texts = [f"{r.get('label', '')}: {r.get('description', '')}" for r in results]
        result_embeddings = self.graph.embedding_model.encode(result_texts, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, result_embeddings)
        for i, result in enumerate(results):
            result['score'] = similarities[0][i].item()
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def _expand_query(self, query: str) -> str:
        prompt = (
            "Please rephrase and expand the following user query to improve search results. "
            "Generate 3 alternative, more detailed queries. "
            "Return a JSON array of strings with the new queries."
            f"query:\n{query}"
        )
        try:
            response = "".join(self.session(prompt))
            expanded_queries = json.loads(response)
            return " ".join([query] + expanded_queries)
        except (json.JSONDecodeError, TypeError):
            self.logger.warning("Failed to expand query, using original query.")
            return query

    def _extract_keywords(self, query: str) -> List[str]:
        prompt = (
            "Extract, from the following query, proper entities and keywords that will be used to deduce a potential answer."
            "Make sure the extracted entities and keywords are conceptually meaningful and relevant to the query."
            "Return a JSON array of strings including the extracted entities and keywords."
            f"query:\n{query}"
        )
        response = "".join(self.session(prompt))
        return json.loads(response)
