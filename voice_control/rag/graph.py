from collections import OrderedDict, deque
import heapq
import random
import sys
import json
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer, util
from typing import Any, Deque, Dict, Tuple, List

from neo4j import GraphDatabase

from ..llm.model import LLM
from ..llm.session import Session
from .triplet import KnowledgeExtractor

from ..common.utils import get_logger, setup_logging


class KnowledgeGraph:
    _rel_addendum: str = "{head: startNode(r).id, tail: endNode(r).id, type: type(r)}"

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_model = SentenceTransformer(
            "avsolatorio/GIST-small-Embedding-v0"
        )

    def close(self):
        self._driver.close()

    def keyword_search(self, keywords: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Performs a keyword search for a list of keywords.

        Args:
            keywords: A list of keywords to search for.
            top_k: The number of top results to return for each keyword.

        Returns:
            A list of lists, where each inner list contains the top-k search
            results (as dictionaries) for the corresponding keyword.
        """
        queries_data = [
            {"id": i, "keyword": keyword} for i, keyword in enumerate(keywords)
        ]

        query = """
            UNWIND $queries_data AS q_data
            CALL {
                WITH q_data
                CALL db.index.fulltext.queryNodes("nodes_label_description_fulltext", q_data.keyword + '~', {limit: $top_k})
                YIELD node, score
                RETURN collect(
                    apoc.map.removeKey(properties(node), 'embedding')
                ) AS result_list
            }
            RETURN q_data.id AS query_id, result_list AS results
            ORDER BY query_id ASC
        """

        with self._driver.session() as session:
            records = session.run(query, queries_data=queries_data, top_k=top_k)
            result = [record["results"] for record in records]

        return result

    def vector_search(
        self, embeddings: List[List[float]], top_k: int = 5
    ) -> List[List[Dict]]:
        """
        Performs a vector similarity search for a list of embeddings.

        Args:
            embeddings: A list of embeddings to search for.
            top_k: The number of top similar results to return for each embedding.

        Returns:
            A list of lists, where each inner list contains the top-k search
            results (as dictionaries) for the corresponding embedding.
        """
        queries_data = [
            {"id": i, "embedding": emb.tolist()} for i, emb in enumerate(embeddings)
        ]

        query = """
            UNWIND $queries_data AS q_data
            CALL (q_data) {
                CALL db.index.vector.queryNodes('embedding', $top_k, q_data.embedding)
                YIELD node, score
                RETURN collect(
                    apoc.map.removeKey(properties(node), 'embedding')
                ) AS result_list
            }
            RETURN q_data.id AS query_id, result_list AS results
            ORDER BY query_id ASC
        """

        with self._driver.session() as session:
            records = session.run(query, queries_data=queries_data, top_k=top_k)
            result = [record["results"] for record in records]

        return result

    def subgraph(self, node_ids: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """
        Retrieves a subgraph of the graph based on a list of node IDs.


        Args:
            node_ids: A list of node IDs to retrieve the subgraph for.

        Returns:
            A tuple containing a list of nodes and a list of connections (relationships).
        """
        query = """
            MATCH (n:Entity) WHERE n.id IN $nodes
            OPTIONAL MATCH (n)-[r]-(m:Entity) WHERE m.id IN $nodes
            WITH COLLECT(DISTINCT n) AS nodes, COLLECT(DISTINCT r) AS rels
            RETURN [n in nodes | apoc.map.removeKey(properties(n), 'embedding')] AS nodes,
                   [r in rels | apoc.map.merge(properties(r), {})] AS relations
        """.format(
            self._rel_addendum
        )
        with self._driver.session() as session:
            record = session.run(query, nodes=node_ids).single()
            result = record.data()

        return result

    def expansion(
        self, frontier_ids: List[str], excluded_ids: List[str]
    ) -> List[Dict[str, Dict[str, Any]]]:
        query = """
            UNWIND $frontier AS sourceId
            MATCH (n:Entity {{id: sourceId}})
            MATCH (n)-[r]-(m:Entity)
            WHERE NOT m.id IN $excluded
            
            RETURN collect({{
                node: apoc.map.removeKey(properties(m), 'embedding'),
                relation: apoc.map.merge(properties(r), {})
            }}) AS results
        """.format(
            self._rel_addendum
        )
        with self._driver.session() as session:
            records = session.run(query, frontier=frontier_ids, excluded=excluded_ids)
            result = [r["results"] for r in records]

        return result

    def triplet_search(self, triplet: Dict) -> List[Dict]:
        """
        Performs a search based on a structured triplet.
        Example: (Subject: '?', Predicate: 'is member of', Object: 'Coldplay')
        """
        s, p, o = triplet.get("subject"), triplet.get("predicate"), triplet.get("object")

        # Basic validation
        if not p or not p.get("name"):
            return []

        query_parts = []
        params = {}

        # Build match clauses based on which parts of the triplet are known
        if s and s.get("name") != "?":
            query_parts.append("(s:Entity {label: $s_name})")
            params["s_name"] = s["name"]
        else:
            query_parts.append("(s:Entity)")

        # Sanitize relationship type
        rel_type = p["name"].upper().replace(" ", "_")
        query_parts.append(f"-[r:{rel_type}]->")

        if o and o.get("name") != "?":
            query_parts.append("(o:Entity {label: $o_name})")
            params["o_name"] = o["name"]
        else:
            query_parts.append("(o:Entity)")

        # Determine what to return
        if s and s.get("name") == "?":
            return_clause = "RETURN s"
        elif o and o.get("name") == "?":
            return_clause = "RETURN o"
        else: # If no unknown, just confirm existence
            return_clause = "RETURN s, o"

        query = f"MATCH {''.join(query_parts)} {return_clause} LIMIT 10"

        with self._driver.session() as session:
            records = session.run(query, **params)
            # The result could be 's' or 'o', so we need to handle both
            results = [record.data()[key] for record in records for key in record.keys()]
        return results


class Exploration:

    class Frontier:
        def __init__(self, maxlen: int | None = None):
            self._d = OrderedDict()
            self._maxlen = maxlen

        def append(self, x):
            k = x["id"]
            if k in self._d:
                self._d.move_to_end(k)
            self._d[k] = x
            if self._maxlen and len(self) > self._maxlen:
                _ = self.pop()

        def extend(self, X):
            for x in X:
                self.append(x)

        def pop(self):
            return self._d.popitem(last=False)

        def clear(self):
            self._d.clear()

        def __iter__(self):
            return iter(self._d.values())

        def __repr__(self):
            return f"Frontier({self._d})"

        def __str__(self):
            return self._d

        def __len__(self):
            return len(self._d)

    def __init__(self, graph: KnowledgeGraph, frontier_size: int = 12):
        self.graph = graph
        self.frontier = self.Frontier(frontier_size)
        self.ancestry = dict()

    def start(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        subgraph = self.graph.subgraph(node_ids)
        self._add(subgraph["nodes"], subgraph["relations"])
        return subgraph

    def expand(
        self,
        candidates: List[Dict[str, Dict[str, Any]]],
        flush_frontier: bool = False,
    ) -> None:
        if flush_frontier:
            self.frontier.clear()
        self._add(candidates["nodes"], candidates["relations"])

    @property
    def candidates(self) -> List[Dict[str, Dict[str, Any]]]:
        frontier_ids = [n["id"] for n in self.frontier]
        excluded_ids = list(self.ancestry.keys())
        candidates = self.graph.expansion(
            frontier_ids=frontier_ids, excluded_ids=excluded_ids
        )
        return candidates

    def _add(self, nodes: List[Dict[str, Any]], relations: List[Dict] = None) -> None:
        self.frontier.extend(nodes)

        for node in nodes:
            nid = node["id"]
            if nid not in self.ancestry:
                self.ancestry[nid] = nid

        if relations is not None:
            for relation in relations:
                a, b = relation["head"], relation["tail"]
                if relation["head"] not in self.ancestry:
                    a, b = b, a
                self.ancestry[b] = self.ancestry[a]


class RAG:
    def __init__(
        self,
        graph: KnowledgeGraph,
        llm: LLM | None = None,
        max_iterations: int = 5,
        max_keywords: int = 3,
        max_retries: int = 3,
    ):
        self.logger = get_logger(__file__)

        self.graph = graph
        self.max_iterations = max_iterations
        self.max_keywords = max_keywords
        self.max_retries = max_retries

        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1
        self.triplet_extractor = KnowledgeExtractor(llm)

    def __call__(self, query: str) -> str:
        """
        Looks up the answer to the given query through knowledge graph search.

        Args:
            query: The query to look up.
        Returns:
            The answer to the query.
        """
        return self._execute_query(query)

    def _execute_query(self, query: str) -> str:

        self.report = {
            "state": "Starting search for clues with initial nodes...",
            "context": "",
            "explicit_mention": [
                {
                    "targeted_part": "{example_query_part}",
                    "relation": "{example_relation}",
                    "subject": "{example_subject_label}",
                    "object": "{example_object_label}",
                    "desc": "{example_description}",
                }
            ],
        }

        # --- Triplet-based Search ---
        triplet_results = []
        try:
            extracted_triplets_str = self.triplet_extractor.extract_triplets(query, retrieval=True)
            extracted_triplets = json.loads(extracted_triplets_str)
            self.logger.debug(f"Extracted triplets: {extracted_triplets}")
            for triplet in extracted_triplets:
                if "?" in str(triplet): # Check if it's a question
                    triplet_results.extend(self.graph.triplet_search(triplet))
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.warning(f"Could not extract or process triplets: {e}")

        expanded_query = self._expand_query(query)
        self.logger.debug(f"Expanded query: {expanded_query}")

        keywords = self._extract_keywords(expanded_query)[: self.max_keywords]
        embeddings = self.graph.embedding_model.encode(keywords)

        initial_results = [
            a + [n for n in b if n["id"] not in (e["id"] for e in a)]
            for a, b in zip(
                self.graph.vector_search(embeddings, top_k=4),
                self.graph.keyword_search(keywords, top_k=2),
            )
        ]

        # Flatten and rerank results
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

        if not initial_nodes:
            return "Could not find any relevant information."

        state = Exploration(self.graph)
        state.start(initial_nodes)
        i = 0
        errors = 0
        while i < self.max_iterations:
            self.logger.info(f"\n--- Iteration {i + 1} ---")
            if not state.frontier:
                self.logger.info("Frontier is empty. Halting exploration.")
                break

            # Step 1: Score candidates
            scoring_prompt = self._build_scoring_prompt(query, state)
            scoring_response = "".join(self.session(scoring_prompt))
            self.logger.debug(f"Scoring Response: {scoring_response}")

            try:
                scored_candidates = json.loads(scoring_response)
                # Filter for candidates with a score > 5
                promising_ids = [c["id"] for c in scored_candidates if c.get("score", 0) > 5]
            except (json.JSONDecodeError, TypeError):
                self.logger.warning(f"Failed to decode scoring JSON: {scoring_response}")
                errors += 1
                if errors >= self.max_retries:
                    break # Break on persistent error
                i += 1
                continue

            nodes_to_expand = [
                n
                for kword_arr in state.candidates
                for c in kword_arr
                if (n := c["node"])["id"] in promising_ids
            ]

            # Step 2: Generate report and answer from promising nodes
            if nodes_to_expand:
                generation_prompt = self._build_generation_prompt(query, self.report, nodes_to_expand)
                generation_response = "".join(self.session(generation_prompt))
                self.logger.debug(f"Generation Response: {generation_response}")

                try:
                    generation_data = json.loads(generation_response)
                    self.report = generation_data.get("report", self.report)
                    final_answer = generation_data.get("answer", None)
                    is_verified = generation_data.get("is_verified", False)
                except (json.JSONDecodeError, TypeError):
                    self.logger.warning(f"Failed to decode generation JSON: {generation_response}")
                    errors += 1
                    if errors >= self.max_retries:
                        break # Break on persistent error
                    i += 1
                    continue
            else: # No promising nodes found
                final_answer = None
                is_verified = False


            if is_verified:
                self.logger.info("Final answer found.")
                break

            if not nodes_to_expand:
                self.logger.info("No nodes to expand. Halting exploration.")
                break

            state.expand({"nodes": nodes_to_expand, "relations": None})
            i += 1
            errors = 0 # Reset errors on successful iteration

        if final_answer and not self._verify_answer(final_answer, self.report):
            final_answer = "I found some relevant information, but could not form a confident answer based on the facts."

        return {"answer": final_answer, "report": self.report}

    def _verify_answer(self, answer: str, report: Dict) -> bool:
        """
        Verifies if the answer is factually supported by the evidence in the report.
        """
        prompt = (
            "You are a fact-checker. Your task is to determine if the provided 'Answer' is fully supported by the 'Evidence'. "
            "Respond with only 'true' or 'false'.\n"
            f" * Evidence: {json.dumps(report, indent=2)}\n"
            f" * Answer: {answer}\n"
            "Is the answer fully and directly supported by the evidence? (true/false)"
        )

        response = "".join(self.session(prompt)).strip().lower()
        self.logger.debug(f"Verification response: {response}")
        return response == 'true'

    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Reranks search results based on semantic similarity to the query.
        """
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
        """
        Expands the user query with alternative phrasings using an LLM.
        """
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

    def _build_scoring_prompt(self, query: str, state: Exploration, max_neighbors: int = 20) -> str:
        """
        Builds a prompt to score candidate nodes for relevance.
        """
        candidates = [
            item
            for kword_arr in state.candidates
            for item in random.sample(kword_arr, min(max_neighbors, len(kword_arr)))
        ]

        # Simplified representation of candidates
        candidate_descs = []
        for item in candidates:
            node = item['node']
            desc = f"ID: {node['id']}, Label: {node['label']}, Description: {node.get('description', 'N/A')}"
            candidate_descs.append(desc)

        candidates_str = "\n".join(candidate_descs)

        return (
            "Task: Score the following candidate nodes based on their relevance to the user's query. "
            "A score of 10 is highly relevant, and 1 is not relevant at all. "
            "Return a JSON array of objects, where each object has 'id' and 'score' keys.\n"
            f" * User Query: '{query}'\n"
            f" * Candidate Nodes:\n{candidates_str}\n"
            "Please provide the JSON array of scores."
        )

    def _build_generation_prompt(self, query: str, report: Dict, nodes: List[Dict]) -> str:
        """
        Builds a prompt to generate an answer and update the report based on promising nodes.
        """
        nodes_info = []
        for node in nodes:
            nodes_info.append(f"- {node['label']} ({node['id']}): {node.get('description', 'N/A')}")

        nodes_str = "\n".join(nodes_info)

        return (
            "Task: Based on the provided evidence, update the report and provide the best possible answer to the user's query. "
            "Return a JSON object with three keys:\n"
            "1. 'report': A dictionary compiling the most relevant evidence found so far.\n"
            "2. 'answer': The best-guess, human-readable answer. Can be null if the answer is not yet known.\n"
            "3. 'is_verified': A boolean that is true only if the answer is completely verified by the evidence.\n"
            f" * User Query: '{query}'\n"
            f" * Current Report: {report}\n"
            f" * New Evidence (Promising Nodes):\n{nodes_str}\n"
            "Please provide the updated JSON object."
        )

    def _extract_keywords(self, query: str) -> List[str]:
        prompt = (
            "Extract, from the following query, proper entities and keywords that will be used to deduce a potential answer."
            "Make sure the extracted entities and keywords are conceptually meaningful and relevant to the query."
            "Return a JSON array of strings including the extracted entities and keywords."
            f"query:\n{query}"
        )

        response = "".join(self.session(prompt))
        return json.loads(response)


def main():

    setup_logging("DEBUG", stream=sys.stdout)
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
