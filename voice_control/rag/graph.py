from collections import deque
import random
import sys
import json
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
from typing import Any, Deque, Dict, Tuple, List

from neo4j import GraphDatabase

from ..llm.model import LLM
from ..llm.session import Session

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


class Exploration:

    def __init__(self, graph: KnowledgeGraph, frontier_size: int = 12):
        self.graph = graph
        self.frontier: Deque[Dict[str, Any]] = deque(maxlen=frontier_size)
        self.ancestry: Dict[str, str] = dict()

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


class Orchestrator:
    def __init__(
        self,
        graph: KnowledgeGraph,
        llm: LLM | None = None,
        max_iterations: int = 5,
    ):
        self.logger = get_logger(__file__)

        self.graph = graph
        self.max_iterations = max_iterations

        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1

    def __call__(self, query: str) -> str:
        return self._execute_query(query)

    def _execute_query(self, query: str) -> str:

        self.answer = ""
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

        keywords = self._extract_keywords(query)
        embeddings = self.graph.embedding_model.encode(keywords)

        initial_results = [
            a + [n for n in b if n["id"] not in (e["id"] for e in a)]
            for a, b in zip(
                self.graph.vector_search(embeddings, top_k=4),
                self.graph.keyword_search(keywords, top_k=2),
            )
        ]

        self.logger.debug("Found initial candidates:")
        initial_nodes = []
        for i, kword_arr in enumerate(initial_results):
            self.logger.debug(f"{keywords[i]}:")
            for r in kword_arr:
                self.logger.debug(f"  - ID: {r['id']}, Label: {r['label']}")
                initial_nodes.append(r["id"])

        if not initial_nodes:
            return "Could not find any relevant information."

        state = Exploration(self.graph)
        state.start(initial_nodes)
        i = 0
        while i < self.max_iterations:
            self.logger.info(f"\n--- Iteration {i + 1} ---")
            if not state.frontier:
                self.logger.info("Frontier is empty. Halting exploration.")
                break

            prompt = self._build_expansion_prompt(query, state)
            response = "".join(self.session(prompt))
            self.logger.debug(f"Response: {response}")

            response = json.loads(response)

            new_frontier = response.get("new_frontier", [])
            nodes_to_expand = [
                n
                for kword_arr in state.candidates
                for c in kword_arr
                if (n := c["node"])["id"] in new_frontier
            ]

            self.report = response.get("report", {})
            final_answer = response.get("answer", None)
            is_verified = response.get("is_verified", False)

            self.answer = final_answer

            if is_verified:
                self.logger.info("Final answer found.")
                break

            if not nodes_to_expand:
                self.logger.info("No nodes to expand. Halting exploration.")
                break

            state.expand({"nodes": nodes_to_expand, "relations": None})
            i += 1

        return {"answer": final_answer, "report": self.report}

    def _build_expansion_prompt(
        self, query: str, state: Exploration, max_neighbors: int = 10
    ) -> str:
        frontier = list(state.frontier)
        candidates = [
            kword_arr[i]
            for kword_arr in state.candidates
            for i in random.sample(
                range(len(kword_arr)), min(max_neighbors, len(kword_arr))
            )
        ]

        id_to_node = {n["id"]: n for n in frontier + [c["node"] for c in candidates]}

        triples = []
        descs = []
        for item in candidates:
            node = item["node"]
            relation = item["relation"]

            head_id = relation["head"]
            tail_id = relation["tail"]

            rel_type = relation["type"]

            ltr = node["id"] == tail_id
            if not ltr:
                head_id, tail_id = tail_id, head_id

            head_label = id_to_node[head_id]["label"]
            tail_label = id_to_node[tail_id]["label"]

            descs_string = f"({node['label']}::{node['id']}): {node['description']}"
            descs.append(descs_string)

            triple_string = (
                f"({head_label}::{head_id}) {'' if ltr else '<'}- "
                f"[{rel_type}]"
                f" -{'>' if ltr else ''} ({tail_label}::{tail_id})"
            )
            triples.append(triple_string)

        nodes_str = "\n".join(descs)
        candidates_str = "\n".join(triples)

        return (
            "Task: Analyze our potential new candidate nodes and their relations with regard to the query. "
            "Consider descriptions, and their connection to our investigation and query. "
            "None of the provided candidates are guaranteed to be relevant. Rely on the query and report for guidance. "
            "Return a JSON object with four keys:\n1. 'new_frontier': a shortlist containing only IDs "
            "(right side of '::' with the 'Q' prefix) of all potentially promising nodes to add to our frontier.\n"
            "2. 'report': a small human-readable (IDs accompanied by labels) dictionary compiling relevant and verified evidence.\n"
            "3. 'answer': the conservative best-guess precise human-readable answer to the query, excluding IDs.\n"
            "4. 'is_verified': a boolean indicator, strictly true only when the objective is met and the answer "
            "to the query is directly and completely verified and cross-referenced with the provided context."
            f" * Query: '{query}'\n"
            f" * Report: {self.report}\n"
            f" * Nodes:\n{nodes_str}\n"
            f" * Outgoing:\n{candidates_str}"
            f" * Query: '{query}'\n"
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
    orchestrator = Orchestrator(graph)

    try:
        i = 0
        while i < len(user_queries):
            user_query = user_queries[i]
            try:
                final_answer = orchestrator(user_query)
                logger.info(final_answer)
                i += 1
            except Exception as e:
                logger.warning(e)
    finally:
        graph.close()


if __name__ == "__main__":
    main()
