import json
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase

from ..llm.model import LLM
from .retriever import RetrievalManager
from .explorer import ExplorationEngine
from .generator import GenerationService
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
        s, p, o = triplet.get("subject"), triplet.get("predicate"), triplet.get("object")
        if not p or not p.get("name"):
            return []
        query_parts = []
        params = {}
        if s and s.get("name") != "?":
            query_parts.append("(s:Entity {label: $s_name})")
            params["s_name"] = s["name"]
        else:
            query_parts.append("(s:Entity)")
        rel_type = p["name"].upper().replace(" ", "_")
        query_parts.append(f"-[r:{rel_type}]->")
        if o and o.get("name") != "?":
            query_parts.append("(o:Entity {label: $o_name})")
            params["o_name"] = o["name"]
        else:
            query_parts.append("(o:Entity)")
        if s and s.get("name") == "?":
            return_clause = "RETURN s"
        elif o and o.get("name") == "?":
            return_clause = "RETURN o"
        else:
            return_clause = "RETURN s, o"
        query = f"MATCH {''.join(query_parts)} {return_clause} LIMIT 10"
        with self._driver.session() as session:
            records = session.run(query, **params)
            results = [record.data()[key] for record in records for key in record.keys()]
        return results


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
        self.retrieval_manager = RetrievalManager(graph, llm, max_keywords)
        self.exploration_engine = ExplorationEngine(graph, llm, max_iterations, max_retries)
        self.generation_service = GenerationService(llm)

    def __call__(self, query: str) -> str:
        return self._execute_query(query)

    def _execute_query(self, query: str) -> str:
        report = {
            "state": "Starting search for clues with initial nodes...",
            "context": "",
            "explicit_mention": [],
        }

        initial_nodes = self.retrieval_manager.retrieve_initial_nodes(query)

        if not initial_nodes:
            return "Could not find any relevant information."

        final_answer, report = self.exploration_engine.explore(
            query, initial_nodes, report, self.generation_service
        )

        final_answer = self.generation_service.verify(final_answer, report)

        return {"answer": final_answer, "report": report}


def main():
    setup_logging("DEBUG")
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
