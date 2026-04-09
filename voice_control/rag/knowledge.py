import time
from typing import Any, Dict, List, Tuple

from ..common.config import config
from loguru import logger

NODE_PROJ = "properties(node) { .*, embedding: null }"
REL_PROJ = "properties(rel) { .*, source: startNode(rel).id, target: endNode(rel).id, type: coalesce(rel.type, type(rel)) }"

class KnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        from sentence_transformers import SentenceTransformer
        from neo4j import GraphDatabase

        self.logger = get_logger(__file__)
        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=100,
            connection_acquisition_timeout=2.0
        )
        embedding_model_name = config.get(
            "llm.models.embedding", "google/embeddinggemma-300m"
        )
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def verify_connectivity(self) -> bool:
        return self._driver.verify_connectivity()

    def k_shortest_paths(
        self, source_id: str, target_id: str, k: int = 3
    ) -> List[Dict]:
        """
        Executes pure Cypher k-shortest paths between two entities to find multi-hop semantic links.
        """
        query = f"""
            MATCH path = (source:Entity {{id: $source_id}})-[*1..3]->(target:Entity {{id: $target_id}})
            RETURN [node in nodes(path) | {NODE_PROJ}] AS nodes,
                   [rel in relationships(path) | {REL_PROJ}] AS relations,
                   length(path) AS weight
            ORDER BY weight ASC
            LIMIT $k
        """

        def _run():
            with self._driver.session() as session:
                records = session.run(
                    query, source_id=source_id, target_id=target_id, k=k
                )
                return [r.data() for r in records]

        return self._execute_with_retry(_run)

    def close(self):
        self._driver.close()

    def _execute_with_retry(self, func, *args, **kwargs):
        """Helper to execute a function with retry logic for Neo4j connection errors."""
        from neo4j.exceptions import (
            ServiceUnavailable,
            SessionExpired,
            DriverError,
        )

        max_retries = 3
        delay = 1

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (ServiceUnavailable, SessionExpired, DriverError) as e:
                self.logger.warning(
                    f"Neo4j connection error: {e}. Retrying {attempt+1}/{max_retries}..."
                )
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    self.logger.error(
                        "Max retries reached for Neo4j connection."
                    )
                    raise e

    def keyword_search(
        self, keywords: List[str], top_k: int = 5
    ) -> List[List[Dict]]:
        queries_data = [
            {"id": i, "keyword": keyword} for i, keyword in enumerate(keywords)
        ]
        query = f"""
            UNWIND $queries_data AS q_data
            CALL db.index.fulltext.queryNodes("nodes_label_description_fulltext", q_data.keyword + '~', {{limit: $top_k}})
            YIELD node, score
            RETURN q_data.id AS query_id, collect({NODE_PROJ}) AS results
            ORDER BY query_id ASC
        """

        def _run():
            with self._driver.session() as session:
                records = session.run(
                    query, queries_data=queries_data, top_k=top_k
                )
                return [record["results"] for record in records]

        return self._execute_with_retry(_run)

    def vector_search(
        self, embeddings: List[List[float]], top_k: int = 5
    ) -> List[List[Dict]]:
        queries_data = [
            {"id": i, "embedding": emb}
            for i, emb in enumerate(embeddings)
        ]
        query = f"""
            UNWIND $queries_data AS q_data
            CALL db.index.vector.queryNodes('embedding', $top_k, q_data.embedding) YIELD node, score
            RETURN q_data.id AS query_id, collect({NODE_PROJ}) AS results
            ORDER BY query_id ASC
        """

        def _run():
            with self._driver.session() as session:
                records = session.run(
                    query, queries_data=queries_data, top_k=top_k
                )
                return [record["results"] for record in records]

        return self._execute_with_retry(_run)

    def subgraph(self, node_ids: List[str]) -> Tuple[List[Dict], List[Dict]]:
        query = f"""
            MATCH (n:Entity) WHERE n.id IN $nodes
            OPTIONAL MATCH (n)-[rel]-(m:Entity) WHERE m.id IN $nodes
            WITH COLLECT(DISTINCT n) AS nodes, COLLECT(DISTINCT rel) AS rels
            RETURN [node in nodes | {NODE_PROJ}] AS nodes,
                   [rel in rels | {REL_PROJ}] AS relations
        """

        def _run():
            with self._driver.session() as session:
                record = session.run(query, nodes=node_ids).single()
                if record:
                    return record.data()
                return {"nodes": [], "relations": []}

        return self._execute_with_retry(_run)

    def expansion(
        self,
        frontier_ids: List[str],
        excluded_ids: List[str],
        n_hops: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        """
        Performs a multi-hop expansion from a set of frontier nodes natively in Cypher.
        """
        query = f"""
            UNWIND $frontier AS sourceId
            MATCH path = (n:Entity {{id: sourceId}})-[*1..{n_hops}]->(m:Entity)
            WHERE NOT m.id IN $excluded
            WITH sourceId, path, m, nodes(path)[-2] AS parent, last(relationships(path)) AS rel
            RETURN sourceId AS query_id, collect(DISTINCT {{
                node: {NODE_PROJ.replace('node', 'm')},
                parent: {NODE_PROJ.replace('node', 'parent')},
                relationship: {REL_PROJ}
            }}) AS results
        """

        def _run():
            with self._driver.session() as session:
                records = session.run(
                    query, frontier=frontier_ids, excluded=excluded_ids
                )
                # Ensure we return empty lists for frontiers that had no results
                result_map = {r["query_id"]: r["results"] for r in records}
                return [result_map.get(fid, []) for fid in frontier_ids]

        return self._execute_with_retry(_run)

    def triplet_search(self, triplet: Dict) -> List[Dict]:
        s, p, o = (
            triplet.get("subject"),
            triplet.get("predicate"),
            triplet.get("object"),
        )
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
            return_clause = f"RETURN {NODE_PROJ.replace('node', 's')} AS result"
        elif o and o.get("name") == "?":
            return_clause = f"RETURN {NODE_PROJ.replace('node', 'o')} AS result"
        else:
            return_clause = f"RETURN {NODE_PROJ.replace('node', 's')} AS s, {NODE_PROJ.replace('node', 'o')} AS o"

        query = f"MATCH {''.join(query_parts)} {return_clause} LIMIT 10"

        def _run():
            with self._driver.session() as session:
                records = session.run(query, **params)
                return [
                    record.data()[key]
                    for record in records
                    for key in record.keys()
                ]

        return self._execute_with_retry(_run)

    def add_triplets(self, triplets: List[Dict[str, str]]):
        """Adds new triplets to the knowledge graph, creating nodes and relationships as needed."""
        if not triplets:
            return

        query = """
        UNWIND $triplets AS triplet
        MERGE (s:Entity {label: triplet.subject})
        ON CREATE SET s.id = randomUUID(), s.description = 'Created by RAG agent'

        MERGE (o:Entity {label: triplet.object})
        ON CREATE SET o.id = randomUUID(), o.description = 'Created by RAG agent'

        // Pure Cypher workaround for dynamic relationship creation is difficult,
        // but if apoc is completely disabled, we use a generic relationship with type property
        // For complete APOC removal without CALL apoc.create.relationship:
        MERGE (s)-[rel:RELATED_TO {type: upper(replace(triplet.predicate, ' ', '_'))}]->(o)
        RETURN count(rel) AS created_relationships
        """

        def _run():
            with self._driver.session() as session:
                result = session.run(query, triplets=triplets)
                rec = result.single()
                if rec:
                    self.logger.info(
                        f"Added {rec['created_relationships']} new relationships to the graph."
                    )

        self._execute_with_retry(_run)
