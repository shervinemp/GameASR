from typing import Any, Dict, List, Tuple
from time import sleep

from ..common.config import config
from ..common.utils import get_logger


class KnowledgeGraph:
    _rel_addendum: str = (
        "{head: startNode(rel).id, tail: endNode(rel).id, type: type(rel)}"
    )

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
        Executes APOC k-shortest paths between two entities to find multi-hop semantic links.
        """
        query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            CALL apoc.algo.kShortestPaths(source, target, '>', '', $k)
            YIELD path, weight
            RETURN [node in nodes(path) | apoc.map.removeKey(properties(node), 'embedding')] AS nodes,
                   [rel in relationships(path) | apoc.map.merge(properties(rel), {self._rel_addendum})] AS relations,
                   weight
            ORDER BY weight ASC
        """

        def _run():
            with self._driver.session() as session:
                records = session.run(
                    query, source_id=source_id, target_id=target_id
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
        import asyncio

        def _sync_exec():
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
                        sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        self.logger.error(
                            "Max retries reached for Neo4j connection."
                        )
                        raise e

        try:
            loop = asyncio.get_running_loop()
            is_running = True
        except RuntimeError:
            loop = asyncio.get_event_loop()
            is_running = False

        if is_running:
            import concurrent.futures
            # The issue explicitly instructed: "Wrap the session.run() execution inside asyncio.get_event_loop().run_in_executor(None, sync_db_call)".
            # We fix the previous double-execution bug by wrapping the future in a proper coroutine before calling run_coroutine_threadsafe.
            async def _wait_for_future():
                return await loop.run_in_executor(None, _sync_exec)

            try:
                return asyncio.run_coroutine_threadsafe(_wait_for_future(), loop).result()
            except Exception as e:
                self.logger.error(f"Error executing DB call in background thread: {e}")
                raise
        else:
            # Synchronous execution fallback if no event loop is active
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(_sync_exec).result()

    def keyword_search(
        self, keywords: List[str], top_k: int = 5
    ) -> List[List[Dict]]:
        queries_data = [
            {"id": i, "keyword": keyword} for i, keyword in enumerate(keywords)
        ]
        query = """
            UNWIND $queries_data AS q_data
            CALL (q_data) {
                CALL db.index.fulltext.queryNodes("nodes_label_description_fulltext", q_data.keyword + '~', {limit: $top_k})
                YIELD node, score
                RETURN collect(
                    apoc.map.removeKey(properties(node), 'embedding')
                ) AS result_list
            }
            RETURN q_data.id AS query_id, result_list AS results
            ORDER BY query_id ASC
        """

        def _run():
            with self._driver.session() as session:
                records = session.run(
                    query, queries_data=queries_data, top_k=top_k
                )
                return [record["results"] for record in records]

        res = self._execute_with_retry(_run)
        import asyncio
        if asyncio.isfuture(res):
            res = asyncio.run_coroutine_threadsafe(asyncio.wrap_future(res), asyncio.get_event_loop()).result()
        return res

    def vector_search(
        self, embeddings: List[List[float]], top_k: int = 5
    ) -> List[List[Dict]]:
        queries_data = [
            {"id": i, "embedding": emb.tolist()}
            for i, emb in enumerate(embeddings)
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

        def _run():
            with self._driver.session() as session:
                records = session.run(
                    query, queries_data=queries_data, top_k=top_k
                )
                return [record["results"] for record in records]

        res = self._execute_with_retry(_run)
        import asyncio
        if asyncio.isfuture(res):
            res = asyncio.run_coroutine_threadsafe(asyncio.wrap_future(res), asyncio.get_event_loop()).result()
        return res

    def subgraph(self, node_ids: List[str]) -> Tuple[List[Dict], List[Dict]]:
        query = f"""
            MATCH (n:Entity) WHERE n.id IN $nodes
            OPTIONAL MATCH (n)-[r]-(m:Entity) WHERE m.id IN $nodes
            WITH COLLECT(DISTINCT n) AS nodes, COLLECT(DISTINCT r) AS rels
            RETURN [node in nodes | apoc.map.removeKey(properties(node), 'embedding')] AS nodes,
                   [rel in rels | apoc.map.merge(properties(rel), {self._rel_addendum})] AS relations
        """

        def _run():
            with self._driver.session() as session:
                record = session.run(query, nodes=node_ids).single()
                return record.data()

        res = self._execute_with_retry(_run)
        import asyncio
        if asyncio.isfuture(res):
            res = asyncio.run_coroutine_threadsafe(asyncio.wrap_future(res), asyncio.get_event_loop()).result()
        return res

    def expansion(
        self,
        frontier_ids: List[str],
        excluded_ids: List[str],
        n_hops: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        """
        Performs a multi-hop expansion from a set of frontier nodes,
        returning the expanded nodes along with their parent and the
        connecting relationship.
        """

        query = f"""
            UNWIND $frontier AS sourceId
            CALL {{
                WITH sourceId
                MATCH (n:Entity {{id: sourceId}})
                CALL apoc.path.expandConfig(n, {{
                    minLevel: 1, // Start from 1 hop away
                    maxLevel: {n_hops},
                    relationshipFilter: ">", // Only traverse outgoing relationships
                    labelFilter: "+Entity",   // Only traverse to Entity nodes
                    uniqueness: "NODE_PATH"  // Ensures paths are simple (no repeated nodes)
                }})
                YIELD path

                WITH last(nodes(path)) AS node,
                     nodes(path)[size(nodes(path))-2] AS parent,
                     last(relationships(path)) AS rel

                WHERE NOT node.id IN $excluded

                RETURN collect(DISTINCT {{
                    node: apoc.map.removeKey(properties(node), 'embedding'),
                    parent: apoc.map.removeKey(properties(parent), 'embedding'),
                    relationship: apoc.map.merge(properties(rel), {self._rel_addendum})
                }}) AS single_result
            }}
            RETURN single_result AS results
        """

        def _run():
            with self._driver.session() as session:
                records = session.run(
                    query, frontier=frontier_ids, excluded=excluded_ids
                )
                return [r["results"] for r in records]

        res = self._execute_with_retry(_run)
        import asyncio
        if asyncio.isfuture(res):
            res = asyncio.run_coroutine_threadsafe(asyncio.wrap_future(res), asyncio.get_event_loop()).result()
        return res

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
            return_clause = "RETURN s"
        elif o and o.get("name") == "?":
            return_clause = "RETURN o"
        else:
            return_clause = "RETURN s, o"
        query = f"MATCH {''.join(query_parts)} {return_clause} LIMIT 10"

        def _run():
            with self._driver.session() as session:
                records = session.run(query, **params)
                return [
                    record.data()[key]
                    for record in records
                    for key in record.keys()
                ]

        res = self._execute_with_retry(_run)
        import asyncio
        if asyncio.isfuture(res):
            res = asyncio.run_coroutine_threadsafe(asyncio.wrap_future(res), asyncio.get_event_loop()).result()
        return res

    def add_triplets(self, triplets: List[Dict[str, str]]):
        """Adds new triplets to the knowledge graph, creating nodes and relationships as needed."""
        if not triplets:
            return

        query = """
        UNWIND $triplets AS triplet
        // Use MERGE for both nodes to avoid creating duplicates
        MERGE (s:Entity {label: triplet.subject})
        ON CREATE SET s.id = apoc.create.uuid(), s.description = 'Created by RAG agent'

        MERGE (o:Entity {label: triplet.object})
        ON CREATE SET o.id = apoc.create.uuid(), o.description = 'Created by RAG agent'

        // Use MERGE for the relationship to avoid duplicates
        // The relationship type is created dynamically from the predicate
        CALL apoc.merge.relationship(s, upper(replace(triplet.predicate, ' ', '_')), {}, {}, o)
        YIELD rel
        RETURN count(rel) AS created_relationships
        """

        def _run():
            with self._driver.session() as session:
                result = session.run(query, triplets=triplets)
                self.logger.info(
                    f"Added {result.single()['created_relationships']} new relationships to the graph."
                )

        res = self._execute_with_retry(_run)
        import asyncio
        if asyncio.isfuture(res):
            asyncio.run_coroutine_threadsafe(asyncio.wrap_future(res), asyncio.get_event_loop()).result()
