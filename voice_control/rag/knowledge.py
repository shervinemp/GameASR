import hashlib
import ipaddress
from contextlib import contextmanager
import re
import threading
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from ..common.config import config
from ..common.utils import get_logger
from .validation import normalize_triplets

NODE_PROJ = "{id: node.id, label: node.label, description: node.description}"
REL_PROJ = "{id: rel.id, source: startNode(rel).id, target: endNode(rel).id, type: type(rel)}"

class KnowledgeGraph:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
        query_timeout: float = 5.0,
    ):
        from sentence_transformers import SentenceTransformer
        from neo4j import GraphDatabase

        self.logger = get_logger(__file__)
        self._validate_uri(uri)
        if not isinstance(database, str) or not database.strip():
            raise ValueError("Neo4j database must be a non-empty string.")
        if not isinstance(query_timeout, (int, float)) or not 1 <= query_timeout <= 30:
            raise ValueError("Neo4j query timeout must be between 1 and 30 seconds.")
        self._database = database
        self._query_timeout = float(query_timeout)
        self._deadline_state = threading.local()
        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=20,
            connection_acquisition_timeout=3.0,
            connection_timeout=3.0,
            max_transaction_retry_time=3.0,
            keep_alive=True,
        )
        embedding_model_name = config.get(
            "llm.models.embedding", "google/embeddinggemma-300m"
        )
        self.embedding_model = SentenceTransformer(embedding_model_name)

    @staticmethod
    def _validate_uri(uri: str) -> None:
        parsed = urlparse(uri)
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Neo4j URI must not contain embedded credentials.")
        secure = parsed.scheme in {"bolt+s", "neo4j+s"}
        local = parsed.hostname.lower() == "localhost"
        if not local:
            try:
                local = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                local = False
        # ASVS 12.3.1 / 12.3.2: plaintext database transport is limited to
        # loopback development; remote certificates are driver-validated.
        if not secure and not (
            local and parsed.scheme in {"bolt", "neo4j"}
        ):
            raise ValueError(
                "Remote Neo4j connections must use bolt+s or neo4j+s."
            )

    @contextmanager
    def deadline(self, value: float | None):
        previous = getattr(self._deadline_state, "value", None)
        self._deadline_state.value = value
        try:
            yield
        finally:
            self._deadline_state.value = previous

    def _remaining_timeout(self) -> float:
        timeout = self._query_timeout
        deadline_state = getattr(self, "_deadline_state", None)
        deadline = getattr(deadline_state, "value", None)
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("Neo4j retrieval deadline expired.")
        return max(0.1, timeout)

    def _timed_query(self, query: str):
        from neo4j import Query
        return Query(query, timeout=self._remaining_timeout())

    def _read_session(self):
        from neo4j import READ_ACCESS
        return self._driver.session(
            database=self._database,
            default_access_mode=READ_ACCESS,
        )

    def _write_session(self):
        from neo4j import WRITE_ACCESS
        return self._driver.session(
            database=self._database,
            default_access_mode=WRITE_ACCESS,
        )

    def verify_connectivity(self) -> bool:
        return self._driver.verify_connectivity()

    def k_shortest_paths(
        self, source_id: str, target_id: str, k: int = 3
    ) -> List[Dict]:
        return self.k_shortest_paths_batch([(source_id, target_id)], k=k)

    def k_shortest_paths_batch(
        self, pairs: List[Tuple[str, str]], k: int = 3
    ) -> List[Dict]:
        """
        Batches multiple k-shortest path queries into a single Cypher call.
        Each pair is (source_id, target_id). Returns flat list of path dicts.
        """
        if not pairs:
            return []
        if not isinstance(k, int) or not 1 <= k <= 5:
            raise ValueError("k must be between 1 and 5.")
        if len(pairs) > 64:
            raise ValueError("At most 64 shortest-path pairs are allowed.")
        if any(
            not isinstance(source, str)
            or not isinstance(target, str)
            or not source
            or not target
            for source, target in pairs
        ):
            raise ValueError("Shortest-path node identifiers must be strings.")

        # ASVS 1.2.4: the only structural query value is an allowlisted integer;
        # all node identifiers remain Cypher parameters.
        query = f"""
            UNWIND $pairs AS p
            MATCH (source:Entity {{id: p.src}}),
                  (target:Entity {{id: p.tgt}})
            CALL (source, target) {{
                MATCH path = SHORTEST {k}
                    (source)-[*1..3]-(target)
                RETURN path
            }}
            RETURN [node in nodes(path) | {NODE_PROJ}] AS nodes,
                   [rel in relationships(path) | {REL_PROJ}] AS relations,
                   length(path) AS weight
        """

        pair_data = [{"src": s, "tgt": t} for s, t in pairs]

        def _run():
            with self._read_session() as session:
                records = session.run(
                    self._timed_query(query), pairs=pair_data
                )
                return [r.data() for r in records]

        return self._execute_with_retry(_run)

    def close(self):
        self._driver.close()

    def _execute_with_retry(self, func):
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
                return func()
            except (ServiceUnavailable, SessionExpired, DriverError) as e:
                self.logger.warning(
                    f"Neo4j connection error: {e}. Retrying {attempt+1}/{max_retries}..."
                )
                if attempt < max_retries - 1:
                    deadline = getattr(
                        getattr(self, "_deadline_state", None),
                        "value",
                        None,
                    )
                    if deadline is not None and time.monotonic() + delay >= deadline:
                        raise TimeoutError(
                            "Neo4j retry would exceed retrieval deadline."
                        ) from e
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
            with self._read_session() as session:
                records = session.run(
                    self._timed_query(query),
                    queries_data=queries_data,
                    top_k=top_k,
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
            with self._read_session() as session:
                records = session.run(
                    self._timed_query(query),
                    queries_data=queries_data,
                    top_k=top_k,
                )
                return [record["results"] for record in records]

        return self._execute_with_retry(_run)

    def exact_label_search(self, labels: List[str]) -> Dict[str, Dict]:
        """Returns a dict mapping normalized label -> full node for exact label matches."""
        if not labels:
            return {}

        clean_labels = list(dict.fromkeys(
            label.strip().lower()
            for label in labels[:64]
            if isinstance(label, str) and label.strip()
        ))
        indexed_query = """
            UNWIND $labels AS label
            MATCH (n:Entity {normalized_label: label})
            RETURN label AS query_label, n.id AS id, n.label AS label,
                   n.description AS description
        """
        legacy_query = """
            UNWIND $labels AS label
            MATCH (n:Entity)
            WHERE n.normalized_label IS NULL
              AND toLower(trim(n.label)) = label
            RETURN label AS query_label, n.id AS id, n.label AS label,
                   n.description AS description
        """

        def _run():
            with self._read_session() as session:
                records = session.run(
                    self._timed_query(indexed_query), labels=clean_labels
                )
                results = {
                    r["query_label"]: {
                        "id": r["id"],
                        "label": r["label"],
                        "description": r["description"],
                    }
                    for r in records
                }
                missing = [
                    label for label in clean_labels if label not in results
                ]
                if missing:
                    legacy_records = session.run(
                        self._timed_query(legacy_query), labels=missing
                    )
                    for record in legacy_records:
                        results[record["query_label"]] = {
                            "id": record["id"],
                            "label": record["label"],
                            "description": record["description"],
                        }
                return results

        return self._execute_with_retry(_run)

    def subgraph(self, node_ids: List[str]) -> Dict[str, List[Dict]]:
        query = f"""
            MATCH (n:Entity) WHERE n.id IN $nodes
            OPTIONAL MATCH (n)-[rel]-(m:Entity) WHERE m.id IN $nodes
            WITH COLLECT(DISTINCT n) AS nodes, COLLECT(DISTINCT rel) AS rels
            RETURN [node in nodes | {NODE_PROJ}] AS nodes,
                   [rel in rels | {REL_PROJ}] AS relations
        """

        def _run():
            with self._read_session() as session:
                record = session.run(
                    self._timed_query(query), nodes=node_ids
                ).single()
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
            with self._read_session() as session:
                records = session.run(
                    self._timed_query(query),
                    frontier=frontier_ids,
                    excluded=excluded_ids,
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
        rel_type = re.sub(r"[^A-Z0-9_]", "", rel_type) or "RELATED_TO"
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
            with self._read_session() as session:
                records = session.run(self._timed_query(query), **params)
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
        triplets = normalize_triplets(triplets, max_items=100)

        from collections import defaultdict

        unique_labels = {}
        for t in triplets:
            sub = str(t.get('subject', '')).strip().lower()
            obj = str(t.get('object', '')).strip().lower()
            sub_clean = re.sub(r'[\W_]+', '', sub)
            obj_clean = re.sub(r'[\W_]+', '', obj)
            t['sub_id'] = hashlib.md5(sub_clean.encode('utf-8')).hexdigest()
            t['obj_id'] = hashlib.md5(obj_clean.encode('utf-8')).hexdigest()
            t["sub_normalized"] = sub
            t["obj_normalized"] = obj
            unique_labels.setdefault(sub, t["subject"])
            unique_labels.setdefault(obj, t["object"])

        label_keys = list(unique_labels)
        vectors = self.embedding_model.encode(
            [unique_labels[key] for key in label_keys],
            normalize_embeddings=True,
        )
        embedding_by_label = {
            key: vectors[index].tolist()
            for index, key in enumerate(label_keys)
        }
        for triplet in triplets:
            triplet["sub_embedding"] = embedding_by_label[
                triplet["sub_normalized"]
            ]
            triplet["obj_embedding"] = embedding_by_label[
                triplet["obj_normalized"]
            ]

        # Group by sanitized relationship type for native Cypher dynamic types
        by_type = defaultdict(list)
        for t in triplets:
            pred = str(t.get("predicate", "")).upper().replace(" ", "_")
            clean_type = re.sub(r"[^A-Z0-9_]", "", pred) or "RELATED_TO"
            by_type[clean_type].append(t)

        total = 0
        for rel_type, group in by_type.items():
            query = f"""
                UNWIND $triplets AS triplet
                MERGE (s:Entity {{id: triplet.sub_id}})
                ON CREATE SET s.label = triplet.subject,
                    s.description = 'Created by RAG agent',
                    s.source = 'extraction',
                    s.created_at = timestamp(),
                    s.normalized_label = triplet.sub_normalized,
                    s.embedding = triplet.sub_embedding
                MERGE (o:Entity {{id: triplet.obj_id}})
                ON CREATE SET o.label = triplet.object,
                    o.description = 'Created by RAG agent',
                    o.source = 'extraction',
                    o.created_at = timestamp(),
                    o.normalized_label = triplet.obj_normalized,
                    o.embedding = triplet.obj_embedding
                MERGE (s)-[rel:{rel_type}]->(o)
                ON CREATE SET rel.id = triplet.sub_id + '_' + triplet.obj_id,
                    rel.source = 'extraction',
                    rel.created_at = timestamp()
                RETURN count(rel) AS created_relationships
            """

            def _run(q=query, g=group):
                with self._write_session() as session:
                    result = session.run(self._timed_query(q), triplets=g)
                    rec = result.single()
                    if rec:
                        self.logger.info(
                            f"Added {rec['created_relationships']} '{rel_type}' relationships."
                        )

            self._execute_with_retry(_run)
            total += len(group)

        self.logger.info(f"Added {total} relationships total.")
