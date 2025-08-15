from typing import Any, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase


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
