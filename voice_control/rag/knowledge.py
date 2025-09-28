from typing import Any, Dict, List, Tuple

from gqlalchemy import Memgraph

from ..common.config import config
from ..common.utils import get_logger


class KnowledgeGraph:
    def __init__(self, host: str, port: int, user: str, password: str):
        from sentence_transformers import SentenceTransformer

        self.logger = get_logger(__file__)
        self.db = Memgraph(host=host, port=port, username=user, password=password)
        embedding_model_name = config.get(
            "llm.models.embedding", "google/embeddinggemma-300m"
        )
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def close(self):
        pass

    def keyword_search(
        self, keywords: List[str], top_k: int = 5
    ) -> List[List[Dict]]:
        queries_data = [
            {"id": i, "keyword": keyword} for i, keyword in enumerate(keywords)
        ]
        query = """
            UNWIND $queries_data AS q_data
            CALL {
                WITH q_data
                CALL mg.fulltext_search("Entity", q_data.keyword, $top_k)
                YIELD node, score
                RETURN collect(
                    apoc.map.removeKey(properties(node), 'embedding')
                ) AS result_list
            }
            RETURN q_data.id AS query_id, result_list AS results
            ORDER BY query_id ASC
        """
        records = self.db.execute_and_fetch(
            query, parameters={"queries_data": queries_data, "top_k": top_k}
        )
        result = [record["results"] for record in records]
        return result

    def vector_search(
        self, embeddings: List[List[float]], top_k: int = 5
    ) -> List[List[Dict]]:
        queries_data = [
            {"id": i, "embedding": emb.tolist()}
            for i, emb in enumerate(embeddings)
        ]
        query = """
            UNWIND $queries_data AS q_data
            CALL {
                WITH q_data
                CALL vector_search.search('embedding', $top_k, q_data.embedding)
                YIELD node, score
                RETURN collect(
                    apoc.map.removeKey(properties(node), 'embedding')
                ) AS result_list
            }
            RETURN q_data.id AS query_id, result_list AS results
            ORDER BY query_id ASC
        """
        records = self.db.execute_and_fetch(
            query, parameters={"queries_data": queries_data, "top_k": top_k}
        )
        result = [record["results"] for record in records]
        return result

    def subgraph(self, node_ids: List[str]) -> Tuple[List[Dict], List[Dict]]:
        query = """
            MATCH (n:Entity) WHERE n.id IN $nodes
            OPTIONAL MATCH (n)-[r]-(m:Entity) WHERE m.id IN $nodes
            WITH COLLECT(DISTINCT n) AS nodes, COLLECT(DISTINCT r) AS rels
            RETURN [n in nodes | apoc.map.removeKey(properties(n), 'embedding')] AS nodes,
                   [r in rels | properties(r)] AS relations
        """
        record = next(self.db.execute_and_fetch(query, parameters={"nodes": node_ids}))
        result = record
        return result

    def expansion(
        self,
        frontier_ids: List[str],
        excluded_ids: List[str],
        n_hops: int = 1,
    ) -> List[Dict[str, Dict[str, Any]]]:
        """Performs a multi-hop expansion from a set of frontier nodes."""
        query = f"""
            UNWIND $frontier AS sourceId
            MATCH (n:Entity {{id: sourceId}})
            CALL {{
                WITH n
                MATCH (n)-[r*1..{n_hops}]->(m:Entity)
                WHERE NOT m.id IN $excluded
                RETURN m
            }}
            RETURN collect({{
                node: apoc.map.removeKey(properties(m), 'embedding')
            }}) AS results
        """
        records = self.db.execute_and_fetch(
            query, parameters={"frontier": frontier_ids, "excluded": excluded_ids}
        )
        result = [r["results"] for r in records]
        return result

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
        records = self.db.execute_and_fetch(query, **params)
        results = [
            record[key]
            for record in records
            for key in record.keys()
        ]
        return results

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
            CALL mg.create_relationship(s, upper(replace(triplet.predicate, ' ', '_')), {}, o)
            YIELD rel
            RETURN count(rel) AS created_relationships
        """
        result = self.db.execute_and_fetch(query, parameters={"triplets": triplets})
        self.logger.info(
            f"Added {next(result)['created_relationships']} new relationships to the graph."
        )