from typing import Any, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

from ..common.config import config
from ..common.utils import get_logger


class KnowledgeGraph:
    _rel_addendum: str = (
        "{head: startNode(r).id, tail: endNode(r).id, type: type(r)}"
    )

    def __init__(self, uri: str, user: str, password: str):
        self.logger = get_logger(__file__)
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        embedding_model_name = config.get(
            "llm.models.embedding", "avsolatorio/GIST-small-Embedding-v0"
        )
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def close(self):
        self._driver.close()

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
            records = session.run(
                query, queries_data=queries_data, top_k=top_k
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
            records = session.run(
                query, queries_data=queries_data, top_k=top_k
            )
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
        self, frontier_ids: List[str], excluded_ids: List[str], max_hops: int = 1
    ) -> List[Dict[str, Dict[str, Any]]]:
        """
        Performs a multi-hop expansion from a set of frontier nodes.

        Uses the APOC path expander to efficiently find all unique nodes up to
        `max_hops` away from the frontier, excluding any nodes in the `excluded_ids` list.

        Args:
            frontier_ids (List[str]): The IDs of nodes to expand from.
            excluded_ids (List[str]): A list of node IDs to exclude from the expansion results.
            max_hops (int, optional): The maximum number of hops to traverse. Defaults to 1.

        Returns:
            List[Dict[str, Dict[str, Any]]]: A list containing the collected neighbor nodes.
        """
        # Ensure max_hops is within a reasonable range to prevent performance issues
        max_hops = max(1, min(max_hops, 3))

        query = f"""
            UNWIND $frontier AS sourceId
            MATCH (n:Entity {{id: sourceId}})
            CALL apoc.path.subgraphNodes(n, {{
                maxLevel: {max_hops},
                relationshipFilter: ">",
                labelFilter: "+Entity"
            }})
            YIELD node
            WHERE NOT node.id IN $excluded
            RETURN collect({{
                node: apoc.map.removeKey(properties(node), 'embedding')
            }}) AS results
        """
        with self._driver.session() as session:
            records = session.run(
                query, frontier=frontier_ids, excluded=excluded_ids
            )
            # The query returns a single list of all nodes found across all paths
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
        with self._driver.session() as session:
            records = session.run(query, **params)
            results = [
                record.data()[key]
                for record in records
                for key in record.keys()
            ]
        return results

    def add_triplets(self, triplets: List[Dict[str, str]]):
        """
        Adds new triplets to the knowledge graph, creating nodes and relationships as needed.

        This method uses `MERGE` to avoid creating duplicate nodes for the same
        entity label and `apoc.merge.relationship` to avoid creating duplicate
        relationships. New nodes are assigned a UUID and a default description.

        Args:
            triplets (List[Dict[str, str]]): A list of triplet dictionaries, where
                each dictionary must have 'subject', 'predicate', and 'object' keys.
        """
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
        with self._driver.session() as session:
            result = session.run(query, triplets=triplets)
            self.logger.info(f"Added {result.single()['created_relationships']} new relationships to the graph.")
