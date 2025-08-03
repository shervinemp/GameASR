from collections import deque
from dataclasses import dataclass
import os
import random
import re
import sys
import requests
import zipfile
import io
import pandas as pd
import json
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
from typing import Any, Deque, Dict, Tuple, Optional, List

from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index

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

        self.report = {"state": "Starting search with initial nodes..."}

    def __call__(self, query: str) -> str:
        return self._execute_query(query)

    def _execute_query(self, query: str) -> str:

        keywords = self._extract_keywords(query)
        embeddings = self.graph.embedding_model.encode(keywords)

        initial_results = [
            a + b
            for a, b in zip(
                self.graph.vector_search(embeddings, top_k=3),
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

            self.report = response.get("report", "")
            final_answer = response.get("answer", None)
            is_verified = response.get("is_verified", False)

            if is_verified:
                self.logger.info("Final answer found.")
                break

            if not nodes_to_expand:
                self.logger.info("No nodes to expand. Halting exploration.")
                break

            state.expand(nodes_to_expand)
            i += 1

        self.logger.info("\n--- Finalizing Answer ---")
        final_subgraph = self.graph.subgraph([n["id"] for n in state.frontier])

        return {
            "answer": final_answer,
            "report": self.report,
            "subgraph": final_subgraph,
            "verified": is_verified,
        }

    def _build_expansion_prompt(
        self, query: str, state: Exploration, max_neighbors: int = 12
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

            head = relation["head"]
            tail = relation["tail"]

            rel_type = relation["type"]

            ltr = node["id"] == tail
            if not ltr:
                head, tail = tail, head

            head_label = id_to_node[head]["label"]
            tail_label = id_to_node[tail]["label"]

            descs_string = f"({node['label']}::{node['id']}): {node['description']}"
            descs.append(descs_string)

            triple_string = f"({head_label}::{head}) {'' if ltr else '<'}- [{rel_type}] -{'>' if ltr else ''} ({tail_label}::{tail})"
            triples.append(triple_string)

        nodes_str = "\n".join(descs)
        candidates_str = "\n".join(triples)

        return (
            f"Query: '{query}'\n"
            "Task: Analyze our potential new candidate nodes and their relations with regard to the query. "
            "Consider descriptions, and their connection to our investigation and query. "
            "None of the provided candidates are guaranteed to be relevant, "
            "so rely on your judgment, the query and any past report."
            "Return a JSON object with four keys:\n1. 'new_frontier': a list containing only the IDs "
            "(right side of '::', with 'Q' prefix) of the most promising new candidate nodes to add to our frontier.\n"
            "2. 'report': a minimal dictionary compiling any verifiable and relevant information gathered so far.\n"
            "3. 'answer': the best-guess answer thus far.\n4. 'is_verified': a boolean indicator, "
            "strictly true only when the objective is met and the answer to the query is directly "
            "and completely verified and cross-referenced with the provided context."
            f" * Query: '{query}'\n"
            f" * Report: {self.report}\n"
            f" * Current nodes:\n{nodes_str}\n"
            f" * Candidates:\n{candidates_str}"
        )

    def _extract_keywords(self, query: str) -> List[str]:
        prompt = (
            "Extract, from the following query, proper entities and keywords that will be used to investigate a potential answer."
            "Return a JSON array of strings including the extracted entities and keywords."
            f"query:\n{query}"
        )

        response = "".join(self.session(prompt))
        return json.loads(response)


class DataLoader:

    @dataclass
    class KnowledgeData:
        triples: pd.DataFrame
        entities: Dict
        relations: Dict

    def load(
        self,
        path: str = ".",
        limit: Optional[int] = None,
    ) -> "KnowledgeData":
        triples_path = os.path.join(path, "triples.txt")
        entities_path = os.path.join(path, "entities.json")
        relations_path = os.path.join(path, "relations.json")
        return self._load_filtered(triples_path, entities_path, relations_path, limit)

    def _load_filtered(
        self,
        triples_path: str,
        entities_path: str,
        relations_path: str,
        limit: Optional[int] = None,
    ) -> "KnowledgeData":
        triples_df = pd.read_csv(
            triples_path,
            sep="\t",
            header=None,
            names=["head_id", "relation_id", "tail_id"],
        )
        entities = self._load_json(entities_path)
        relations = self._load_json(relations_path)

        if limit and limit < len(triples_df):
            triples_subset_df = triples_df.head(limit).copy()
        else:
            triples_subset_df = triples_df

        required_entity_ids = set(
            pd.concat(
                [triples_subset_df["head_id"], triples_subset_df["tail_id"]]
            ).unique()
        )
        required_relation_ids = set(triples_subset_df["relation_id"].unique())

        filtered_entities = {
            eid: entities[eid] for eid in required_entity_ids if eid in entities
        }
        filtered_relations = {
            rid: relations[rid] for rid in required_relation_ids if rid in relations
        }

        return self.KnowledgeData(
            triples_subset_df,
            filtered_entities,
            filtered_relations,
        )

    def _load_json(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class CodexDataLoader(DataLoader):
    def load(
        self,
        path: str = ".",
        size: str = "s",
        lang: str = "en",
        limit: Optional[int] = None,
    ) -> DataLoader.KnowledgeData:
        """
        Loads a filtered dataset from the CoDEx repository.
        """
        repo_path = os.path.join(path, "codex-master")
        data_path = os.path.join(repo_path, "data")
        triples_path = os.path.join(data_path, "triples", f"codex-{size}", "train.txt")
        entities_path = os.path.join(data_path, "entities", lang, "entities.json")
        relations_path = os.path.join(data_path, "relations", lang, "relations.json")

        print(f"\n--- Loading CoDEx-{size.upper()} Dataset ---")
        self._download_and_unzip(repo_path)
        data = self._load_filtered(triples_path, entities_path, relations_path, limit)

        return data

    def _download_and_unzip(self, repo_path):
        """
        Downloads and unzips the CoDEx repository from GitHub if it's not already present.
        This is idempotent and will skip downloading if the directory exists.
        """
        if os.path.exists(repo_path):
            print(f"'{repo_path}' already exists. Skipping download.")
            return

        print("Downloading CoDEx repository...")
        repo_url = "https://github.com/tsafavi/codex/archive/refs/heads/master.zip"
        try:
            response = requests.get(repo_url, stream=True)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(self.extract_to)
            print("Download and extraction complete.")
        except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:
            raise RuntimeError(f"Failed during download or extraction: {e}")


class Neo4jImporter:
    """
    Manages the connection, embedding generation, and data import into Neo4j.
    """

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Initializing embedding model...")
        self._embedding_model = SentenceTransformer(
            "avsolatorio/GIST-small-Embedding-v0"
        )
        print("Model ready.")

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _run_query(self, query: str, parameters: Optional[Dict] = None):
        with self._driver.session() as session:
            session.run(query, parameters or {})

    def _ensure_indexes(self):
        print("Ensuring indexes are in place...")
        query = "CREATE INDEX entity_id_index IF NOT EXISTS FOR (n:Entity) ON (n.id)"
        self._run_query(query)

        query = "CREATE FULLTEXT INDEX nodes_label_description_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.label, n.description]"
        self._run_query(query)

        create_vector_index(
            self._driver,
            name="embedding",
            label="Entity",
            embedding_property="embedding",
            dimensions=384,
            similarity_fn="cosine",
        )
        print("Indexes are ready.")

    def clear_database(self):
        print("Clearing database...")
        self._run_query("DROP INDEX entity_id_index IF EXISTS")
        self._run_query("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

    def _generate_entity_embeddings(self, entities_meta: Dict) -> List[Dict]:
        """
        Generates embeddings for entity labels and returns a list of dictionaries
        ready for import.
        """
        print(f"Generating embeddings for {len(entities_meta)} entities...")

        entity_ids = list(entities_meta.keys())
        labels_to_embed = [entities_meta[eid].get("label", "") for eid in entity_ids]

        embeddings = self._embedding_model.encode(
            labels_to_embed, show_progress_bar=True
        )

        entities_with_embeddings = []
        for i, eid in enumerate(entity_ids):
            entity_data = entities_meta[eid]
            entity_data["id"] = eid
            entity_data["embedding"] = embeddings[i].tolist()
            entities_with_embeddings.append(entity_data)

        return entities_with_embeddings

    def import_graph_data(self, knowledge_data: DataLoader.KnowledgeData):
        """Orchestrates the entire import process including embedding generation."""
        triples_df = knowledge_data.triples
        entities_meta = knowledge_data.entities
        relations_meta = knowledge_data.relations

        self._ensure_indexes()

        print("Starting data preparation for Neo4j...")
        entities_with_embeddings = self._generate_entity_embeddings(entities_meta)

        self._import_entities(entities_with_embeddings)
        self._import_relationships(triples_df, relations_meta)

    def _import_entities(self, entities_with_embeddings: List[Dict]):
        """Imports entity nodes with their properties and a vector embedding."""
        print(f"Importing {len(entities_with_embeddings)} entities with embeddings...")
        query = """
            UNWIND $entities AS entity
            MERGE (e:Entity {id: entity.id})
            SET e.label = entity.label, 
                e.description = entity.description,
                e.embedding = entity.embedding
        """
        self._run_query(query, {"entities": entities_with_embeddings})
        print("Entity import complete.")

    def _import_relationships(self, triples_df: pd.DataFrame, relations_meta: Dict):
        """Imports relationships using dynamic types via APOC."""

        def sanitize_label(label: str) -> str:
            label = label.upper().replace(" ", "_").replace("-", "_")
            return re.sub(r"[^A-Z0-9_]", "", label)

        triples_list = []
        for _, row in triples_df.iterrows():
            rel_id = row["relation_id"]
            rel_meta = relations_meta.get(rel_id, {})
            rel_type = sanitize_label(rel_meta.get("label", "RELATED_TO"))
            triples_list.append(
                {
                    "head": row["head_id"],
                    "tail": row["tail_id"],
                    "type": rel_type,
                    **rel_meta,
                }
            )

        query = """
            UNWIND $triples AS triple
            MATCH (head:Entity {id: triple.head})
            MATCH (tail:Entity {id: triple.tail})
            CALL apoc.create.relationship(head, triple.type, {id: triple.id}, tail)
            YIELD rel
            RETURN count(rel)
        """
        print("Importing relationships with dynamic types (using APOC)...")
        self._run_query(query, {"triples": triples_list})
        print(f"Imported {len(triples_list)} relationships.")

    def get_node_count(self) -> int:
        with self._driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            return result.single()["count"]


def main():
    """Main execution function to run the entire ETL process."""
    # --- CONFIGURATION ---
    FORCE_DOWNLOAD = False
    DATASET_SIZE = "s"
    TRIPLE_LIMIT = None
    # ---------------------

    setup_logging("DEBUG", stream=sys.stdout)
    logger = get_logger(__file__)

    env = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))
    NEO4J_URI = env.get("NEO4J_URI")
    NEO4J_USER = env.get("NEO4J_USER")
    NEO4J_PASSWORD = env.get("NEO4J_PASSWORD")

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise ValueError("Neo4j credentials not found in .env file.")

    with Neo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as importer:
        if not FORCE_DOWNLOAD and importer.get_node_count():
            logger.info("Database already contains data. Skipping import.")
        else:
            data_loader = CodexDataLoader()
            knowledge_data = data_loader.load(size=DATASET_SIZE, limit=TRIPLE_LIMIT)

            importer.clear_database()
            importer.import_graph_data(knowledge_data)

            node_count = importer.get_node_count()
            logger.info(f"\n✅ Import complete. Total nodes in database: {node_count}")

    # user_query = "Which American presidents had a background in law before taking office, like Obama?"
    # user_query = "Who are the members of the band Coldplay?"
    user_query = "Give me all the information you have on Justin Bieber."

    graph = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    orchestrator = Orchestrator(graph)

    try:
        final_answer = orchestrator(user_query)
        print("\n--- Final Answer ---")
        print(final_answer)
    finally:
        graph.close()


if __name__ == "__main__":
    main()
