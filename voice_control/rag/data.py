from dataclasses import dataclass
import os
import re
import requests
import zipfile
import io
import pandas as pd
import json
import sys
from dotenv import dotenv_values
from typing import Dict, Optional, List

from sentence_transformers import SentenceTransformer

from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index

from ..common.utils import get_logger, setup_logging


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
    # --- CONFIGURATION ---
    FORCE_DOWNLOAD = False
    DATASET_SIZE = "s"
    TRIPLE_LIMIT = None
    # ---------------------

    setup_logging("DEBUG", stream=sys.stdout)
    logger = get_logger(__file__)

    env = dotenv_values(os.path.join(".env"))
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


if __name__ == "__main__":
    main()
