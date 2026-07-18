from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import zipfile

import pandas as pd
import sys
from typing import Dict, Optional, List
from dotenv import load_dotenv


from sentence_transformers import SentenceTransformer

from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index

from ..common.utils import download_file, get_logger, setup_logging
from ..common.config import config


class DataLoader:

    CODEX_REVISION = "3132e426c2a6b643b70bad679905a3a6270be440"
    CODEX_ARCHIVE_SHA256 = (
        "18a0721753e1e137a0d3ab1c79dde6964d628fdf8eae38dc5178c726ba469b02"
    )
    CODEX_ARCHIVE_MAX_BYTES = 200_000_000
    CODEX_EXTRACTED_MAX_BYTES = 1_000_000_000

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
        return self._load_filtered(
            triples_path, entities_path, relations_path, limit
        )

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
        relations = {rid: {"id": rid, **relations[rid]} for rid in relations}

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
            eid: entities[eid]
            for eid in required_entity_ids
            if eid in entities
        }
        filtered_relations = {
            rid: relations[rid]
            for rid in required_relation_ids
            if rid in relations
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
        limit: Optional[int] = None,
        *,
        size: str = "s",
        lang: str = "en",
    ) -> DataLoader.KnowledgeData:
        """
        Loads a filtered dataset from the CoDEx repository.
        """
        repo_path = os.path.join(path, "codex-master")
        data_path = os.path.join(repo_path, "data")
        triples_path = os.path.join(
            data_path, "triples", f"codex-{size}", "train.txt"
        )
        entities_path = os.path.join(
            data_path, "entities", lang, "entities.json"
        )
        relations_path = os.path.join(
            data_path, "relations", lang, "relations.json"
        )

        print(f"\n--- Loading CoDEx-{size.upper()} Dataset ---")
        self._download_and_unzip(repo_path)
        data = self._load_filtered(
            triples_path, entities_path, relations_path, limit
        )

        return data

    def _download_and_unzip(self, repo_path):
        """
        Downloads and unzips the CoDEx repository from GitHub if it's not already present.
        This is idempotent and will skip downloading if the directory exists.
        """
        if os.path.exists(repo_path):
            print(f"'{repo_path}' already exists. Skipping download.")
            return

        print("Downloading pinned CoDEx dataset archive...")
        repo_url = (
            "https://github.com/tsafavi/codex/archive/"
            f"{self.CODEX_REVISION}.zip"
        )
        parent = os.path.abspath(os.path.dirname(repo_path))
        os.makedirs(parent, exist_ok=True)
        archive_path = os.path.join(parent, f".codex-{self.CODEX_REVISION}.zip")
        try:
            download_file(
                repo_url,
                archive_path,
                expected_sha256=self.CODEX_ARCHIVE_SHA256,
                allowed_hosts={"github.com", "codeload.github.com"},
                max_bytes=self.CODEX_ARCHIVE_MAX_BYTES,
            )
            with tempfile.TemporaryDirectory(
                dir=parent, prefix=".codex-extract-"
            ) as extraction_dir:
                self._extract_archive_safely(archive_path, extraction_dir)
                extracted_root = os.path.join(
                    extraction_dir, f"codex-{self.CODEX_REVISION}"
                )
                if not os.path.isdir(extracted_root):
                    raise ValueError("Dataset archive has an unexpected layout.")
                os.replace(extracted_root, repo_path)
            print("Download and extraction complete.")
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            raise RuntimeError(f"Failed during download or extraction: {e}")
        finally:
            if os.path.exists(archive_path):
                os.unlink(archive_path)

    def _extract_archive_safely(self, archive_path: str, destination: str):
        """Extract a bounded ZIP without path traversal or symbolic links."""
        destination_root = Path(destination).resolve()
        extracted_bytes = 0
        with zipfile.ZipFile(archive_path) as archive:
            if len(archive.infolist()) > 20_000:
                raise ValueError("Dataset archive contains too many entries.")
            for member in archive.infolist():
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("Dataset archive contains a symbolic link.")
                extracted_bytes += member.file_size
                if extracted_bytes > self.CODEX_EXTRACTED_MAX_BYTES:
                    raise ValueError("Dataset archive exceeds the extraction limit.")

                target = (destination_root / member.filename).resolve()
                if os.path.commonpath((destination_root, target)) != str(
                    destination_root
                ):
                    raise ValueError("Dataset archive contains an unsafe path.")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                # ASVS 5.3.2 / 15.4.2: copy only validated regular-file paths.
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)


class Neo4jImporter:
    """
    Manages the connection, embedding generation, and data import into Neo4j.
    """

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Initializing embedding model...")
        embedding_model_name = config.get(
            "llm.models.embedding", "google/embeddinggemma-300m"
        )
        self._embedding_model = SentenceTransformer(embedding_model_name)
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

        query = "CREATE INDEX entity_normalized_label_index IF NOT EXISTS FOR (n:Entity) ON (n.normalized_label)"
        self._run_query(query)

        # Backfill graphs imported before normalized labels were introduced.
        self._run_query(
            "MATCH (n:Entity) WHERE n.normalized_label IS NULL "
            "SET n.normalized_label = toLower(trim(n.label))"
        )

        query = "CREATE FULLTEXT INDEX nodes_label_description_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.label, n.description]"
        self._run_query(query)

        dimensions = self._embedding_model.get_sentence_embedding_dimension()
        create_vector_index(
            self._driver,
            name="embedding",
            label="Entity",
            embedding_property="embedding",
            dimensions=dimensions,
            similarity_fn="cosine",
            fail_if_exists=False,
        )
        print("Indexes are ready.")

    def clear_database(self):
        print("Clearing database...")
        for index in [
            "entity_id_index",
            "entity_normalized_label_index",
            "nodes_label_description_fulltext",
            "embedding",
        ]:
            self._run_query(f"DROP INDEX {index} IF EXISTS")
        self._run_query("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

    def _generate_entity_embeddings(self, entities_meta: Dict) -> List[Dict]:
        """
        Generates embeddings for entity labels and returns a list of dictionaries
        ready for import.
        """
        print(f"Generating embeddings for {len(entities_meta)} entities...")

        entity_ids = list(entities_meta.keys())
        documents_to_embed = []
        for entity_id in entity_ids:
            entity = entities_meta[entity_id]
            label = str(entity.get("label", "")).strip()
            description = str(entity.get("description", "")).strip()
            documents_to_embed.append(
                f"{label}. {description}" if description else label
            )

        embeddings = self._embedding_model.encode(
            documents_to_embed,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        entities_with_embeddings = []
        for i, eid in enumerate(entity_ids):
            entity_data = entities_meta[eid]
            entity_data["id"] = eid
            entity_data["normalized_label"] = str(
                entity_data.get("label", "")
            ).strip().lower()
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
        entities_with_embeddings = self._generate_entity_embeddings(
            entities_meta
        )

        self._import_entities(entities_with_embeddings)
        self._import_relationships(triples_df, relations_meta)

    def _import_entities(self, entities_with_embeddings: List[Dict]):
        """Imports entity nodes with their properties and a vector embedding."""
        print(
            f"Importing {len(entities_with_embeddings)} entities with embeddings..."
        )
        query = """
            UNWIND $entities AS entity
            MERGE (e:Entity {id: entity.id})
            SET e.label = entity.label, 
                e.normalized_label = entity.normalized_label,
                e.description = entity.description,
                e.embedding = entity.embedding,
                e.source = 'import',
                e.created_at = timestamp()
        """
        self._run_query(query, {"entities": entities_with_embeddings})
        print("Entity import complete.")

    def _import_relationships(
        self, triples_df: pd.DataFrame, relations_meta: Dict
    ):
        """Imports relationships using native Cypher dynamic types."""

        from collections import defaultdict

        def sanitize_label(label: str) -> str:
            label = label.upper().replace(" ", "_").replace("-", "_")
            cleaned = re.sub(r"[^A-Z0-9_]", "", label)
            if not cleaned:
                return "RELATED_TO"
            return cleaned

        # Group by sanitized relationship type
        by_type = defaultdict(list)
        for row in triples_df.itertuples(index=False):
            rel_meta = relations_meta.get(row.relation_id, {})
            rel_type = sanitize_label(rel_meta.get("label", "RELATED_TO"))
            by_type[rel_type].append(
                {
                    "head": row.head_id,
                    "tail": row.tail_id,
                    "id": row.relation_id,
                }
            )

        print("Importing relationships with native dynamic types...")
        total = 0
        for rel_type, group in by_type.items():
            query = f"""
                UNWIND $triples AS t
                MATCH (h:Entity {{id: t.head}})
                MATCH (tl:Entity {{id: t.tail}})
                CREATE (h)-[r:{rel_type}]->(tl)
                SET r.id = t.id,
                    r.source = 'import',
                    r.created_at = timestamp()
                RETURN count(r)
            """
            self._run_query(query, {"triples": group})
            total += len(group)
            print(f"  Imported {len(group)} '{rel_type}' relationships.")

        print(f"Imported {total} relationships total.")

    def get_node_count(self) -> int:
        with self._driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            record = result.single()
            return record["count"] if record else 0


def main():
    # --- CONFIGURATION ---
    FORCE_DOWNLOAD = True
    DATASET_SIZE = "s"
    TRIPLE_LIMIT = None
    # ---------------------

    setup_logging("DEBUG", stream=sys.stdout)
    logger = get_logger(__file__)

    load_dotenv()

    # Load Neo4j credentials from the central config
    config_id = "database.neo4j"
    if not config.get(config_id):
        raise ValueError("Neo4j configuration not found in config file.")

    uri = config.get(f"{config_id}.uri")
    user = config.get(f"{config_id}.user")
    password = config.get(f"{config_id}.password")

    if not all([uri, user, password]):
        raise ValueError(
            "Neo4j credentials not fully configured. Check your config file."
        )

    with Neo4jImporter(uri, user, password) as importer:
        if not FORCE_DOWNLOAD and importer.get_node_count():
            logger.info("Database already contains data. Skipping import.")
        else:
            data_loader = CodexDataLoader()
            knowledge_data = data_loader.load(
                size=DATASET_SIZE, limit=TRIPLE_LIMIT
            )

            importer.clear_database()
            importer.import_graph_data(knowledge_data)

            node_count = importer.get_node_count()
            logger.info(
                f"\n✅ Import complete. Total nodes in database: {node_count}"
            )


if __name__ == "__main__":
    main()
