from abc import ABC, abstractmethod
import os
import re
import requests
import zipfile
import io
import pandas as pd
import json
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple, Optional, List, Set

from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index

from ..llm.session import Session


class KnowledgeGraph(ABC):
    @abstractmethod
    def vector_search(self, query_text: str, top_k: int = 5) -> List[Dict]: ...

    @abstractmethod
    def vector_search_multi(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[Dict]]: ...


class KnowledgeGraphManager:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def close(self):
        self._driver.close()

    def search(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Performs a vector similarity search for a list of query texts efficiently.

        Args:
            queries: A list of strings to search for.
            top_k: The number of top similar results to return for each query.

        Returns:
            A list of lists, where each inner list contains the top-k search
            results (as dictionaries) for the corresponding query text.
        """
        embeddings = self.embedding_model.encode(queries)
        queries_data = [
            {"id": i, "embedding": emb.tolist()} for i, emb in enumerate(embeddings)
        ]

        query = """
            UNWIND $queries_data AS q_data
            CALL (q_data) {
                CALL db.index.vector.queryNodes('embedding', $top_k, q_data.embedding)
                YIELD node, score
                RETURN collect(
                    apoc.map.merge(
                        apoc.map.removeKey(properties(node), 'embedding'),
                        {match_score: score}
                    )
                ) AS result_list
            }
            RETURN q_data.id AS query_id, result_list AS results
            ORDER BY query_id ASC
        """

        with self._driver.session() as session:
            records = session.run(query, queries_data=queries_data, top_k=top_k)
            result = [record["results"] for record in records]

        return result

    def subgraph(self, node_ids: List[str]) -> Dict[str, List[Dict]]:
        """
        Retrieves a subgraph of the graph based on a list of node IDs.


        Args:
            node_ids: A list of node IDs to retrieve the subgraph for.

        Returns:
            A dictionary containing nodes and relations of the subgraph.
        """
        if not node_ids:
            return {}
        query = """
            MATCH (n:Entity) WHERE n.id IN $nodes
            OPTIONAL MATCH (n)-[r]-(m:Entity) WHERE m.id IN $nodes
            WITH COLLECT(DISTINCT n) AS nodes, COLLECT(DISTINCT r) AS rels
            RETURN [n in nodes | apoc.map.removeKey(properties(n), 'embedding')] AS nodes,
                   [r in rels | apoc.map.merge(properties(r), {head: startNode(r).id, tail: endNode(r).id})] AS relations
        """
        with self._driver.session() as session:
            record = session.run(query, nodes=node_ids).single()
            result = record.data()

        return result

    def expand(self, frontier_ids: List[str], excluded_ids: List[str]) -> List[Dict]:
        if not frontier_ids:
            return []
        query = """
            UNWIND $frontier AS sourceId
            MATCH (n:Entity {id: sourceId})
            MATCH (n)-[r]-(m:Entity)
            WHERE NOT m.id IN $excluded

            WITH m, head(collect({
                relation: apoc.map.merge(properties(r), {head: startNode(r).id, tail: endNode(r).id}),
                node: apoc.map.removeKey(properties(m), 'embedding')
            })) AS result_map

            RETURN result_map AS results
        """
        with self._driver.session() as session:
            records = session.run(query, frontier=frontier_ids, excluded=excluded_ids)
            result = [r["results"] for r in records]

        return result


class ExplorationState:
    def __init__(self, initial_nodes: List[str]):
        self.frontier: Set[str] = set(initial_nodes)
        self.ancestry: Dict[str, str] = {node_id: node_id for node_id in initial_nodes}
        self.summary: str = "Starting search with initial nodes..."
        self.step: int = 0

    def apply_expansion(
        self,
        expansion: List[Dict],
    ):
        self.frontier = set()
        for r, n in zip(expansion["relation"], expansion["node"]):
            tail = n["id"]
            self.frontier.add(tail)
            head = r["head"] if r["head"] != tail else r["tail"]
            self.ancestry[tail] = self.ancestry[head]

        self.step += 1


class RAGOrchestrator:
    def __init__(
        self,
        graph_manager: KnowledgeGraphManager,
        session: Session,
        max_iterations: int = 5,
    ):
        self.graph = graph_manager
        self.session = session
        self.max_iterations = max_iterations

    def _build_expansion_prompt(
        self, query: str, summary: str, frontier_data: List[Dict]
    ) -> str:
        candidates_str = "\n\n".join(
            [
                f"{i+1}. Node: {{id: '{c['node']['id']}', label: '{c['node']['label']}'}}\n"
                f"   Connection: {c['connection_path']}\n"
                f"   Anchor Distances to Active Seeds: {c['distances']}"
                for i, c in enumerate(frontier_data)
            ]
        )

        return (
            f"Investigation Summary: {summary}\n\n"
            f"Original Query: '{query}'\n\n"
            "Task: Analyze the following frontier of candidate nodes. Consider their description, "
            "their connection to our investigation, and their community-level distances to our active concepts. "
            "Return a JSON object with two keys: 'nodes_to_expand', a list containing only the IDs of the most "
            "promising candidates to add to our evidence board, and 'investigation_summary', a new one-sentence summary of what "
            "was learned from the candidates you selected.\n\n"
            f"Frontier Candidates:\n{candidates_str}"
        )

    def execute_query(self, query: str) -> str:
        print("\nPerforming initial vector search using Neo4j...")
        initial_search_results = self.graph.vector_search(query, top_k=5)

        print("Found initial candidates:")
        for result in initial_search_results:
            print(f"  - ID: {result['id']}, Score: {result['score']:.4f}")

        initial_nodes = [result["id"] for result in initial_search_results]
        if not initial_nodes:
            return "Could not find any relevant starting points in the knowledge graph for your query."

        state = ExplorationState(initial_nodes)
        seed_contexts = self.graph.get_context(list(state.active_seed_nodes))
        seed_communities = {
            sid: ctx["communityId"]
            for sid, ctx in seed_contexts.items()
            if "communityId" in ctx
        }

        while state.iteration < self.max_iterations:
            print(f"\n--- Iteration {state.iteration + 1} ---")
            if not state.frontier:
                print("Frontier is empty. Halting exploration.")
                break

            frontier_contexts = self.graph.get_context(list(state.frontier))
            enriched_frontier = []

            all_frontier_connections = (
                self.graph.get_frontier(state.visited) if state.iteration > 0 else []
            )

            for node_id in state.frontier:
                if node_id not in frontier_contexts:
                    continue

                distances = self.graph.get_community_distances(
                    frontier_contexts[node_id]["communityId"], seed_communities
                )

                if state.iteration == 0:
                    connection_path = "Original seed node"
                else:
                    connection = next(
                        (
                            c
                            for c in all_frontier_connections
                            if c["target_id"] == node_id
                        ),
                        None,
                    )
                    if not connection:
                        continue
                    source_context = self.graph.get_context([connection["source_id"]])
                    source_label = source_context.get(connection["source_id"], {}).get(
                        "label", "Unknown"
                    )
                    connection_path = (
                        f"via ({source_label}) -[:{connection['rel_label']}]->"
                    )

                enriched_frontier.append(
                    {
                        "node": frontier_contexts[node_id],
                        "connection_path": connection_path,
                        "distances": distances,
                    }
                )

            if not enriched_frontier:
                print("Could not enrich any frontier nodes. Halting exploration.")
                break

            prompt = self._build_expansion_prompt(
                query, state.investigation_summary, enriched_frontier
            )
            response = self.session.get_expansion_decision(prompt)

            nodes_to_expand = response.get("nodes_to_expand", [])
            new_summary = response.get("investigation_summary", "No summary provided.")

            print(f"LLM Summary: {new_summary}")
            print(f"LLM decided to expand: {nodes_to_expand}")

            if not nodes_to_expand:
                print("LLM returned no nodes to expand. Halting exploration.")
                break

            if state.iteration == 0:
                state.visited = set(nodes_to_expand)
                state.node_ancestry = {nid: nid for nid in nodes_to_expand}

            state.update_after_expansion(
                nodes_to_expand, new_summary, all_frontier_connections
            )

        print("\n--- Finalizing Answer ---")
        final_subgraph_context = self.graph.get_subgraph(state.visited)

        final_prompt = (
            f"Answer the following user query based ONLY on the provided knowledge graph context.\n\n"
            f"User Query: '{query}'\n\n"
            f"Knowledge Graph Context:\n{final_subgraph_context}\n\n"
            "Answer:"
        )

        return self.session.get_final_answer(final_prompt)


class CodexDataLoader:
    """
    Handles downloading the CoDEx repository and loading a filtered dataset into memory.
    This class is responsible for all file-based data retrieval and preparation.
    """

    def __init__(self, extract_to: str = "."):
        """
        Initializes the data loader.

        Args:
            extract_to (str): The directory where the repository will be extracted.
        """
        self.extract_to = extract_to
        self.repo_path = os.path.join(self.extract_to, "codex-master")
        self.data_path = os.path.join(self.repo_path, "data")

    def _download_and_unzip(self):
        """
        Downloads and unzips the CoDEx repository from GitHub if it's not already present.
        This is idempotent and will skip downloading if the directory exists.
        """
        if os.path.exists(self.repo_path):
            print(f"'{self.repo_path}' already exists. Skipping download.")
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

    def load_filtered_data(
        self, size: str = "s", lang: str = "en", limit: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Loads a specified (and optionally limited) set of triples and intelligently
        filters the full metadata files to only include entities and relations
        present in that selected subset.
        """
        self._download_and_unzip()
        print(f"\n--- Loading CoDEx-{size.upper()} Dataset ---")

        triples_file = os.path.join(
            self.data_path, "triples", f"codex-{size}", "train.txt"
        )
        full_triples_df = pd.read_csv(
            triples_file,
            sep="\t",
            header=None,
            names=["head_id", "relation_id", "tail_id"],
        )
        with open(
            os.path.join(self.data_path, "entities", lang, "entities.json"),
            "r",
            encoding="utf-8",
        ) as f:
            full_entities_meta = json.load(f)
        with open(
            os.path.join(self.data_path, "relations", lang, "relations.json"),
            "r",
            encoding="utf-8",
        ) as f:
            full_relations_meta = json.load(f)

        if limit and limit < len(full_triples_df):
            print(f"Applying a limit of {limit} triples.")
            triples_subset_df = full_triples_df.head(limit).copy()
        else:
            triples_subset_df = full_triples_df

        required_entity_ids = set(
            pd.concat(
                [triples_subset_df["head_id"], triples_subset_df["tail_id"]]
            ).unique()
        )
        required_relation_ids = set(triples_subset_df["relation_id"].unique())

        filtered_entities = {
            eid: full_entities_meta[eid]
            for eid in required_entity_ids
            if eid in full_entities_meta
        }
        filtered_relations = {
            rid: full_relations_meta[rid]
            for rid in required_relation_ids
            if rid in full_relations_meta
        }

        print(
            f"Filtered to {len(filtered_entities)} entities and {len(filtered_relations)} relations based on the triple selection."
        )

        return triples_subset_df, filtered_entities, filtered_relations


class Neo4jImporter:
    """
    Manages the connection, embedding generation, and data import into Neo4j.
    """

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Initializing embedding model (all-MiniLM-L6-v2)...")
        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
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
        query = "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)"
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

    def import_graph_data(
        self, triples_df: pd.DataFrame, entities_meta: Dict, relations_meta: Dict
    ):
        """Orchestrates the entire import process including embedding generation."""
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
        MATCH (head:Entity {id: triple.head_id})
        MATCH (tail:Entity {id: triple.tail_id})
        CALL apoc.create.relationship(head, triple.rel_type, {id: triple.rel_id}, tail)
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
    DATASET_SIZE = "s"
    TRIPLE_LIMIT = None
    # ---------------------

    try:
        env = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))
        NEO4J_URI = env.get("NEO4J_URI")
        NEO4J_USER = env.get("NEO4J_USER")
        NEO4J_PASSWORD = env.get("NEO4J_PASSWORD")

        if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
            raise ValueError("Neo4j credentials not found in .env file.")

        data_loader = CodexDataLoader()
        triples, entities, relations = data_loader.load_filtered_data(
            size=DATASET_SIZE, limit=TRIPLE_LIMIT
        )

        with Neo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as importer:
            importer.clear_database()
            importer.import_graph_data(triples, entities, relations)

            node_count = importer.get_node_count()
            print(f"\n✅ Import complete. Total nodes in database: {node_count}")

        user_query = "Which American presidents had a background in law before taking office, like Obama?"

        print("\nPerforming initial vector search...")
        initial_search_results = vector_db.search(user_query, top_k=5)
        print(f"Found initial candidates: {initial_search_results}")

        graph_manager = KnowledgeGraphManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        llm_interface = LLMInterface(OPENAI_API_KEY)
        orchestrator = RAGOrchestrator(graph_manager, llm_interface)

        try:
            final_answer = orchestrator.execute_query(
                user_query, initial_search_results
            )
            print("\n--- Final Answer ---")
            print(final_answer)
        finally:
            graph_manager.close()

    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
