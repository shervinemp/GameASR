import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ...common.utils import get_logger
from ...exceptions import StorageError
from .base import StorageBackend


class SQLiteBackend(StorageBackend):

    def __init__(self, db_path: str, vector_dim: int = 384):
        self.logger = get_logger(__name__)
        self.db_path = db_path
        self.vector_dim = vector_dim
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._load_vec_extension()
        self._init_schema()

    def _load_vec_extension(self):
        try:
            import sqlite_vec
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
        except Exception as e:
            raise StorageError(
                "sqlite-vec extension required for vector search. "
                "Install: pip install sqlite-vec"
            ) from e

    def _init_schema(self):
        cur = self._conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='entities'
        """)
        if cur.fetchone():
            return

        with self._conn:
            self._conn.executescript("""
                CREATE TABLE entities (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    normalized_label TEXT,
                    source TEXT DEFAULT 'rag',
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE relationships (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES entities(id),
                    target_id TEXT NOT NULL REFERENCES entities(id),
                    type TEXT NOT NULL,
                    source TEXT DEFAULT 'rag',
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX idx_entities_norm ON entities(normalized_label);
                CREATE INDEX idx_rel_source ON relationships(source_id);
                CREATE INDEX idx_rel_target ON relationships(target_id);
                CREATE INDEX idx_rel_type ON relationships(type);

                CREATE VIRTUAL TABLE entities_fts USING fts5(
                    label, description, content=entities
                );

                CREATE VIRTUAL TABLE entities_vec USING vec0(
                    embedding F32($vector_dim)
                );
            """.replace("$vector_dim", str(self.vector_dim)))

    def exact_label_search(self, labels: List[str]) -> Dict[str, Dict]:
        if not labels:
            return {}
        clean = list(dict.fromkeys(
            label.strip().lower()
            for label in labels[:64]
            if isinstance(label, str) and label.strip()
        ))
        if not clean:
            return {}

        placeholders = ",".join("?" for _ in clean)
        cur = self._conn.execute(
            f"SELECT id, label, description, normalized_label "
            f"FROM entities WHERE normalized_label IN ({placeholders})",
            clean,
        )
        results = {}
        for row in cur.fetchall():
            key = row[3] or row[1].strip().lower()
            results[key] = {
                "id": row[0],
                "label": row[1],
                "description": row[2],
            }
        return results

    def vector_search(
        self, embeddings: List[List[float]], top_k: int = 5
    ) -> List[List[Dict]]:
        results = []
        for emb in embeddings:
            blob = struct_pack_f32(emb)
            cur = self._conn.execute(
                "SELECT e.id, e.label, e.description, vec.distance "
                "FROM entities_vec AS vec "
                "JOIN entities e ON e.rowid = vec.rowid "
                "WHERE embedding MATCH ? "
                "ORDER BY distance LIMIT ?",
                (blob, top_k),
            )
            batch = []
            for row in cur.fetchall():
                batch.append({
                    "id": row[0],
                    "label": row[1],
                    "description": row[2],
                })
            results.append(batch)
        return results

    def keyword_search(
        self, keywords: List[str], top_k: int = 5
    ) -> List[List[Dict]]:
        results = []
        for kw in keywords:
            query = f'"{kw}"~'  # fuzzy FTS5
            try:
                cur = self._conn.execute(
                    "SELECT e.id, e.label, e.description "
                    "FROM entities_fts f JOIN entities e ON e.rowid = f.rowid "
                    "WHERE entities_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (query, top_k),
                )
            except sqlite3.OperationalError:
                results.append([])
                continue
            batch = [
                {"id": r[0], "label": r[1], "description": r[2]}
                for r in cur.fetchall()
            ]
            results.append(batch)
        return results

    def subgraph(self, node_ids: List[str]) -> Dict[str, List[Dict]]:
        if not node_ids:
            return {"nodes": [], "relations": []}
        placeholders = ",".join("?" for _ in node_ids)

        cur = self._conn.execute(
            f"SELECT id, label, description FROM entities WHERE id IN ({placeholders})",
            node_ids,
        )
        nodes = [
            {"id": r[0], "label": r[1], "description": r[2]}
            for r in cur.fetchall()
        ]

        cur = self._conn.execute(
            f"SELECT r.id, r.source_id, r.target_id, r.type "
            f"FROM relationships r "
            f"WHERE r.source_id IN ({placeholders}) "
            f"AND r.target_id IN ({placeholders})",
            node_ids + node_ids,
        )
        relations = [
            {
                "id": r[0],
                "source": r[1],
                "target": r[2],
                "type": r[3],
            }
            for r in cur.fetchall()
        ]

        return {"nodes": nodes, "relations": relations}

    def expansion(
        self,
        frontier_ids: List[str],
        excluded_ids: List[str],
        n_hops: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        if not frontier_ids:
            return [[] for _ in frontier_ids]

        excluded_set = set(excluded_ids)
        frontier_map: Dict[str, List[Dict]] = {fid: [] for fid in frontier_ids}

        for fid in frontier_ids:
            if fid in excluded_set:
                continue
            seen = {fid}
            current = {fid}
            for hop in range(n_hops):
                if not current:
                    break
                placeholders = ",".join("?" for _ in current)
                cur = self._conn.execute(
                    f"SELECT r.id, r.source_id, r.target_id, r.type, "
                    f"e.id, e.label, e.description "
                    f"FROM relationships r "
                    f"JOIN entities e ON e.id = r.target_id "
                    f"WHERE r.source_id IN ({placeholders}) "
                    f"AND r.target_id NOT IN ({','.join('?' for _ in excluded_set) if excluded_set else 'NULL'})",
                    list(current) + (list(excluded_set) if excluded_set else []),
                )
                next_ids = set()
                for row in cur.fetchall():
                    if row[4] in seen:
                        continue
                    seen.add(row[4])
                    next_ids.add(row[4])
                    frontier_map[fid].append({
                        "node": {"id": row[4], "label": row[5], "description": row[6]},
                        "parent": {"id": row[1], "label": "", "description": ""},
                        "relationship": {
                            "id": row[0],
                            "source": row[1],
                            "target": row[2],
                            "type": row[3],
                        },
                    })
                current = next_ids

        return [frontier_map.get(fid, []) for fid in frontier_ids]

    def k_shortest_paths_batch(
        self, pairs: List[Tuple[str, str]], k: int = 3
    ) -> List[Dict]:
        if not pairs:
            return []
        k = min(k, 5)

        results = []
        for src, tgt in pairs:
            paths = self._find_paths(src, tgt, max_depth=3, max_results=k)
            for nodes, rels, weight in paths:
                results.append({
                    "nodes": [
                        {"id": n[0], "label": n[1], "description": n[2]}
                        for n in nodes
                    ],
                    "relations": [
                        {
                            "id": r[0],
                            "source": r[1],
                            "target": r[2],
                            "type": r[3],
                        }
                        for r in rels
                    ],
                    "weight": weight,
                })
        return results

    def _find_paths(self, src: str, tgt: str, max_depth: int = 3, max_results: int = 3):
        """BFS-based path finding between two nodes."""
        queue = [(src, [src], [])]
        visited = {src}
        found = []

        while queue and len(found) < max_results:
            current, nodes, rels = queue.pop(0)
            if len(nodes) - 1 >= max_depth:
                continue
            cur = self._conn.execute(
                "SELECT r.id, r.source_id, r.target_id, r.type, "
                "e.id, e.label, e.description "
                "FROM relationships r "
                "JOIN entities e ON e.id = r.target_id "
                "WHERE r.source_id = ?",
                (current,),
            )
            for row in cur.fetchall():
                next_id = row[4]
                next_node = (row[4], row[5], row[6])
                next_rel = (row[0], row[1], row[2], row[3])
                if next_id == tgt:
                    found.append((
                        nodes + [next_node],
                        rels + [next_rel],
                        len(nodes),
                    ))
                    if len(found) >= max_results:
                        break
                elif next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, nodes + [next_node], rels + [next_rel]))

        return found

    def triplet_search(self, triplet: Dict) -> List[Dict]:
        s = triplet.get("subject", {})
        p = triplet.get("predicate", {})
        o = triplet.get("object", {})
        if not p or not p.get("name"):
            return []

        rel_type = p["name"].upper().replace(" ", "_")
        rel_type = re.sub(r"[^A-Z0-9_]", "", rel_type) or "RELATED_TO"

        s_name = s.get("name") if s else None
        o_name = o.get("name") if o else None

        if s_name and s_name != "?" and o_name and o_name != "?":
            cur = self._conn.execute(
                "SELECT e.id, e.label, e.description FROM entities e "
                "JOIN relationships r ON r.source_id = e.id "
                "WHERE e.label = ? AND r.type = ?",
                (s_name, rel_type),
            )
            return [{"id": r[0], "label": r[1], "description": r[2]} for r in cur]
        if s_name and s_name != "?":
            cur = self._conn.execute(
                "SELECT e.id, e.label, e.description FROM entities e "
                "JOIN relationships r ON r.target_id = e.id "
                "WHERE r.source_id IN (SELECT id FROM entities WHERE label = ?) AND r.type = ?",
                (s_name, rel_type),
            )
            return [{"id": r[0], "label": r[1], "description": r[2]} for r in cur]
        if o_name and o_name != "?":
            cur = self._conn.execute(
                "SELECT e.id, e.label, e.description FROM entities e "
                "JOIN relationships r ON r.source_id = e.id "
                "WHERE r.target_id IN (SELECT id FROM entities WHERE label = ?) AND r.type = ?",
                (o_name, rel_type),
            )
            return [{"id": r[0], "label": r[1], "description": r[2]} for r in cur]
        return []

    def add_triplets(self, triplets: List[Dict[str, str]]):
        from ..validation import normalize_triplets
        triplets = normalize_triplets(triplets, max_items=100)

        with self._lock:
            for t in triplets:
                sub = str(t.get("subject", "")).strip()
                obj = str(t.get("object", "")).strip()
                pred = str(t.get("predicate", "")).upper().replace(" ", "_")
                pred = re.sub(r"[^A-Z0-9_]", "", pred) or "RELATED_TO"

                sub_id = hashlib.md5(sub.lower().encode()).hexdigest()
                obj_id = hashlib.md5(obj.lower().encode()).hexdigest()

                self._conn.execute(
                    "INSERT OR IGNORE INTO entities (id, label, description, normalized_label) "
                    "VALUES (?, ?, ?, ?)",
                    (sub_id, sub, "Created by RAG agent", sub.lower()),
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO entities (id, label, description, normalized_label) "
                    "VALUES (?, ?, ?, ?)",
                    (obj_id, obj, "Created by RAG agent", obj.lower()),
                )

                rel_id = f"{sub_id}_{obj_id}"
                self._conn.execute(
                    "INSERT OR IGNORE INTO relationships (id, source_id, target_id, type) "
                    "VALUES (?, ?, ?, ?)",
                    (rel_id, sub_id, obj_id, pred),
                )
            self._conn.commit()

    def close(self):
        self._conn.close()

    def verify_connectivity(self) -> bool:
        if not os.path.exists(self.db_path):
            return False
        try:
            self._conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def deadline(self, value: float | None):
        """SQLite is fast enough that explicit deadlines are usually unnecessary."""
        return super().deadline(value)


def struct_pack_f32(vec: List[float]) -> bytes:
    """Pack a list of floats into raw F32 bytes for sqlite-vec."""
    import struct
    return struct.pack(f"{len(vec)}f", *vec)
