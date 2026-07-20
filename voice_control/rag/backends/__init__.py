from typing import Optional

from ...common.config import config
from ...common.utils import get_logger
from ...exceptions import ConfigError
from .base import StorageBackend


def create_backend(name: Optional[str] = None) -> StorageBackend:
    """Create a storage backend based on config or explicit name."""
    backend_name = name or config.get("rag.runtime.backend", "neo4j")
    logger = get_logger(__name__)

    if backend_name == "neo4j":
        neo4j_config = config.get("database.neo4j")
        if not neo4j_config or not all([
            neo4j_config.uri, neo4j_config.user, neo4j_config.password
        ]):
            raise ConfigError(
                "Neo4j credentials not fully configured for storage backend."
            )

        from ..knowledge import KnowledgeGraph
        return KnowledgeGraph(
            uri=neo4j_config.uri,
            user=neo4j_config.user,
            password=neo4j_config.password,
            database=neo4j_config.database,
            query_timeout=neo4j_config.query_timeout_seconds,
        )

    if backend_name == "sqlite":
        db_path = config.get("rag.runtime.sqlite_path", "data/rag.sqlite")
        from .sqlite import SQLiteBackend
        return SQLiteBackend(db_path=db_path)

    raise ConfigError(f"Unknown storage backend: {backend_name!r}.")
