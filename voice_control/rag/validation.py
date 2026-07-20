"""Validation and quarantine helpers for model-generated knowledge."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List

from ..exceptions import StorageError


_QUEUE_LOCK = threading.Lock()
_PREDICATE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,63}")


def normalize_triplets(raw: Any, max_items: int = 20) -> List[Dict[str, str]]:
    """Return a strict, bounded triplet list or raise for malformed output."""
    # ASVS 1.5.2 / 2.2.1: model JSON is untrusted external data. Only the
    # expected scalar fields and bounded values are admitted to persistence.
    if not isinstance(raw, list):
        raise StorageError("Triplets must be a JSON array.")
    if len(raw) > max_items:
        raise StorageError(f"Triplet count exceeds the limit of {max_items}.")

    normalized = []
    for item in raw:
        if not isinstance(item, dict):
            raise StorageError("Each triplet must be a JSON object.")
        if set(item) != {"subject", "predicate", "object"}:
            raise StorageError(
                "Each triplet must contain only subject, predicate, and object."
            )

        subject = item["subject"]
        predicate = item["predicate"]
        obj = item["object"]
        if not all(isinstance(value, str) for value in (subject, predicate, obj)):
            raise StorageError("Triplet fields must be strings.")

        subject = subject.strip()
        obj = obj.strip()
        predicate = re.sub(r"[^A-Z0-9_]", "", predicate.upper().replace(" ", "_"))
        if not subject or not obj or not predicate:
            raise StorageError("Triplet fields must not be empty.")
        if len(subject) > 200 or len(obj) > 200:
            raise StorageError("Triplet entities must not exceed 200 characters.")
        if not _PREDICATE_PATTERN.fullmatch(predicate):
            raise StorageError("Triplet predicate must be a valid relationship name.")

        normalized.append(
            {"subject": subject, "predicate": predicate, "object": obj}
        )
    return normalized


def queue_triplets(
    triplets: List[Dict[str, str]],
    queue_path: str,
    *,
    query: str,
    provenance: str,
) -> Path:
    """Append validated triplets to a project-local JSONL review queue."""
    project_root = Path.cwd().resolve()
    candidate = Path(queue_path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()

    # ASVS 5.3.2: a configured path may not escape the project directory.
    if os.path.commonpath((str(project_root), str(candidate))) != str(project_root):
        raise StorageError("The review queue must be inside the project directory.")
    if candidate.suffix.lower() != ".jsonl":
        raise StorageError("The review queue must use a .jsonl extension.")

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending_review",
        "provenance": provenance,
        "query": query[:1_000],
        "triplets": triplets,
    }
    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise StorageError("Review queue record exceeds 64 KiB.")

    candidate.parent.mkdir(parents=True, exist_ok=True)
    # ASVS 15.4.1: serialize append operations from concurrent voice/RPC flows.
    with _QUEUE_LOCK, candidate.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
    return candidate
