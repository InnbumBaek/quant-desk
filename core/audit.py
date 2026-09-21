"""Append-only, hash-chained audit log.

Every order, approval, limit change and gate verdict lands here. The chain
makes silent edits detectable: each record carries the hash of the previous
one, so rewriting history breaks verification at the first tampered record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _digest(prev_hash: str, payload: str) -> str:
    return hashlib.sha256(f"{prev_hash}{payload}".encode()).hexdigest()


def _last_hash(path: Path) -> str:
    if not path.exists():
        return GENESIS
    last = GENESIS
    for record in read(path):
        last = record["hash"]
    return last


def append(event: str, data: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Append one record and return it."""
    from core.config import AUDIT_LOG

    path = path or AUDIT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event,
        "data": data,
    }
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False)
    record = {**body, "prev": _last_hash(path), "hash": ""}
    record["hash"] = _digest(record["prev"], payload)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return record


def read(path: Path | None = None) -> Iterator[dict[str, Any]]:
    from core.config import AUDIT_LOG

    path = path or AUDIT_LOG
    if not path.exists():
        return iter(())
    with path.open(encoding="utf-8") as fh:
        return iter([json.loads(line) for line in fh if line.strip()])


def verify(path: Path | None = None) -> tuple[bool, str | None]:
    """Return (ok, first_broken_record_hash)."""
    prev = GENESIS
    for record in read(path):
        body = {"ts": record["ts"], "event": record["event"], "data": record["data"]}
        payload = json.dumps(body, sort_keys=True, ensure_ascii=False)
        if record["prev"] != prev or record["hash"] != _digest(prev, payload):
            return False, record["hash"]
        prev = record["hash"]
    return True, None
