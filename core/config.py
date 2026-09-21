"""Paths and environment for the quant-desk monorepo.

Secrets never live in the repository: every API key is read from the
environment (see .env.example). The repository is public, so a committed
credential is a disclosed credential.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
STATE = ROOT / "state"
AUDIT_LOG = STATE / "audit.log"
ORDER_STATE = STATE / "orders"
LIMITS_FILE = ROOT / "core" / "risk" / "limits.yaml"

# Trading is paper-only until a human flips this and the G8 gate is signed off.
LIVE_TRADING = os.environ.get("QD_LIVE_TRADING", "0") == "1"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing environment variable {name}; see .env.example")
    return value
