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

#: How every fetcher identifies itself. One string, in one place, because two
#: of them differed and it cost a week of the literature sweep: the fetcher
#: sending `quant-desk/0.1 (research; +https://...)` was refused 406 with an
#: empty body by two arXiv hosts, while the probe sending this string got 200
#: from one of them minutes later (registry/probes/2026-09-27.json). A
#: `name/version` token reads as a script to a bot filter; a sentence with a
#: name and a way to reach us reads as a person running something.
#:
#: A contact URL and not an address: a personal email does not belong in a
#: public file, and the issue tracker is a place somebody can actually reach us.
#: Never a browser string -- a 200 obtained by lying is a fetcher that breaks
#: the moment it matters and a refusal nobody can diagnose.
USER_AGENT = "quant-desk research (InnbumBaek; https://github.com/InnbumBaek/quant-desk/issues)"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing environment variable {name}; see .env.example")
    return value
