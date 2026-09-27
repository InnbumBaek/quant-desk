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
#: of them differed and we could not tell a host that filters us from a typo.
#:
#: That difference was once blamed for arXiv's 406, and the blame was wrong:
#: the refusal turned out to be decided by the request's content, not by who
#: sent it (ADR-0020). The rule outlived its rejected reason, and the reason
#: is written down so nobody re-runs the experiment.
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
