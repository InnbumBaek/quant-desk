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

#: The one host that will not take the string above. `data.sec.gov` answered the
#: 2026-09-26 probe with **"Your Request Originates from an Undeclared Automated
#: Tool"**, which is a different refusal from the "Request Rate Threshold
#: Exceeded" that `www.sec.gov` gave in the same run -- and for eight days this
#: desk recorded the pair as one rate problem, because the second body was never
#: read (ADR-0030).
#:
#: SEC asks automated clients for a declared **contact address**, not a URL. So
#: the address is read from the environment and never committed: the repository
#: is public and the address is a person's. Absent, `sec_user_agent` raises, and
#: a caller is expected to skip the SEC source with a recorded reason rather than
#: send a request SEC has already said it will refuse.
SEC_CONTACT_ENV = "SEC_CONTACT_EMAIL"


def sec_user_agent() -> str:
    """The User-Agent `sec.gov` asks for, or a refusal naming what is missing.

    SEC's own guidance is a company name followed by a contact address. Keeping
    the desk's name in front means a refusal is still attributable to us, and the
    address is what makes the client declared rather than anonymous.
    """
    contact = os.environ.get(SEC_CONTACT_ENV, "").strip()
    if not contact or "@" not in contact:
        raise RuntimeError(
            f"{SEC_CONTACT_ENV} is not set to a contact address, and sec.gov refuses an "
            "undeclared automated tool by name. Set it in the runner's secrets; it is a "
            "contact address, not a credential, and it is never committed."
        )
    return f"quant-desk research (InnbumBaek) {contact}"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing environment variable {name}; see .env.example")
    return value
