"""Canary contract. See README.md in this directory.

The gates do not exist yet (P2), so each canary is marked strict-xfail: the
build stays green while the engine is missing, and turns red the moment a
canary starts passing without having been implemented properly.
"""

import pytest

CANARIES = ["canary_random", "canary_lookahead", "canary_overfit", "canary_factor"]


@pytest.mark.parametrize("canary", CANARIES)
@pytest.mark.xfail(strict=True, reason="gate engine lands in P1/P2")
def test_canary_is_rejected(canary):
    from core.backtest import gates  # noqa: F401  (does not exist yet)

    raise AssertionError("unreachable until the gate engine exists")
