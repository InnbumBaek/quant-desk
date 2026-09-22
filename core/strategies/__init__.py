"""Editable strategies, registered so a pod cannot keep a private one.

Importing this package registers the library. The registry is the only way in:

    from core import strategies
    trend = strategies.get("ts_momentum")
    weights = trend.build(lookback=120)(panel.close)   # edit any parameter here
    grid = trend.search_grid()                          # the pre-registered grid G1 counts

Overriding a parameter the strategy does not have raises rather than silently
running the default, and a family whose data we do not hold is registered as
unavailable rather than approximated. See `library.py` for what is in and what
is deliberately out.
"""

from core.strategies import library as _library  # noqa: F401 - registers the library
from core.strategies.base import (
    DEFAULT_GROSS,
    REGISTRY,
    MissingData,
    Strategy,
    UnknownParameter,
    blend,
    get,
    names,
    register,
)
from core.strategies.overlay import vol_target

__all__ = [
    "DEFAULT_GROSS",
    "REGISTRY",
    "MissingData",
    "Strategy",
    "UnknownParameter",
    "blend",
    "get",
    "names",
    "register",
    "vol_target",
]
