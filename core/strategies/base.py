"""The strategy registry: parameterised, editable, point-in-time by construction.

A strategy here is a pure function of the close panel and a parameter mapping,
returning a weight matrix of the same shape. `weights[t]` is the position held
into bar `t -> t+1`, which is the alignment `core/backtest/engine.simulate`
charges and pays. Nothing in this package reads a bar after `t`.

Three rules the registry enforces rather than documents:

1. **A parameter override must name a real parameter.** `build(lookbak=60)` raises
   instead of silently running the default. A typo that quietly changes nothing
   is the worst outcome available: the run succeeds and the result is attributed
   to a parameter nobody used.
2. **Every strategy declares the pre-registered grid it may be searched over.**
   G1 counts trials, and the deflated Sharpe deflates by that count. A grid that
   lives in the caller can be widened after the fact; one that lives with the
   strategy cannot.
3. **Every strategy declares what data it needs.** A family that needs
   fundamentals or yields is registered as unavailable rather than approximated
   from prices. A value factor faked out of returns is not a value factor.

Weights are normalised here, never rescaled by the engine: the engine refuses a
book above the pod gross limit instead of shrinking it (ADR-0004).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

#: (close, params) -> weights, same shape as close.
StrategyFn = Callable[[np.ndarray, Mapping[str, float]], np.ndarray]

#: Default target gross. Deliberately below the 1.0 pod limit so the G5
#: robustness gate can actually test the +/-20% neighbourhood: a strategy sitting
#: exactly at the limit has neighbours the engine must drop as infeasible, and a
#: plateau that was never sampled is not evidence of a plateau.
DEFAULT_GROSS = 0.8


class UnknownParameter(KeyError):
    """An override named a parameter the strategy does not have."""


class MissingData(RuntimeError):
    """A strategy was requested whose inputs this repository cannot supply."""


@dataclass(frozen=True)
class Strategy:
    """One editable strategy: a function, its defaults, and its search grid."""

    name: str
    family: str
    fn: StrategyFn
    defaults: Mapping[str, float]
    grid: tuple[Mapping[str, float], ...]
    citation: str
    requires: tuple[str, ...] = ("close",)
    available: bool = True
    unavailable_because: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def params(self, **overrides: float) -> dict[str, float]:
        """Defaults with overrides applied. An unknown parameter name raises."""
        unknown = set(overrides) - set(self.defaults)
        if unknown:
            raise UnknownParameter(
                f"{self.name} has no parameter(s) {sorted(unknown)}; it takes {sorted(self.defaults)}"
            )
        return {**self.defaults, **overrides}

    def build(self, **overrides: float) -> Callable[[np.ndarray], np.ndarray]:
        """Bind parameters and return the function the engine and G0 scan call."""
        if not self.available:
            raise MissingData(f"{self.name}: {self.unavailable_because}")
        bound = self.params(**overrides)
        return lambda close: self.fn(np.asarray(close, dtype=float), bound)

    def search_grid(self, **overrides: float) -> list[dict[str, float]]:
        """The pre-registered grid, with any fixed parameter overridden in every point.

        Overriding narrows the grid rather than widening it: the number of trials
        is what it always was, so the deflated Sharpe deflates by the same count.
        """
        return [{**self.params(**point), **overrides} for point in self.grid]


REGISTRY: dict[str, Strategy] = {}


def register(strategy: Strategy) -> Strategy:
    if strategy.name in REGISTRY:
        raise ValueError(f"{strategy.name} is already registered")
    if strategy.available and not strategy.grid:
        raise ValueError(f"{strategy.name} has no pre-registered grid; G1 needs one")
    for point in strategy.grid:
        unknown = set(point) - set(strategy.defaults)
        if unknown:
            raise ValueError(f"{strategy.name} grid point names unknown parameter(s) {sorted(unknown)}")
    REGISTRY[strategy.name] = strategy
    return strategy


def get(name: str) -> Strategy:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"no strategy {name!r}; registered: {sorted(REGISTRY)}") from None


def names(family: str | None = None, available_only: bool = False) -> list[str]:
    out = [
        s.name
        for s in REGISTRY.values()
        if (family is None or s.family == family) and (s.available or not available_only)
    ]
    return sorted(out)


# --- shared shaping ---------------------------------------------------------


def scale_to_gross(raw: np.ndarray, gross: float) -> np.ndarray:
    """Scale each row so its absolute weights sum to `gross`.

    A row that is entirely zero stays zero: no position is a position, and
    dividing by a zero norm to force one would invent a book out of no signal.
    """
    out = np.zeros_like(raw)
    norm = np.sum(np.abs(raw), axis=1)
    live = norm > 0
    out[live] = raw[live] / norm[live, None] * gross
    return out


def dollar_neutral(scores: np.ndarray, gross: float) -> np.ndarray:
    """Cross-sectionally demeaned scores scaled to `gross`.

    Demeaning per row is what makes the book dollar neutral, which is the point
    of a cross-sectional signal: the market return is not the alpha.
    """
    centred = scores - scores.mean(axis=1, keepdims=True)
    return scale_to_gross(centred, gross)


def cross_sectional_rank(scores: np.ndarray) -> np.ndarray:
    """Per-row ranks centred on zero, or a row of zeros when any score is missing.

    Ranks rather than raw scores because the cross-section here is a handful of
    instruments, where one outlier return would otherwise take the whole book.
    A row with a NaN is dropped entirely rather than ranked around the gap: a
    partial cross-section is a different universe, and ranking it would compare
    today's four names against yesterday's five.
    """
    out = np.zeros_like(scores)
    n = scores.shape[1]
    if n < 2:
        return out
    for t in range(scores.shape[0]):
        row = scores[t]
        if not np.all(np.isfinite(row)):
            continue
        order = np.argsort(np.argsort(row))
        out[t] = order - (n - 1) / 2.0
    return out


def trailing_return(close: np.ndarray, lookback: int) -> np.ndarray:
    """`close[t] / close[t - lookback] - 1`, and NaN before enough history.

    NaN rather than 0.0 for the warm-up: a zero would be read as a measured
    flat return and put into a cross-sectional rank alongside real ones.
    """
    out = np.full_like(close, np.nan)
    if lookback < 1 or lookback >= close.shape[0]:
        return out
    out[lookback:] = close[lookback:] / close[:-lookback] - 1.0
    return out


def bar_returns(close: np.ndarray) -> np.ndarray:
    """Simple returns aligned to the close panel, first row NaN.

    Row `t` is the return earned *into* `t`, so it is known at `t`. The engine's
    `bar_returns` is the forward series and is a different thing; mixing the two
    up is exactly the look-ahead G0 exists to catch.
    """
    out = np.full_like(close, np.nan)
    out[1:] = close[1:] / close[:-1] - 1.0
    return out


def blend(parts: Sequence[np.ndarray], gross: float) -> np.ndarray:
    """Average several weight matrices and renormalise to `gross`.

    This is the plain equal-weight blend, and it is deliberately not risk
    weighted. Weighting sub-strategies by their realised risk needs each one's
    return series, which exists one layer up in `core/portfolio/allocate.py`
    where the correlation shrinkage and the risk parity already live. Doing it
    here from prices alone would either duplicate that or fake it.
    """
    if not parts:
        raise ValueError("blend needs at least one strategy")
    shapes = {part.shape for part in parts}
    if len(shapes) != 1:
        raise ValueError(f"cannot blend weight matrices of shapes {sorted(shapes)}")
    stacked = np.mean(np.stack(parts, axis=0), axis=0)
    return scale_to_gross(stacked, gross)
