"""Deterministic limit engine.

No LLM is in this path. The engine takes a snapshot of the book and returns
breaches; callers (the pre-trade hook, the intraday poller) must treat any
breach as blocking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Breach:
    code: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.code}: {self.detail}"


def load_limits(path: Path | None = None) -> dict[str, Any]:
    from core.config import LIMITS_FILE

    with (path or LIMITS_FILE).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def check_pod(snapshot: dict[str, Any], limits: dict[str, Any] | None = None) -> list[Breach]:
    """Check one pod's snapshot against the pod-level limits.

    snapshot keys: gross, net, weights {symbol: w}, sector_weights, style_betas,
    liquidation_days, drawdown, backtest_dd_pct.
    """
    lim = (limits or load_limits())["pod"]
    out: list[Breach] = []

    gross = _number(snapshot.get("gross"))
    if gross is None:
        out.append(Breach("GROSS_UNMEASURED", f"gross is {snapshot.get('gross')!r}"))
    elif gross > lim["gross_leverage_max"]:
        out.append(Breach("GROSS", f"{gross:.2f}x > {lim['gross_leverage_max']}x"))

    low, high = lim["net_exposure"]
    net = _number(snapshot.get("net"))
    if net is None:
        out.append(Breach("NET_UNMEASURED", f"net exposure is {snapshot.get('net')!r}"))
    elif not low <= net <= high:
        out.append(Breach("NET", f"{net:+.2%} outside [{low:+.0%}, {high:+.0%}]"))

    book = snapshot.get("weights") or {}
    for symbol, weight in book.items():
        if abs(weight) > lim["single_name_max"]:
            out.append(Breach("SINGLE_NAME", f"{symbol} {weight:+.2%}"))

    out.extend(sector_breaches(snapshot, lim, book))
    out.extend(style_beta_breaches(snapshot, lim))
    out.extend(liquidity_breaches(snapshot, lim))
    out.extend(volatility_breaches(snapshot, lim))
    out.extend(drawdown_breaches(snapshot, lim))
    return out


def _number(value: Any) -> float | None:
    """The value as a float, or `None` when it is not a measurement.

    `None`, a string, a bool and NaN are all "not measured". Bools are excluded
    deliberately: `True` is 1.0 to Python and would read as a 100% exposure.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return None if number != number else number


def sector_breaches(
    snapshot: dict[str, Any], pod_limits: dict[str, Any], book: dict[str, float]
) -> list[Breach]:
    """Bucket concentration. Absence blocks (ADR-0015).

    An empty mapping is a measurement only for a book with nothing in it: a flat
    book has no bucket weights. With positions on, an empty mapping means nobody
    computed them, and reading that as "no concentration" is how a limit that
    checks nothing looks like a limit that passes.
    """
    weights = snapshot.get("sector_weights")
    held = any(abs(float(weight)) > 0.0 for weight in book.values())
    if weights is None or (not weights and held):
        return [
            Breach(
                "SECTOR_UNMEASURED",
                f"sector weights are {weights!r} for a book of {len(book)} name(s); "
                f"the {pod_limits['sector_max']:.0%} bucket cap cannot be checked",
            )
        ]
    out: list[Breach] = []
    for sector, weight in weights.items():
        value = _number(weight)
        if value is None:
            out.append(Breach("SECTOR_UNMEASURED", f"{sector} weight is {weight!r}"))
        elif abs(value) > pod_limits["sector_max"]:
            out.append(Breach("SECTOR", f"{sector} {value:+.2%}"))
    return out


def style_beta_breaches(snapshot: dict[str, Any], pod_limits: dict[str, Any]) -> list[Breach]:
    """Factor betas against the band. Absence blocks (ADR-0015).

    A market-neutral mandate carrying a market beta of 0.8 is running a strategy
    it was not approved for, and that is exactly the state an unmeasured beta
    hides. An empty mapping is not a flat book -- a measured flat book has a beta
    of zero on every factor -- so it is absence too.
    """
    betas = snapshot.get("style_betas")
    if not betas:
        return [
            Breach(
                "STYLE_BETA_UNMEASURED",
                f"style betas are {betas!r}; the +/-{pod_limits['style_beta_abs_max']:.2f} band "
                "cannot be checked. A factor file is needed (ADR-0014)",
            )
        ]
    out: list[Breach] = []
    for factor, beta in betas.items():
        value = _number(beta)
        if value is None:
            out.append(Breach("STYLE_BETA_UNMEASURED", f"{factor} beta is {beta!r}"))
        elif abs(value) > pod_limits["style_beta_abs_max"]:
            out.append(Breach("STYLE_BETA", f"{factor} {value:+.2f}"))
    return out


def liquidity_breaches(snapshot: dict[str, Any], pod_limits: dict[str, Any]) -> list[Breach]:
    """Days to unwind. Absence blocks (ADR-0015).

    This one read a missing value as 0.0, which is the most forgiving number
    available: an unmeasured book was treated as instantly liquidatable.
    """
    days = _number(snapshot.get("liquidation_days"))
    if days is None:
        return [
            Breach(
                "LIQUIDITY_UNMEASURED",
                f"liquidation days is {snapshot.get('liquidation_days')!r}; "
                f"the {pod_limits['liquidation_days_max']}-day cap cannot be checked",
            )
        ]
    if days < 0.0:
        return [Breach("LIQUIDITY_UNMEASURED", f"liquidation days {days} is not a horizon")]
    if days > pod_limits["liquidation_days_max"]:
        return [Breach("LIQUIDITY", f"{days:.1f}d > {pod_limits['liquidation_days_max']}d")]
    return []


def volatility_breaches(snapshot: dict[str, Any], pod_limits: dict[str, Any]) -> list[Breach]:
    """Realised volatility against the target band.

    Absence blocks. A target nobody can measure is a target nobody is keeping,
    and reading a missing value as 0.0 would pass every book ever submitted.

    Below the band is deliberately not a breach. The band is an operating
    target, and undershooting it costs return, not capital; a limit exists to
    stop a loss. Treating an undershoot as a breach would also invite raising
    gross to clear it, which is the one thing the gross limit is there to stop.
    """
    low, high = pod_limits["target_volatility"]
    realised = _number(snapshot.get("realised_volatility"))
    if realised is None:
        return [
            Breach(
                "VOL_UNMEASURED",
                f"realised volatility is {snapshot.get('realised_volatility')!r}; "
                f"the band [{low:.0%}, {high:.0%}] cannot be checked",
            )
        ]
    if realised < 0.0:
        return [Breach("VOL_UNMEASURED", f"realised volatility {realised} is not a volatility")]
    if realised > high:
        return [Breach("VOL_ABOVE_TARGET", f"{realised:.2%} > {high:.0%}")]
    return []


def drawdown_breaches(snapshot: dict[str, Any], pod_limits: dict[str, Any]) -> list[Breach]:
    """Drawdown needs BOTH the absolute level and the distribution condition.

    A pod inside its own historical drawdown distribution is having a normal bad
    run; cutting it there is how platforms kill good strategies at the bottom.
    """
    dd = _number(snapshot.get("drawdown"))
    pct = _number(snapshot.get("backtest_dd_pct"))
    if dd is None or pct is None:
        # Absence blocks (ADR-0015). This is the limit that halts a losing pod, so
        # "we could not measure the drawdown" must not be the quietest way past it.
        return [
            Breach(
                "DD_UNMEASURED",
                f"drawdown is {snapshot.get('drawdown')!r} and its backtest percentile is "
                f"{snapshot.get('backtest_dd_pct')!r}; both are needed to act on either",
            )
        ]
    out: list[Breach] = []
    for name in ("stop", "cut", "warn"):
        tier = pod_limits["drawdown"][name]
        if dd <= tier["abs"] and pct > tier["backtest_dd_pct"]:
            detail = f"{dd:+.2%}, p{pct:.0f} of backtest DD -> {tier['action']}"
            out.append(Breach(f"DD_{name.upper()}", detail))
            break
    return out
