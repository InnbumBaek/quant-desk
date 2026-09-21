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

    if snapshot.get("gross", 0.0) > lim["gross_leverage_max"]:
        out.append(Breach("GROSS", f"{snapshot['gross']:.2f}x > {lim['gross_leverage_max']}x"))

    low, high = lim["net_exposure"]
    net = snapshot.get("net", 0.0)
    if not low <= net <= high:
        out.append(Breach("NET", f"{net:+.2%} outside [{low:+.0%}, {high:+.0%}]"))

    for symbol, weight in (snapshot.get("weights") or {}).items():
        if abs(weight) > lim["single_name_max"]:
            out.append(Breach("SINGLE_NAME", f"{symbol} {weight:+.2%}"))

    for sector, weight in (snapshot.get("sector_weights") or {}).items():
        if abs(weight) > lim["sector_max"]:
            out.append(Breach("SECTOR", f"{sector} {weight:+.2%}"))

    for factor, beta in (snapshot.get("style_betas") or {}).items():
        if abs(beta) > lim["style_beta_abs_max"]:
            out.append(Breach("STYLE_BETA", f"{factor} {beta:+.2f}"))

    if snapshot.get("liquidation_days", 0.0) > lim["liquidation_days_max"]:
        out.append(Breach("LIQUIDITY", f"{snapshot['liquidation_days']:.1f}d"))

    out.extend(drawdown_breaches(snapshot, lim))
    return out


def drawdown_breaches(snapshot: dict[str, Any], pod_limits: dict[str, Any]) -> list[Breach]:
    """Drawdown needs BOTH the absolute level and the distribution condition.

    A pod inside its own historical drawdown distribution is having a normal bad
    run; cutting it there is how platforms kill good strategies at the bottom.
    """
    dd = snapshot.get("drawdown")
    pct = snapshot.get("backtest_dd_pct")
    if dd is None or pct is None:
        return []
    out: list[Breach] = []
    for name in ("stop", "cut", "warn"):
        tier = pod_limits["drawdown"][name]
        if dd <= tier["abs"] and pct > tier["backtest_dd_pct"]:
            detail = f"{dd:+.2%}, p{pct:.0f} of backtest DD -> {tier['action']}"
            out.append(Breach(f"DD_{name.upper()}", detail))
            break
    return out
