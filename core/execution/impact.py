"""Price impact estimated from daily bars, and the capacity it implies.

`core/risk/capacity.py` enforces CLAUDE.md 7항 -- no capital above 80% of the
estimated capacity -- and its estimate is
`tested_capital x participation_cap / observed_participation`. That arithmetic
is exact for the question it asks, because participation is linear in capital by
construction. **The question is the approximation.** Real capacity is the capital
at which impact eats the alpha, and a strategy can sit at 5% of ADV with its edge
already gone. ADR-0026 called the participation estimate an over-estimate for
exactly this reason and left the cost side unbuilt.

**What this module will not do.** It will not invent a coefficient. A
square-root impact law with a textbook constant typed in from memory would put a
made-up number behind a capital limit, which is the same failure as a sector
bucket nobody measured (ADR-0029) and a scenario size nobody observed
(ADR-0027). Realised impact cannot be calibrated here at all: calibration needs
fills, and this desk has none.

**So the estimator is Amihud's, and the reason is the data.** `PricePanel`
carries closes and dollar volume and no intraday range, which rules out the
high-low spread estimators (Corwin-Schultz 2012, Abdi-Ranaldo 2017). Amihud
(2002) illiquidity -- the mean of `|return| / dollar volume` over a window -- is
computable from exactly these two series, is a published derivation rather than
a fitted constant, and reads directly as *fractional price move per dollar
traded*, which is what a capacity question needs.

**Linear, and conservative in the direction that matters.** Amihud's lambda is
linear in size by construction, and this module charges the terminal price
displacement rather than the average one a trade actually walks through. Impact
is empirically concave, and the average displacement is roughly half the
terminal one, so both approximations push the same way: the cost comes out high
and the capacity comes out low. That is the side a capital cap should be wrong
on, and it is stated rather than hidden -- what this module reports is an upper
bound, not a forecast of what an order will pay.

**Nothing here is a limit and nothing here blocks.** `capacity.py` keeps
enforcing on participation and this module only reports the gap between the two
estimates, as findings. A `capacity.cost_based_*` block is proposed in ADR-0031
and left for the owner: the threshold is what share of a strategy's gross edge
may be spent on impact, and that is a risk-appetite decision, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.data.sources import PricePanel

#: A round trip is two crossings. Entering moves the price against you and so
#: does leaving, and a capacity question is about holding a position, which
#: implies both. Not a tunable: it is the count of crossings, not a coefficient.
ROUND_TRIP_CROSSINGS = 2

#: The window Amihud's illiquidity is averaged over, in sessions. Sixty matches
#: `pod_snapshot`'s liquidity window so the two measurements describe the same
#: stretch of trading; a capacity built on a different window than the unwind
#: horizon would be two answers about two markets.
DEFAULT_WINDOW = 60

#: Below this many usable observations a symbol gets no lambda. Amihud's measure
#: is a mean of a heavy-tailed ratio, and a mean of five of those is noise with a
#: number attached. Twenty is a month of sessions, and it is a floor rather than
#: a recommendation: the reported `observations` says what it actually had.
MIN_OBSERVATIONS = 20


@dataclass(frozen=True)
class Illiquidity:
    """Amihud's lambda for one symbol, or the reason there is none.

    `lam` is a fractional price move per dollar traded, so `lam * 1e6` is the
    fractional move a million-dollar order is charged. `unmeasured` is non-empty
    exactly when `lam` is None, and it says which input was missing -- a capacity
    that silently became infinite is what ADR-0026 exists to stop.
    """

    symbol: str
    lam: float | None = None
    observations: int = 0
    median_dollar_volume: float | None = None
    unmeasured: str = ""

    @property
    def measured(self) -> bool:
        return self.lam is not None

    def cost_fraction(self, dollars: float, crossings: int = ROUND_TRIP_CROSSINGS) -> float | None:
        """The fraction of notional a trade of `dollars` is charged, or None.

        **Two deliberate conservatisms, named so neither is a surprise.** This
        charges the *terminal* displacement `lam * dollars` per crossing, where a
        trade walked into the book pays closer to the average displacement, about
        half of it; and Amihud's lambda is linear where impact is empirically
        concave. Both err the same way -- the cost comes out high, so the
        capacity comes out low -- which is the side a capital cap should be wrong
        on. It also means this is not a fill forecast and must not be reported as
        one (ADR-0031).
        """
        if self.lam is None or dollars <= 0.0:
            return None
        return float(crossings * self.lam * dollars)


def amihud_lambda(
    panel: PricePanel,
    symbol: str,
    window: int = DEFAULT_WINDOW,
    min_observations: int = MIN_OBSERVATIONS,
) -> Illiquidity:
    """Amihud (2002) illiquidity for one symbol over the last `window` sessions.

    Every refusal is named rather than returned as a number: no dollar-volume
    panel, a symbol the panel does not carry, or too few usable observations.
    """
    if symbol not in panel.symbols:
        return Illiquidity(symbol=symbol, unmeasured=f"{symbol!r} is not in the panel")
    if panel.dollar_volume is None:
        return Illiquidity(
            symbol=symbol,
            unmeasured="the panel carries no dollar volume, so impact per dollar is unmeasurable",
        )
    if window < 2:
        raise ValueError(f"a window of {window} session(s) produces no return")

    column = panel.symbols.index(symbol)
    close = np.asarray(panel.close, dtype=float)[:, column]
    volume = np.asarray(panel.dollar_volume, dtype=float)[:, column]

    # One more close than returns, so the window of returns is `window` long.
    close = close[-(window + 1) :]
    volume = volume[-window:]
    if close.size < 2:
        return Illiquidity(symbol=symbol, unmeasured="fewer than two closes in the window")
    returns = np.diff(close) / close[:-1]
    volume = volume[-returns.size :]

    usable = np.isfinite(returns) & np.isfinite(volume) & (volume > 0.0)
    count = int(np.count_nonzero(usable))
    median_volume = float(np.median(volume[usable])) if count else None
    if count < min_observations:
        return Illiquidity(
            symbol=symbol,
            observations=count,
            median_dollar_volume=median_volume,
            unmeasured=(
                f"{count} usable session(s) in a {window}-session window, below the "
                f"{min_observations} floor; a mean of that few heavy-tailed ratios is noise"
            ),
        )
    lam = float(np.mean(np.abs(returns[usable]) / volume[usable]))
    if not np.isfinite(lam) or lam <= 0.0:
        return Illiquidity(
            symbol=symbol,
            observations=count,
            median_dollar_volume=median_volume,
            unmeasured=f"lambda came out {lam}, which is not a price move per dollar",
        )
    return Illiquidity(
        symbol=symbol,
        lam=lam,
        observations=count,
        median_dollar_volume=median_volume,
    )


def panel_illiquidity(
    panel: PricePanel,
    window: int = DEFAULT_WINDOW,
    min_observations: int = MIN_OBSERVATIONS,
) -> dict[str, Illiquidity]:
    """Every symbol's lambda, measured or refused. One entry per column, always."""
    return {
        symbol: amihud_lambda(panel, symbol, window=window, min_observations=min_observations)
        for symbol in panel.symbols
    }


def book_cost_fraction(
    weights: np.ndarray,
    panel: PricePanel,
    capital: float,
    window: int = DEFAULT_WINDOW,
    crossings: int = ROUND_TRIP_CROSSINGS,
) -> tuple[float | None, tuple[str, ...]]:
    """What a round trip of this book costs, as a fraction of `capital`.

    Returns `(None, unmeasured_symbols)` when **any** held name has no lambda.
    Charging zero for the names nobody could measure and summing the rest is how
    a cost estimate quietly becomes an argument for more capital: the illiquid
    name is exactly the one that is expensive, and dropping it makes the book
    look cheaper the less we know about it.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size != len(panel.symbols):
        raise ValueError(f"expected one weight per panel symbol ({len(panel.symbols)}), got {weights.shape}")
    if capital <= 0.0:
        raise ValueError(f"capital {capital} is not a book size")

    measures = panel_illiquidity(panel, window=window)
    held = [symbol for symbol, weight in zip(panel.symbols, weights, strict=True) if weight != 0.0]
    unmeasured = tuple(sorted(s for s in held if not measures[s].measured))
    if unmeasured:
        return None, unmeasured

    total = 0.0
    for symbol, weight in zip(panel.symbols, weights, strict=True):
        if weight == 0.0:
            continue
        notional = abs(float(weight)) * capital
        cost = measures[symbol].cost_fraction(notional, crossings=crossings)
        # `measured` is True for every held name here, so `cost` is not None.
        total += float(cost) * notional  # type: ignore[arg-type]
    return total / capital, ()


def cost_capacity(
    weights: np.ndarray,
    panel: PricePanel,
    gross_edge: float,
    cost_share_max: float,
    window: int = DEFAULT_WINDOW,
    crossings: int = ROUND_TRIP_CROSSINGS,
) -> tuple[float | None, str]:
    """The capital at which round-trip cost reaches `cost_share_max` of the edge.

    `gross_edge` is the strategy's gross return per round trip, before costs, and
    it comes from a backtest artifact rather than from this module: CLAUDE.md 2항
    means the number in a report cites where it was produced. `cost_share_max` is
    the share of that edge the desk is willing to pay away, and it is an argument
    with no default on purpose -- there is no such key in `limits.yaml`, and
    inventing one here would be a limit nobody approved (ADR-0031).

    Cost is linear in capital, so the budget is reached at exactly one capital
    and the arithmetic closes: `budget / cost_at_unit_capital`.
    """
    if gross_edge <= 0.0:
        return None, f"a gross edge of {gross_edge} leaves no room for any cost"
    if not 0.0 < cost_share_max < 1.0:
        raise ValueError(f"cost_share_max {cost_share_max} is not a share of the edge")

    unit = 1_000_000.0
    fraction, unmeasured = book_cost_fraction(weights, panel, unit, window=window, crossings=crossings)
    if fraction is None:
        return None, f"no illiquidity measure for {list(unmeasured)[:5]}"
    if fraction <= 0.0:
        return None, "the book's round-trip cost came out non-positive, which is not a cost"

    budget = gross_edge * cost_share_max
    # `fraction` is the cost share of capital at `unit`; it scales linearly, so
    # cost_share(capital) = fraction * capital / unit.
    return float(unit * budget / fraction), ""


@dataclass(frozen=True)
class CapacityGap:
    """The participation estimate beside the cost estimate, and neither enforced.

    `findings` is prose for a human. There are deliberately no `Breach` objects:
    `core/risk/capacity.py` enforces the participation cap and this is a second
    opinion, not a second limit (ADR-0031).
    """

    participation_capacity: float | None = None
    cost_capacity: float | None = None
    cost_share_max: float | None = None
    gross_edge: float | None = None
    unmeasured: str = ""
    findings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def binding(self) -> float | None:
        """The smaller of the two, which is the one a capacity cap should use."""
        pair = [x for x in (self.participation_capacity, self.cost_capacity) if x is not None]
        return min(pair) if pair else None


def capacity_gap(
    participation_capacity: float | None,
    weights: np.ndarray,
    panel: PricePanel,
    gross_edge: float | None,
    cost_share_max: float | None,
    window: int = DEFAULT_WINDOW,
) -> CapacityGap:
    """Both capacity estimates and what their difference means, as findings.

    Says "unmeasured" rather than falling back to the participation number when
    the cost side cannot be computed. The participation estimate already exists
    and is already enforced; repeating it here as though it were a cost answer
    would turn a missing measurement into a confirmation.
    """
    if gross_edge is None or cost_share_max is None:
        missing = "gross edge" if gross_edge is None else "cost share"
        return CapacityGap(
            participation_capacity=participation_capacity,
            gross_edge=gross_edge,
            cost_share_max=cost_share_max,
            unmeasured=f"no {missing}, so cost-based capacity is not a number",
        )

    cost, why = cost_capacity(weights, panel, gross_edge, cost_share_max, window=window)
    if cost is None:
        return CapacityGap(
            participation_capacity=participation_capacity,
            gross_edge=gross_edge,
            cost_share_max=cost_share_max,
            unmeasured=why,
        )

    findings: list[str] = []
    if participation_capacity is None:
        findings.append(f"participation capacity is unmeasured; the cost estimate alone says {cost:,.0f}")
    elif cost < participation_capacity:
        ratio = cost / participation_capacity
        findings.append(
            f"cost-based capacity {cost:,.0f} is {ratio:.0%} of the participation-based "
            f"{participation_capacity:,.0f}; the enforced cap is the larger of the two, so "
            f"CLAUDE.md 7항's 80% is being applied to the more forgiving estimate"
        )
    else:
        findings.append(
            f"cost-based capacity {cost:,.0f} exceeds the participation-based "
            f"{participation_capacity:,.0f}, so participation is the binding constraint and "
            "the enforced cap is the conservative one"
        )
    findings.append(
        "linear in size (Amihud), so this charges a large order more than a concave law "
        "would; it is an upper bound on cost and not a forecast of fills"
    )
    return CapacityGap(
        participation_capacity=participation_capacity,
        cost_capacity=cost,
        cost_share_max=cost_share_max,
        gross_edge=gross_edge,
        findings=tuple(findings),
    )
