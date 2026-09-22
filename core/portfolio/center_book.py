"""Center book: internal netting and a risk-reducing concentration overlay.

Large platforms do not send every pod's order to the market. A firm-level book
nets offsetting flows first and trims positions the pods have crowded into.
Two pods trading the same name in opposite directions otherwise pay the spread
twice to achieve nothing.

The overlay here only ever reduces risk. Adding to a pod's conviction — which
real center books also do — needs explicit approval and is out of scope until
the platform has a track record.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NettingResult:
    net_targets: dict[str, float]
    gross_pod_turnover: float
    netted_turnover: float
    trimmed: dict[str, float]

    @property
    def turnover_saved(self) -> float:
        if self.gross_pod_turnover == 0:
            return 0.0
        return 1.0 - self.netted_turnover / self.gross_pod_turnover


def net_orders(
    pod_targets: dict[str, dict[str, float]],
    single_name_max: float | None = None,
) -> NettingResult:
    """Combine pod target weights into one book.

    pod_targets: {pod_id: {symbol: target weight of NAV}}
    single_name_max: when given, net exposure per symbol is trimmed to this
        absolute weight. Trimming only shrinks positions.
    """
    net: dict[str, float] = {}
    gross = 0.0
    for targets in pod_targets.values():
        for symbol, weight in targets.items():
            net[symbol] = net.get(symbol, 0.0) + weight
            gross += abs(weight)

    trimmed: dict[str, float] = {}
    if single_name_max is not None:
        for symbol, weight in net.items():
            if abs(weight) > single_name_max:
                capped = single_name_max if weight > 0 else -single_name_max
                trimmed[symbol] = weight - capped
                net[symbol] = capped

    netted = sum(abs(w) for w in net.values())
    return NettingResult(
        net_targets={s: w for s, w in net.items() if w != 0.0},
        gross_pod_turnover=gross,
        netted_turnover=netted,
        trimmed=trimmed,
    )
