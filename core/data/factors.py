"""Reading a factor file, and lining it up with a panel's bars.

G4 regresses a strategy's net returns on factor returns and reads the intercept's
t-statistic. Two things decide whether that number means anything, and neither is
visible in the output if it goes wrong.

**The alignment.** `panel.bar_returns[t]` is the return from `dates[t]` to
`dates[t + 1]`, so the factor return that belongs beside it is the one earned on
`dates[t + 1]` -- the day the bar closed. Ken French dates a daily factor return
by the day it was earned, so the join is on `dates[1:]`, not `dates[:-1]`. Off by
one in either direction still produces a t-statistic, a plausible one, and
nothing anywhere says the regression compared a strategy's Tuesday with the
market's Monday. `tests/data/test_factors.py` pins the convention by feeding the
panel's own market return back in as a factor and requiring the residual to
vanish.

**The holes.** A factor file covers the days its vendor published. A panel date
with no factor row, or a row the vendor marked missing, is a hole, and there is
no honest way to fill it: a zero is a fabricated observation and dropping the bar
silently changes the sample the t-statistic describes. So a hole raises. The
caller can then trim the panel, which is a decision that leaves a trace, rather
than inherit a matrix that quietly disagrees with the returns beside it.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.backtest.engine import PricePanel

#: The market the Ken French library describes. A US factor model does not price a
#: Korean book: regressing one on the other would return a t-statistic about
#: nothing. Other markets keep the panel proxy until they have a factor source.
FRENCH_MARKET = "US"


@dataclass(frozen=True)
class FactorPanel:
    """Factor returns in decimals, one row per day they were earned."""

    dates: np.ndarray  # datetime64[D]
    names: tuple[str, ...]
    values: np.ndarray  # (n_dates, n_factors), NaN where the vendor had no value
    source: str
    digest: str

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("factor values must be a 2-D (dates x factors) matrix")
        if self.values.shape != (self.dates.shape[0], len(self.names)):
            raise ValueError(
                f"values {self.values.shape} do not match {self.dates.shape[0]} dates "
                f"and {len(self.names)} factors"
            )

    def drop(self, names: tuple[str, ...]) -> FactorPanel:
        """Without the named columns, for the ones that are not risk factors."""
        keep = [i for i, name in enumerate(self.names) if name not in names]
        if not keep:
            raise ValueError(f"dropping {names} would leave no factors")
        return FactorPanel(
            dates=self.dates,
            names=tuple(self.names[i] for i in keep),
            values=self.values[:, keep],
            source=self.source,
            digest=self.digest,
        )

    def align_to_bars(self, panel_dates: np.ndarray) -> np.ndarray:
        """The factor matrix for a panel's bar returns: row `t` is `panel_dates[t + 1]`.

        Raises on any bar whose closing day has no usable factor row. See the
        module docstring for why a hole is not filled.
        """
        if panel_dates.ndim != 1 or panel_dates.shape[0] < 2:
            raise ValueError("a panel needs at least two dates to have a bar")
        wanted = [str(day) for day in panel_dates.astype("datetime64[D]")[1:]]
        position = {str(day): index for index, day in enumerate(self.dates.astype("datetime64[D]"))}

        absent = [day for day in wanted if day not in position]
        if absent:
            raise ValueError(
                f"{len(absent)} of {len(wanted)} bar dates have no factor row "
                f"(first: {', '.join(absent[:5])}); the factor file covers "
                f"{self.dates[0]}..{self.dates[-1]}"
            )

        rows = [position[day] for day in wanted]
        matrix = self.values[rows, :]
        if not np.all(np.isfinite(matrix)):
            holes = [
                f"{wanted[i]} {self.names[j]}" for i, j in zip(*np.where(~np.isfinite(matrix)), strict=True)
            ][:5]
            raise ValueError(
                f"the factor file has {int(np.sum(~np.isfinite(matrix)))} missing value(s) inside "
                f"the panel's range (first: {', '.join(holes)}); a missing factor return is not a zero"
            )
        return matrix

    def coverage_slice(self, panel_dates: np.ndarray) -> tuple[int, int]:
        """The `[start, stop)` slice of panel dates whose bars this file can price.

        A factor file lags its vendor by days or weeks, so a panel fetched today
        usually runs past the end of it. Refusing the whole panel for that would
        mean the factor model never gets used; silently regressing the covered
        part against all the returns would be worse. So the caller gets a slice to
        trim to, which is a decision it can record.

        An uncovered day *inside* the range is different: that is a hole in the
        vendor's file, not a lag, and it raises. Trimming around it would drop
        real bars from the middle of a sample without saying so.
        """
        if panel_dates.ndim != 1 or panel_dates.shape[0] < 2:
            raise ValueError("a panel needs at least two dates to have a bar")
        usable = {
            str(day)
            for day, row in zip(self.dates.astype("datetime64[D]"), self.values, strict=True)
            if np.all(np.isfinite(row))
        }
        days = [str(day) for day in panel_dates.astype("datetime64[D]")]
        covered = [index for index, day in enumerate(days) if day in usable]
        if len(covered) < 3:
            raise ValueError(
                f"the factor file covers {len(covered)} of the panel's {len(days)} dates, "
                f"too few to run; it spans {self.dates[0]}..{self.dates[-1]}"
            )
        start, stop = covered[0], covered[-1] + 1
        gaps = [days[index] for index in range(start, stop) if days[index] not in usable]
        if gaps:
            raise ValueError(
                f"{len(gaps)} date(s) inside the factor file's range have no usable row "
                f"(first: {', '.join(gaps[:5])}); that is a hole in the file, not a lag"
            )
        return start, stop

    def covers(self, panel_dates: np.ndarray) -> bool:
        """Whether `align_to_bars` would succeed, for a caller deciding to use it."""
        try:
            self.align_to_bars(panel_dates)
        except ValueError:
            return False
        return True


def load_factors(path: Path, source: str = "ken-french-data-library") -> FactorPanel:
    """Read the normalised CSV `scripts/fetch_factors.py` writes.

    That file is already in decimals; the percent conversion happens once, in the
    fetcher. An empty field means the vendor published no value and becomes NaN,
    which `align_to_bars` refuses rather than treats as zero.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or header[0].strip() != "Date" or len(header) < 2:
            raise ValueError(f"{path.name} does not start with a Date column and factors: {header}")
        names = tuple(field.strip() for field in header[1:])
        days: list[str] = []
        rows: list[list[float]] = []
        for line in reader:
            if not line or not line[0].strip():
                continue
            if len(line) != len(names) + 1:
                raise ValueError(f"{path.name} row {line[0]} has {len(line) - 1} values for {len(names)}")
            day = line[0].strip()
            if days and day <= days[-1]:
                raise ValueError(f"{path.name} is not in ascending date order at {day}")
            days.append(day)
            rows.append([float("nan") if not field.strip() else float(field) for field in line[1:]])
    if len(days) < 2:
        raise ValueError(f"{path.name} holds {len(days)} rows; a factor file needs a history")

    return FactorPanel(
        dates=np.array(days, dtype="datetime64[D]"),
        names=names,
        values=np.array(rows, dtype=float),
        source=source,
        digest=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def trim_panel_to_factors(panel: PricePanel, factors: FactorPanel) -> tuple[PricePanel, int]:
    """The part of a panel the factor file can price, and how many bars that cost.

    Returns the panel itself and 0 when the file already covers it. The count is
    what a run record has to carry: a Sharpe measured on a trimmed sample is a
    different number from one measured on the whole panel, and the difference is
    not visible in either.
    """
    start, stop = factors.coverage_slice(panel.dates)
    if (start, stop) == (0, panel.dates.shape[0]):
        return panel, 0
    trimmed = PricePanel(
        dates=panel.dates[start:stop],
        symbols=panel.symbols,
        close=panel.close[start:stop, :],
        dollar_volume=None if panel.dollar_volume is None else panel.dollar_volume[start:stop, :],
    )
    return trimmed, panel.dates.shape[0] - trimmed.dates.shape[0]
