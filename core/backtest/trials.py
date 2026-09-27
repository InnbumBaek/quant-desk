"""How many configurations this desk searched, across every alpha it declared.

G1 counts the trials **one** alpha declared and G4 deflates its Sharpe by that
count. Both are right about one alpha and both miss the thing a fund does: run
six families, and allocate to whichever one clears. The selection that matters is
then over every configuration of every family, not over the five this submission
happens to carry, and a Sharpe deflated by 5 when the desk searched 30 is
deflated by the wrong number in the flattering direction (ADR-0039).

**Why the count comes from the declarations rather than from the runs.** A run
records what it searched; the declarations record what the desk *committed to*
searching, before any of it ran, and they are committed files so the count is
pinned by the git SHA like every other input. A submission's desk count is
therefore reproducible from its own pin, which a count assembled from whatever
runs happened to exist at that moment would not be.

**Why an unreadable declaration makes the count unmeasured rather than smaller.**
Skipping a declaration lowers N, and a lower N is the forgiving direction: it
raises the deflated Sharpe with nothing about any strategy having changed. So one
declaration that will not load, or is uncommitted, or declares an empty grid,
makes the whole desk count a refusal (CLAUDE.md 4항, and the same reasoning
`core/risk/cost.py` gives for an uncharged pod).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.backtest import prereg


@dataclass(frozen=True)
class DeskTrials:
    """The desk's total declared trial count, or the reasons it is not a number."""

    total: int = 0
    per_alpha: dict[str, int] = field(default_factory=dict)
    unusable: tuple[str, ...] = ()

    @property
    def measured(self) -> bool:
        """True only when every declaration contributed a count.

        One missing contribution means the total is an undercount, and an
        undercount is the direction that flatters every alpha on the desk.
        """
        return not self.unusable and self.total > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "measured": self.measured,
            "per_alpha": dict(sorted(self.per_alpha.items())),
            "unusable": list(self.unusable),
        }


def desk_trials(alphas: Path | None = None, repo: Path | None = None) -> DeskTrials:
    """Sum the declared grids of every alpha this repository has declared."""
    directory = alphas or prereg.DEFAULT_DIRECTORY
    per_alpha: dict[str, int] = {}
    unusable: list[str] = []

    ids = prereg.declared_ids(directory)
    if not ids:
        return DeskTrials(0, {}, ("no alpha is declared, so there is no desk-wide count",))

    for alpha_id in ids:
        try:
            declaration = prereg.load(alpha_id, directory=directory, repo=repo)
        except prereg.PreregistrationError as error:
            unusable.append(f"{alpha_id}: {error}")
            continue
        if declaration is None:
            unusable.append(f"{alpha_id}: has a file that does not load as a declaration")
            continue
        if not declaration.committed:
            unusable.append(
                f"{alpha_id}: {declaration.path} is uncommitted or modified, so its grid could "
                "have been trimmed after the fact"
            )
            continue
        count = declaration.declared_trials
        if count <= 0:
            unusable.append(f"{alpha_id}: declares no grid, so it contributes an unknown count")
            continue
        per_alpha[alpha_id] = count

    return DeskTrials(sum(per_alpha.values()), per_alpha, tuple(unusable))
