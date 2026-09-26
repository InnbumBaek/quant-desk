"""Nasdaq's own sector label -> concentration bucket, as the fallback to SIC.

`core/data/sic.py` is the preferred source and this is not a replacement for it.
SIC comes from the filer's own SEC registration, so a wrong bucket is wrong in a
way anybody can check against EDGAR. This one is a vendor's opinion.

It exists because the preferred source is unreachable. `www.sec.gov` refused
fourteen consecutive requests from the GitHub Actions address range across an
hour and four separate runs, including a fresh runner at a different time
(ADR-0019). Without a bucket nothing can be ordered, so an all-symbol universe
with no sector source is an all-symbol universe that trades nothing.

Two things keep this honest. The universe's sidecar records which source each
bucket came from, so a vendor label is never mistaken for a filing. And the
mapping is declared here, in the same coarse eleven-plus-one buckets as the SIC
table, rather than passing the vendor's own strings through -- a bucket name
that appears in a concentration limit is ours to name.

`Miscellaneous` maps to nothing. It is Nasdaq's "we did not classify this", and
reading somebody else's shrug as a sector is how unrelated risk ends up under
one limit.
"""

from __future__ import annotations

#: Nasdaq's sector strings, lower-cased, to our buckets. A label not in here is
#: unmapped rather than guessed, and `unmapped_labels` reports it so a vendor
#: renaming a sector shows up as a coverage drop instead of silence.
NASDAQ_SECTORS: dict[str, str] = {
    "basic materials": "materials",
    "consumer discretionary": "consumer_discretionary",
    "consumer staples": "consumer_staples",
    "energy": "energy",
    "finance": "financials",
    "health care": "health_care",
    "industrials": "industrials",
    "real estate": "real_estate",
    "technology": "technology",
    "telecommunications": "communication_services",
    "utilities": "utilities",
    # Deliberately absent: "miscellaneous" and the empty label, which is what
    # the screener carries for funds. Both mean "no bucket", not "other".
}


def bucket_for_nasdaq_sector(label: str | None) -> str | None:
    """The bucket a vendor sector label counts against, or None if unmapped."""
    if not label:
        return None
    return NASDAQ_SECTORS.get(str(label).strip().lower())


def unmapped_labels(labels: object) -> tuple[str, ...]:
    """The distinct labels no bucket covers, for the coverage report."""
    out = set()
    for label in labels:  # type: ignore[union-attr]
        text = str(label or "").strip()
        if text and bucket_for_nasdaq_sector(text) is None:
            out.add(text)
    return tuple(sorted(out))
