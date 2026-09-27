"""Yahoo's sector label -> concentration bucket, as the fallback that is reachable.

`core/data/sic.py` is still the preferred source and this does not replace it.
A filer's SIC comes from its own SEC registration, so a wrong bucket is wrong in
a way anybody can check against EDGAR. This one is a vendor's opinion.

It exists because every better source refuses this runner. Measured in one
minute rather than argued (registry/probes/2026-09-27.json): `www.sec.gov` and
`data.sec.gov` both 403, `api.nasdaq.com` and `www.nasdaq.com` do not answer at
all, and `stooq.com` serves a JavaScript challenge. What does answer from the
runner is `query1.finance.yahoo.com`, which is already where the price snapshot
comes from (ADR-0007). Without a bucket nothing can be ordered, so an all-symbol
universe with no sector source is an all-symbol universe that trades nothing.

The same two things keep it honest as with any vendor. The universe's sidecar
records which source each bucket came from, so a vendor label is never mistaken
for a filing. And the mapping is declared here, in the same eleven buckets the
SIC table uses, rather than passing the vendor's own strings through -- a bucket
name that appears in a concentration limit is ours to name.

Yahoo's own vocabulary differs from ours in three places and the differences are
the point: "Consumer Cyclical" and "Consumer Defensive" are discretionary and
staples, and "Financial Services" is financials. Mapping them is a judgement we
are making explicitly here rather than leaving to whoever reads a limit report.
"""

from __future__ import annotations

#: Yahoo's sector strings, lower-cased, to our buckets. A label not in here is
#: unmapped rather than guessed, and `unmapped_labels` reports it so a vendor
#: renaming a sector shows up as a coverage drop instead of silence.
YAHOO_SECTORS: dict[str, str] = {
    "basic materials": "materials",
    "communication services": "communication_services",
    "consumer cyclical": "consumer_discretionary",
    "consumer defensive": "consumer_staples",
    "energy": "energy",
    "financial services": "financials",
    "healthcare": "health_care",
    "industrials": "industrials",
    "real estate": "real_estate",
    "technology": "technology",
    "utilities": "utilities",
    # Deliberately absent: the empty label, which is what a fund, a trust, a
    # SPAC or a shell carries. That means "no bucket", not "other", and an
    # all-symbol universe holds a great many of them.
}


def bucket_for_yahoo_sector(label: str | None) -> str | None:
    """The bucket a vendor sector label counts against, or None if unmapped."""
    if not label:
        return None
    return YAHOO_SECTORS.get(str(label).strip().lower())


def unmapped_labels(labels: object) -> tuple[str, ...]:
    """The distinct labels no bucket covers, for the coverage report."""
    out = set()
    for label in labels:  # type: ignore[union-attr]
        text = str(label or "").strip()
        if text and bucket_for_yahoo_sector(text) is None:
            out.add(text)
    return tuple(sorted(out))
