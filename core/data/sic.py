"""SIC code -> concentration bucket, so a listings file can classify itself.

`core/data/universe.py` excludes a name with no sector bucket from the order
list, which is right and, with a universe of every listed company, would exclude
almost all of it. Something has to assign the bucket, and the choice is between
buying a classification and deriving one from a field the filings already carry.

**SIC is the SEC's own classification of the filer, not our guess.** Every
company that files with the SEC carries a four-digit Standard Industrial
Classification code assigned at registration, and it is published in the DERA
financial-statement data sets. Using it means the bucket comes from the same
public record as the listing itself, and when it is wrong it is wrong in a way
anybody can check against EDGAR rather than in a way only we can see.

**The mapping is coarse on purpose.** `sector_max` caps a bucket at 20% of NAV.
A classification that splits finely enough never binds, which is the same as
having no limit, so SIC's ~400 industries collapse into the eleven buckets
`core/data/classification.py` already uses plus `communication_services`, which
single stocks need and an ETF universe never did.

**An unmapped code returns None, not a default bucket.** SIC 9995 is literally
"nonclassifiable establishments", and a shell company with no operations is
exactly the kind of name an all-listings universe fills up with. Dropping those
into a catch-all bucket would put unrelated risk under one limit; returning None
sends them down the path a missing bucket already takes -- loaded, not orderable.

The ranges are ordered narrow-first and the first match wins, because the carve
outs are the whole point: 3571 is a computer maker inside a machinery block,
2834 is a drug maker inside a chemicals block, and 8731 is where most of biotech
files. A change to this table moves a concentration limit, so it is reviewed as
a table rather than as code.

**Five of the carve-outs are here because Korea is.** `core/data/ksic.py` maps
KIND's industry labels to these same buckets, and `sector_max` is a fund-level
cap: a battery maker that counts as technology in New York and industrials in
Seoul makes the cap under-measure the concentration it exists to catch. Where
the two tables disagreed, the one that read the business wrong was changed. The
one disagreement left standing is SIC 7370, which puts a search engine and a
systems integrator on the same code; KSIC names 포털 separately and does not
have to guess (ADR-0029).
"""

from __future__ import annotations

#: The buckets this module can produce. `us_equity_broad` and the other
#: fund-level buckets in `classification.SECTORS` describe ETFs and cannot come
#: out of a filer's SIC code, so they are not here.
BUCKETS = (
    "communication_services",
    "consumer_discretionary",
    "consumer_staples",
    "energy",
    "financials",
    "health_care",
    "industrials",
    "materials",
    "real_estate",
    "technology",
    "utilities",
)

#: (low, high, bucket), inclusive, first match wins. Narrow carve-outs precede
#: the broad block they sit inside; reordering this list changes which limit a
#: position counts against.
RANGES: tuple[tuple[int, int, str], ...] = (
    # --- carve-outs, before the blocks that contain them --------------------
    # Added 2026-09-27 when `core/data/ksic.py` was written and five businesses
    # would otherwise have counted against a different limit in Seoul than in
    # New York. `sector_max` is fund-level, so that gap makes the cap
    # under-measure real concentration (ADR-0029).
    (3691, 3692, "industrials"),  # storage and primary batteries -- electrical equipment
    (3661, 3669, "technology"),  # telephone, broadcast and communications equipment
    (4950, 4959, "industrials"),  # sanitary and waste services -- commercial services
    (5140, 5159, "consumer_staples"),  # groceries and farm-product wholesale
    (5180, 5182, "consumer_staples"),  # beer, wine and distilled beverage wholesale
    (2830, 2836, "health_care"),  # pharmaceutical preparations and biologics
    (2840, 2844, "consumer_staples"),  # soap, detergents, cosmetics
    (3570, 3579, "technology"),  # computer and office equipment
    (3670, 3679, "technology"),  # semiconductors and electronic components
    (3680, 3699, "technology"),  # computers, peripherals, electronic systems
    (3710, 3716, "consumer_discretionary"),  # motor vehicles and bodies
    (3751, 3751, "consumer_discretionary"),  # motorcycles and bicycles
    (3840, 3851, "health_care"),  # surgical, medical and dental instruments
    (5122, 5122, "health_care"),  # drugs and druggists' sundries, wholesale
    (5171, 5172, "energy"),  # petroleum wholesale
    (5912, 5912, "consumer_staples"),  # drug stores
    (6798, 6798, "real_estate"),  # real estate investment trusts
    (7370, 7379, "technology"),  # software, data processing, computer services
    (8731, 8731, "health_care"),  # commercial research -- where most biotech files
    # --- the blocks ---------------------------------------------------------
    (100, 999, "consumer_staples"),  # agriculture, forestry, fishing
    (1000, 1099, "materials"),  # metal mining
    (1200, 1299, "energy"),  # coal mining
    (1300, 1399, "energy"),  # oil and gas extraction
    (1400, 1499, "materials"),  # nonmetallic minerals
    (1500, 1799, "industrials"),  # construction
    (2000, 2199, "consumer_staples"),  # food, beverages, tobacco
    (2200, 2399, "consumer_discretionary"),  # textiles and apparel
    (2400, 2499, "materials"),  # lumber and wood
    (2500, 2599, "consumer_discretionary"),  # furniture
    (2600, 2699, "materials"),  # paper
    (2700, 2799, "communication_services"),  # printing and publishing
    (2800, 2899, "materials"),  # chemicals
    (2900, 2999, "energy"),  # petroleum refining
    (3000, 3099, "materials"),  # rubber and plastics
    (3100, 3199, "consumer_discretionary"),  # leather
    (3200, 3399, "materials"),  # stone, clay, glass, primary metals
    (3400, 3569, "industrials"),  # fabricated metal and industrial machinery
    (3580, 3669, "industrials"),  # machinery and electrical equipment
    (3700, 3799, "industrials"),  # transportation equipment
    (3800, 3839, "industrials"),  # measuring and control instruments
    (3852, 3999, "consumer_discretionary"),  # photographic, jewellery, toys
    (4000, 4599, "industrials"),  # rail, trucking, water, air transport
    (4600, 4699, "energy"),  # pipelines
    (4700, 4799, "industrials"),  # transportation services
    (4800, 4899, "communication_services"),  # telephone, broadcasting
    (4900, 4999, "utilities"),  # electric, gas, water, sanitary
    (5000, 5199, "industrials"),  # wholesale
    (5200, 5399, "consumer_discretionary"),  # building supply, general merchandise
    (5400, 5499, "consumer_staples"),  # food stores
    (5500, 5899, "consumer_discretionary"),  # auto dealers, apparel, restaurants
    (5900, 5999, "consumer_discretionary"),  # miscellaneous retail
    (6000, 6499, "financials"),  # banks, credit, securities, insurance
    (6500, 6599, "real_estate"),  # real estate
    (6700, 6799, "financials"),  # holding and investment offices
    (7000, 7099, "consumer_discretionary"),  # hotels
    (7200, 7299, "consumer_discretionary"),  # personal services
    (7300, 7369, "industrials"),  # business services
    (7380, 7399, "industrials"),  # business services
    (7500, 7699, "consumer_discretionary"),  # auto and repair services
    (7800, 7999, "communication_services"),  # motion pictures, entertainment
    (8000, 8099, "health_care"),  # health services
    (8100, 8199, "industrials"),  # legal services
    (8200, 8299, "consumer_discretionary"),  # educational services
    (8300, 8399, "health_care"),  # social services
    (8400, 8499, "consumer_discretionary"),  # museums, galleries
    (8600, 8699, "industrials"),  # membership organisations
    (8700, 8799, "industrials"),  # engineering, accounting, management
    (8900, 8999, "industrials"),  # services, not elsewhere classified
)


def bucket_for_sic(code: int | str | None) -> str | None:
    """The concentration bucket a SIC code counts against, or None if unmapped.

    None covers three real cases and they all want the same treatment: no code
    in the filing, a code outside every block (9995 "nonclassifiable", the 9100
    to 9700 public-administration range that a handful of filers carry), and a
    code that is not a number at all. The name loads and cannot be ordered.
    """
    if code is None or code == "":
        return None
    try:
        value = int(str(code).strip())
    except ValueError:
        return None
    if not 0 < value <= 9999:
        return None
    for low, high, bucket in RANGES:
        if low <= value <= high:
            return bucket
    return None


def unmapped(codes: object) -> tuple[int, ...]:
    """The codes in `codes` that no range covers, for a coverage report."""
    out = []
    for code in codes:  # type: ignore[union-attr]
        if bucket_for_sic(code) is None:
            try:
                out.append(int(str(code).strip()))
            except ValueError:
                continue
    return tuple(sorted(set(out)))
