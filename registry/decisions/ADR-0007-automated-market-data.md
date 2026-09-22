# ADR-0007 — Automated market data: fetch in CI, commit only the manifest

- Date: 2026-09-22
- Status: accepted
- Context: the owner asked whether market data can be pulled in automatically.
  ADR-0006 built the file-based loader because Claude's research container has no
  route to any vendor host. That constraint is real but narrow: it is the
  *container's* egress policy, not GitHub's.
- Authorised by: the owner's standing delegation ("너가 판단한대로 해, 근데 어떠한
  판단에는 항상 근거가 있어야 해") and their question of 2026-09-22
  ("시세 데이터를 자동으로 끌고 올 수 없나?").

## What was measured, not assumed

From the research container, every market-data host tested is refused by the
egress proxy at the CONNECT stage: `stooq.com`, `query1.finance.yahoo.com`,
`www.alphavantage.co`, `api.tiingo.com`, `data.nasdaq.com`, `fred.stlouisfed.org`,
`mba.tuck.dartmouth.edu` (the Ken French library), `www.sec.gov`,
`api.polygon.io`. Only `pypi.org` (200) and `raw.githubusercontent.com` (301)
answer. So no amount of code makes a live fetch work *here*.

A GitHub Actions runner is a different network. That is where the fetch belongs.

## Decision

`.github/workflows/data-snapshot.yml` fetches on the runner —
`workflow_dispatch` for a manual run, a weekly `schedule`, and `push` on the
fetcher's own paths so a change to it is exercised immediately.

`scripts/fetch_prices.py` writes per-symbol daily CSVs into git-ignored `data/`
in exactly the shape `core/data/sources.py` reads. Two keyless sources are
implemented, `stooq` and `yahoo`, and `--source auto` tries them in order:
a run that needs no secret is a run the owner does not have to set up before it
works. Adding a keyed vendor later is one function plus a repository secret.

`scripts/data_snapshot.py` then loads the panel, takes the reproducibility pin
(ADR-0006), writes the manifest, and runs one throwaway momentum strategy through
G0–G6 so the path from bytes to verdict is exercised end to end rather than
assumed.

## The licence rule: prices stay on the runner

This repository is **public** (ADR-0001). That decides the data question, because
on a public repository anything committed *and any workflow artifact* is a public
download — so uploading prices as an artifact is redistribution just as much as
committing them is.

No free EOD equity source permits redistribution. Stooq and Yahoo both provide
data for the user's own use; Alpha Vantage, Tiingo and Polygon all prohibit
redistribution in their terms; the one CC-licensed US equity set (Nasdaq Data
Link's Wiki EOD) was discontinued in 2018 and is useless for current research.
Public-domain sources exist (SEC EDGAR filings, Federal Reserve series) but they
are not price panels.

So the rule is: **the vendor's bytes never leave the runner.** `data/` is
git-ignored, the workflow has no `upload-artifact` step, and what gets committed
is:

- `registry/snapshots/<id>.json` — the manifest: file hashes, symbols, date
  range, dropped dates, row counts. Facts *about* the data, containing none of it.
- `registry/snapshots/<id>.smoke.json` — gate verdicts and metrics. Derived
  statistics, which are our output, not the vendor's.

That is enough to reproduce a result: anyone can re-fetch from the same source
for the same dates and check the hashes match. It is the same discipline the
audit log uses — keep the proof, not the copy.

## Alternatives rejected

- **Commit the price CSVs.** Rejected: redistribution on a public repo, against
  every candidate vendor's terms, and it makes the repository grow without bound.
- **Upload the CSVs as a workflow artifact.** Rejected for the same reason —
  artifacts of a public repository are public downloads. This one is easy to miss,
  which is why it is written down here.
- **Make the repository private to allow committing data.** Rejected: public was
  a deliberate decision (ADR-0001) whose point is that no alpha here depends on
  secrecy. Data licensing is not a reason to reverse it.
- **Require a paid vendor key up front.** Rejected as the default: it makes the
  automated path depend on the owner buying something before anything runs. The
  keyless sources are enough to prove the pipeline; a keyed vendor is the upgrade
  when paper trading starts, and the plan already names Polygon.io for that.
- **Fetch inside the research container through a different route.** Rejected:
  the block is the environment's network policy. The owner can change it when
  creating the environment; working around it is not ours to do.
- **Cache the data in the repository between runs to save vendor calls.** Rejected
  for the licence reason above. Each run re-fetches; the manifest is what persists.

## Honest limits

- **Free sources are not survivorship-corrected and not corporate-action
  perfect.** Stooq and Yahoo give adjusted-ish closes for symbols that exist
  today. That is fine for ETFs (the default universe is SPY/QQQ/IWM/TLT/GLD) and
  not fine for a single-name equity cross-section, which is what G0's minimum
  history and survivorship checks exist to catch. A real equity panel needs a
  paid vendor.
- **A keyless endpoint can disappear or start refusing datacenter IPs.** That is
  why the fetcher refuses anything that is not the expected shape and why `auto`
  falls through to a second source. A run that cannot get a complete panel fails;
  it never writes a short one.
- **G4's factor regression is still a panel proxy, not FF5+MOM.** The Ken French
  library is free but unreachable from here; fetching it on the runner is the
  obvious next step, and until then every submission carries the
  `PANEL_PROXY` note (ADR-0004).

## Verification

`tests/test_fetch_prices.py` (16 tests), all offline against recorded response
bodies — a test that needed the network would fail here for the wrong reason.
The load-bearing ones:
`test_a_rate_limit_notice_is_an_error_not_an_empty_panel` and
`test_an_html_error_page_is_an_error` (a vendor's error text must not become a
price history), `test_yahoo_padding_rows_are_dropped_not_filled`,
`test_a_partial_panel_is_not_accepted` (four of five symbols is a different
universe, so it is not success), `test_auto_falls_through_to_the_next_source`,
and `test_written_csv_loads_as_a_panel` (the fetcher's output is what the loader
reads, pinned rather than hoped for).

The full path was also run locally against a synthetic panel of 5 symbols and
286 rows: manifest written, pin taken, and the smoke strategy rejected by
G2/G3/G4/G5 with reasons — which is the expected outcome for a throwaway
strategy and the proof that the wiring is real.
