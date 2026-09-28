"""A submission is mostly a set of refusals, so that is mostly what is tested here.

The one thing a submission script must never do is produce a number that looks
like evidence when the declaration behind it does not support it. Every test below
is one way that could happen: a grid wider than the declaration, a reported
configuration nobody declared, a declaration that is still editable, a strategy
this repository cannot honestly run. The happy path is tested too, but it is the
smaller half.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from core.backtest import prereg
from core.backtest.engine import PricePanel
from scripts import submit_alpha

DECLARED = {"lookback": [20, 40, 60, 120, 250], "gross": [0.8]}
CHOSEN = {"lookback": 60.0, "gross": 0.8}


def panel(
    n_rows: int = 700,
    n_cols: int = 5,
    seed: int = 11,
    symbols: tuple[str, ...] | None = None,
) -> PricePanel:
    """A synthetic panel. `symbols` builds one shaped like a declared universe.

    The default five are the legacy universe. An alpha that declares twenty-six
    needs twenty-six columns, because `submit()` restricts the panel to the
    declared names and a missing one is a different universe (ADR-0041).
    """
    names = symbols or ("SPY", "QQQ", "IWM", "TLT", "GLD")[:n_cols]
    n_cols = len(names)
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.011, size=(n_rows, n_cols))
    return PricePanel(
        dates=np.datetime64("2018-01-01") + np.arange(n_rows),
        symbols=tuple(names),
        close=100.0 * np.exp(np.cumsum(steps, axis=0)),
        dollar_volume=np.full((n_rows, n_cols), 8e8),
    )


def declaration(committed: bool = True, **overrides) -> prereg.Preregistration:
    body = {
        "alpha_id": "tsmom-001",
        "economic_rationale": "trend persists because position adjustment is slow",
        "universe": "five US ETFs",
        "horizon": "weeks to months",
        "parameters_declared": dict(DECLARED),
        "committed": committed,
        "path": "registry/alphas/tsmom-001.yaml",
    }
    return prereg.Preregistration(**(body | overrides))


def pin():
    from core.repro import ReproPin

    return ReproPin(git_sha="a" * 40, snapshot_id="b" * 32, seed=0)


# --- the grid may not step outside the declaration -----------------------------


def test_the_registered_grid_matches_the_committed_declaration():
    from core.strategies.base import get

    grid = get("ts_momentum").search_grid()
    assert submit_alpha.grid_within_declaration(grid, DECLARED) == []


@pytest.mark.parametrize(
    ("grid", "fragment"),
    [
        ([{"lookback": 30, "gross": 0.8}], "not among the declared"),
        ([{"lookback": 60, "gross": 0.8, "skip": 5}], "does not mention"),
        ([{"lookback": 60, "gross": 1.0}], "not among the declared"),
    ],
)
def test_a_grid_outside_the_declaration_is_named_not_silently_run(grid, fragment):
    problems = submit_alpha.grid_within_declaration(grid, DECLARED)
    assert problems and any(fragment in problem for problem in problems)


def test_a_declared_axis_that_is_not_a_list_gives_no_n_and_says_so():
    problems = submit_alpha.grid_within_declaration([{"lookback": 60}], {"lookback": 60})
    assert problems and "not a list of values" in problems[0]


# --- the reported configuration comes out of the declaration -------------------


def test_the_declared_configuration_is_located_in_the_grid():
    from core.strategies.base import get

    grid = get("ts_momentum").search_grid()
    index = submit_alpha.chosen_index(grid, CHOSEN)
    assert grid[index]["lookback"] == 60


def test_a_declared_configuration_the_run_never_visits_is_refused():
    with pytest.raises(submit_alpha.NotSubmittable, match="not a point in the grid"):
        submit_alpha.chosen_index([{"lookback": 20}, {"lookback": 40}], {"lookback": 60})


def test_a_duplicated_grid_point_does_not_identify_a_run():
    with pytest.raises(submit_alpha.NotSubmittable, match="matches 2 grid points"):
        submit_alpha.chosen_index([{"lookback": 60}, {"lookback": 60}], {"lookback": 60})


# --- submit() ------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    return submit_alpha.submit(
        alpha_id="tsmom-001",
        strategy_name="ts_momentum",
        panel=panel(),
        pin=pin(),
        declaration=declaration(),
        chosen=CHOSEN,
    )


def test_the_record_carries_the_declaration_it_was_judged_against(record):
    declared = record["declaration"]
    assert declared["declared_trials"] == 5
    assert record["n_trials"] == 5
    assert declared["chosen_declared"] == CHOSEN
    assert record["chosen_params"]["lookback"] == 60


def test_g1_passes_only_when_the_declaration_is_committed_and_unmodified(record):
    verdicts = {v["gate"]: v for v in record["verdicts"]}
    assert verdicts["G1_preregistration"]["passed"]

    editable = submit_alpha.submit(
        alpha_id="tsmom-001",
        strategy_name="ts_momentum",
        panel=panel(),
        pin=pin(),
        declaration=declaration(committed=False),
        chosen=CHOSEN,
    )
    g1 = {v["gate"]: v for v in editable["verdicts"]}["G1_preregistration"]
    assert not g1["passed"]
    assert "uncommitted or modified" in g1["reason"]


def test_every_gate_reports_and_g7_is_among_them(record):
    gates_seen = {v["gate"] for v in record["verdicts"]}
    assert {
        "G0_data",
        "G2_in_sample",
        "G3_oos",
        "G4_statistics",
        "G5_robustness",
        "G6_capacity",
    } <= gates_seen
    assert {"G1_preregistration", "G7_paper"} <= gates_seen


def test_live_capital_is_never_one_run_away(record):
    """G8 is the owner's, so the blocker list can never be empty (ADR-0032)."""
    assert record["live_blockers"]
    assert any("G8_live" in blocker for blocker in record["live_blockers"])
    assert any("G7_paper" in blocker for blocker in record["live_blockers"])


def test_the_record_is_json_with_no_nan(record, tmp_path):
    found: list[str] = []
    written = submit_alpha.json_safe(record, found=found)
    text = json.dumps(written, allow_nan=False, sort_keys=True)
    assert "NaN" not in text


def test_a_grid_wider_than_the_declaration_stops_the_submission():
    narrow = declaration(parameters_declared={"lookback": [20, 40], "gross": [0.8]})
    with pytest.raises(submit_alpha.NotSubmittable, match="steps outside"):
        submit_alpha.submit(
            alpha_id="tsmom-001",
            strategy_name="ts_momentum",
            panel=panel(),
            pin=pin(),
            declaration=narrow,
            chosen=CHOSEN,
        )


def test_a_family_this_repository_cannot_run_is_refused_not_approximated():
    with pytest.raises(submit_alpha.NotSubmittable, match="value"):
        submit_alpha.submit(
            alpha_id="tsmom-001",
            strategy_name="value",
            panel=panel(),
            pin=pin(),
            declaration=declaration(),
            chosen=CHOSEN,
        )


# --- CLAUDE.md rule 6: the signal is in the shared catalogue -------------------


def test_the_submitted_signal_is_registered_beside_the_other_families(record):
    catalogue = record["catalogue"]
    assert "ts_momentum" in catalogue["registered"]
    assert catalogue["submitted_feature_accepted"] is True
    # The de-duplication check is only a check when there is something to be a
    # duplicate of, so more than one family has to be in there.
    assert len(catalogue["registered"]) > 1


def test_an_ensemble_is_not_catalogued_as_a_peer_of_its_own_components(record):
    assert "multi_signal" not in record["catalogue"]["registered"]


def test_the_catalogue_threshold_is_recorded_so_the_rejections_can_be_read(record):
    from core.features.catalog import MAX_ABS_CORRELATION

    assert record["catalogue"]["max_abs_correlation"] == MAX_ABS_CORRELATION


def test_grid_crowding_is_measured_rather_than_assumed(record):
    crowding = record["grid_crowding"]
    assert len(crowding) == 10  # five configurations, ten pairs
    assert all(-1.0 <= value <= 1.0 for value in crowding.values())


# --- markdown ------------------------------------------------------------------


def test_the_summary_leads_with_the_verdict_and_lists_what_blocks_live(record):
    text = submit_alpha.markdown(record)
    assert "tsmom-001" in text
    assert "G8_live" in text
    assert "| G1_preregistration |" in text
    assert ("연구 게이트 통과" in text) or ("기각" in text)


def test_a_rejected_submission_names_the_gates_that_caught_it(record):
    rejected = dict(record) | {"approved": False, "failed_gates": ["G4_statistics"]}
    assert "기각 -- G4_statistics" in submit_alpha.markdown(rejected)


# --- main() refuses before it touches the data --------------------------------


def write_declaration(directory, **hypothesis_overrides):
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "id": "tsmom-001",
        "hypothesis": {
            "economic_rationale": "trend persists",
            "universe": "five US ETFs",
            "horizon": "weeks",
            "parameters_declared": dict(DECLARED),
            "chosen_declared": {"lookback": 60, "gross": 0.8},
        }
        | hypothesis_overrides,
    }
    path = directory / "tsmom-001.yaml"
    path.write_text(yaml.safe_dump(body, allow_unicode=True), encoding="utf-8")
    return path


def write_implementations(directory, table=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "_implementations.yaml"
    path.write_text(
        yaml.safe_dump({"implementations": table or {"tsmom-001": "ts_momentum"}}),
        encoding="utf-8",
    )
    return path


def test_no_declaration_means_no_run_at_all(tmp_path, capsys):
    code = submit_alpha.main(["--alpha", "tsmom-001", "--alphas", str(tmp_path / "alphas")])
    assert code == 2
    assert "no declaration" in capsys.readouterr().out


def test_a_declaration_with_no_reported_configuration_is_refused(tmp_path, capsys):
    alphas = tmp_path / "alphas"
    write_declaration(alphas, chosen_declared=None)
    code = submit_alpha.main(["--alpha", "tsmom-001", "--alphas", str(alphas)])
    assert code == 2
    assert "chosen_declared" in capsys.readouterr().out


def test_missing_price_data_is_a_refusal_not_an_empty_result(tmp_path, capsys):
    alphas = tmp_path / "alphas"
    write_declaration(alphas)
    write_implementations(alphas)
    code = submit_alpha.main(
        ["--alpha", "tsmom-001", "--alphas", str(alphas), "--data", str(tmp_path / "data")]
    )
    assert code == 2
    assert "no CSV files" in capsys.readouterr().out


# --- the declaration's own reader ---------------------------------------------


def test_the_repository_declaration_is_committed_and_declares_five_trials():
    """The file this desk actually submits, read the way the gate reads it."""
    loaded = prereg.load("tsmom-001")
    assert loaded is not None
    assert loaded.declared_trials == 5
    assert loaded.complete == ()
    assert prereg.declared_chosen("tsmom-001") == CHOSEN


# --- the whole desk in one run -------------------------------------------------


def test_all_and_a_named_alpha_are_not_both_a_request(tmp_path):
    assert submit_alpha.main(["--all", "--alpha", "tsmom-001", "--alphas", str(tmp_path)]) == 64


def test_a_declaration_with_no_implementation_stops_the_batch(tmp_path, capsys):
    """Skipping it would leave the desk's trial count above what anyone searched."""
    alphas = tmp_path / "alphas"
    write_declaration(alphas)
    (alphas / "ghost-001.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "ghost-001",
                "hypothesis": {
                    "economic_rationale": "something",
                    "universe": "five US ETFs",
                    "horizon": "weeks",
                    "parameters_declared": dict(DECLARED),
                    "chosen_declared": {"lookback": 60, "gross": 0.8},
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    write_implementations(alphas)
    code = submit_alpha.main(["--all", "--alphas", str(alphas), "--data", str(tmp_path / "data")])
    assert code == 2
    out = capsys.readouterr().out
    assert "ghost-001" in out and "never be run" in out
    assert "no CSV files" not in out, "the batch read data before checking its own wiring"


def test_the_declarations_are_read_before_any_price_data(tmp_path, capsys):
    """Ordering the script's docstring promises: a submission that cannot be
    evaluated honestly stops before it computes anything."""
    alphas = tmp_path / "alphas"
    write_declaration(alphas, chosen_declared=None)
    write_implementations(alphas)
    code = submit_alpha.main(
        ["--alpha", "tsmom-001", "--alphas", str(alphas), "--data", str(tmp_path / "data")]
    )
    assert code == 2
    out = capsys.readouterr().out
    assert "chosen_declared" in out and "no CSV files" not in out


def test_an_empty_registry_is_a_refusal_rather_than_an_empty_batch(tmp_path, capsys):
    alphas = tmp_path / "alphas"
    alphas.mkdir()
    assert submit_alpha.main(["--all", "--alphas", str(alphas)]) == 2
    assert "no alpha is declared" in capsys.readouterr().out


def test_the_batch_table_reports_the_numbers_the_gates_measured(record):
    """Read from the verdicts, not from `performance`: the book's own Sharpe is not
    the in-sample half G2 judged, and printing it under that heading would be a
    different number wearing the right label."""
    table = submit_alpha.batch_markdown([record])
    row = next(line for line in table.splitlines() if line.startswith("| `"))
    metrics = {v["gate"]: v["metrics"] for v in record["verdicts"]}
    assert f"{metrics['G2_in_sample']['is_sharpe']:.2f}" in row
    assert f"{metrics['G4_statistics']['deflated_sharpe_probability']:.3f}" in row
    assert "n/a" not in row, "a measured number was reported as missing"


def test_an_unmeasured_cell_is_not_a_zero(record):
    """The other direction: a metric that could not be computed says so."""
    blank = dict(record)
    blank["verdicts"] = [
        {**v, "metrics": {}} if v["gate"] == "G4_statistics" else v for v in record["verdicts"]
    ]
    row = next(line for line in submit_alpha.batch_markdown([blank]).splitlines() if line.startswith("| `"))
    assert "n/a" in row
    assert "0.000" not in row


def test_every_alpha_this_desk_declares_is_wired_and_preparable():
    """The live registry, read the way `--all` reads it.

    Counted against the declarations rather than a literal: a declaration nobody
    runs inflates the desk's trial count above the search anyone looked at, and a
    number written here would have to be edited every time the desk declares
    again -- which is the edit nobody remembers to make.
    """
    from core.alphas import implementations

    declared = sorted(prereg.declared_ids())
    plan, problems = implementations.for_all()
    assert not problems, problems
    assert sorted(plan) == declared
    prepared, refusals = submit_alpha._prepare(declared, prereg.DEFAULT_DIRECTORY)
    assert not refusals, refusals
    assert sorted(prepared) == declared


# --- an ensemble is measured, not admitted -------------------------------------


def test_an_ensemble_submission_is_not_catalogued_as_its_own_peer():
    """The crash this pair of tests was written for: the submitted signal was
    looked up in the registrations, and an ensemble is deliberately never
    registered, so submitting one raised instead of reporting."""
    from core.strategies.base import get

    strategy = get("multi_signal")
    catalogue = submit_alpha.catalogue_the_signal(panel(700), strategy, strategy.defaults)
    assert catalogue["submitted_is_ensemble"]
    assert catalogue["submitted_feature_accepted"] is None
    assert "multi_signal" not in catalogue["registered"]


def test_an_ensemble_is_measured_against_every_component_it_blends():
    """Its hypothesis is that the legs are decorrelated, so the number that tests
    it has to be in the record."""
    from core.strategies.base import get

    strategy = get("multi_signal")
    catalogue = submit_alpha.catalogue_the_signal(panel(700), strategy, strategy.defaults)
    against = catalogue["submitted_vs_components"]
    assert set(against) == set(catalogue["registered"])
    worst = catalogue["submitted_worst_component_correlation"]
    assert worst is not None
    assert abs(worst) == max(abs(value) for value in against.values())


def test_an_unjudged_signal_is_not_reported_as_admitted():
    """`True` here would read as a de-duplication check that never ran."""
    rendered = submit_alpha._admitted({"submitted_feature_accepted": None})
    assert "not judged" in rendered
    assert submit_alpha._admitted({"submitted_feature_accepted": True}) == "True"


@pytest.mark.parametrize("alpha_id", sorted(prereg.declared_ids()))
def test_every_declared_alpha_survives_a_submission_on_a_synthetic_panel(alpha_id):
    """Each declared family, end to end, without market data.

    The reason this test exists: `submit()` was only ever exercised on one family,
    and the second one the desk declared crashed on the runner instead of being
    judged. A submission path that is only tested where the data is, is tested
    hours after the commit that breaks it (ADR-0040).
    """
    from core.alphas import implementations, universes

    declared = prereg.load(alpha_id)
    chosen = prereg.declared_chosen(alpha_id)
    assert declared is not None and chosen is not None
    symbols, _ = universes.for_alpha(alpha_id)
    record = submit_alpha.submit(
        alpha_id=alpha_id,
        strategy_name=implementations.for_alpha(alpha_id),
        panel=panel(700, symbols=symbols),
        pin=pin(),
        declaration=declared,
        chosen=chosen,
    )
    assert record["alpha_id"] == alpha_id
    assert isinstance(record["approved"], bool)
    # The record has to serialise: a metric that cannot be written is a metric the
    # reviewer never sees.
    json.dumps(submit_alpha.json_safe(record, found=[]), allow_nan=False)
