"""Every declared hypothesis is matched to code, and the code stays inside it.

Two checks that need no market data and would otherwise only be discovered on the
runner. The second is the one that matters: `scripts/submit_alpha.py` compares the
code grid against the declaration at submission time, but that check only runs
where the data is. A grid widened in `library.py` should fail in CI, on the commit
that widens it, rather than on a runner hours later (ADR-0040).
"""

from __future__ import annotations

import pytest
import yaml

from core.alphas import implementations
from core.backtest import prereg
from core.strategies.base import get
from scripts.submit_alpha import chosen_index, grid_within_declaration


def test_every_declared_alpha_has_an_implementation():
    table, problems = implementations.for_all()
    assert not problems, problems
    assert set(table) == set(prereg.declared_ids())


def test_the_map_is_missing_rather_than_empty_when_the_file_is_gone(tmp_path):
    with pytest.raises(implementations.ImplementationError):
        implementations.load(tmp_path / "nowhere.yaml")


def test_a_map_with_no_entries_is_a_refusal(tmp_path):
    path = tmp_path / "_implementations.yaml"
    path.write_text("implementations: {}\n", encoding="utf-8")
    with pytest.raises(implementations.ImplementationError):
        implementations.load(path)


def test_an_entry_with_no_declaration_is_named(tmp_path):
    alphas = tmp_path / "alphas"
    alphas.mkdir()
    (alphas / "a-001.yaml").write_text("id: a-001\n", encoding="utf-8")
    path = tmp_path / "_implementations.yaml"
    path.write_text(
        yaml.safe_dump({"implementations": {"a-001": "ts_momentum", "ghost-001": "ts_momentum"}}),
        encoding="utf-8",
    )
    table, problems = implementations.for_all(alphas, path)
    assert table == {"a-001": "ts_momentum"}
    assert any("ghost-001" in problem for problem in problems)


def test_a_declaration_with_no_entry_is_named(tmp_path):
    alphas = tmp_path / "alphas"
    alphas.mkdir()
    (alphas / "a-001.yaml").write_text("id: a-001\n", encoding="utf-8")
    (alphas / "b-001.yaml").write_text("id: b-001\n", encoding="utf-8")
    path = tmp_path / "_implementations.yaml"
    path.write_text(yaml.safe_dump({"implementations": {"a-001": "ts_momentum"}}), encoding="utf-8")
    _, problems = implementations.for_all(alphas, path)
    assert any("b-001" in problem and "never be run" in problem for problem in problems)


@pytest.mark.parametrize("alpha_id", prereg.declared_ids())
def test_the_code_grid_stays_inside_the_declared_grid(alpha_id):
    """The submission's pre-run check, run in CI where the code changes."""
    declaration = prereg.load(alpha_id)
    assert declaration is not None
    strategy = get(implementations.for_alpha(alpha_id))
    assert not grid_within_declaration(strategy.search_grid(), declaration.parameters_declared)


@pytest.mark.parametrize("alpha_id", prereg.declared_ids())
def test_the_declared_configuration_is_one_point_in_the_code_grid(alpha_id):
    chosen = prereg.declared_chosen(alpha_id)
    assert chosen is not None, f"{alpha_id} declares no chosen_declared"
    strategy = get(implementations.for_alpha(alpha_id))
    assert chosen_index(strategy.search_grid(), chosen) >= 0


@pytest.mark.parametrize("alpha_id", prereg.declared_ids())
def test_every_declaration_is_committed_and_carries_a_hypothesis(alpha_id):
    declaration = prereg.load(alpha_id)
    assert declaration is not None
    assert declaration.committed, f"{declaration.path} is uncommitted or modified"
    assert not declaration.complete, declaration.complete
    assert declaration.declared_trials > 0
