"""The desk's trial count: every declared grid, and a refusal when one is missing.

The direction is what these tests pin. A smaller N raises every deflated Sharpe
on the desk, so any way of losing a declaration has to end in a refusal rather
than in a smaller number.
"""

from __future__ import annotations

import subprocess

import yaml

from core.backtest import prereg
from core.backtest.trials import desk_trials


def repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    return tmp_path


def declare(directory, alpha_id, grid, *, valid=True):
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "id": alpha_id,
        "hypothesis": {
            "economic_rationale": "because",
            "universe": "five ETFs",
            "horizon": "weeks",
            "parameters_declared": grid,
        },
    }
    path = directory / f"{alpha_id}.yaml"
    path.write_text(
        yaml.safe_dump(body, allow_unicode=True) if valid else "id: [not, a, mapping\n",
        encoding="utf-8",
    )
    return path


def commit(root, message="declare"):
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)


def test_the_count_is_the_sum_of_every_declared_grid(tmp_path):
    root = repo(tmp_path)
    alphas = root / "registry" / "alphas"
    declare(alphas, "a-001", {"lookback": [1, 2, 3], "gross": [0.8]})
    declare(alphas, "b-001", {"lookback": [1, 2], "skip": [0, 5]})
    commit(root)
    result = desk_trials(alphas, repo=root)
    assert result.per_alpha == {"a-001": 3, "b-001": 4}
    assert result.total == 7
    assert result.measured


def test_an_uncommitted_declaration_makes_the_count_unmeasured(tmp_path):
    root = repo(tmp_path)
    alphas = root / "registry" / "alphas"
    declare(alphas, "a-001", {"lookback": [1, 2, 3]})
    commit(root)
    declare(alphas, "b-001", {"lookback": [1, 2]})  # written, never committed
    result = desk_trials(alphas, repo=root)
    assert not result.measured
    assert any("b-001" in reason for reason in result.unusable)
    # The total is not silently the committed part.
    assert result.total == 3 and not result.measured


def test_a_modified_declaration_makes_the_count_unmeasured(tmp_path):
    root = repo(tmp_path)
    alphas = root / "registry" / "alphas"
    path = declare(alphas, "a-001", {"lookback": [1, 2, 3]})
    commit(root)
    path.write_text(path.read_text(encoding="utf-8") + "# touched\n", encoding="utf-8")
    result = desk_trials(alphas, repo=root)
    assert not result.measured
    assert any("uncommitted or modified" in reason for reason in result.unusable)


def test_a_declaration_with_no_grid_is_a_refusal_not_a_zero(tmp_path):
    root = repo(tmp_path)
    alphas = root / "registry" / "alphas"
    declare(alphas, "a-001", {"lookback": [1, 2, 3]})
    declare(alphas, "b-001", {})
    commit(root)
    result = desk_trials(alphas, repo=root)
    assert not result.measured
    assert any("declares no grid" in reason for reason in result.unusable)


def test_a_file_that_does_not_parse_is_named(tmp_path):
    root = repo(tmp_path)
    alphas = root / "registry" / "alphas"
    declare(alphas, "a-001", {"lookback": [1]})
    declare(alphas, "b-001", {}, valid=False)
    commit(root)
    result = desk_trials(alphas, repo=root)
    assert not result.measured
    assert any("b-001" in reason for reason in result.unusable)


def test_no_declaration_at_all_is_not_a_count_of_zero(tmp_path):
    result = desk_trials(tmp_path / "nowhere")
    assert not result.measured
    assert result.total == 0
    assert result.unusable


def test_the_template_and_the_ledger_do_not_contribute(tmp_path):
    root = repo(tmp_path)
    alphas = root / "registry" / "alphas"
    declare(alphas, "a-001", {"lookback": [1, 2]})
    declare(alphas, "_template", {"lookback": [1, 2, 3, 4, 5, 6, 7, 8]})
    (alphas / "lifecycle.yaml").write_text("entries: []\n", encoding="utf-8")
    commit(root)
    result = desk_trials(alphas, repo=root)
    assert result.per_alpha == {"a-001": 2}
    assert prereg.declared_ids(alphas) == ["a-001"]


def test_the_record_round_trips_as_json_safe_data(tmp_path):
    root = repo(tmp_path)
    alphas = root / "registry" / "alphas"
    declare(alphas, "a-001", {"lookback": [1, 2]})
    commit(root)
    payload = desk_trials(alphas, repo=root).as_dict()
    assert payload == {"total": 2, "measured": True, "per_alpha": {"a-001": 2}, "unusable": []}


def test_this_repository_declares_a_measurable_desk_count():
    """The live count, read the way a submission reads it."""
    result = desk_trials()
    assert result.measured, result.unusable
    assert result.per_alpha.get("tsmom-001") == 5
