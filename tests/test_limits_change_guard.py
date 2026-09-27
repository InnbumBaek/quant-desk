from scripts.check_limits_change import violation


def test_unrelated_change_is_fine():
    assert violation(["core/pipeline.py", "tests/test_pipeline_failclosed.py"]) is None


def test_limits_change_without_adr_is_blocked():
    message = violation(["core/risk/limits.yaml"])
    assert message and "limits.yaml" in message


def test_limits_change_with_adr_passes():
    assert violation(["core/risk/limits.yaml", "registry/decisions/ADR-0007-raise-gross.md"]) is None


def test_adr_must_be_a_document():
    assert violation(["core/risk/limits.yaml", "registry/decisions/notes.txt"]) is not None


# --- the classification tables are controlled too (ADR-0029) ------------------
#
# They decide which bucket a position counts against, so a line moved in either
# changes what `sector_max` measures without changing the number it measures
# against. That is the harder of the two changes to notice.


def test_a_us_bucket_table_change_without_a_record_is_blocked():
    message = violation(["core/data/sic.py"])
    assert message and "sic.py" in message


def test_a_korean_bucket_table_change_without_a_record_is_blocked():
    message = violation(["core/data/ksic.py"])
    assert message and "ksic.py" in message


def test_a_bucket_table_change_with_a_record_passes():
    assert (
        violation(
            [
                "core/data/ksic.py",
                "core/data/sic.py",
                "registry/decisions/ADR-0029-korean-industry-labels-become-buckets.md",
            ]
        )
        is None
    )


def test_every_controlled_file_is_named_in_the_message():
    """So a commit touching two of them is not fixed one at a time."""
    message = violation(["core/risk/limits.yaml", "core/data/sic.py", "core/data/ksic.py"])
    assert message
    for name in ("limits.yaml", "sic.py", "ksic.py"):
        assert name in message


def test_the_parser_that_reads_the_labels_is_not_controlled():
    """`core/data/kind.py` reads what the vendor sends and has no opinion about
    what a label means, so a change there cannot move a bucket on its own."""
    assert violation(["core/data/kind.py"]) is None


# --- the gate module is controlled too (ADR-0032) -----------------------------
#
# Not every gate threshold is in the table: `g5_robustness` carries the 30% decay
# and the double-cost sign test in code. CLAUDE.md already required an ADR for a
# change to a gate criterion, and this is the mechanism for the half of that rule
# that does not live in YAML.


def test_a_gate_change_without_a_record_is_blocked():
    message = violation(["core/backtest/gates.py"])
    assert message and "gates.py" in message


def test_a_gate_change_with_a_record_passes():
    assert (
        violation(
            [
                "core/backtest/gates.py",
                "registry/decisions/ADR-0032-a-gate-with-no-code.md",
            ]
        )
        is None
    )


def test_the_modules_that_only_consume_a_verdict_are_not_controlled():
    """`core/pipeline.py` acts on a verdict and cannot change what produced it."""
    assert violation(["core/pipeline.py", "core/backtest/engine.py"]) is None
