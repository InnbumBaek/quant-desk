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
