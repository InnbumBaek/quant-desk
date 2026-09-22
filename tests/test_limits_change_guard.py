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
