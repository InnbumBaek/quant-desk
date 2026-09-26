"""The prober's job is to never lie about what a host said, and to never raise.

Whether a given host answers is settled on the runner; these fix what the
prober records when it does and when it does not.
"""

from __future__ import annotations

import email.message
import gzip
import io
import urllib.error

import pytest

from scripts.probe_hosts import (
    TARGETS,
    USER_AGENT,
    Target,
    as_markdown,
    probe,
    run,
)


class Response:
    def __init__(self, body: bytes, status: int = 200, headers: dict[str, str] | None = None, url=""):
        self._body, self.status, self.url = body, status, url
        self.headers = email.message.Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value

    def read(self, size: int | None = None) -> bytes:
        return self._body if size is None else self._body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def http_error(code: int, body: bytes, headers: dict[str, str] | None = None):
    message = email.message.Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError("https://x.test/", code, "refused", message, io.BytesIO(body))


def serving(result):
    def opener(request, timeout=None):
        if isinstance(result, Exception):
            raise result
        return result

    return opener


TARGET = Target(label="x", url="https://x.test/", expect="<?xml", note="a test target")


# --- the happy path ---------------------------------------------------------


def test_a_good_answer_is_recorded_as_ok():
    outcome = probe(TARGET, opener=serving(Response(b"<?xml version='1.0'?><feed/>")))
    assert outcome.looks_right is True
    assert outcome.verdict == "ok"
    assert outcome.status == 200


def test_a_gzipped_answer_is_read_before_it_is_judged():
    packed = gzip.compress(b"<?xml version='1.0'?><feed/>")
    response = Response(packed, headers={"Content-Encoding": "gzip"})
    assert probe(TARGET, opener=serving(response)).looks_right is True


def test_the_probe_declares_who_we_are():
    seen = {}

    def opener(request, timeout=None):
        seen.update(request.headers)
        return Response(b"<?xml?>")

    probe(TARGET, opener=opener)
    headers = {key.lower(): value for key, value in seen.items()}
    assert "quant-desk" in headers["user-agent"]
    assert "Mozilla" not in headers["user-agent"]


def test_no_target_is_probed_as_a_browser():
    """A 200 obtained by lying sends the next adapter down a path that breaks in earnest."""
    assert "Mozilla" not in USER_AGENT
    assert "https://" in USER_AGENT


# --- the refusals, which are the point --------------------------------------


def test_a_refusal_keeps_the_status_and_the_body():
    outcome = probe(TARGET, opener=serving(http_error(403, b"Request Rate Threshold Exceeded")))
    assert outcome.status == 403
    assert "Rate Threshold" in outcome.body_head
    assert outcome.looks_right is False


def test_an_empty_refusal_says_that_the_host_explained_nothing():
    """arXiv's 406 carries no body; a report that hides that hides the whole problem."""
    outcome = probe(TARGET, opener=serving(http_error(406, b"")))
    assert "empty body" in outcome.verdict


def test_an_unreachable_host_is_recorded_rather_than_raised():
    outcome = probe(TARGET, opener=serving(urllib.error.URLError("no route")))
    assert outcome.status is None
    assert outcome.verdict == "unreachable"
    assert "URLError" in outcome.reason


def test_a_200_carrying_an_error_page_is_not_a_success():
    """This is the case an adapter accepts silently and turns into an empty universe."""
    outcome = probe(TARGET, opener=serving(Response(b"<html>Access Denied</html>")))
    assert outcome.looks_right is False
    assert "not what we asked for" in outcome.verdict


def test_a_waf_that_signs_its_refusal_is_recorded():
    response = http_error(403, b"blocked", {"Server": "AkamaiGHost", "X-Served-By": "cache-icn"})
    outcome = probe(TARGET, opener=serving(response))
    assert outcome.headers_of_interest["server"] == "AkamaiGHost"


def test_a_redirect_is_noted_because_it_changes_what_was_measured():
    response = Response(b"<?xml?>", url="https://x.test/elsewhere")
    assert probe(TARGET, opener=serving(response)).redirected_to == "https://x.test/elsewhere"


# --- the report -------------------------------------------------------------


def test_the_report_splits_reachable_from_refused(monkeypatch):
    targets = (
        Target(label="good", url="https://good.test/", expect="ok"),
        Target(label="bad", url="https://bad.test/", expect="ok"),
    )
    answers = {"https://good.test/": Response(b"ok"), "https://bad.test/": http_error(406, b"")}

    def opener(request, timeout=None):
        answer = answers[request.full_url]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("scripts.probe_hosts.urllib.request.urlopen", opener)
    report = run(targets)
    assert report["reachable"] == ["good"]
    assert report["refused"] == ["bad"]


def test_a_failing_control_invalidates_the_whole_run(monkeypatch):
    """If our own network is the problem, every other line in the report is noise."""
    targets = (Target(label="control", url="https://c.test/", expect="ok", control=True),)
    monkeypatch.setattr("scripts.probe_hosts.urllib.request.urlopen", serving(urllib.error.URLError("down")))
    report = run(targets)
    assert report["control_ok"] is False
    assert "measured our own network" in as_markdown(report)


def test_the_markdown_survives_a_body_containing_a_pipe():
    report = {
        "control_ok": True,
        "results": [
            {
                "label": "x",
                "verdict": "ok",
                "status": 200,
                "elapsed_ms": 12,
                "body_head": "Symbol|Security Name",
                "reason": "",
            }
        ],
    }
    row = as_markdown(report).splitlines()[-1]
    assert "Symbol\\|Security Name" in row
    assert row.count("|") - row.count("\\|") == 6  # five cells, and the body's pipe escaped


# --- the table itself -------------------------------------------------------


def test_the_table_has_a_control():
    """Without one, a run where everything failed cannot be told from a broken runner."""
    assert any(target.control for target in TARGETS)


def test_every_target_says_why_it_is_in_the_table():
    for target in TARGETS:
        assert target.note, f"{target.label} has no note"


@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.label)
def test_every_target_is_an_https_url(target):
    assert target.url.startswith("https://")


def test_no_target_carries_a_credential():
    """The report is committed, so a probe URL with a key in it is a disclosed key."""
    for target in TARGETS:
        lowered = target.url.lower()
        assert "key=" not in lowered
        assert "token=" not in lowered
        assert "@" not in lowered.split("://", 1)[1].split("/", 1)[0]
