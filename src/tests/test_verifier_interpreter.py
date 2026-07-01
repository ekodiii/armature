"""vr-result-interpreter regression tests — RawTestOutcome -> ObligationResult.

Covers the verdict mapping: a clean run is 'verified'; a falsification is
'refuted' and carries the counterexample; a timeout and an un-collectable
program are both 'error' (never a silent 'verified'); the obligation_id is
threaded through; and, end to end, a real falsifiable property routed through
the runner comes back 'refuted' with the counterexample intact.
"""

import textwrap

from verifier.models import RawTestOutcome, TestProgram
from verifier.runner import run_test_program
from verifier.interpreter import interpret_outcome


def test_passed_outcome_is_verified():
    outcome = RawTestOutcome(status="passed", returncode=0, stdout="1 passed", stderr="", duration_s=0.4)
    result = interpret_outcome(outcome, obligation_id="ob-1")
    assert result.status == "verified"
    assert result.obligation_id == "ob-1"
    assert result.counterexample is None
    assert "1 passed" in result.evidence


def test_failed_outcome_is_refuted_with_counterexample():
    outcome = RawTestOutcome(
        status="failed",
        returncode=1,
        stdout="Falsifying example: test_p(x=100)",
        stderr="",
        falsifying_example="Falsifying example: test_p(x=100)",
    )
    result = interpret_outcome(outcome, obligation_id="ob-2")
    assert result.status == "refuted"
    assert result.counterexample == "Falsifying example: test_p(x=100)"
    assert "violated" in result.detail


def test_timeout_outcome_is_error_not_verified():
    outcome = RawTestOutcome(status="timeout", returncode=None, stdout="", stderr="", duration_s=2.0)
    result = interpret_outcome(outcome)
    assert result.status == "error"
    assert "time budget" in result.detail


def test_no_tests_collected_is_error_with_helpful_detail():
    outcome = RawTestOutcome(status="error", returncode=5, stdout="no tests ran", stderr="")
    result = interpret_outcome(outcome)
    assert result.status == "error"
    assert "no runnable properties" in result.detail


def test_generic_error_reports_returncode():
    outcome = RawTestOutcome(status="error", returncode=2, stdout="", stderr="boom")
    result = interpret_outcome(outcome)
    assert result.status == "error"
    assert "returncode 2" in result.detail
    assert "boom" in result.evidence


def test_end_to_end_refute_carries_counterexample():
    """The two engine halves together: run a falsifiable property, interpret it."""
    program = TestProgram(
        obligation_id="always-small",
        source=textwrap.dedent(
            """
            from hypothesis import given, strategies as st

            @given(st.integers())
            def test_always_small(x):
                assert x < 100
            """
        ),
    )
    result = interpret_outcome(run_test_program(program, timeout_s=60), obligation_id="always-small")
    assert result.status == "refuted"
    assert result.counterexample is not None
    assert "x=100" in result.counterexample
    assert result.obligation_id == "always-small"
