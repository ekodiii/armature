"""vr-test-runner regression tests — the un-fakeable execution step.

Covers the contract TestProgram -> RawTestOutcome: a property that holds
reports 'passed'; a property Hypothesis can falsify reports 'failed' with the
counterexample extracted; import_roots let the generated test exercise REAL
code in a subprocess; a runaway program is cut off as 'timeout'; an empty
program (no tests collected) is reported as 'error', not a silent pass.
"""

import textwrap
from pathlib import Path

from verifier.models import TestProgram
from verifier.runner import run_test_program


def test_passing_property_reports_passed():
    program = TestProgram(
        obligation_id="add-commutes",
        source=textwrap.dedent(
            """
            from hypothesis import given, strategies as st

            @given(st.integers(), st.integers())
            def test_add_commutes(a, b):
                assert a + b == b + a
            """
        ),
    )
    outcome = run_test_program(program, timeout_s=60)
    assert outcome.status == "passed"
    assert outcome.returncode == 0
    assert outcome.falsifying_example is None


def test_falsifiable_property_reports_failed_with_counterexample():
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
    outcome = run_test_program(program, timeout_s=60)
    assert outcome.status == "failed"
    assert outcome.returncode == 1
    assert outcome.falsifying_example is not None
    assert "Falsifying example" in outcome.falsifying_example
    # Hypothesis shrinks to the boundary value.
    assert "x=100" in outcome.falsifying_example


def test_import_roots_exercise_real_code(tmp_path):
    # A real module the generated test must import and property-check.
    module_dir = tmp_path / "pkg"
    module_dir.mkdir()
    (module_dir / "leaf.py").write_text(
        textwrap.dedent(
            """
            def double(x):
                return x + x
            """
        )
    )
    program = TestProgram(
        obligation_id="leaf-double",
        import_roots=[str(module_dir)],
        source=textwrap.dedent(
            """
            from hypothesis import given, strategies as st
            from leaf import double

            @given(st.integers())
            def test_double_is_twice(x):
                assert double(x) == 2 * x
            """
        ),
    )
    outcome = run_test_program(program, timeout_s=60)
    assert outcome.status == "passed", outcome.stdout + outcome.stderr
    assert outcome.returncode == 0


def test_runaway_program_times_out():
    program = TestProgram(
        obligation_id="slow",
        source=textwrap.dedent(
            """
            import time

            def test_slow():
                time.sleep(30)
            """
        ),
    )
    outcome = run_test_program(program, timeout_s=2)
    assert outcome.status == "timeout"
    assert outcome.returncode is None


def test_empty_program_reports_error_not_pass():
    program = TestProgram(
        obligation_id="empty",
        source="x = 1  # no test functions here\n",
    )
    outcome = run_test_program(program, timeout_s=60)
    assert outcome.status == "error"
    assert outcome.returncode == 5
