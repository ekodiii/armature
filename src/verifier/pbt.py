"""vr-pbt-generator — the MVP checker: synthesize -> run -> interpret.

Discharges a property_testable obligation by generating the Hypothesis test,
running it against the real code (the un-fakeable step), and interpreting the
outcome. It bundles the generated source alongside the run output as evidence
(the interpreter only sees the outcome; the source is the parent's to attach).
"""

from .interpreter import interpret_outcome
from .models import ObligationResult, RoutedObligation
from .runner import run_test_program
from .synthesizer import SynthesisError, synthesize


def check(routed: RoutedObligation, *, timeout_s: float = 30.0) -> ObligationResult:
    oid = routed.obligation.obligation_id
    try:
        program = synthesize(routed)
    except SynthesisError as e:
        return ObligationResult(obligation_id=oid, status="error", detail=str(e))

    result = interpret_outcome(
        run_test_program(program, timeout_s=timeout_s), obligation_id=oid
    )
    result.evidence = (
        program.source.rstrip() + "\n# --- run ---\n" + result.evidence
    ).strip()
    return result
