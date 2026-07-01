"""vr-result-interpreter — turn a RawTestOutcome into an ObligationResult.

The rendering step: the runner reported what pytest mechanically did; here we
say what it MEANS for the obligation. A clean run is 'verified'; a falsification
is 'refuted' with the counterexample carried through; a timeout or a run that
never produced a real verdict is 'error' -- never silently 'verified'.

The generated test source is not attached here (the interpreter only sees the
outcome); the parent vr-pbt-generator bundles the source alongside this verdict.
"""

from .models import ObligationResult, RawTestOutcome


def interpret_outcome(
    outcome: RawTestOutcome,
    *,
    obligation_id: str = "",
) -> ObligationResult:
    """Map a RawTestOutcome onto an ObligationResult verdict + evidence."""
    evidence = _assemble_evidence(outcome)

    if outcome.status == "passed":
        return ObligationResult(
            obligation_id=obligation_id,
            status="verified",
            evidence=evidence,
            detail=f"All generated cases held ({outcome.duration_s:.2f}s).",
        )
    if outcome.status == "failed":
        return ObligationResult(
            obligation_id=obligation_id,
            status="refuted",
            counterexample=outcome.falsifying_example,
            evidence=evidence,
            detail="A generated case violated the property.",
        )
    if outcome.status == "timeout":
        return ObligationResult(
            obligation_id=obligation_id,
            status="error",
            evidence=evidence,
            detail=f"The property test exceeded its time budget ({outcome.duration_s:.1f}s).",
        )
    # "error" (or any unrecognised status): the run never yielded a real verdict.
    return ObligationResult(
        obligation_id=obligation_id,
        status="error",
        evidence=evidence,
        detail=_error_detail(outcome),
    )


def _assemble_evidence(outcome: RawTestOutcome) -> str:
    parts = []
    if outcome.stdout and outcome.stdout.strip():
        parts.append(outcome.stdout.rstrip())
    if outcome.stderr and outcome.stderr.strip():
        parts.append("[stderr]\n" + outcome.stderr.rstrip())
    return "\n".join(parts)


def _error_detail(outcome: RawTestOutcome) -> str:
    if outcome.returncode == 5:  # pytest: no tests collected
        return "The test program defined no runnable properties (nothing was collected)."
    return f"The property test failed to run (pytest returncode {outcome.returncode})."
