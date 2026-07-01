"""Verification-router domain models — the data contracts between the
extract -> route -> generate/run/interpret stages.

Only the contracts needed by the components built so far live here; LeafContract
joins when the obligation-extractor / verify dispatch surface are implemented.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestProgram:
    """A self-contained pytest + Hypothesis module ready to execute.

    Emitted by vr-strategy-synthesizer, consumed by vr-test-runner.
    """
    __test__ = False  # not a pytest test class despite the Test* name

    source: str                       # full module source (defines test_* functions)
    obligation_id: str = ""           # traceability back to the obligation / leaf
    # dirs to prepend to PYTHONPATH so the generated test can import the REAL
    # leaf callable (the run must exercise real code, not a mock).
    import_roots: list[str] = field(default_factory=list)


@dataclass
class RawTestOutcome:
    """The mechanical result of running a TestProgram — what pytest did, not
    yet what it means for the obligation.

    Emitted by vr-test-runner, consumed by vr-result-interpreter (which maps
    it onto the verified / refuted / error verdict).
    """
    status: str                       # "passed" | "failed" | "error" | "timeout"
    returncode: Optional[int]         # pytest exit code; None on timeout
    stdout: str
    stderr: str
    falsifying_example: Optional[str] = None  # Hypothesis counterexample block, if any
    duration_s: float = 0.0


@dataclass
class ObligationResult:
    """The verdict for one obligation after its check ran.

    Terminal output of the verification-router (LeafContract -> ObligationResult).
    vr-result-interpreter emits it from a RawTestOutcome; the 'unverifiable'
    status is set upstream by vr-checker-router for IO/boundary leaves that are
    never run.
    """
    obligation_id: str
    status: str                           # "verified" | "refuted" | "error" | "unverifiable"
    counterexample: Optional[str] = None  # minimal falsifying example, on refute
    evidence: str = ""                    # captured run output for operator / fidelity report
    detail: str = ""                      # short human-readable summary


@dataclass
class Property:
    """One operator-authored, executable property over a leaf's inputs and output.

    `expression` is a Python boolean expression evaluated inside the generated
    test, with the fuzzed parameter names, `result` (the leaf's return value),
    and the imported leaf symbol all in scope.
    """
    name: str
    expression: str
    kind: str = "postcondition"           # postcondition | invariant | metamorphic


@dataclass
class CallableSignature:
    """How to import and call the real leaf, recovered from its code anchor."""
    import_module: str                    # dotted module, e.g. "verifier.runner"
    symbol: str                           # callable name, e.g. "run_test_program"
    params: list[tuple[str, str]]         # ordered (name, type_hint) of params to fuzz


@dataclass
class Obligation:
    """A leaf's checkable obligation: what to call and the properties it must hold."""
    obligation_id: str
    signature: CallableSignature
    properties: list[Property]
    import_roots: list[str] = field(default_factory=list)   # PYTHONPATH for the leaf
    strategies: dict[str, str] = field(default_factory=dict)  # operator strategy overrides, keyed by type string


@dataclass
class RoutedObligation:
    """An Obligation tagged with the checker vr-checker-router routed it to."""
    obligation: Obligation
    checker_kind: str                     # "property_testable" | "formally_checkable" | "unverifiable"


@dataclass
class LeafContract:
    """A leaf component's projection — the input to the verification-router.

    Assembled from the graph node by the verify dispatch surface: its typed
    ports, processing prose, external flag, and code anchor (locations).
    """
    component_id: str
    input_types: list[str]
    output_types: list[str]
    processing: str
    locations: list[dict]                 # [{"path", "start_line"?, "end_line"?}]
    external: bool = False
