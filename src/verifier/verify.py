"""verification-router — behavioral contract verification for one leaf.

The root orchestration, operator-driven end to end: the operator authors the
properties; this composes extract (package + recover signature) -> route
(classify) -> discharge (run the property test) and returns the ObligationResult.
No embedded LLM; the server assembles and runs.
"""

from .extractor import ExtractionError, extract_obligation
from .models import LeafContract, ObligationResult, Property
from .pbt import check
from .router import route


def verify_leaf(
    leaf: LeafContract,
    properties: list[Property],
    *,
    project_root: str,
    external: bool = False,
    strategies: dict | None = None,
    symbol: str | None = None,
    timeout_s: float = 30.0,
) -> ObligationResult:
    try:
        ob = extract_obligation(
            leaf, properties, project_root=project_root,
            symbol=symbol, strategies=strategies,
        )
    except ExtractionError as e:
        return ObligationResult(obligation_id=leaf.component_id, status="error", detail=str(e))

    routed = route(ob, external=external)
    if routed.checker_kind == "property_testable":
        return check(routed, timeout_s=timeout_s)
    if routed.checker_kind == "formally_checkable":
        return ObligationResult(
            ob.obligation_id, "unverifiable",
            detail="formal/SMT checking is deferred (vr-formal-checker, post-beta).",
        )
    return ObligationResult(
        ob.obligation_id, "unverifiable",
        detail="Leaf is an external / IO boundary; flagged, not run.",
    )
