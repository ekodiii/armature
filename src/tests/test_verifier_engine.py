"""End-to-end verification-router engine tests — extract -> route -> discharge.

Covers the pieces that tie the checker together: the router's MVP classification
(property_testable vs. unverifiable for external leaves); the extractor recovering
a real callable's signature from a code anchor (and demanding a symbol when the
file is ambiguous); and verify_leaf running the whole chain from a LeafContract —
verified when an operator-authored property holds, refuted with a counterexample
when it doesn't, and short-circuited to 'unverifiable' for external boundaries.
"""

import pytest

from verifier.extractor import ExtractionError, extract_obligation
from verifier.models import (
    CallableSignature,
    LeafContract,
    Obligation,
    Property,
    RoutedObligation,
)
from verifier.router import route
from verifier.pbt import check
from verifier.verify import verify_leaf


def _leaf(tmp_path, filename, body, component_id):
    (tmp_path / filename).write_text(body)
    return LeafContract(
        component_id=component_id,
        input_types=[], output_types=[], processing="",
        locations=[{"path": filename}],
    )


# --------------------------------------------------------------------------- #
# router
# --------------------------------------------------------------------------- #

def test_router_defaults_property_testable():
    ob = Obligation("o", CallableSignature("m", "f", [("x", "int")]), [Property("p", "True")])
    assert route(ob).checker_kind == "property_testable"


def test_router_flags_external_unverifiable():
    ob = Obligation("o", CallableSignature("m", "f", [("x", "int")]), [Property("p", "True")])
    assert route(ob, external=True).checker_kind == "unverifiable"


# --------------------------------------------------------------------------- #
# extractor
# --------------------------------------------------------------------------- #

def test_extractor_recovers_signature(tmp_path):
    leaf = _leaf(tmp_path, "m.py", "def f(a: int, b: str):\n    return b * a\n", "f")
    ob = extract_obligation(leaf, [Property("p", "True")], project_root=str(tmp_path))
    assert ob.signature.import_module == "m"
    assert ob.signature.symbol == "f"
    assert ob.signature.params == [("a", "int"), ("b", "str")]
    assert ob.import_roots == [str(tmp_path)]


def test_extractor_ignores_private_helpers(tmp_path):
    body = "def public(x: int):\n    return _helper(x)\n\ndef _helper(x):\n    return x\n"
    leaf = _leaf(tmp_path, "m.py", body, "m")
    ob = extract_obligation(leaf, [Property("p", "True")], project_root=str(tmp_path))
    assert ob.signature.symbol == "public"


def test_extractor_ambiguous_requires_symbol(tmp_path):
    leaf = _leaf(tmp_path, "m.py", "def f(x):\n    return x\n\ndef g(y):\n    return y\n", "m")
    with pytest.raises(ExtractionError):
        extract_obligation(leaf, [Property("p", "True")], project_root=str(tmp_path))
    ob = extract_obligation(leaf, [Property("p", "True")], project_root=str(tmp_path), symbol="g")
    assert ob.signature.symbol == "g"


def test_extractor_requires_locations():
    leaf = LeafContract("x", [], [], "", locations=[])
    with pytest.raises(ExtractionError):
        extract_obligation(leaf, [Property("p", "True")], project_root="/tmp")


# --------------------------------------------------------------------------- #
# pbt.check bundles source into evidence
# --------------------------------------------------------------------------- #

def test_check_bundles_generated_source(tmp_path):
    (tmp_path / "leaf.py").write_text("def double(x):\n    return x + x\n")
    ob = Obligation(
        "double", CallableSignature("leaf", "double", [("x", "int")]),
        [Property("doubles", "result == 2 * x")], import_roots=[str(tmp_path)],
    )
    result = check(RoutedObligation(ob, "property_testable"), timeout_s=60)
    assert result.status == "verified", result.evidence
    assert "def test_double" in result.evidence  # generated source is bundled


# --------------------------------------------------------------------------- #
# verify_leaf — the whole chain from a LeafContract
# --------------------------------------------------------------------------- #

def test_verify_leaf_verified(tmp_path):
    leaf = _leaf(tmp_path, "leaf.py", "def double(x: int):\n    return x + x\n", "double")
    result = verify_leaf(
        leaf, [Property("doubles", "result == 2 * x")],
        project_root=str(tmp_path), timeout_s=60,
    )
    assert result.status == "verified", result.evidence
    assert result.obligation_id == "double"


def test_verify_leaf_untyped_param_via_operator_strategy(tmp_path):
    """Un-annotated real code verifies when the operator names the param's strategy."""
    leaf = _leaf(tmp_path, "leaf.py", "def double(x):\n    return x + x\n", "double")
    result = verify_leaf(
        leaf, [Property("doubles", "result == 2 * x")],
        project_root=str(tmp_path), strategies={"x": "st.integers()"}, timeout_s=60,
    )
    assert result.status == "verified", result.evidence


def test_verify_leaf_refuted(tmp_path):
    leaf = _leaf(tmp_path, "leaf.py", "def double(x: int):\n    return x + x\n", "double")
    result = verify_leaf(
        leaf, [Property("too_big", "result < 100")],
        project_root=str(tmp_path), timeout_s=60,
    )
    assert result.status == "refuted"
    assert result.counterexample is not None
    assert "x=" in result.counterexample


def test_verify_leaf_external_is_unverifiable(tmp_path):
    leaf = _leaf(tmp_path, "leaf.py", "def double(x: int):\n    return x + x\n", "double")
    leaf.external = True
    result = verify_leaf(
        leaf, [Property("doubles", "result == 2 * x")],
        project_root=str(tmp_path), external=True, timeout_s=60,
    )
    assert result.status == "unverifiable"
    assert "boundary" in result.detail
