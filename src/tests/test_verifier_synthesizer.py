"""vr-strategy-synthesizer regression tests — RoutedObligation -> TestProgram.

Covers the mechanical assembly: type hints map to Hypothesis strategies
(scalars, recursive containers, unions/optional), operator overrides win,
unknown types fail loudly; synthesis refuses non-property_testable or
property-less obligations; and, end to end, an operator-authored property is
assembled, RUN against real code, and interpreted — verified when it holds,
refuted with a counterexample when it does not.
"""

import pytest

from verifier.models import CallableSignature, Obligation, Property, RoutedObligation
from verifier.synthesizer import synthesize, SynthesisError, _strategy_expr
from verifier.runner import run_test_program
from verifier.interpreter import interpret_outcome


# --------------------------------------------------------------------------- #
# strategy mapping
# --------------------------------------------------------------------------- #

def test_strategy_scalars():
    assert _strategy_expr("int", {}) == "st.integers()"
    assert _strategy_expr("str", {}) == "st.text()"
    assert _strategy_expr("bool", {}) == "st.booleans()"


def test_strategy_containers_recurse():
    assert _strategy_expr("list[int]", {}) == "st.lists(st.integers())"
    assert _strategy_expr("dict[str, int]", {}) == "st.dictionaries(st.text(), st.integers())"


def test_strategy_optional_and_union():
    assert _strategy_expr("int | None", {}) == "st.one_of(st.integers(), st.none())"
    assert _strategy_expr("Optional[int]", {}) == "st.one_of(st.none(), st.integers())"


def test_strategy_override_wins():
    assert _strategy_expr("Widget", {"Widget": "st.builds(Widget)"}) == "st.builds(Widget)"


def test_unknown_type_raises():
    with pytest.raises(SynthesisError):
        _strategy_expr("Widget", {})


# --------------------------------------------------------------------------- #
# assembly + guards
# --------------------------------------------------------------------------- #

def _routed(properties=None, **ob_kw):
    sig = CallableSignature("leaf", "double", [("x", "int")])
    ob = Obligation(
        "ob", sig,
        properties=properties if properties is not None else [Property("p", "result == 2 * x")],
        **ob_kw,
    )
    return RoutedObligation(ob, "property_testable")


def test_synthesize_emits_given_import_and_assert():
    prog = synthesize(_routed())
    assert "from hypothesis import given" in prog.source
    assert "from leaf import double" in prog.source
    assert "@given(x=st.integers())" in prog.source
    assert "result = double(x)" in prog.source
    assert "assert (result == 2 * x)" in prog.source
    assert prog.obligation_id == "ob"


def test_synthesize_refuses_non_property_testable():
    routed = _routed()
    routed.checker_kind = "unverifiable"
    with pytest.raises(SynthesisError):
        synthesize(routed)


def test_synthesize_requires_properties():
    with pytest.raises(SynthesisError):
        synthesize(_routed(properties=[]))


# --------------------------------------------------------------------------- #
# end to end: assemble -> run -> interpret
# --------------------------------------------------------------------------- #

def _write_leaf(tmp_path, body):
    (tmp_path / "leaf.py").write_text(body)
    return str(tmp_path)


def test_end_to_end_verified(tmp_path):
    root = _write_leaf(tmp_path, "def double(x):\n    return x + x\n")
    ob = Obligation(
        "double-ok",
        CallableSignature("leaf", "double", [("x", "int")]),
        properties=[
            Property("doubles", "result == 2 * x"),
            Property("metamorphic", "double(x) + double(x) == double(2 * x)"),
        ],
        import_roots=[root],
    )
    program = synthesize(RoutedObligation(ob, "property_testable"))
    result = interpret_outcome(run_test_program(program, timeout_s=60), obligation_id="double-ok")
    assert result.status == "verified", result.evidence
    assert result.obligation_id == "double-ok"


def test_end_to_end_refuted_with_counterexample(tmp_path):
    root = _write_leaf(tmp_path, "def double(x):\n    return x + x\n")
    ob = Obligation(
        "double-bad",
        CallableSignature("leaf", "double", [("x", "int")]),
        properties=[Property("too_big", "result < 100")],
        import_roots=[root],
    )
    program = synthesize(RoutedObligation(ob, "property_testable"))
    result = interpret_outcome(run_test_program(program, timeout_s=60))
    assert result.status == "refuted"
    assert result.counterexample is not None
    assert "x=" in result.counterexample


def test_end_to_end_custom_type_via_operator_strategy(tmp_path):
    """A domain type with no builtin strategy works via an operator override."""
    root = _write_leaf(
        tmp_path,
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Point:\n"
        "    x: int\n"
        "    y: int\n"
        "def flip(p):\n"
        "    return Point(p.y, p.x)\n",
    )
    ob = Obligation(
        "flip-involutive",
        CallableSignature("leaf", "flip", [("p", "Point")]),
        properties=[Property("involutive", "flip(result) == p")],
        import_roots=[root],
        strategies={"Point": "st.builds(__import__('leaf').Point, st.integers(), st.integers())"},
    )
    program = synthesize(RoutedObligation(ob, "property_testable"))
    result = interpret_outcome(run_test_program(program, timeout_s=60), obligation_id="flip-involutive")
    assert result.status == "verified", result.evidence
