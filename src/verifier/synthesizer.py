"""vr-strategy-synthesizer — assemble a runnable Hypothesis test from an obligation.

Operator-driven and mechanical. The property statements are authored by the
operator (upstream, against the leaf's surfaced contract/code) and arrive on the
Obligation; this step binds each fuzzed parameter to a Hypothesis strategy,
embeds those properties as asserts, and imports the REAL leaf so vr-test-runner
exercises actual code. No embedded LLM: unknown types are resolved from the
operator-supplied strategy overrides, or synthesis fails loudly.
"""

import re

from .models import RoutedObligation, TestProgram


class SynthesisError(ValueError):
    """The obligation cannot be turned into a runnable property test."""


_SCALAR_STRATEGIES = {
    "int": "st.integers()",
    "float": "st.floats(allow_nan=False, allow_infinity=False)",
    "str": "st.text()",
    "bool": "st.booleans()",
    "bytes": "st.binary()",
    "none": "st.none()",
    "nonetype": "st.none()",
}

_BARE_CONTAINERS = {
    "list": "st.lists(st.integers())",
    "sequence": "st.lists(st.integers())",
    "set": "st.sets(st.integers())",
    "dict": "st.dictionaries(st.text(), st.integers())",
    "mapping": "st.dictionaries(st.text(), st.integers())",
}


def synthesize(routed: RoutedObligation) -> TestProgram:
    """Turn a property_testable RoutedObligation into a self-contained TestProgram."""
    if routed.checker_kind != "property_testable":
        raise SynthesisError(
            f"cannot synthesize a property test for checker_kind={routed.checker_kind!r}"
        )
    ob = routed.obligation
    if not ob.properties:
        raise SynthesisError(f"obligation {ob.obligation_id!r} has no properties to check")

    return TestProgram(
        source=_module_source(ob),
        obligation_id=ob.obligation_id,
        import_roots=list(ob.import_roots),
    )


def _module_source(ob) -> str:
    sig = ob.signature
    param_names = [name for name, _ in sig.params]
    call = f"{sig.symbol}(" + ", ".join(param_names) + ")"

    lines = [
        f"# auto-generated property test for obligation {ob.obligation_id!r}",
        "from hypothesis import given, strategies as st",
        f"from {sig.import_module} import {sig.symbol}",
    ]
    for i, prop in enumerate(ob.properties):
        fname = f"test_{_slug(ob.obligation_id)}_{i}_{_slug(prop.name)}"
        lines.append("")
        if sig.params:
            given_args = ", ".join(
                f"{name}={_param_strategy(name, hint, ob.strategies)}" for name, hint in sig.params
            )
            lines.append(f"@given({given_args})")
            lines.append(f"def {fname}({', '.join(param_names)}):")
        else:
            lines.append(f"def {fname}():")
        lines.append(f"    result = {call}")
        lines.append(f"    assert ({prop.expression})")
    return "\n".join(lines) + "\n"


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_") or "x"


def _split_top(s: str, sep: str) -> list[str]:
    """Split on `sep` only at bracket depth 0 (so 'dict[str, int]' splits cleanly)."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "[":
            depth += 1
            cur.append(ch)
        elif ch == "]":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _param_strategy(name: str, type_hint: str, overrides: dict) -> str:
    """A per-parameter strategy: an operator override keyed by the param NAME
    wins (needed for un-annotated code), else resolve from the type hint (which
    itself honours type-keyed overrides)."""
    if name in overrides:
        return overrides[name]
    return _strategy_expr(type_hint, overrides)


def _strategy_expr(type_hint: str, overrides: dict) -> str:
    """Map a type-hint string to a Hypothesis strategy expression (as source text).

    Operator overrides win (exact type-string match), then scalars, then common
    containers (recursively), then bare containers. Anything else fails loudly so
    the operator supplies a strategy rather than the test silently drifting.
    """
    t = (type_hint or "").strip()
    if t in overrides:
        return overrides[t]

    union = [p.strip() for p in _split_top(t, "|")]
    if len(union) > 1:  # "X | None", "A | B"
        return "st.one_of(" + ", ".join(_strategy_expr(u, overrides) for u in union) + ")"

    low = t.lower()
    if low in _SCALAR_STRATEGIES:
        return _SCALAR_STRATEGIES[low]

    if "[" in t and t.endswith("]"):
        base = t[: t.index("[")].strip().lower()
        args = [a.strip() for a in _split_top(t[t.index("[") + 1 : -1], ",")]
        if base in ("list", "sequence", "iterable"):
            return f"st.lists({_strategy_expr(args[0], overrides)})"
        if base == "set":
            return f"st.sets({_strategy_expr(args[0], overrides)})"
        if base == "frozenset":
            return f"st.frozensets({_strategy_expr(args[0], overrides)})"
        if base == "optional":
            return f"st.one_of(st.none(), {_strategy_expr(args[0], overrides)})"
        if base in ("dict", "mapping"):
            return (
                f"st.dictionaries({_strategy_expr(args[0], overrides)}, "
                f"{_strategy_expr(args[1], overrides)})"
            )
        if base == "tuple":
            if len(args) == 2 and args[1] == "...":
                return f"st.lists({_strategy_expr(args[0], overrides)}).map(tuple)"
            return "st.tuples(" + ", ".join(_strategy_expr(a, overrides) for a in args) + ")"

    if low in _BARE_CONTAINERS:
        return _BARE_CONTAINERS[low]

    raise SynthesisError(
        f"no Hypothesis strategy for type {type_hint!r}; add an override to the "
        f"obligation's strategies (keyed by type name or parameter name)"
    )
