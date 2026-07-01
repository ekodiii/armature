"""vr-obligation-extractor — package a leaf's contract + operator properties.

Operator-driven: the operator authors the property statements (against the
leaf's contract/code, surfaced by get_work_context / get_component_code); this
step is the mechanical packaging. It recovers the real callable's signature
(params + type hints) and import target from the leaf's code anchor so the
checker can import and exercise the ACTUAL code, then bundles that with the
operator's properties into an Obligation.
"""

import ast
import os

from .models import CallableSignature, LeafContract, Obligation, Property


class ExtractionError(ValueError):
    """The leaf cannot be turned into a checkable obligation."""


def extract_obligation(
    leaf: LeafContract,
    properties: list[Property],
    *,
    project_root: str,
    symbol: str | None = None,
    strategies: dict | None = None,
) -> Obligation:
    if not leaf.locations:
        raise ExtractionError(f"leaf {leaf.component_id!r} has no code anchor (locations)")
    if not properties:
        raise ExtractionError(f"leaf {leaf.component_id!r}: no properties authored")

    loc = leaf.locations[0]
    rel_path = loc["path"]
    abs_path = os.path.join(project_root, rel_path)
    sig = _recover_signature(
        abs_path, rel_path, symbol=symbol,
        line_range=(loc.get("start_line"), loc.get("end_line")),
    )
    return Obligation(
        obligation_id=leaf.component_id,
        signature=sig,
        properties=list(properties),
        import_roots=[project_root],
        strategies=dict(strategies or {}),
    )


def _module_name(rel_path: str) -> str:
    p = rel_path.replace(os.sep, "/")
    if p.endswith(".py"):
        p = p[:-3]
    return p.replace("/", ".")


def _recover_signature(abs_path, rel_path, *, symbol, line_range):
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError) as e:
        raise ExtractionError(f"cannot parse {rel_path}: {e}") from e

    funcs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    target = _select_func(funcs, tree, symbol=symbol, line_range=line_range, rel_path=rel_path)
    return CallableSignature(
        import_module=_module_name(rel_path),
        symbol=target.name,
        params=_fuzzable_params(target),
    )


def _select_func(funcs, tree, *, symbol, line_range, rel_path):
    if symbol:
        matches = [f for f in funcs if f.name == symbol]
        if not matches:
            raise ExtractionError(f"symbol {symbol!r} not found in {rel_path}")
        return matches[0]

    top = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    start, end = line_range
    if start is not None:
        end = end or start
        span = [f for f in funcs if start <= f.lineno <= end]
        candidates = [f for f in span if f in top] or span
    else:
        candidates = top

    # a lone public function is the obvious target; helpers (_foo) are ignored
    public = [f for f in candidates if not f.name.startswith("_")] or candidates
    if len(public) == 1:
        return public[0]
    raise ExtractionError(
        f"cannot uniquely identify the target callable in {rel_path}; pass symbol="
    )


def _fuzzable_params(func):
    a = func.args
    params = []
    for arg in list(a.posonlyargs) + list(a.args):
        if arg.arg in ("self", "cls"):
            continue
        hint = ast.unparse(arg.annotation) if arg.annotation else ""
        params.append((arg.arg, hint))
    return params
