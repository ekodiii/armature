"""Skeleton extraction + the import/scope-aware call resolver.

Regression focus:
  - a bare callee name no longer matches any same-named symbol anywhere
    (the old first-match-by-name bug that fabricated edges)
  - same-file definitions and imports resolve correctly
  - ambiguous method names resolve to nothing rather than inventing an edge
  - resolved paths are source_root-relative (so resolution works when
    source_root != ".", i.e. every region pass)
"""

from translator.skeleton import build_skeleton
from translator.source_ingestion import ingest


def make_tree(tmp_path, files: dict):
    """files: {relative_path: source_text}. Returns the root dir as str."""
    for rel, text in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text)
    return str(tmp_path)


def callees(skel, caller_substr):
    cid = next(s.id for s in skel.symbols if caller_substr in s.id)
    return sorted({e.callee_id for e in skel.call_edges if e.caller_id == cid})


def test_no_dangling_call_edges(tmp_path):
    root = make_tree(tmp_path, {
        "a.py": "def helper():\n    return 1\n\ndef run():\n    return helper()\n",
    })
    skel = build_skeleton(ingest(root), root)
    ids = {s.id for s in skel.symbols}
    assert all(e.caller_id in ids and e.callee_id in ids for e in skel.call_edges)


def test_same_file_name_wins_over_other_file(tmp_path):
    # Both files define helper(); a.run() must resolve to a.helper, not b.helper.
    root = make_tree(tmp_path, {
        "a.py": "def helper():\n    return 'a'\n\ndef run():\n    return helper()\n",
        "b.py": "def helper():\n    return 'b'\n",
    })
    skel = build_skeleton(ingest(root), root)
    assert callees(skel, "a.py::run") == ["a.py::helper"]


def test_import_based_resolution(tmp_path):
    root = make_tree(tmp_path, {
        "a.py": "from b import compute\n\ndef run():\n    return compute()\n",
        "b.py": "def compute():\n    return 42\n",
    })
    skel = build_skeleton(ingest(root), root)
    assert callees(skel, "a.py::run") == ["b.py::compute"]


def test_ambiguous_method_name_not_fabricated(tmp_path):
    # Two unrelated classes both define save(); a bare module-level call to
    # save() is ambiguous and must resolve to nothing (no invented edge).
    root = make_tree(tmp_path, {
        "m.py": (
            "class A:\n    def save(self):\n        return 1\n\n"
            "class B:\n    def save(self):\n        return 2\n\n"
            "def run():\n    return save()\n"
        ),
    })
    skel = build_skeleton(ingest(root), root)
    assert callees(skel, "m.py::run") == []


def test_self_method_resolves_to_own_class(tmp_path):
    # self.helper() inside C1 must resolve to C1.helper even though C2 also has
    # a helper() (same-class preference).
    root = make_tree(tmp_path, {
        "m.py": (
            "class C1:\n"
            "    def helper(self):\n        return 1\n"
            "    def run(self):\n        return self.helper()\n\n"
            "class C2:\n"
            "    def helper(self):\n        return 2\n"
        ),
    })
    skel = build_skeleton(ingest(root), root)
    assert callees(skel, "m.py::C1::run") == ["m.py::C1::helper"]


def test_resolution_works_with_nested_source_root(tmp_path):
    # Regression: when source_root is a subdir, resolved import paths must be
    # normalized source_root-relative to match SymbolRecord.path, or every
    # cross-file edge silently vanishes.
    make_tree(tmp_path, {
        "pkg/a.py": "from pkg.b import compute\n\ndef run():\n    return compute()\n",
        "pkg/b.py": "def compute():\n    return 42\n",
    })
    root = str(tmp_path / "pkg")
    skel = build_skeleton(ingest(root), root)
    # symbol paths are source_root-relative
    assert {s.path for s in skel.symbols if s.kind == "module"} == {"a.py", "b.py"}
    assert callees(skel, "a.py::run") == ["b.py::compute"]


def test_relative_import_resolution(tmp_path):
    make_tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "from .b import compute\n\ndef run():\n    return compute()\n",
        "pkg/b.py": "def compute():\n    return 1\n",
    })
    root = str(tmp_path)
    skel = build_skeleton(ingest(root), root)
    assert callees(skel, "pkg/a.py::run") == ["pkg/b.py::compute"]


def test_dataflow_edge_uses_resolver(tmp_path):
    # produce() result flows into consume() as an argument -> a dataflow edge,
    # with both endpoints resolved by import/scope (not a name collision).
    root = make_tree(tmp_path, {
        "a.py": (
            "def produce():\n    return 1\n\n"
            "def consume(x):\n    return x\n\n"
            "def run():\n    v = produce()\n    return consume(v)\n"
        ),
    })
    skel = build_skeleton(ingest(root), root)
    pairs = {(e.from_callee, e.to_callee) for e in skel.dataflow_edges}
    assert ("a.py::produce", "a.py::consume") in pairs


def test_submodule_import_resolves_to_submodule_file(tmp_path):
    # `from pkg import sub` binds pkg/sub.py, not a symbol in pkg/__init__.py.
    # The resolver must emit a resolution for the submodule file so grounding
    # and call analysis see the real dependency.
    root = make_tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/sub.py": "def go():\n    return 1\n",
        "user.py": "from pkg import sub\n\ndef run():\n    return sub.go()\n",
    })
    skel = build_skeleton(ingest(root), root)
    resolved_paths = {
        r.resolved_path for r in skel.resolved if r.importer_file == "user.py"
    }
    assert "pkg/sub.py" in resolved_paths


def test_stdlib_classified_not_in_scope(tmp_path):
    root = make_tree(tmp_path, {
        "a.py": "import os\n\ndef run():\n    return os.getcwd()\n",
    })
    skel = build_skeleton(ingest(root), root)
    os_classes = [sc for sc in skel.scope_classes if sc.external_module == "os"]
    assert os_classes and os_classes[0].scope == "stdlib"
