"""Token-diet response shaping: the _line formatter, get_orient depth bounds,
search limit, and the get_component_code max_lines budget.

These exist because the JSON-dict terse views were ~4-5x the tokens of the
information they carried (measured on the 113-node django graph), and
get_component_code could dump whole files unbounded.
"""

import pytest

import server
from models import Component, Edge, EdgeType, FileLocation, Graph
from store import add_component, add_edge


def comp(cid, z=0, parent=None, itypes=("a",), otypes=("b",), desc="one line\nsecond line"):
    return Component(
        component_id=cid,
        description=desc,
        processing="p",
        input_types=list(itypes),
        output_types=list(otypes),
        z_level=z,
        parent_id=parent,
    )


@pytest.fixture
def small_graph(monkeypatch):
    """root -> (kid1, kid2), kid1 -> grandkid; root2 standalone."""
    g = Graph.new()
    add_component(g, comp("root"))
    add_component(g, comp("root2"))
    add_component(g, comp("kid1", z=1, parent="root"))
    add_component(g, comp("kid2", z=1, parent="root"))
    add_component(g, comp("grandkid", z=2, parent="kid1"))
    add_edge(g, Edge(EdgeType.FLOW, "kid1", "kid2"))
    monkeypatch.setattr(server, "GRAPH", g)
    return g


# --------------------------------------------------------------------------- #
# _line formatter
# --------------------------------------------------------------------------- #

def test_line_is_single_line_with_contract(small_graph):
    line = server._line(small_graph.components["kid1"])
    assert "\n" not in line
    assert line.startswith("kid1 [a -> b]")
    assert "1ch" in line  # grandkid
    assert ":: one line" in line
    assert "second line" not in line  # first description line only


def test_line_omits_implemented_status_but_flags_planned(small_graph):
    c = small_graph.components["kid2"]
    assert "(PLANNED)" in server._line(c)
    c.implemented_version = c.version
    assert "(PLANNED)" not in server._line(c)
    assert "(IMPLEMENTED)" not in server._line(c)


def test_line_indent_and_locations_cap(small_graph):
    c = small_graph.components["kid1"]
    c.locations = [FileLocation(path=f"f{i}.py", start_line=1, end_line=9) for i in range(8)]
    line = server._line(c, locations=True, indent=2)
    assert line.startswith("    kid1")
    assert "@ f0.py:1-9" in line
    assert "+3 more" in line
    assert "f5.py" not in line


# --------------------------------------------------------------------------- #
# get_orient depth bounds
# --------------------------------------------------------------------------- #

def test_orient_default_depth_hides_deeper_levels(small_graph):
    out = server.get_orient()
    assert out["count"] == 4  # root, root2, kid1, kid2 -- not grandkid
    assert out["total"] == 5
    assert "hidden" in out["note"]
    assert all(isinstance(l, str) for l in out["map"])
    assert out["legend"] == server.LINE_LEGEND


def test_orient_full_depth_shows_everything_indented(small_graph):
    out = server.get_orient(depth=-1)
    assert out["count"] == out["total"] == 5
    assert "note" not in out
    assert any(l.startswith("    grandkid") for l in out["map"])


# --------------------------------------------------------------------------- #
# list tools emit lines + legend; full dicts still opt-in
# --------------------------------------------------------------------------- #

def test_neighbors_terse_lines_and_full_dict_optout(small_graph):
    terse = server.get_neighbors("kid1", "FLOW")
    assert terse["neighbors"] == [server._line(small_graph.components["kid2"])]
    assert terse["legend"] == server.LINE_LEGEND
    full = server.get_neighbors("kid1", "FLOW", terse=False)
    assert full["neighbors"][0]["component_id"] == "kid2"
    assert "legend" not in full


def test_search_limit_truncates_with_note(small_graph):
    out = server.search_components("one line", limit=2)
    assert len(out["matches"]) == 2
    assert out["total"] == 5
    assert "narrow the query" in out["note"]


# --------------------------------------------------------------------------- #
# get_component_code budget
# --------------------------------------------------------------------------- #

@pytest.fixture
def coded_graph(small_graph, tmp_path, monkeypatch):
    (tmp_path / "big.py").write_text("".join(f"line{i}\n" for i in range(1, 301)))
    (tmp_path / "small.py").write_text("x = 1\ny = 2\n")
    monkeypatch.setattr(server, "PROJECT_ROOT_OVERRIDE", str(tmp_path))
    return small_graph


def test_code_within_budget_untouched(coded_graph):
    coded_graph.components["kid1"].locations = [FileLocation(path="small.py")]
    out = server.get_component_code("kid1")
    assert out["locations"][0]["code"] == "x = 1\ny = 2\n"
    assert "note" not in out
    assert "omitted_lines" not in out["locations"][0]


def test_code_over_budget_truncates_with_counts(coded_graph):
    coded_graph.components["kid1"].locations = [FileLocation(path="big.py")]
    out = server.get_component_code("kid1", max_lines=50)
    loc = out["locations"][0]
    assert loc["code"].splitlines()[-1] == "line50"
    assert loc["omitted_lines"] == 250
    assert "max_lines=50" in out["note"]


def test_code_budget_spans_locations_and_lists_all_anchors(coded_graph):
    coded_graph.components["kid1"].locations = [
        FileLocation(path="big.py", start_line=1, end_line=195),
        FileLocation(path="small.py"),
    ]
    out = server.get_component_code("kid1")  # default 200
    first, second = out["locations"]
    assert "omitted_lines" not in first  # 195 lines fit
    assert second["path"] == "small.py"  # anchor still listed
    assert second["code"] == "x = 1\ny = 2\n"  # 2 lines fit in remaining 5
    out2 = server.get_component_code("kid1", max_lines=100)
    first2, second2 = out2["locations"]
    assert first2["omitted_lines"] == 95
    assert second2["code"] == ""  # budget spent, code omitted
    assert second2["omitted_lines"] == 2
