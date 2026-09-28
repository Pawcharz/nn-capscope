"""Acceptance tests for capscope on the deliberately mismatched toy model."""
from __future__ import annotations

import http.server
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from toy_model import (TRUE_EDGES, ToyGraphNet, baseline_loss, edge_index_fn, forward_fn,  # noqa: E402
                       get_trained, loss_fn, make_loader)
from capscope import inspect  # noqa: E402
from capscope.capture import is_relative  # noqa: E402


@pytest.fixture(scope="session")
def trained():
    model, batches = get_trained(verbose=True)
    # genuine convergence: far below the "predict the mean" baseline
    with torch.no_grad():
        loss = float(sum(loss_fn(model(b), b) for b in batches) / len(batches))
    base = baseline_loss(batches)
    print(f"\ntrained loss {loss:.4f} vs predict-mean baseline {base:.4f}")
    assert loss < 0.35 * base, "toy model did not converge; diagnostics would be meaningless (trap 7)"
    return model, batches


@pytest.fixture(scope="session")
def report(trained):
    model, batches = trained
    rep = inspect(model, batches, forward_fn=forward_fn, loss_fn=loss_fn,
                  edge_index_fn=edge_index_fn, n_batches=8)
    rep.summary()
    return rep


# ---------------------------------------------------------------------------
# ground-truth assertions
# ---------------------------------------------------------------------------

def test_bottleneck_is_top_and_saturated(report):
    ranked = report.ranked()
    top = ranked[0]
    assert top["name"] == "bottleneck", [ (m["name"], m["verdict"], round(m["priority"], 2)) for m in ranked[:5] ]
    assert top["verdict"] == "saturated"
    assert report["bottleneck"]["rank_cap"] == "output"          # trap 4: evidence to widen *here*
    assert report["bottleneck"]["used_frac"] > 0.85


def test_overwide_relay_is_upstream(report):
    for name in ("encoder", "encoder.0"):
        m = report[name]
        assert m["verdict"] == "upstream", (name, m["verdict"], m["sentence"])
        assert m["verdict"] != "saturated"
    assert report["encoder.0"]["rank_cap"] == "input"            # trap 4: capped by its 8-dim input
    assert report["encoder.0"]["carry"] == 8


def test_parameter_free_modules_are_passthrough(report):
    for name in ("encoder.1", "mp1.act", "mp2.act", "mp3.act"):
        m = report[name]
        assert m["verdict"] == "passthrough", (name, m["verdict"])
        assert m["priority"] == 0.0
        assert m["n_params_own"] == 0


def test_heads_are_narrow(report):
    for h in ("heads.a", "heads.b", "heads.c", "heads.d"):
        assert report[h]["verdict"] == "narrow"


def test_dataflow_graph_matches_architecture(report):
    got = set(report.edges)
    true = set(TRUE_EDGES)
    assert got == true, f"missing={sorted(true - got)} extra={sorted(got - true)}"


def test_no_container_redundant_against_own_child(report):
    for m in report.modules:
        partner = m.get("cka_partner")
        if partner is not None:
            assert not is_relative(m["name"], partner), (m["name"], partner)
        if m["verdict"] == "redundant" and m["reason"] == "CKA duplicate":
            assert not is_relative(m["name"], m["cka_partner"])
    # the Sequential 'encoder' shares its output with 'encoder.1' (CKA == 1) and must not be flagged for it
    names = report.cka["names"]
    i, j = names.index("encoder"), names.index("encoder.1")
    assert report.cka["matrix"][i][j] > 0.999
    assert report["encoder"]["verdict"] != "redundant"


def test_message_passing_layers_are_not_upstream_or_passthrough(report):
    # the layers doing the real work must be judged on capacity, not dismissed
    for name in ("mp2", "mp3"):
        assert report[name]["verdict"] in ("saturated", "tight", "spare"), (name, report[name]["verdict"], report[name]["sentence"])


def test_no_undertrained_warning_on_converged_model(report):
    assert not any("UNDERTRAINED" in w or "undertrained" in w for w in report.warnings), report.warnings


def test_untrained_checkpoint_is_detected():
    torch.manual_seed(1)
    model = ToyGraphNet()
    batches = make_loader(n_graphs=4, n_nodes=600)
    rep = inspect(model, batches, forward_fn=forward_fn, loss_fn=loss_fn, edge_index_fn=edge_index_fn,
                  n_batches=4)
    assert any("undertrained" in w.lower() for w in rep.warnings), rep.warnings


def test_summary_and_serialisation(report, tmp_path):
    txt = report.summary(print_it=False)
    assert "bottleneck" in txt and "saturated" in txt
    js = report.to_json()
    assert "NaN" not in js
    out = report.to_html(tmp_path / "r.html")
    assert Path(out).stat().st_size > 50_000


class _SumConv(torch.nn.Module):
    """A PyG-style SAGEConv: the container itself builds the output (a sum of two
    Linears), so it must become a graph node, not a hole in the graph."""

    def __init__(self, din, dout):
        super().__init__()
        self.lin_l = torch.nn.Linear(din, dout)
        self.lin_r = torch.nn.Linear(din, dout, bias=False)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x).index_add_(0, dst, x[src])
        return self.lin_l(agg) + self.lin_r(x)


class _SumStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList([_SumConv(8, 32), _SumConv(32, 32)])
        self.trunk = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.GELU())
        self.head = torch.nn.Linear(32, 4)

    def forward(self, batch):
        x, ei = batch
        for i, c in enumerate(self.convs):
            x = c(x, ei)
            if i == 0:
                x = torch.relu(x)                      # functional: no module boundary
        return self.head(self.trunk(x))


def test_emitter_containers_are_graph_nodes():
    torch.manual_seed(0)
    model = _SumStack()
    g = torch.Generator().manual_seed(0)
    batches = [(torch.randn(50, 8, generator=g), torch.randint(0, 50, (2, 120), generator=g)) for _ in range(2)]
    rep = inspect(model, batches, forward_fn=lambda m, b: m(b), edge_index_fn=lambda b: b[1], n_batches=2)
    assert rep["convs.0"]["emits"] and rep["convs.1"]["emits"]
    assert not rep["trunk"]["emits"] and "convs" not in rep        # handed through / never called
    expected = {
        ("convs.0.lin_l", "convs.0"), ("convs.0.lin_r", "convs.0"),
        ("convs.0", "convs.1.lin_l"), ("convs.0", "convs.1.lin_r"),
        ("convs.1.lin_l", "convs.1"), ("convs.1.lin_r", "convs.1"),
        ("convs.1", "trunk.0"), ("trunk.0", "trunk.1"), ("trunk.1", "head"),
    }
    assert set(rep.edges) == expected, f"missing={sorted(expected - set(rep.edges))} extra={sorted(set(rep.edges) - expected)}"
    assert rep["convs.0"]["consumers"] == ["convs.1.lin_l", "convs.1.lin_r"]
    assert rep["head"]["is_sink"] and not rep["convs.1"]["is_sink"]
    # a container's rank cap comes from the leaf that produces its output, not its narrowest weight
    assert rep["convs.0"]["max_weight_rank"] == 8            # both Linears take 8 inputs
    assert rep["convs.1"]["max_weight_rank"] == 32
    assert rep["trunk"]["rank_cap"] == "square" and rep["trunk"]["max_weight_rank"] == 32
    js = rep.to_json()
    assert '"emits": true' in js


def test_reload_from_saved_report(report, tmp_path):
    from capscope import Report
    from capscope.cli import main
    html = report.to_html(tmp_path / "r.html")
    js = report.to_json_file(tmp_path / "r.json")
    for path in (html, js):
        again = Report.load(path)
        assert again.names() == report.names()
        assert again.edges == report.edges
        assert again["bottleneck"]["verdict"] == "saturated"
        assert again.to_json() == report.to_json()
    # the CLI reopens a report without a factory and re-renders it with the current template
    out = tmp_path / "again.html"
    assert main([html, "--no-gui", "--quiet", "--html", str(out)]) == 0
    assert "bottleneck" in out.read_text(encoding="utf-8")


def test_data_free_mode():
    model = ToyGraphNet()
    rep = inspect(model)             # weight spectra only, no loader
    assert "bottleneck" in rep
    assert rep["bottleneck"]["alpha"] is not None
    assert rep.to_json()
    # the compact export must survive a report with no activations, sweep or edges
    d = rep.to_llm_dict()
    b = next(m for m in d["modules"] if m["name"] == "bottleneck")
    assert b["rank"] is None and b["sweep"] is None and b["alpha"] is not None
    assert d["edges"] == [] and d["cka_top_pairs"] == []
    assert "bottleneck" in rep.to_llm("md")


def test_llm_export(report, tmp_path):
    import json
    from capscope.cli import main
    from capscope.export import FIELDS, VERDICTS, FORMAT
    from capscope.verdict import VERDICT_ORDER
    d = report.to_llm_dict(source="toy")
    assert d["format"] == FORMAT
    assert d["meta"]["source"] == "toy" and d["meta"]["capscope_version"] and d["meta"]["generated_at"]
    # every module, ranked by priority, with the same verdicts as the report
    assert [m["name"] for m in d["modules"]] == [m["name"] for m in report.ranked()] == d["ranked"]
    by = {m["name"]: m for m in d["modules"]}
    for m in report.modules:
        assert by[m["name"]]["verdict"] == m["verdict"]
        assert by[m["name"]]["sentence"] == m["sentence"]
    b = by["bottleneck"]
    assert b["verdict"] == "saturated" and b["used_rank"] == report["bottleneck"]["used_rank"]
    assert set(b["rank"]) == {"effective_rank", "stable_rank", "participation_ratio", "numerical_rank", "rank90", "rank99", "n_rows"}
    assert b["sweep"]["curve"] and all(isinstance(v, float) for v in b["sweep"]["curve"].values())
    assert b["producers"] == report["bottleneck"]["producers"] and b["consumers"] == report["bottleneck"]["consumers"]
    assert [tuple(e) for e in d["edges"]] == report.edges
    assert d["cka_top_pairs"] and d["cka_top_pairs"][0]["cka"] >= d["cka_top_pairs"][-1]["cka"]
    from capscope.capture import is_relative
    assert not any(is_relative(p["a"], p["b"]) for p in d["cka_top_pairs"])   # no container/child pairs
    # self-describing: every exported field and verdict has a legend entry; no bulk arrays
    js = report.to_llm("json")
    assert "singular_values" not in js and "cum_var" not in js and '"hist"' not in js and '"hierarchy"' not in js
    for k in b:
        assert k in FIELDS or k in ("rank", "sweep", "redundancy", "weights"), k
    for k in b["rank"]:
        assert f"rank.{k}" in FIELDS, k
    assert set(VERDICTS) == set(VERDICT_ORDER) == set(d["legend"]["verdict_order"])
    assert "\n" not in report.to_llm("jsonl").strip()
    assert len(report.to_llm("jsonl")) < len(report.to_json())
    # markdown: header, ranked table, one section per module, edges, legend
    md = report.to_llm("md")
    assert md.startswith("# capscope report:")
    assert "## Modules by growth priority" in md and "## Legend" in md
    assert md.count("\n### `") == len(report.modules)
    assert "`mp1.lin_self` →" in md or "→ `bottleneck`" in md
    assert "**saturated**" in md
    # CLI: suffix picks the format; .jsonl appends one line per run; '-' prints
    html = report.to_html(tmp_path / "r.html")
    assert main([html, "--no-gui", "--quiet", "--export", str(tmp_path / "s.md")]) == 0
    assert (tmp_path / "s.md").read_text(encoding="utf-8").startswith("# capscope report:")
    assert main([html, "--no-gui", "--quiet", "--export", str(tmp_path / "s.json")]) == 0
    loaded = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert loaded["format"] == FORMAT and loaded["meta"]["source"] == html
    log = tmp_path / "runs.jsonl"
    for _ in range(2):
        assert main([html, "--no-gui", "--quiet", "--export", str(log)]) == 0
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(json.loads(l)["format"] == FORMAT for l in lines)


# ---------------------------------------------------------------------------
# GUI: zero console errors in a real browser
# ---------------------------------------------------------------------------

def _browser_launch_kwargs():
    """Use an already-installed browser so no download is needed."""
    for channel in ("msedge", "chrome"):
        yield {"channel": channel}
    yield {}


def test_html_renders_without_console_errors(report, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    path = Path(report.to_html(tmp_path / "report.html"))
    errors, page_errors = [], []
    with playwright.sync_playwright() as p:
        browser = None
        for kw in _browser_launch_kwargs():
            try:
                browser = p.chromium.launch(headless=True, **kw)
                break
            except Exception:
                continue
        if browser is None:
            pytest.skip("no chromium-based browser available for Playwright")
        page = browser.new_page(viewport={"width": 1500, "height": 900})
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.goto(path.resolve().as_uri())
        page.wait_for_selector("#graph .node")
        # exercise the interaction model
        assert page.locator("#graph .node").count() >= 10
        page.locator("table.mods tr[data-name='mp2']").click()
        assert page.locator("#right h2").inner_text() == "mp2"
        page.locator("#collapseAll").click()
        page.locator("#expandAll").click()
        page.locator("#leavesOnly").check()
        page.locator("#tab-heat").click()
        page.wait_for_selector("#heatsvg")
        page.locator("#heatsvg rect[data-a='mp1'][data-b='mp1']").click(force=True) if not page.locator("#leavesOnly").is_checked() else None
        page.locator("#tab-graph").click()
        page.locator("#leavesOnly").uncheck()
        page.locator("#graph .node[data-name='bottleneck']").click()
        assert "bottleneck" in page.locator("#right h2").inner_text()
        page.mouse.wheel(0, -300)
        page.mouse.move(700, 400); page.mouse.down(); page.mouse.move(760, 430); page.mouse.up()
        page.locator("#search").fill("mp3")
        assert page.locator("table.mods tr[data-name]").count() >= 4
        page.locator("#fit").click()
        # detail panel: move below the graph and back
        page.locator("#moveDetail").click()
        assert page.locator("#below #right h2").count() == 1
        assert page.evaluate("document.body.classList.contains('below')")
        page.locator("#search").fill("")
        page.locator("table.mods tr[data-name='mp2']").click()
        assert page.locator("#below #right h2").inner_text() == "mp2"
        # the page scrolls to the last section and at most two cards share a row
        page.locator("[data-secs='open']").click()
        last = page.locator("#right .sec").last
        last.scroll_into_view_if_needed()
        assert last.bounding_box()["y"] + last.bounding_box()["height"] <= page.viewport_size["height"] + 1
        assert page.evaluate("getComputedStyle(document.querySelector('#right .secs')).gridTemplateColumns.split(' ').length") == 2
        assert page.locator("#right .cell .d").count() >= 20
        # in panel-below mode each section carries its guide text
        assert page.locator("#right .sec[data-sec='rank'] .explain").is_visible()
        assert "Effective rank" in page.locator("#right .sec[data-sec='rank'] .explain").inner_text()
        assert page.locator("#right .sec[data-sec='capacity'] .explain table").count() == 1
        page.locator("#moveDetail").click()
        assert page.locator("main #right h2").count() == 1
        # sidebars: drag the gutters
        left_before = page.locator("#left").bounding_box()["width"]
        gut = page.locator("#gutL").bounding_box()
        page.mouse.move(gut["x"] + 3, gut["y"] + 200); page.mouse.down(); page.mouse.move(gut["x"] + 120, gut["y"] + 200, steps=4); page.mouse.up()
        assert page.locator("#left").bounding_box()["width"] > left_before + 60
        right_before = page.locator("#right").bounding_box()["width"]
        gut = page.locator("#gutR").bounding_box()
        page.mouse.move(gut["x"] + 3, gut["y"] + 200); page.mouse.down(); page.mouse.move(gut["x"] + 100, gut["y"] + 200, steps=4); page.mouse.up()
        assert page.locator("#right").bounding_box()["width"] < right_before - 60
        # guide modal: opens page-size, has the verdict table, closes on Escape
        assert not page.locator("#guide").is_visible()
        page.locator("#guideBtn").click()
        assert page.locator("#guide").is_visible()
        box = page.locator("#guide .box").bounding_box()
        assert box["width"] > 1300 and box["height"] > 700
        assert page.locator("#guide h2").count() >= 9
        assert page.locator("#g-svd").count() == 1
        assert "saturated" in page.locator("#guide table").nth(2).inner_text() or "saturated" in page.locator("#guide").inner_text()
        page.keyboard.press("Escape")
        assert not page.locator("#guide").is_visible()
        # detail sections: present, collapsible, and their "?" opens the guide at the anchor
        assert page.locator("#right .sec").count() >= 6
        page.locator("#right .sec[data-sec='rank'] .sech h3").click()
        assert "closed" in page.locator("#right .sec[data-sec='rank']").get_attribute("class")
        page.locator("#right .sec[data-sec='rank'] .sech h3").click()
        page.locator("#right .sec[data-sec='sweep'] .help").click()
        assert page.locator("#guide").is_visible()
        assert page.evaluate("document.getElementById('g-sweep').getBoundingClientRect().top < 200")
        page.keyboard.press("Escape")
        # chart hover shows live values; sections reorder by drag and the order sticks
        sv = page.locator("#right .sec[data-sec='rank'] svg[data-chart]").first
        sv.scroll_into_view_if_needed()
        bb = sv.bounding_box(); page.mouse.move(bb["x"] + bb["width"] * 0.6, bb["y"] + bb["height"] * 0.5)
        assert page.locator("#tip").is_visible() and "σ[" in page.locator("#tip").inner_text()
        first = page.locator("#right .sec").first.get_attribute("data-sec")
        page.evaluate("""() => { const secs = [...document.querySelectorAll('#right .sec')]; const a = secs[0], b = secs[1];
          const dt = new DataTransfer();
          a.dispatchEvent(new DragEvent('dragstart', {bubbles: true, dataTransfer: dt}));
          const r = b.getBoundingClientRect();
          b.dispatchEvent(new DragEvent('dragover', {bubbles: true, dataTransfer: dt, clientX: r.left + r.width - 2, clientY: r.top + r.height - 2}));
          b.dispatchEvent(new DragEvent('drop', {bubbles: true, dataTransfer: dt, clientX: r.left + r.width - 2, clientY: r.top + r.height - 2}));
          a.dispatchEvent(new DragEvent('dragend', {bubbles: true, dataTransfer: dt})); }""")
        assert page.locator("#right .sec").first.get_attribute("data-sec") != first
        page.locator("table.mods tr[data-name='mp3']").click()          # re-render keeps the order
        assert page.locator("#right .sec").first.get_attribute("data-sec") != first
        page.locator("[data-secs='reset']").click()
        assert page.locator("#right .sec").first.get_attribute("data-sec") == first
        browser.close()
    assert not page_errors, page_errors
    assert not errors, errors


def test_gui_server_serves_report(report):
    url = report.show(open_browser=False, block=False)
    try:
        import urllib.request
        body = urllib.request.urlopen(url, timeout=5).read().decode("utf-8")
        assert "capscope" in body and "bottleneck" in body
        js = urllib.request.urlopen(url + "data.json", timeout=5).read().decode("utf-8")
        assert '"modules"' in js
    finally:
        report.close()
