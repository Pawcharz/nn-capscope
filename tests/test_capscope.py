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


def test_data_free_mode():
    model = ToyGraphNet()
    rep = inspect(model)             # weight spectra only, no loader
    assert "bottleneck" in rep
    assert rep["bottleneck"]["alpha"] is not None
    assert rep.to_json()


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
